//! The transformer: forward pass, hand-written backward pass, AdamW, sampling.
//!
//! Architecture is GPT-2 small-shaped: pre-LN blocks, causal multi-head
//! self-attention, GELU MLP with 4x expansion, learned position embeddings and
//! a tied output head (`logits = x @ wte^T`).
//!
//! Backprop is written by hand against an explicit activation cache instead of
//! using an autograd tape: it keeps allocations predictable and makes the hot
//! path readable. `gpt.test.gradients match finite differences` guards it.
const std = @import("std");
const tensor = @import("tensor.zig");
const model_mod = @import("model.zig");
const Model = model_mod.Model;

pub const OptState = struct {
    m: []f32,
    v: []f32,
    t: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, n: usize) !OptState {
        const m = try allocator.alloc(f32, n);
        errdefer allocator.free(m);
        const v = try allocator.alloc(f32, n);
        @memset(m, 0);
        @memset(v, 0);
        return .{ .m = m, .v = v };
    }

    pub fn deinit(self: *OptState, allocator: std.mem.Allocator) void {
        allocator.free(self.m);
        allocator.free(self.v);
    }
};

pub const SampleParams = struct {
    temperature: f32 = 0.8,
    top_k: usize = 40,
    seed: u64 = 0,
};

/// Per-layer key/value cache for autoregressive decoding.
///
/// K and V for already-seen positions never change, so decoding a new token only
/// computes one row of q/k/v and attends over the cache: O(T) instead of O(T²)
/// per token.
pub const KvCache = struct {
    k: [][]f32,
    v: [][]f32,
    pos: usize = 0,
    max_seq: usize,

    pub fn init(allocator: std.mem.Allocator, cfg: model_mod.Config) !KvCache {
        const L: usize = cfg.n_layer;
        const len: usize = cfg.block_size * cfg.n_embd;
        const k = try layerBufs(allocator, L, len);
        errdefer {
            for (k) |b| allocator.free(b);
            allocator.free(k);
        }
        const v = try layerBufs(allocator, L, len);
        return .{ .k = k, .v = v, .pos = 0, .max_seq = cfg.block_size };
    }

    pub fn deinit(self: *KvCache, allocator: std.mem.Allocator) void {
        for (self.k) |b| allocator.free(b);
        allocator.free(self.k);
        for (self.v) |b| allocator.free(b);
        allocator.free(self.v);
    }

    pub fn reset(self: *KvCache) void {
        self.pos = 0;
    }
};

/// Callback invoked for every generated token (for streaming UIs / CLI echo).
pub const TokenSink = struct {
    ctx: *anyopaque,
    emit: *const fn (ctx: *anyopaque, token: u32) void,

    pub fn send(self: TokenSink, token: u32) void {
        self.emit(self.ctx, token);
    }
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    batch: usize,
    max_seq: usize,

    grads: []f32,

    // per-layer activation cache
    xin: [][]f32,
    ln1y: [][]f32,
    ln1mean: [][]f32,
    ln1rstd: [][]f32,
    q: [][]f32,
    k: [][]f32,
    v: [][]f32,
    probs: [][]f32, // [B*H*T*T]
    ctx: [][]f32,
    ln2in: [][]f32,
    ln2y: [][]f32,
    ln2mean: [][]f32,
    ln2rstd: [][]f32,
    h1: [][]f32,
    h2: [][]f32,

    // globals
    x: []f32,
    lnfin: []f32,
    lnfy: []f32,
    lnfmean: []f32,
    lnfrstd: []f32,
    logits: []f32,
    dlogits: []f32,

    // scratch
    s1: []f32,
    s2: []f32,
    s3: []f32,
    s4: []f32,
    tmp_bt_c: []f32,
    tmp_t: []f32,
    dq: []f32,
    dk: []f32,
    dv: []f32,

    pub fn init(allocator: std.mem.Allocator, model: *Model, batch: usize) !Session {
        const cfg = model.cfg;
        const T: usize = cfg.block_size;
        const C: usize = cfg.n_embd;
        const M: usize = cfg.mlpDim();
        const V: usize = cfg.vocab_size;
        const B = batch;
        const BT = B * T;
        const L: usize = cfg.n_layer;

        const session = Session{
            .allocator = allocator,
            .model = model,
            .batch = batch,
            .max_seq = T,
            .grads = try allocator.alloc(f32, model.blob.len),
            .xin = try layerBufs(allocator, L, BT * C),
            .ln1y = try layerBufs(allocator, L, BT * C),
            .ln1mean = try layerBufs(allocator, L, BT),
            .ln1rstd = try layerBufs(allocator, L, BT),
            .q = try layerBufs(allocator, L, BT * C),
            .k = try layerBufs(allocator, L, BT * C),
            .v = try layerBufs(allocator, L, BT * C),
            .probs = try layerBufs(allocator, L, B * cfg.n_head * T * T),
            .ctx = try layerBufs(allocator, L, BT * C),
            .ln2in = try layerBufs(allocator, L, BT * C),
            .ln2y = try layerBufs(allocator, L, BT * C),
            .ln2mean = try layerBufs(allocator, L, BT),
            .ln2rstd = try layerBufs(allocator, L, BT),
            .h1 = try layerBufs(allocator, L, BT * M),
            .h2 = try layerBufs(allocator, L, BT * M),
            .x = try allocator.alloc(f32, BT * C),
            .lnfin = try allocator.alloc(f32, BT * C),
            .lnfy = try allocator.alloc(f32, BT * C),
            .lnfmean = try allocator.alloc(f32, BT),
            .lnfrstd = try allocator.alloc(f32, BT),
            .logits = try allocator.alloc(f32, BT * V),
            .dlogits = try allocator.alloc(f32, BT * V),
            .s1 = try allocator.alloc(f32, BT * @max(C, M)),
            .s2 = try allocator.alloc(f32, BT * @max(C, M)),
            .s3 = try allocator.alloc(f32, BT * @max(C, M)),
            .s4 = try allocator.alloc(f32, BT * @max(C, M)),
            .tmp_bt_c = try allocator.alloc(f32, BT * C),
            .tmp_t = try allocator.alloc(f32, T),
            .dq = try allocator.alloc(f32, BT * C),
            .dk = try allocator.alloc(f32, BT * C),
            .dv = try allocator.alloc(f32, BT * C),
        };
        @memset(session.grads, 0);
        return session;
    }

    pub fn deinit(self: *Session) void {
        const a = self.allocator;
        a.free(self.grads);
        for ([_][][]f32{ self.xin, self.ln1y, self.ln1mean, self.ln1rstd, self.q, self.k, self.v, self.probs, self.ctx, self.ln2in, self.ln2y, self.ln2mean, self.ln2rstd, self.h1, self.h2 }) |outer| {
            for (outer) |b| a.free(b);
            a.free(outer);
        }
        a.free(self.x);
        a.free(self.lnfin);
        a.free(self.lnfy);
        a.free(self.lnfmean);
        a.free(self.lnfrstd);
        a.free(self.logits);
        a.free(self.dlogits);
        a.free(self.s1);
        a.free(self.s2);
        a.free(self.s3);
        a.free(self.s4);
        a.free(self.tmp_bt_c);
        a.free(self.tmp_t);
        a.free(self.dq);
        a.free(self.dk);
        a.free(self.dv);
    }

    pub fn paramCount(self: *const Session) usize {
        return self.grads.len;
    }

    // -- forward ------------------------------------------------------------

    /// Run the transformer over `tokens` (batch*seq ids, row-major), filling
    /// `self.logits`. `T` may be smaller than the model's block size.
    pub fn forward(self: *Session, tokens: []const u32, T: usize) void {
        const model = self.model;
        const cfg = model.cfg;
        const C: usize = cfg.n_embd;
        const H: usize = cfg.n_head;
        const hd: usize = cfg.headDim();
        const V: usize = cfg.vocab_size;
        const M: usize = cfg.mlpDim();
        const L: usize = cfg.n_layer;
        const B = self.batch;
        const BT = B * T;
        std.debug.assert(tokens.len >= BT);
        std.debug.assert(T <= self.max_seq);

        const p = model.blob;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        // token + position embeddings
        for (0..BT) |r| {
            const tok: usize = tokens[r];
            std.debug.assert(tok < V);
            const row = self.x[r * C ..][0..C];
            @memcpy(row, p[model.layout.wte + tok * C ..][0..C]);
            const pos = r % T;
            const prow = p[model.layout.wpe + pos * C ..][0..C];
            for (row, prow) |*xv, pv| xv.* += pv;
        }

        for (0..L) |l| {
            const lo = model.layout.layers[l];
            const xn = self.x[0 .. BT * C];

            // --- attention block ---
            @memcpy(self.xin[l][0 .. BT * C], xn);
            tensor.layerNorm(
                xn,
                p[lo.ln1g..][0..C],
                p[lo.ln1b..][0..C],
                self.ln1y[l][0 .. BT * C],
                self.ln1mean[l][0..BT],
                self.ln1rstd[l][0..BT],
                BT,
                C,
            );
            tensor.matmul(BT, C, C, self.ln1y[l][0 .. BT * C], p[lo.wq..][0 .. C * C], self.q[l][0 .. BT * C]);
            tensor.matmul(BT, C, C, self.ln1y[l][0 .. BT * C], p[lo.wk..][0 .. C * C], self.k[l][0 .. BT * C]);
            tensor.matmul(BT, C, C, self.ln1y[l][0 .. BT * C], p[lo.wv..][0 .. C * C], self.v[l][0 .. BT * C]);

            for (0..B) |b| {
                for (0..H) |h| {
                    const off = h * hd;
                    const pbase = (b * H + h) * T * T;
                    for (0..T) |t| {
                        const r = b * T + t;
                        const qv = self.q[l][r * C + off ..][0..hd];
                        const prow = self.probs[l][pbase + t * T ..][0..T];
                        for (0..t + 1) |s| {
                            const kv = self.k[l][(b * T + s) * C + off ..][0..hd];
                            prow[s] = tensor.dot(qv, kv) * scale;
                        }
                        for (t + 1..T) |s| prow[s] = 0;
                        tensor.softmax(prow[0 .. t + 1]);
                        const cv = self.ctx[l][r * C + off ..][0..hd];
                        @memset(cv, 0);
                        for (0..t + 1) |s| {
                            const vv = self.v[l][(b * T + s) * C + off ..][0..hd];
                            const w = prow[s];
                            for (cv, vv) |*c, val| c.* += w * val;
                        }
                    }
                }
            }

            tensor.matmul(BT, C, C, self.ctx[l][0 .. BT * C], p[lo.wo..][0 .. C * C], self.s1[0 .. BT * C]);
            tensor.addInPlace(xn, self.s1[0 .. BT * C]);

            // --- mlp block ---
            @memcpy(self.ln2in[l][0 .. BT * C], xn);
            tensor.layerNorm(
                xn,
                p[lo.ln2g..][0..C],
                p[lo.ln2b..][0..C],
                self.ln2y[l][0 .. BT * C],
                self.ln2mean[l][0..BT],
                self.ln2rstd[l][0..BT],
                BT,
                C,
            );
            tensor.matmul(BT, M, C, self.ln2y[l][0 .. BT * C], p[lo.fc..][0 .. C * M], self.h1[l][0 .. BT * M]);
            @memcpy(self.h2[l][0 .. BT * M], self.h1[l][0 .. BT * M]);
            tensor.geluInPlace(self.h2[l][0 .. BT * M]);
            tensor.matmul(BT, C, M, self.h2[l][0 .. BT * M], p[lo.proj..][0 .. M * C], self.s1[0 .. BT * C]);
            tensor.addInPlace(xn, self.s1[0 .. BT * C]);
        }

        // final norm + tied head
        @memcpy(self.lnfin[0 .. BT * C], self.x[0 .. BT * C]);
        tensor.layerNorm(
            self.lnfin[0 .. BT * C],
            p[model.layout.lnfg..][0..C],
            p[model.layout.lnfb..][0..C],
            self.lnfy[0 .. BT * C],
            self.lnfmean[0..BT],
            self.lnfrstd[0..BT],
            BT,
            C,
        );
        tensor.matmulNt(BT, V, C, self.lnfy[0 .. BT * C], p[model.layout.wte..][0 .. V * C], self.logits[0 .. BT * V]);
    }

    /// Mean cross-entropy loss for the current batch (no gradients touched).
    pub fn lossOnly(self: *Session, tokens: []const u32, targets: []const u32, T: usize) f32 {
        self.forward(tokens, T);
        const V: usize = self.model.cfg.vocab_size;
        const BT = self.batch * T;
        var loss: f32 = 0;
        for (0..BT) |r| {
            const lr = self.logits[r * V ..][0..V];
            var max: f32 = lr[0];
            for (lr) |v| max = @max(max, v);
            var sum: f32 = 0;
            for (lr) |v| sum += @exp(v - max);
            loss += @log(sum) + max - lr[targets[r]];
        }
        return loss / @as(f32, @floatFromInt(BT));
    }

    /// Run `tokens` through the model and fill the KV cache with their keys and
    /// values, so the next `forwardStep` only has to look at one new token.
    /// Requires a batch-1 session.
    pub fn forwardPrefill(self: *Session, cache: *KvCache, tokens: []const u32, T: usize) void {
        std.debug.assert(self.batch == 1);
        std.debug.assert(T <= cache.max_seq);
        self.forward(tokens, T);
        const C: usize = self.model.cfg.n_embd;
        for (0..self.model.cfg.n_layer) |l| {
            @memcpy(cache.k[l][0 .. T * C], self.k[l][0 .. T * C]);
            @memcpy(cache.v[l][0 .. T * C], self.v[l][0 .. T * C]);
        }
        cache.pos = T;
    }

    /// Decode a single new token at position `pos` (== cache.pos), reading keys
    /// and values out of `cache`. Writes the next-token logits into
    /// `self.logits[0..vocab]`.
    pub fn forwardStep(self: *Session, cache: *KvCache, token: u32, pos: usize) void {
        const model = self.model;
        const cfg = model.cfg;
        const C: usize = cfg.n_embd;
        const H: usize = cfg.n_head;
        const hd: usize = cfg.headDim();
        const V: usize = cfg.vocab_size;
        const M: usize = cfg.mlpDim();
        const L: usize = cfg.n_layer;
        std.debug.assert(self.batch == 1);
        std.debug.assert(pos < cache.max_seq);

        const p = model.blob;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const x = self.x[0..C];

        // embedding for this single position
        std.debug.assert(token < V);
        @memcpy(x, p[model.layout.wte + @as(usize, token) * C ..][0..C]);
        const prow = p[model.layout.wpe + pos * C ..][0..C];
        for (x, prow) |*xv, pv| xv.* += pv;

        for (0..L) |l| {
            const lo = model.layout.layers[l];
            const ln1 = self.ln1y[l][0..C];
            tensor.layerNorm(x, p[lo.ln1g..][0..C], p[lo.ln1b..][0..C], ln1, self.ln1mean[l][0..1], self.ln1rstd[l][0..1], 1, C);

            const qrow = self.q[l][0..C];
            const krow = self.k[l][0..C];
            const vrow = self.v[l][0..C];
            tensor.matmul(1, C, C, ln1, p[lo.wq..][0 .. C * C], qrow);
            tensor.matmul(1, C, C, ln1, p[lo.wk..][0 .. C * C], krow);
            tensor.matmul(1, C, C, ln1, p[lo.wv..][0 .. C * C], vrow);
            @memcpy(cache.k[l][pos * C ..][0..C], krow);
            @memcpy(cache.v[l][pos * C ..][0..C], vrow);

            const ctxrow = self.ctx[l][0..C];
            for (0..H) |h| {
                const off = h * hd;
                const qv = qrow[off..][0..hd];
                const scores = self.tmp_t[0 .. pos + 1];
                for (0..pos + 1) |t| {
                    const kv = cache.k[l][t * C + off ..][0..hd];
                    scores[t] = tensor.dot(qv, kv) * scale;
                }
                tensor.softmax(scores);
                const cv = ctxrow[off..][0..hd];
                @memset(cv, 0);
                for (0..pos + 1) |t| {
                    const vv = cache.v[l][t * C + off ..][0..hd];
                    const w = scores[t];
                    for (cv, vv) |*c, val| c.* += w * val;
                }
            }

            tensor.matmul(1, C, C, ctxrow, p[lo.wo..][0 .. C * C], self.s1[0..C]);
            for (x, self.s1[0..C]) |*xv, sv| xv.* += sv;

            const ln2 = self.ln2y[l][0..C];
            tensor.layerNorm(x, p[lo.ln2g..][0..C], p[lo.ln2b..][0..C], ln2, self.ln2mean[l][0..1], self.ln2rstd[l][0..1], 1, C);
            const hrow = self.h2[l][0..M];
            tensor.matmul(1, M, C, ln2, p[lo.fc..][0 .. C * M], hrow);
            tensor.geluInPlace(hrow);
            tensor.matmul(1, C, M, hrow, p[lo.proj..][0 .. M * C], self.s1[0..C]);
            for (x, self.s1[0..C]) |*xv, sv| xv.* += sv;
        }

        const lnf = self.lnfy[0..C];
        tensor.layerNorm(x, p[model.layout.lnfg..][0..C], p[model.layout.lnfb..][0..C], lnf, self.lnfmean[0..1], self.lnfrstd[0..1], 1, C);
        tensor.matmulNt(1, V, C, lnf, p[model.layout.wte..][0 .. V * C], self.logits[0..V]);

        cache.pos = pos + 1;
    }

    // -- backward -----------------------------------------------------------

    pub fn zeroGrads(self: *Session) void {
        @memset(self.grads, 0);
    }

    /// Forward + backward. Gradients accumulate into `self.grads`, so call
    /// `zeroGrads` first for a clean step.
    pub fn fwdBwd(self: *Session, tokens: []const u32, targets: []const u32, T: usize) f32 {
        self.forward(tokens, T);
        const cfg = self.model.cfg;
        const C: usize = cfg.n_embd;
        const H: usize = cfg.n_head;
        const hd: usize = cfg.headDim();
        const V: usize = cfg.vocab_size;
        const M: usize = cfg.mlpDim();
        const L: usize = cfg.n_layer;
        const B = self.batch;
        const BT = B * T;

        const model = self.model;
        const p = model.blob;
        const g = self.grads;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        const loss = tensor.softmaxCrossEntropy(
            self.logits[0 .. BT * V],
            targets,
            self.dlogits[0 .. BT * V],
            BT,
            V,
        );

        // dx is the gradient of the residual stream; s2 = dx
        const dx = self.s2[0 .. BT * C];

        // tied head: dlnfy = dlogits @ wte ; dwte += dlogits^T @ lnfy
        tensor.matmul(BT, C, V, self.dlogits[0 .. BT * V], p[model.layout.wte..][0 .. V * C], self.s3[0 .. BT * C]);
        {
            const dwte = g[model.layout.wte..][0 .. V * C];
            for (0..BT) |r| {
                const dl = self.dlogits[r * V ..][0..V];
                const xr = self.lnfy[r * C ..][0..C];
                for (dl, 0..) |av, n| {
                    if (av == 0) continue;
                    const wrow = dwte[n * C ..][0..C];
                    for (wrow, xr) |*wv, xv| wv.* += av * xv;
                }
            }
        }

        // final layer norm
        @memset(dx, 0);
        tensor.layerNormBwd(
            self.s3[0 .. BT * C],
            self.lnfin[0 .. BT * C],
            p[model.layout.lnfg..][0..C],
            self.lnfmean[0..BT],
            self.lnfrstd[0..BT],
            dx,
            g[model.layout.lnfg..][0..C],
            g[model.layout.lnfb..][0..C],
            BT,
            C,
        );

        var l: usize = L;
        while (l > 0) {
            l -= 1;
            const lo = model.layout.layers[l];

            // ---- mlp branch ----
            const dh2 = self.s3[0 .. BT * M];
            const dh1 = self.s1[0 .. BT * M];
            @memset(dh2, 0);
            tensor.matmulNtAcc(BT, M, C, dx, p[lo.proj..][0 .. M * C], dh2);
            tensor.matmulTnAcc(M, C, BT, self.h2[l][0 .. BT * M], dx, g[lo.proj..][0 .. M * C]);
            @memset(dh1, 0);
            tensor.geluBwdAcc(dh2, self.h1[l][0 .. BT * M], dh1);

            tensor.matmulTnAcc(C, M, BT, self.ln2y[l][0 .. BT * C], dh1, g[lo.fc..][0 .. C * M]);
            const dln2y = self.s4[0 .. BT * C];
            @memset(dln2y, 0);
            tensor.matmulNtAcc(BT, C, M, dh1, p[lo.fc..][0 .. C * M], dln2y);

            @memset(self.tmp_bt_c, 0);
            tensor.layerNormBwd(
                dln2y,
                self.ln2in[l][0 .. BT * C],
                p[lo.ln2g..][0..C],
                self.ln2mean[l][0..BT],
                self.ln2rstd[l][0..BT],
                self.tmp_bt_c,
                g[lo.ln2g..][0..C],
                g[lo.ln2b..][0..C],
                BT,
                C,
            );
            tensor.addInPlace(dx, self.tmp_bt_c[0 .. BT * C]);

            // ---- attention branch ----
            const dctx = self.s4[0 .. BT * C];
            @memset(dctx, 0);
            tensor.matmulNtAcc(BT, C, C, dx, p[lo.wo..][0 .. C * C], dctx);
            tensor.matmulTnAcc(C, C, BT, self.ctx[l][0 .. BT * C], dx, g[lo.wo..][0 .. C * C]);

            const dq = self.dq;
            const dk = self.dk;
            const dv = self.dv;
            @memset(dq, 0);
            @memset(dk, 0);
            @memset(dv, 0);

            for (0..B) |b| {
                for (0..H) |h| {
                    const off = h * hd;
                    const pbase = (b * H + h) * T * T;
                    for (0..T) |t| {
                        const r = b * T + t;
                        const dOv = dctx[r * C + off ..][0..hd];
                        const prow = self.probs[l][pbase + t * T ..][0..T];
                        const dP = self.tmp_t[0..T];
                        for (0..t + 1) |s| {
                            const rr = b * T + s;
                            const dvs = dv[rr * C + off ..][0..hd];
                            const w = prow[s];
                            for (dvs, dOv) |*dvi, oi| dvi.* += w * oi;
                            dP[s] = tensor.dot(dOv, self.v[l][rr * C + off ..][0..hd]);
                        }
                        var sum: f32 = 0;
                        for (0..t + 1) |s| sum += prow[s] * dP[s];
                        const qv = self.q[l][r * C + off ..][0..hd];
                        const dqv = dq[r * C + off ..][0..hd];
                        for (0..t + 1) |s| {
                            const rr = b * T + s;
                            const ds = (dP[s] - sum) * prow[s] * scale;
                            const kv = self.k[l][rr * C + off ..][0..hd];
                            for (dqv, kv) |*dqi, ki| dqi.* += ds * ki;
                            const dkv = dk[rr * C + off ..][0..hd];
                            for (dkv, qv) |*dki, qi| dki.* += ds * qi;
                        }
                    }
                }
            }

            tensor.matmulTnAcc(C, C, BT, self.ln1y[l][0 .. BT * C], dq, g[lo.wq..][0 .. C * C]);
            tensor.matmulTnAcc(C, C, BT, self.ln1y[l][0 .. BT * C], dk, g[lo.wk..][0 .. C * C]);
            tensor.matmulTnAcc(C, C, BT, self.ln1y[l][0 .. BT * C], dv, g[lo.wv..][0 .. C * C]);

            const dln1y = self.s3[0 .. BT * C];
            @memset(dln1y, 0);
            tensor.matmulNtAcc(BT, C, C, dq, p[lo.wq..][0 .. C * C], dln1y);
            tensor.matmulNtAcc(BT, C, C, dk, p[lo.wk..][0 .. C * C], dln1y);
            tensor.matmulNtAcc(BT, C, C, dv, p[lo.wv..][0 .. C * C], dln1y);

            @memset(self.tmp_bt_c, 0);
            tensor.layerNormBwd(
                dln1y,
                self.xin[l][0 .. BT * C],
                p[lo.ln1g..][0..C],
                self.ln1mean[l][0..BT],
                self.ln1rstd[l][0..BT],
                self.tmp_bt_c,
                g[lo.ln1g..][0..C],
                g[lo.ln1b..][0..C],
                BT,
                C,
            );
            tensor.addInPlace(dx, self.tmp_bt_c[0 .. BT * C]);
        }

        // embedding grads
        {
            const dwte = g[model.layout.wte..][0 .. V * C];
            const dwpe = g[model.layout.wpe..][0 .. self.max_seq * C];
            for (0..BT) |r| {
                const tok: usize = tokens[r];
                const drow = dx[r * C ..][0..C];
                const wrow = dwte[tok * C ..][0..C];
                for (wrow, drow) |*wv, dv| wv.* += dv;
                const pos = r % T;
                const prow = dwpe[pos * C ..][0..C];
                for (prow, drow) |*wv, dv| wv.* += dv;
            }
            _ = self.s4;
        }

        return loss;
    }

    // -- optimisation -------------------------------------------------------

    pub fn gradNorm(self: *Session) f32 {
        var sum: f32 = 0;
        for (self.grads) |v| sum += v * v;
        return @sqrt(sum);
    }

    pub fn clipGrads(self: *Session, max_norm: f32) f32 {
        const norm = self.gradNorm();
        if (norm > max_norm and norm > 0) {
            const s = max_norm / norm;
            for (self.grads) |*v| v.* *= s;
        }
        return norm;
    }

    /// AdamW step: decoupled weight decay, bias-corrected moments.
    pub fn adamStep(self: *Session, opt: *OptState, lr: f32, weight_decay: f32, beta1: f32, beta2: f32) void {
        opt.t += 1;
        const t: f32 = @floatFromInt(opt.t);
        const bc1 = 1.0 - std.math.pow(f32, beta1, t);
        const bc2 = 1.0 - std.math.pow(f32, beta2, t);
        const eps: f32 = 1e-8;
        const params = self.model.blob;
        for (params, self.grads, opt.m, opt.v) |*pv, gv, *mv, *vv| {
            mv.* = beta1 * mv.* + (1.0 - beta1) * gv;
            vv.* = beta2 * vv.* + (1.0 - beta2) * gv * gv;
            const mhat = mv.* / bc1;
            const vhat = vv.* / bc2;
            pv.* -= lr * (mhat / (@sqrt(vhat) + eps) + weight_decay * pv.*);
        }
    }

    // -- sampling -----------------------------------------------------------

    /// Autoregressive generation with a KV cache.
    ///
    /// The prompt is prefilled in one batched pass, then each new token costs a
    /// single-row forward over the cached keys/values. When the context fills up
    /// the newest half is kept and prefilled again.
    pub fn generate(
        self: *Session,
        allocator: std.mem.Allocator,
        prompt: []const u32,
        max_new_tokens: usize,
        params: SampleParams,
        sink: ?TokenSink,
        out: *std.ArrayList(u32),
    ) !void {
        const cfg = self.model.cfg;
        const V: usize = cfg.vocab_size;
        const block: usize = cfg.block_size;
        std.debug.assert(self.batch == 1);
        if (prompt.len == 0) return error.EmptyPrompt;

        var rng = std.Random.DefaultPrng.init(params.seed);
        var cache = try KvCache.init(allocator, cfg);
        defer cache.deinit(allocator);

        const ctx_buf = try allocator.alloc(u32, block);
        defer allocator.free(ctx_buf);
        const cand = try allocator.alloc(f32, V);
        defer allocator.free(cand);
        const order = try allocator.alloc(u32, V);
        defer allocator.free(order);

        // prefill the tail of the prompt
        const pre_len = @min(prompt.len, block);
        @memcpy(ctx_buf[0..pre_len], prompt[prompt.len - pre_len ..]);
        var ctx_len: usize = pre_len;
        self.forwardPrefill(&cache, ctx_buf[0..ctx_len], ctx_len);
        var logits_row = self.logits[(ctx_len - 1) * V ..][0..V];

        for (0..max_new_tokens) |_| {
            if (cache.pos >= block) {
                // context full: keep the newest half, re-prefill, positions restart
                const keep = @max(block / 2, 1);
                std.mem.copyForwards(u32, ctx_buf[0..keep], ctx_buf[ctx_len - keep .. ctx_len]);
                ctx_len = keep;
                cache.reset();
                self.forwardPrefill(&cache, ctx_buf[0..ctx_len], ctx_len);
                logits_row = self.logits[(ctx_len - 1) * V ..][0..V];
            }

            @memcpy(cand, logits_row);
            const inv_temp = 1.0 / @max(params.temperature, 1e-5);
            for (cand) |*v| v.* *= inv_temp;
            const next = pickToken(&rng, cand, order, params.top_k);

            ctx_buf[ctx_len] = next;
            ctx_len += 1;
            try out.append(allocator, next);
            if (sink) |s| s.send(next);

            if (cache.pos < block) {
                self.forwardStep(&cache, next, cache.pos);
                logits_row = self.logits[0..V];
            }
        }
    }

    fn pickToken(rng: *std.Random.DefaultPrng, scores: []f32, order: []u32, top_k: usize) u32 {
        const r = rng.random();
        const vocab = scores.len;
        const k = if (top_k == 0) vocab else @min(top_k, vocab);

        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            s: []const f32,
            fn lessThan(self: @This(), a: u32, b: u32) bool {
                return self.s[a] > self.s[b];
            }
        };
        std.mem.sort(u32, order, Ctx{ .s = scores }, Ctx.lessThan);

        var total: f32 = 0;
        const best = order[0..k];
        const top = scores[best[0]];
        for (best) |i| {
            const e = @exp(scores[i] - top);
            scores[i] = e;
            total += e;
        }
        var target = r.float(f32) * total;
        for (best) |i| {
            target -= scores[i];
            if (target <= 0) return i;
        }
        return best[best.len - 1];
    }
};

fn layerBufs(allocator: std.mem.Allocator, n: usize, len: usize) ![][]f32 {
    const outer = try allocator.alloc([]f32, n);
    var i: usize = 0;
    errdefer {
        for (outer[0..i]) |b| allocator.free(b);
        allocator.free(outer);
    }
    while (i < n) : (i += 1) {
        outer[i] = try allocator.alloc(f32, len);
    }
    return outer;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

pub const tiny_test_cfg = model_mod.Config{
    .n_layer = 2,
    .n_head = 2,
    .n_embd = 16,
    .block_size = 8,
    .vocab_size = 8,
};

test "forward produces finite logits of the right shape" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var m = try Model.initRandom(allocator, tiny_test_cfg, "abcdefgh", 3);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 5;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 1, 2 };
    s.forward(&tokens, T);
    for (s.logits[0 .. 2 * T * tiny_test_cfg.vocab_size]) |v| {
        try testing.expect(std.math.isFinite(v));
    }
    try testing.expect(s.logits[0] != s.logits[1]);
}

test "gradients match finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 8, .block_size = 4, .vocab_size = 6 };
    var m = try Model.initRandom(allocator, cfg, "abcdef", 99);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 3;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 0 };

    s.zeroGrads();
    _ = s.fwdBwd(&tokens, &targets, T);

    // Probe a spread of parameters: embeddings, attention, mlp and layernorm.
    const indices = [_]usize{
        m.layout.wte + 1,
        m.layout.wpe + 1,
        m.layout.layers[0].wq + 3,
        m.layout.layers[0].fc + 5,
        m.layout.layers[0].proj + 7,
        m.layout.lnfg,
        m.layout.layers[0].ln1g + 1,
    };

    const h: f32 = 1e-2;
    for (indices) |idx| {
        const orig = m.blob[idx];
        m.blob[idx] = orig + h;
        const lp = s.lossOnly(&tokens, &targets, T);
        m.blob[idx] = orig - h;
        const lm = s.lossOnly(&tokens, &targets, T);
        m.blob[idx] = orig;

        const numeric = (lp - lm) / (2 * h);
        const analytic = s.grads[idx];
        const tol = 2e-2 * @max(@abs(numeric), 1e-2) + 1e-3;
        if (@abs(numeric - analytic) >= tol) {
            std.debug.print("gradient mismatch at {d}: numeric {d:.6} analytic {d:.6}\n", .{ idx, numeric, analytic });
            return error.GradientMismatch;
        }
    }
}

test "repeated backward passes are identical (scratch is not carried over)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 4, .vocab_size = 8 };
    var m = try Model.initRandom(allocator, cfg, "abcdefgh", 77);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 4;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0 };

    s.zeroGrads();
    const first_loss = s.fwdBwd(&tokens, &targets, T);
    const first = try allocator.dupe(f32, s.grads);
    defer allocator.free(first);

    var iter: usize = 0;
    while (iter < 4) : (iter += 1) {
        s.zeroGrads();
        const again = s.fwdBwd(&tokens, &targets, T);
        try testing.expectApproxEqAbs(first_loss, again, 1e-6);
        for (first, s.grads) |a, b| {
            try testing.expectApproxEqAbs(a, b, 1e-6 * @max(@abs(a), 1.0));
        }
    }
}

test "one optimisation step reduces the loss" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 16, .block_size = 8, .vocab_size = 6 };
    var m = try Model.initRandom(allocator, cfg, "abcdef", 5);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 2);
    defer s.deinit();
    var opt = try OptState.init(allocator, m.blob.len);
    defer opt.deinit(allocator);

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4 };

    const before = s.lossOnly(&tokens, &targets, 8);
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        s.zeroGrads();
        _ = s.fwdBwd(&tokens, &targets, 8);
        _ = s.clipGrads(1.0);
        s.adamStep(&opt, 3e-3, 0.0, 0.9, 0.99);
    }
    const after = s.lossOnly(&tokens, &targets, 8);
    try testing.expect(after < before);
}

test "cached decoding matches a full forward pass (greedy)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 16, .vocab_size = 8 };
    var m = try Model.initRandom(allocator, cfg, "abcdefgh", 21);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 1);
    defer s.deinit();

    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const n_new = 10;

    // cached path (what the CLI and server use)
    var cached: std.ArrayList(u32) = .empty;
    defer cached.deinit(allocator);
    try s.generate(allocator, &prompt, n_new, .{ .temperature = 0.0, .top_k = 0, .seed = 1 }, null, &cached);
    try testing.expectEqual(@as(usize, n_new), cached.items.len);

    // reference path: re-run the whole context for every token
    var ctx: std.ArrayList(u32) = .empty;
    defer ctx.deinit(allocator);
    try ctx.appendSlice(allocator, &prompt);
    var reference: std.ArrayList(u32) = .empty;
    defer reference.deinit(allocator);
    for (0..n_new) |_| {
        s.forward(ctx.items, ctx.items.len);
        const row = s.logits[(ctx.items.len - 1) * cfg.vocab_size ..][0..cfg.vocab_size];
        var best: u32 = 0;
        for (row, 0..) |v, i| {
            if (v > row[best]) best = @intCast(i);
        }
        try reference.append(allocator, best);
        try ctx.append(allocator, best);
    }

    try testing.expectEqualSlices(u32, reference.items, cached.items);
}

test "generate streams the requested number of tokens" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 8, .block_size = 8, .vocab_size = 6 };
    var m = try Model.initRandom(allocator, cfg, "abcdef", 11);
    defer m.deinit();
    var s = try Session.init(allocator, &m, 1);
    defer s.deinit();

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(allocator);
    try s.generate(allocator, &[_]u32{ 1, 2, 3 }, 12, .{ .temperature = 0.9, .top_k = 6, .seed = 4 }, null, &out);
    try testing.expectEqual(@as(usize, 12), out.items.len);
    for (out.items) |t| try testing.expect(t < cfg.vocab_size);
}

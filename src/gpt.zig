//! The transformer, now expressed on Fucina's eager autograd runtime.
//!
//! Architecture is unchanged: GPT-2 small-shaped, pre-LN blocks, causal
//! multi-head self-attention, GELU MLP with 4x expansion, learned position
//! embeddings and a tied output head (`logits = x @ wte^T`).
//!
//! Every parameter is a Fucina `variable` that borrows its slice of
//! `Model.blob` in place, so the `.vzm` checkpoint stays the single source of
//! truth: the optimizer updates the blob directly and `Model.save` needs no
//! copy-back. Forward, backward and AdamW are Fucina's; this file only states
//! the block wiring.
const std = @import("std");
const fucina = @import("fucina");
const model_mod = @import("model.zig");
const Model = model_mod.Model;

const ExecContext = fucina.ExecContext;
const optim = fucina.optim;

/// LayerNorm epsilon, matching the original local kernels.
pub const eps: f32 = 1e-5;

const Emb = fucina.Tensor(.{ .vocab, .model });
const PosEmb = fucina.Tensor(.{ .pos, .model });
/// Any 2-D weight, stored [in, out] as in the `.vzm` registry.
const Mat = fucina.Tensor(.{ .in, .out });
const Norm = fucina.Tensor(.{.model});
const Stream = fucina.Tensor(.{ .seq, .model });
const AttnStream = fucina.Tensor(.{ .seq, .attn });
const Logits = fucina.Tensor(.{ .seq, .vocab });

pub const SampleParams = struct {
    temperature: f32 = 0.8,
    top_k: usize = 40,
    seed: u64 = 0,
};

/// Per-layer key/value cache for autoregressive decoding.
///
/// K and V for already-seen positions never change, so decoding a new token only
/// computes one row of q/k/v and attends over the cache: O(T) instead of O(T²)
/// per token. The cache is plain host memory; Fucina reads it through a
/// borrowed zero-copy view.
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

/// One transformer layer's trainable leaves, borrowed from the model blob.
const LayerParams = struct {
    ln1g: Norm,
    ln1b: Norm,
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    ln2g: Norm,
    ln2b: Norm,
    fc: Mat,
    proj: Mat,

    fn init(ctx: *ExecContext, model: *const Model, l: usize) !LayerParams {
        const cfg = model.cfg;
        const c: usize = cfg.n_embd;
        const m: usize = cfg.mlpDim();
        const lo = model.layout.layers[l];
        const b = model.blob;

        var ln1g = try varFrom(Norm, ctx, .{c}, b, lo.ln1g, c);
        errdefer ln1g.deinit();
        var ln1b = try varFrom(Norm, ctx, .{c}, b, lo.ln1b, c);
        errdefer ln1b.deinit();
        var wq = try varFrom(Mat, ctx, .{ c, c }, b, lo.wq, c * c);
        errdefer wq.deinit();
        var wk = try varFrom(Mat, ctx, .{ c, c }, b, lo.wk, c * c);
        errdefer wk.deinit();
        var wv = try varFrom(Mat, ctx, .{ c, c }, b, lo.wv, c * c);
        errdefer wv.deinit();
        var wo = try varFrom(Mat, ctx, .{ c, c }, b, lo.wo, c * c);
        errdefer wo.deinit();
        var ln2g = try varFrom(Norm, ctx, .{c}, b, lo.ln2g, c);
        errdefer ln2g.deinit();
        var ln2b = try varFrom(Norm, ctx, .{c}, b, lo.ln2b, c);
        errdefer ln2b.deinit();
        var fc = try varFrom(Mat, ctx, .{ c, m }, b, lo.fc, c * m);
        errdefer fc.deinit();
        var proj = try varFrom(Mat, ctx, .{ m, c }, b, lo.proj, m * c);
        errdefer proj.deinit();

        return .{
            .ln1g = ln1g,
            .ln1b = ln1b,
            .wq = wq,
            .wk = wk,
            .wv = wv,
            .wo = wo,
            .ln2g = ln2g,
            .ln2b = ln2b,
            .fc = fc,
            .proj = proj,
        };
    }

    fn deinit(self: *LayerParams) void {
        self.proj.deinit();
        self.fc.deinit();
        self.ln2b.deinit();
        self.ln2g.deinit();
        self.wo.deinit();
        self.wv.deinit();
        self.wk.deinit();
        self.wq.deinit();
        self.ln1b.deinit();
        self.ln1g.deinit();
    }
};

const Params = struct {
    wte: Emb,
    wpe: PosEmb,
    lnfg: Norm,
    lnfb: Norm,
    layers: []LayerParams,

    fn init(allocator: std.mem.Allocator, ctx: *ExecContext, model: *const Model) !Params {
        const cfg = model.cfg;
        const c: usize = cfg.n_embd;
        const v: usize = cfg.vocab_size;
        const t: usize = cfg.block_size;
        const b = model.blob;

        var wte = try varFrom(Emb, ctx, .{ v, c }, b, model.layout.wte, v * c);
        errdefer wte.deinit();
        var wpe = try varFrom(PosEmb, ctx, .{ t, c }, b, model.layout.wpe, t * c);
        errdefer wpe.deinit();
        var lnfg = try varFrom(Norm, ctx, .{c}, b, model.layout.lnfg, c);
        errdefer lnfg.deinit();
        var lnfb = try varFrom(Norm, ctx, .{c}, b, model.layout.lnfb, c);
        errdefer lnfb.deinit();

        const layers = try allocator.alloc(LayerParams, cfg.n_layer);
        var built: usize = 0;
        errdefer {
            for (layers[0..built]) |*lp| lp.deinit();
            allocator.free(layers);
        }
        while (built < layers.len) : (built += 1) {
            layers[built] = try LayerParams.init(ctx, model, built);
        }

        return .{ .wte = wte, .wpe = wpe, .lnfg = lnfg, .lnfb = lnfb, .layers = layers };
    }

    fn deinit(self: *Params) void {
        // The slice itself is freed by the owning Session.
        for (self.layers) |*lp| lp.deinit();
        self.lnfb.deinit();
        self.lnfg.deinit();
        self.wpe.deinit();
        self.wte.deinit();
    }
};

/// A training or inference session over one `Model`.
///
/// Heap-allocated because Fucina's `ExecContext` is pinned after `init` (it is
/// self-referential) and the optimizer retains pointers to the parameter
/// fields; returning it by value would move both. The model must outlive the
/// session: every parameter borrows a slice of `model.blob`.
pub const Session = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    batch: usize,
    max_seq: usize,

    ctx: ExecContext = undefined,
    params: Params = undefined,
    adam: optim.AdamW = undefined,
    head_map: []usize = &.{},
    /// Next-token logits for the last `forwardStep` call.
    logits: []f32 = &.{},

    pub fn init(allocator: std.mem.Allocator, model: *Model, batch: usize) !*Session {
        const self = try allocator.create(Session);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .model = model,
            .batch = batch,
            .max_seq = model.cfg.block_size,
        };

        self.ctx.init(allocator);
        errdefer self.ctx.deinit();

        self.params = try Params.init(allocator, &self.ctx, model);
        errdefer {
            self.params.deinit();
            allocator.free(self.params.layers);
        }

        self.adam = try optim.AdamW.init(allocator, .{
            .lr = 3e-3,
            .beta1 = 0.9,
            .beta2 = 0.99,
            .weight_decay = 0.01,
        });
        errdefer self.adam.deinit();
        try self.registerParams();

        self.head_map = try allocator.alloc(usize, model.cfg.n_head);
        errdefer allocator.free(self.head_map);
        for (self.head_map, 0..) |*h, i| h.* = i;

        self.logits = try allocator.alloc(f32, model.cfg.vocab_size);
        return self;
    }

    pub fn deinit(self: *Session) void {
        const a = self.allocator;
        self.adam.deinit();
        self.params.deinit();
        a.free(self.params.layers);
        a.free(self.head_map);
        a.free(self.logits);
        self.ctx.deinit();
        a.destroy(self);
    }

    fn registerParams(self: *Session) !void {
        const p = &self.params;
        try self.adam.addParam(&p.wte);
        try self.adam.addParam(&p.wpe);
        try self.adam.addParam(&p.lnfg);
        try self.adam.addParam(&p.lnfb);
        for (p.layers) |*lp| {
            try self.adam.addParam(&lp.ln1g);
            try self.adam.addParam(&lp.ln1b);
            try self.adam.addParam(&lp.wq);
            try self.adam.addParam(&lp.wk);
            try self.adam.addParam(&lp.wv);
            try self.adam.addParam(&lp.wo);
            try self.adam.addParam(&lp.ln2g);
            try self.adam.addParam(&lp.ln2b);
            try self.adam.addParam(&lp.fc);
            try self.adam.addParam(&lp.proj);
        }
    }

    pub fn paramCount(self: *const Session) usize {
        return self.model.blob.len;
    }

    // -- forward ------------------------------------------------------------

    /// Run the transformer over `tokens` (batch*seq ids, row-major) and return
    /// `[batch*seq, vocab]` logits. Must be called inside an exec scope.
    fn forwardTrain(self: *Session, tokens: []const u32, T: usize) !Logits {
        const cfg = self.model.cfg;
        const BT = self.batch * T;
        std.debug.assert(tokens.len >= BT);
        std.debug.assert(T <= self.max_seq);

        const alloc = self.ctx.allocator();
        const tok_idx = try alloc.alloc(usize, BT);
        defer alloc.free(tok_idx);
        const pos_idx = try alloc.alloc(usize, BT);
        defer alloc.free(pos_idx);
        for (0..BT) |r| {
            std.debug.assert(tokens[r] < cfg.vocab_size);
            tok_idx[r] = tokens[r];
            pos_idx[r] = r % T;
        }

        var x = try self.params.wte.gather(&self.ctx, .vocab, tok_idx, .seq);
        const pe = try self.params.wpe.gather(&self.ctx, .pos, pos_idx, .seq);
        x = try x.add(&self.ctx, &pe);

        for (0..cfg.n_layer) |l| {
            x = try self.forwardBlock(l, &x, T);
        }

        const nf = try x.layerNorm(&self.ctx, .model, eps, .{
            .weight = &self.params.lnfg,
            .bias = &self.params.lnfb,
        });
        return nf.matmul(&self.ctx, &self.params.wte, .trans_b, .{ .seq, .vocab });
    }

    fn forwardBlock(self: *Session, l: usize, x: *const Stream, T: usize) !Stream {
        const lp = &self.params.layers[l];
        const ctx = &self.ctx;

        const n1 = try x.layerNorm(ctx, .model, eps, .{ .weight = &lp.ln1g, .bias = &lp.ln1b });
        const q = try n1.matmul(ctx, &lp.wq, .plain, .{ .seq, .model });
        const k = try n1.matmul(ctx, &lp.wk, .plain, .{ .seq, .model });
        const v = try n1.matmul(ctx, &lp.wv, .plain, .{ .seq, .model });

        const attn = try self.attention(&q, &k, &v, T);
        const o = try attn.matmul(ctx, &lp.wo, .plain, .{ .seq, .model });
        const x1 = try x.add(ctx, &o);

        const n2 = try x1.layerNorm(ctx, .model, eps, .{ .weight = &lp.ln2g, .bias = &lp.ln2b });
        const h = try n2.matmul(ctx, &lp.fc, .plain, .{ .seq, .hidden });
        const hg = try h.gelu(ctx);
        const p = try hg.matmul(ctx, &lp.proj, .plain, .{ .seq, .model });
        return x1.add(ctx, &p);
    }

    /// Causal attention for every batch row, then re-concatenated along seq.
    fn attention(self: *Session, q: *const Stream, k: *const Stream, v: *const Stream, T: usize) !AttnStream {
        const cfg = self.model.cfg;
        const H: usize = cfg.n_head;
        const hd: usize = cfg.headDim();
        const B = self.batch;
        const alloc = self.ctx.allocator();
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

        const parts = try alloc.alloc(AttnStream, B);
        defer alloc.free(parts);
        for (0..B) |b| {
            const qb = try q.narrow(&self.ctx, .seq, b * T, T);
            const kb = try k.narrow(&self.ctx, .seq, b * T, T);
            const vb = try v.narrow(&self.ctx, .seq, b * T, T);
            const q3 = try qb.split(&self.ctx, .model, .{ .head, .d }, .{ H, hd });
            const k3 = try kb.split(&self.ctx, .model, .{ .kv_head, .d }, .{ H, hd });
            const v3 = try vb.split(&self.ctx, .model, .{ .kv_head, .d }, .{ H, hd });
            parts[b] = try q3.groupedAttention(&self.ctx, &k3, &v3, self.head_map, .attn, scale, .{});
        }
        if (B == 1) return parts[0];

        const refs = try alloc.alloc(*const AttnStream, B - 1);
        defer alloc.free(refs);
        for (1..B) |i| refs[i - 1] = &parts[i];
        return parts[0].concat(&self.ctx, .seq, refs);
    }

    /// Mean cross-entropy loss for the current batch (no gradients touched).
    pub fn lossOnly(self: *Session, tokens: []const u32, targets: []const u32, T: usize) !f32 {
        const scope = self.ctx.openExecScope();
        defer self.ctx.closeExecScope(scope);

        const logits = try self.forwardTrain(tokens, T);
        const label_ids = try self.labels(targets, self.batch * T);
        defer self.ctx.allocator().free(label_ids);
        var loss = try logits.crossEntropy(&self.ctx, .vocab, label_ids, .{});
        return loss.item();
    }

    /// Copy of the `[batch*seq, vocab]` logits (used by the cached-decoding
    /// parity test and other diagnostics).
    pub fn logitsAlloc(self: *Session, allocator: std.mem.Allocator, tokens: []const u32, T: usize) ![]f32 {
        const scope = self.ctx.openExecScope();
        defer self.ctx.closeExecScope(scope);
        const logits = try self.forwardTrain(tokens, T);
        return allocator.dupe(f32, try logits.dataConst());
    }

    fn labels(self: *Session, targets: []const u32, n: usize) ![]usize {
        std.debug.assert(targets.len >= n);
        const out = try self.ctx.allocator().alloc(usize, n);
        for (0..n) |i| out[i] = targets[i];
        return out;
    }

    /// Run `tokens` through the model and fill the KV cache with their keys and
    /// values, so the next `forwardStep` only has to look at one new token.
    pub fn forwardPrefill(self: *Session, cache: *KvCache, tokens: []const u32, T: usize) !void {
        std.debug.assert(T <= cache.max_seq);
        for (tokens[0..T], 0..) |tok, i| {
            try self.forwardStep(cache, tok, i);
        }
    }

    /// Decode a single new token at position `pos` (== cache.pos), reading keys
    /// and values out of `cache`. Writes the next-token logits into
    /// `self.logits[0..vocab]`.
    pub fn forwardStep(self: *Session, cache: *KvCache, token: u32, pos: usize) !void {
        const cfg = self.model.cfg;
        const C: usize = cfg.n_embd;
        const H: usize = cfg.n_head;
        const hd: usize = cfg.headDim();
        std.debug.assert(pos < cache.max_seq);
        std.debug.assert(token < cfg.vocab_size);

        const scope = self.ctx.openExecScope();
        defer self.ctx.closeExecScope(scope);
        var no_grad = fucina.noGrad();
        defer no_grad.close();

        const ctx = &self.ctx;
        const p = &self.params;
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const tok_idx = [_]usize{token};
        const pos_idx = [_]usize{pos};

        var x = try p.wte.gather(ctx, .vocab, &tok_idx, .seq);
        const pe = try p.wpe.gather(ctx, .pos, &pos_idx, .seq);
        x = try x.add(ctx, &pe);

        for (0..cfg.n_layer) |l| {
            const lp = &p.layers[l];
            const n1 = try x.layerNorm(ctx, .model, eps, .{ .weight = &lp.ln1g, .bias = &lp.ln1b });
            const q = try n1.matmul(ctx, &lp.wq, .plain, .{ .seq, .model });
            const k = try n1.matmul(ctx, &lp.wk, .plain, .{ .seq, .model });
            const v = try n1.matmul(ctx, &lp.wv, .plain, .{ .seq, .model });

            const q3 = try q.split(ctx, .model, .{ .head, .d }, .{ H, hd });
            const k3 = try k.split(ctx, .model, .{ .kv_head, .d }, .{ H, hd });
            const v3 = try v.split(ctx, .model, .{ .kv_head, .d }, .{ H, hd });
            @memcpy(cache.k[l][pos * C ..][0..C], try k3.dataConst());
            @memcpy(cache.v[l][pos * C ..][0..C], try v3.dataConst());

            var kview = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(
                ctx,
                .{ pos + 1, H, hd },
                cache.k[l][0 .. (pos + 1) * C],
            );
            defer kview.deinit();
            var vview = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromBorrowedConstSlice(
                ctx,
                .{ pos + 1, H, hd },
                cache.v[l][0 .. (pos + 1) * C],
            );
            defer vview.deinit();

            const attn = try q3.groupedAttention(ctx, &kview, &vview, self.head_map, .attn, scale, .{});
            const o = try attn.matmul(ctx, &lp.wo, .plain, .{ .seq, .model });
            const x1 = try x.add(ctx, &o);

            const n2 = try x1.layerNorm(ctx, .model, eps, .{ .weight = &lp.ln2g, .bias = &lp.ln2b });
            const h = try n2.matmul(ctx, &lp.fc, .plain, .{ .seq, .hidden });
            const hg = try h.gelu(ctx);
            const pr = try hg.matmul(ctx, &lp.proj, .plain, .{ .seq, .model });
            x = try x1.add(ctx, &pr);
        }

        const nf = try x.layerNorm(ctx, .model, eps, .{ .weight = &p.lnfg, .bias = &p.lnfb });
        const logits = try nf.matmul(ctx, &p.wte, .trans_b, .{ .seq, .vocab });
        @memcpy(self.logits, try logits.dataConst());

        cache.pos = pos + 1;
    }

    // -- backward -----------------------------------------------------------

    pub fn zeroGrads(self: *Session) void {
        self.adam.zeroGrad();
    }

    /// Forward + backward. Gradients accumulate into the parameter leaves, so
    /// call `zeroGrads` first for a clean step.
    pub fn fwdBwd(self: *Session, tokens: []const u32, targets: []const u32, T: usize) !f32 {
        const scope = self.ctx.openExecScope();
        defer self.ctx.closeExecScope(scope);

        const logits = try self.forwardTrain(tokens, T);
        const label_ids = try self.labels(targets, self.batch * T);
        defer self.ctx.allocator().free(label_ids);
        var loss = try logits.crossEntropy(&self.ctx, .vocab, label_ids, .{});
        try loss.backward(&self.ctx);
        return loss.item();
    }

    // -- optimisation -------------------------------------------------------

    pub fn gradNorm(self: *Session) !f32 {
        return @floatCast(@sqrt(try self.adam.gradSquaredNorm(&self.ctx)));
    }

    pub fn clipGrads(self: *Session, max_norm: f32) !f32 {
        return self.adam.clipGradNorm(&self.ctx, max_norm);
    }

    /// AdamW step with decoupled weight decay.
    pub fn adamStep(self: *Session, lr: f32, weight_decay: f32, beta1: f32, beta2: f32) !void {
        self.adam.config.lr = lr;
        self.adam.config.weight_decay = weight_decay;
        self.adam.config.beta1 = beta1;
        self.adam.config.beta2 = beta2;
        try self.adam.step(&self.ctx);
    }

    // -- sampling -----------------------------------------------------------

    /// Autoregressive generation with a KV cache.
    ///
    /// The prompt is prefilled one token at a time (each step fills one cache
    /// row), then each new token costs a single-row forward over the cached
    /// keys/values. When the context fills up the newest half is kept and
    /// replayed.
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
        try self.forwardPrefill(&cache, ctx_buf[0..ctx_len], ctx_len);
        var logits_row = self.logits[0..V];

        for (0..max_new_tokens) |_| {
            if (cache.pos >= block) {
                // context full: keep the newest half, re-prefill, positions restart
                const keep = @max(block / 2, 1);
                std.mem.copyForwards(u32, ctx_buf[0..keep], ctx_buf[ctx_len - keep .. ctx_len]);
                ctx_len = keep;
                cache.reset();
                try self.forwardPrefill(&cache, ctx_buf[0..ctx_len], ctx_len);
                logits_row = self.logits[0..V];
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
                try self.forwardStep(&cache, next, cache.pos);
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

fn varFrom(comptime T: type, ctx: *ExecContext, shape: anytype, blob: []f32, offset: usize, len: usize) !T {
    var raw = try ctx.fromBorrowedSlice(.f32, shape, blob[offset..][0..len]);
    errdefer raw.deinit();
    return T.variable(ctx, raw);
}

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

    var m = try Model.initRandomChar(allocator, tiny_test_cfg, "abcdefgh", 3);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 5;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 1, 2 };
    const logits = try s.logitsAlloc(allocator, &tokens, T);
    defer allocator.free(logits);
    try testing.expectEqual(@as(usize, 2 * T * tiny_test_cfg.vocab_size), logits.len);
    for (logits) |v| try testing.expect(std.math.isFinite(v));
    try testing.expect(logits[0] != logits[1]);
}

test "gradients match finite differences" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 8, .block_size = 4, .vocab_size = 6 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdef", 99);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 3;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 0 };

    s.zeroGrads();
    _ = try s.fwdBwd(&tokens, &targets, T);

    const Probe = struct { idx: usize, grad: f32 };
    var probes: [7]Probe = undefined;
    {
        var g = (try s.params.wte.grad(&s.ctx)).?;
        defer g.deinit();
        probes[0] = .{ .idx = m.layout.wte + 1, .grad = (try g.dataConst())[1] };
    }
    {
        var g = (try s.params.wpe.grad(&s.ctx)).?;
        defer g.deinit();
        probes[1] = .{ .idx = m.layout.wpe + 1, .grad = (try g.dataConst())[1] };
    }
    {
        var g = (try s.params.layers[0].wq.grad(&s.ctx)).?;
        defer g.deinit();
        probes[2] = .{ .idx = m.layout.layers[0].wq + 3, .grad = (try g.dataConst())[3] };
    }
    {
        var g = (try s.params.layers[0].fc.grad(&s.ctx)).?;
        defer g.deinit();
        probes[3] = .{ .idx = m.layout.layers[0].fc + 5, .grad = (try g.dataConst())[5] };
    }
    {
        var g = (try s.params.layers[0].proj.grad(&s.ctx)).?;
        defer g.deinit();
        probes[4] = .{ .idx = m.layout.layers[0].proj + 7, .grad = (try g.dataConst())[7] };
    }
    {
        var g = (try s.params.lnfg.grad(&s.ctx)).?;
        defer g.deinit();
        probes[5] = .{ .idx = m.layout.lnfg, .grad = (try g.dataConst())[0] };
    }
    {
        var g = (try s.params.layers[0].ln1g.grad(&s.ctx)).?;
        defer g.deinit();
        probes[6] = .{ .idx = m.layout.layers[0].ln1g + 1, .grad = (try g.dataConst())[1] };
    }

    const h: f32 = 1e-2;
    for (probes) |p| {
        const orig = m.blob[p.idx];
        m.blob[p.idx] = orig + h;
        const lp = try s.lossOnly(&tokens, &targets, T);
        m.blob[p.idx] = orig - h;
        const lm = try s.lossOnly(&tokens, &targets, T);
        m.blob[p.idx] = orig;

        const numeric = (lp - lm) / (2 * h);
        const tol = 2e-2 * @max(@abs(numeric), 1e-2) + 1e-3;
        if (@abs(numeric - p.grad) >= tol) {
            std.debug.print("gradient mismatch at {d}: numeric {d:.6} analytic {d:.6}\n", .{ p.idx, numeric, p.grad });
            return error.GradientMismatch;
        }
    }
}

test "repeated backward passes are identical (scratch is not carried over)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 4, .vocab_size = 8 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdefgh", 77);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const T = 4;
    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 0 };

    s.zeroGrads();
    const first_loss = try s.fwdBwd(&tokens, &targets, T);
    const first_norm = try s.gradNorm();

    var iter: usize = 0;
    while (iter < 4) : (iter += 1) {
        s.zeroGrads();
        const again = try s.fwdBwd(&tokens, &targets, T);
        try testing.expectApproxEqAbs(first_loss, again, 1e-6);
        try testing.expectApproxEqAbs(first_norm, try s.gradNorm(), 1e-6 * @max(first_norm, 1.0));
    }
}

test "one optimisation step reduces the loss" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 16, .block_size = 8, .vocab_size = 6 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdef", 5);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 2);
    defer s.deinit();

    const tokens = [_]u32{ 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3 };
    const targets = [_]u32{ 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4 };

    const before = try s.lossOnly(&tokens, &targets, 8);
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        s.zeroGrads();
        _ = try s.fwdBwd(&tokens, &targets, 8);
        _ = try s.clipGrads(1.0);
        try s.adamStep(3e-3, 0.0, 0.9, 0.99);
    }
    const after = try s.lossOnly(&tokens, &targets, 8);
    try testing.expect(after < before);
}

test "cached decoding matches a full forward pass (greedy)" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 16, .vocab_size = 8 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdefgh", 21);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 1);
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
        const logits = try s.logitsAlloc(allocator, ctx.items, ctx.items.len);
        defer allocator.free(logits);
        const row = logits[(ctx.items.len - 1) * cfg.vocab_size ..][0..cfg.vocab_size];
        var best: u32 = 0;
        for (row, 0..) |v, i| {
            if (v > row[best]) best = @intCast(i);
        }
        try reference.append(allocator, best);
        try ctx.append(allocator, best);
    }

    try testing.expectEqualSlices(u32, reference.items, cached.items);
}

test "a training-batch session can still sample" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 8, .block_size = 8, .vocab_size = 6 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdef", 13);
    defer m.deinit();
    // Decode always uses the single-row path, even on a session sized for
    // training batches (the CLI samples during training this way).
    const s = try Session.init(allocator, &m, 4);
    defer s.deinit();

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(allocator);
    try s.generate(allocator, &[_]u32{ 1, 2 }, 5, .{ .temperature = 0.8, .top_k = 4, .seed = 2 }, null, &out);
    try testing.expectEqual(@as(usize, 5), out.items.len);
    for (out.items) |t| try testing.expect(t < cfg.vocab_size);
}

test "generate streams the requested number of tokens" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = model_mod.Config{ .n_layer = 1, .n_head = 2, .n_embd = 8, .block_size = 8, .vocab_size = 6 };
    var m = try Model.initRandomChar(allocator, cfg, "abcdef", 11);
    defer m.deinit();
    const s = try Session.init(allocator, &m, 1);
    defer s.deinit();

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(allocator);
    try s.generate(allocator, &[_]u32{ 1, 2, 3 }, 12, .{ .temperature = 0.9, .top_k = 6, .seed = 4 }, null, &out);
    try testing.expectEqual(@as(usize, 12), out.items.len);
    for (out.items) |t| try testing.expect(t < cfg.vocab_size);
}

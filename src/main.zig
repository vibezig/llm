//! `llm` CLI: generate tensor models, train them, run inference, serve the web UI.
const std = @import("std");
const llm = @import("llm");
const model_mod = llm.model;
const gpt = llm.gpt;
const tokenizer = llm.tokenizer;

const default_data = "data/input.txt";
const default_model = "models/tiny.vzm";

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [16 * 1024]u8 = undefined;
    var out_file = std.Io.File.stdout().writer(io, &out_buf);
    const out = &out_file.interface;

    var err_buf: [4096]u8 = undefined;
    var err_file = std.Io.File.stderr().writer(io, &err_buf);
    const err = &err_file.interface;

    const cmd: []const u8 = if (args.len > 1) args[1] else "help";
    const rest = if (args.len > 2) args[2..] else args[0..0];
    const flags = Flags{ .args = rest };

    if (std.mem.eql(u8, cmd, "help") or std.mem.eql(u8, cmd, "--help") or std.mem.eql(u8, cmd, "-h")) {
        try printHelp(out);
    } else if (std.mem.eql(u8, cmd, "gen")) {
        try cmdGen(gpa, io, flags, out);
    } else if (std.mem.eql(u8, cmd, "train")) {
        try cmdTrain(gpa, io, flags, out, err);
    } else if (std.mem.eql(u8, cmd, "infer")) {
        try cmdInfer(gpa, io, flags, out);
    } else if (std.mem.eql(u8, cmd, "info")) {
        try cmdInfo(gpa, io, flags, out);
    } else if (std.mem.eql(u8, cmd, "serve")) {
        try llm.server.serve(gpa, io, flags.get("model") orelse default_model, flags.getInt(u16, "port", 8080), out);
    } else {
        try err.print("unknown command '{s}'\n", .{cmd});
        try err.flush();
        try printHelp(out);
        try out.flush();
        std.process.exit(1);
    }
    try out.flush();
    try err.flush();
}

fn printHelp(out: *std.Io.Writer) !void {
    try out.writeAll(
        \\vibezig/llm — a fullstack LLM engine in Zig
        \\
        \\usage:
        \\  llm gen   [--out models/tiny.vzm] [--layers 3] [--heads 4] [--embd 96]
        \\            [--block 64] [--data data/input.txt] [--seed 1337]
        \\  llm train [--data data/input.txt] [--out models/tiny.vzm] [--model in.vzm]
        \\            [--steps 3000] [--batch 8] [--lr 0.003] [--wd 0.01]
        \\            [--eval-every 100] [--sample-every 250] [--seed 1337]
        \\  llm infer [--model models/tiny.vzm] [--prompt "ROMEO:"] [--tokens 200]
        \\            [--temp 0.8] [--top-k 40] [--seed 0]
        \\  llm serve [--model models/tiny.vzm] [--port 8080]
        \\  llm info  [--model models/tiny.vzm]
        \\
        \\quickstart:
        \\  llm gen && llm train --steps 2000 && llm infer --prompt "ROMEO:"
        \\  llm serve            # then open http://localhost:8080
        \\
    );
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

fn cmdGen(gpa: std.mem.Allocator, io: std.Io, flags: Flags, out: *std.Io.Writer) !void {
    const data_path = flags.get("data") orelse default_data;
    const out_path = flags.get("out") orelse default_model;
    const seed = flags.getInt(u64, "seed", 1337);

    const corpus = std.Io.Dir.cwd().readFileAlloc(io, data_path, gpa, .limited(64 << 20)) catch |e| {
        try out.print("cannot read corpus '{s}': {t}\n", .{ data_path, e });
        return e;
    };
    defer gpa.free(corpus);

    const vocab = try tokenizer.buildVocab(gpa, corpus);
    defer gpa.free(vocab);

    const cfg = model_mod.Config{
        .n_layer = flags.getInt(u32, "layers", 3),
        .n_head = flags.getInt(u32, "heads", 4),
        .n_embd = flags.getInt(u32, "embd", 96),
        .block_size = flags.getInt(u32, "block", 64),
        .vocab_size = @intCast(vocab.len),
    };

    var m = try model_mod.Model.initRandom(gpa, cfg, vocab, seed);
    defer m.deinit();

    try ensureDir(io, std.fs.path.dirname(out_path) orelse ".");
    try m.save(io, out_path);

    try out.print(
        "generated model: {s}\n  params   {d}\n  layers   {d}\n  heads    {d} (head dim {d})\n  embd     {d}\n  block    {d}\n  vocab    {d}\n  bytes    {d}\n",
        .{
            out_path,
            m.paramCount(),
            cfg.n_layer,
            cfg.n_head,
            cfg.headDim(),
            cfg.n_embd,
            cfg.block_size,
            cfg.vocab_size,
            m.paramCount() * 4,
        },
    );
}

fn cmdTrain(gpa: std.mem.Allocator, io: std.Io, flags: Flags, out: *std.Io.Writer, err: *std.Io.Writer) !void {
    const data_path = flags.get("data") orelse default_data;
    const out_path = flags.get("out") orelse default_model;
    const steps = flags.getInt(usize, "steps", 3000);
    const batch = flags.getInt(usize, "batch", 8);
    const lr = flags.getFloat(f32, "lr", 3e-3);
    const wd = flags.getFloat(f32, "wd", 0.01);
    const eval_every = flags.getInt(usize, "eval-every", 100);
    const sample_every = flags.getInt(usize, "sample-every", 250);
    const seed = flags.getInt(u64, "seed", 1337);

    const corpus = std.Io.Dir.cwd().readFileAlloc(io, data_path, gpa, .limited(64 << 20)) catch |e| {
        try err.print("cannot read corpus '{s}': {t}\n", .{ data_path, e });
        return e;
    };
    defer gpa.free(corpus);
    if (corpus.len < 1024) return error.CorpusTooSmall;

    // 95% train / 5% validation
    const split = corpus.len - corpus.len / 20;
    const train_text = corpus[0..split];
    const val_text = corpus[split..];

    var m = if (flags.get("model")) |path|
        try model_mod.Model.load(gpa, io, path, 512 << 20)
    else blk: {
        const vocab = try tokenizer.buildVocab(gpa, corpus);
        defer gpa.free(vocab);
        const cfg = model_mod.Config{
            .n_layer = flags.getInt(u32, "layers", 3),
            .n_head = flags.getInt(u32, "heads", 4),
            .n_embd = flags.getInt(u32, "embd", 96),
            .block_size = flags.getInt(u32, "block", 64),
            .vocab_size = @intCast(vocab.len),
        };
        break :blk try model_mod.Model.initRandom(gpa, cfg, vocab, seed);
    };
    defer m.deinit();

    const tk = tokenizer.Tokenizer.init(m.vocab);
    const T: usize = m.cfg.block_size;

    const train_ids = try tk.encode(gpa, train_text);
    defer gpa.free(train_ids);
    const val_ids = try tk.encode(gpa, val_text);
    defer gpa.free(val_ids);
    if (train_ids.len < T + 1 or val_ids.len < T + 1) return error.CorpusTooSmall;

    const s = try gpt.Session.init(gpa, &m, batch);
    defer s.deinit();

    var tokens = try gpa.alloc(u32, batch * T);
    defer gpa.free(tokens);
    var targets = try gpa.alloc(u32, batch * T);
    defer gpa.free(targets);
    var val_tokens = try gpa.alloc(u32, batch * T);
    defer gpa.free(val_tokens);
    var val_targets = try gpa.alloc(u32, batch * T);
    defer gpa.free(val_targets);

    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();

    const t0 = nowNs(io);
    var tokens_seen: usize = 0;

    try out.print(
        "training on {s}\n  params {d} | batch {d} x seq {d} | steps {d} | lr {d:.4}\n",
        .{ data_path, m.paramCount(), batch, T, steps, lr },
    );
    try out.flush();

    var step: usize = 0;
    while (step < steps) : (step += 1) {
        // learning-rate schedule: 5% linear warmup, cosine decay to 10%
        const progress = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(@max(steps, 1)));
        const warmup = @as(f32, @floatFromInt(@max(steps / 20, 1)));
        const step_f = @as(f32, @floatFromInt(step));
        const cur_lr = if (step_f < warmup)
            lr * (step_f + 1.0) / warmup
        else
            lr * (0.1 + 0.9 * 0.5 * (1.0 + @cos(std.math.pi * (progress - 0.05) / 0.95)));

        for (0..batch) |b| {
            const start = rng.intRangeAtMost(usize, 0, train_ids.len - T - 1);
            @memcpy(tokens[b * T ..][0..T], train_ids[start..][0..T]);
            @memcpy(targets[b * T ..][0..T], train_ids[start + 1 ..][0..T]);
        }

        s.zeroGrads();
        const loss = try s.fwdBwd(tokens, targets, T);
        const norm = try s.clipGrads(1.0);
        try s.adamStep(cur_lr, wd, 0.9, 0.99);
        tokens_seen += batch * T;

        if (step % 10 == 0 or step == steps - 1) {
            const elapsed_s = secondsSince(io, t0);
            try out.print(
                "step {d:>5}/{d} | loss {d:.4} | lr {d:.5} | gnorm {d:.2} | {d:.0} tok/s | {d:.1}s\n",
                .{ step + 1, steps, loss, cur_lr, norm, @as(f64, @floatFromInt(tokens_seen)) / elapsed_s, elapsed_s },
            );
            try out.flush();
        }

        if (eval_every > 0 and (step + 1) % eval_every == 0 and val_ids.len > T) {
            // every batch row gets a different validation window
            for (0..batch) |b| {
                const start = (b * 997 + step * 31) % (val_ids.len - T - 1);
                @memcpy(val_tokens[b * T ..][0..T], val_ids[start..][0..T]);
                @memcpy(val_targets[b * T ..][0..T], val_ids[start + 1 ..][0..T]);
            }
            const val_loss = try s.lossOnly(val_tokens, val_targets, T);
            try out.print("   val loss {d:.4}\n", .{val_loss});
            try out.flush();
        }

        if (sample_every > 0 and (step + 1) % sample_every == 0) {
            try out.writeAll("   sample: ");
            try sampleInto(gpa, s, tk, "ROMEO:", 48, seed +% step, out);
            try out.writeAll("\n");
            try out.flush();
        }
    }

    try ensureDir(io, std.fs.path.dirname(out_path) orelse ".");
    try m.save(io, out_path);
    try out.print("saved {d} params to {s}\n", .{ m.paramCount(), out_path });
    try out.flush();
}

fn cmdInfer(gpa: std.mem.Allocator, io: std.Io, flags: Flags, out: *std.Io.Writer) !void {
    const path = flags.get("model") orelse default_model;
    const prompt = flags.get("prompt") orelse "\n";
    const n_tokens = flags.getInt(usize, "tokens", 200);
    const temp = flags.getFloat(f32, "temp", 0.8);
    const top_k = flags.getInt(usize, "top-k", 40);
    const seed = flags.getInt(u64, "seed", 0);

    var m = try model_mod.Model.load(gpa, io, path, 512 << 20);
    defer m.deinit();
    const tk = tokenizer.Tokenizer.init(m.vocab);

    var s = try gpt.Session.init(gpa, &m, 1);
    defer s.deinit();

    const prompt_ids = try tk.encode(gpa, prompt);
    defer gpa.free(prompt_ids);

    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);

    try out.print("{s}", .{prompt});
    try out.flush();

    var stream_ctx = StreamCtx{ .w = out, .vocab = m.vocab };
    const t0 = nowNs(io);
    const sink = gpt.TokenSink{ .ctx = @ptrCast(&stream_ctx), .emit = &emitToken };
    try s.generate(gpa, prompt_ids, n_tokens, .{ .temperature = temp, .top_k = top_k, .seed = seed }, sink, &ids);
    const elapsed = secondsSince(io, t0);
    try out.print("\n---\n{d} tokens in {d:.2}s ({d:.1} tok/s)\n", .{
        n_tokens,
        elapsed,
        @as(f64, @floatFromInt(n_tokens)) / @max(elapsed, 1e-6),
    });
}

const StreamCtx = struct {
    w: *std.Io.Writer,
    vocab: []const u8,
};

/// Streams a token as its raw byte straight to the writer, no allocation.
fn emitToken(ctx: *anyopaque, token: u32) void {
    const sc: *StreamCtx = @ptrCast(@alignCast(ctx));
    const b: u8 = if (token < sc.vocab.len) sc.vocab[token] else '?';
    sc.w.writeByte(b) catch {};
}

fn sampleInto(gpa: std.mem.Allocator, s: *gpt.Session, tk: tokenizer.Tokenizer, prompt: []const u8, n: usize, seed: u64, out: *std.Io.Writer) !void {
    const prompt_ids = try tk.encode(gpa, prompt);
    defer gpa.free(prompt_ids);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    try s.generate(gpa, prompt_ids, n, .{ .temperature = 0.8, .top_k = 40, .seed = seed }, null, &ids);
    const text = try tk.decode(gpa, ids.items);
    defer gpa.free(text);
    // keep samples on one line for the training log
    for (text) |ch| {
        try out.writeByte(if (ch == '\n') ' ' else ch);
    }
}

fn cmdInfo(gpa: std.mem.Allocator, io: std.Io, flags: Flags, out: *std.Io.Writer) !void {
    const path = flags.get("model") orelse default_model;
    var m = try model_mod.Model.load(gpa, io, path, 512 << 20);
    defer m.deinit();
    const cfg = m.cfg;
    try out.print(
        "model: {s}\n  params   {d}\n  layers   {d}\n  heads    {d} (head dim {d})\n  embd     {d}\n  block    {d}\n  vocab    {d}\n  tensors  {d}\n",
        .{ path, m.paramCount(), cfg.n_layer, cfg.n_head, cfg.headDim(), cfg.n_embd, cfg.block_size, cfg.vocab_size, m.tensors.len },
    );
    try out.writeAll("  tensors:\n");
    for (m.tensors) |t| {
        try out.print("    {s:<18} [{d}, {d}]\n", .{ t.name, t.rows, t.cols });
    }
}

// ---------------------------------------------------------------------------
// tiny helpers
// ---------------------------------------------------------------------------

const Flags = struct {
    args: []const [:0]const u8,

    fn get(self: Flags, name: []const u8) ?[]const u8 {
        var i: usize = 0;
        while (i < self.args.len) : (i += 1) {
            const a = self.args[i];
            if (!std.mem.startsWith(u8, a, "--")) continue;
            const key = a[2..];
            if (std.mem.indexOfScalar(u8, key, '=')) |eq| {
                if (std.mem.eql(u8, key[0..eq], name)) return key[eq + 1 ..];
            } else if (std.mem.eql(u8, key, name)) {
                if (i + 1 < self.args.len) return self.args[i + 1];
                return "";
            }
        }
        return null;
    }

    fn getInt(self: Flags, comptime T: type, name: []const u8, default: T) T {
        const v = self.get(name) orelse return default;
        return std.fmt.parseInt(T, v, 10) catch default;
    }

    fn getFloat(self: Flags, comptime T: type, name: []const u8, default: T) T {
        const v = self.get(name) orelse return default;
        return std.fmt.parseFloat(T, v) catch default;
    }
};

/// Monotonic clock reading in nanoseconds (uses `std.Io` clocks).
fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

fn secondsSince(io: std.Io, t0: i96) f64 {
    const delta: i96 = nowNs(io) - t0;
    return @as(f64, @floatFromInt(delta)) / 1e9;
}

fn ensureDir(io: std.Io, path: []const u8) !void {
    if (path.len == 0 or std.mem.eql(u8, path, ".")) return;
    std.Io.Dir.cwd().createDirPath(io, path) catch {};
}

test "flag parsing handles --k v and --k=v" {
    const a = [_][:0]const u8{ "--steps", "42", "--model=x.vzm", "--temp", "0.5" };
    const f = Flags{ .args = &a };
    try std.testing.expectEqual(@as(usize, 42), f.getInt(usize, "steps", 0));
    try std.testing.expectEqualStrings("x.vzm", f.get("model").?);
    try std.testing.expectEqual(@as(f32, 0.5), f.getFloat(f32, "temp", 1.0));
    try std.testing.expect(f.get("nope") == null);
}

test "emitToken streams the token byte to the writer" {
    var out_buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&out_buf);
    var ctx = StreamCtx{ .w = &w, .vocab = "abcde" };
    const sink = gpt.TokenSink{ .ctx = @ptrCast(&ctx), .emit = &emitToken };
    sink.send(2);
    sink.send(1);
    try std.testing.expectEqualStrings("cb", out_buf[0..2]);
}

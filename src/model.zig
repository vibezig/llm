//! Model description, tensor registry and the `.vzm` checkpoint format.
//!
//! A model is a flat `f32` blob plus a registry that names every tensor and
//! records its offset/shape. Keeping one blob means:
//!   * gradients and optimizer state are the same size and layout (offsets are
//!     shared, so `grad[i]` always belongs to `param[i]`);
//!   * a checkpoint is a header + one `write` of the blob.
//!
//! `.vzm` layout (little-endian):
//!
//!   magic       4 bytes  "VZLM"
//!   version     u32      1
//!   n_layer     u32
//!   n_head      u32
//!   n_embd      u32
//!   block_size  u32
//!   vocab_size  u32
//!   vocab_len   u32      number of bytes following
//!   vocab       [vocab_len]u8
//!   n_tensors   u32
//!   repeated n_tensors times:
//!       name_len u32, name [name_len]u8, rows u32, cols u32, dtype u32(0 = f32)
//!   blob        [total_floats]f32
const std = @import("std");
const tensor = @import("tensor.zig");

pub const magic = "VZLM";
pub const version: u32 = 1;

pub const Config = struct {
    n_layer: u32 = 3,
    n_head: u32 = 4,
    n_embd: u32 = 96,
    block_size: u32 = 64,
    vocab_size: u32 = 65,

    pub fn headDim(self: Config) usize {
        return self.n_embd / self.n_head;
    }

    pub fn mlpDim(self: Config) usize {
        return self.n_embd * 4;
    }

    pub fn validate(self: Config) !void {
        if (self.n_layer == 0) return error.InvalidConfig;
        if (self.n_head == 0) return error.InvalidConfig;
        if (self.n_embd == 0 or self.n_embd % self.n_head != 0) return error.InvalidConfig;
        if (self.block_size == 0) return error.InvalidConfig;
        if (self.vocab_size < 2) return error.InvalidConfig;
    }
};

pub const TensorInfo = struct {
    name: []const u8,
    offset: usize,
    rows: usize,
    cols: usize,

    pub fn len(self: TensorInfo) usize {
        return self.rows * self.cols;
    }
};

/// Resolved offsets of every tensor the transformer touches, so the hot loop
/// never does string lookups.
pub const LayerOffsets = struct {
    ln1g: usize,
    ln1b: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    ln2g: usize,
    ln2b: usize,
    fc: usize,
    proj: usize,
};

pub const Layout = struct {
    wte: usize,
    wpe: usize,
    lnfg: usize,
    lnfb: usize,
    layers: []LayerOffsets,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    vocab: []u8,
    tensors: []TensorInfo,
    blob: []f32,
    layout: Layout,

    pub fn initRandom(allocator: std.mem.Allocator, cfg: Config, vocab: []const u8, seed: u64) !Model {
        try cfg.validate();
        if (vocab.len != cfg.vocab_size) return error.VocabSizeMismatch;

        const tensors = try buildRegistry(allocator, cfg);
        errdefer freeRegistry(allocator, tensors);
        const layout = try resolveLayout(allocator, cfg, tensors);

        var total: usize = 0;
        for (tensors) |t| total += t.len();
        const blob = try allocator.alloc(f32, total);
        @memset(blob, 0);

        const vocab_copy = try allocator.dupe(u8, vocab);

        var m = Model{
            .allocator = allocator,
            .cfg = cfg,
            .vocab = vocab_copy,
            .tensors = tensors,
            .blob = blob,
            .layout = layout,
        };
        m.initWeights(seed);
        return m;
    }

    pub fn deinit(self: *Model) void {
        self.allocator.free(self.blob);
        self.allocator.free(self.vocab);
        self.allocator.free(self.layout.layers);
        freeRegistry(self.allocator, self.tensors);
    }

    /// Scaled normal init, GPT-2 style: residual projections get a smaller std
    /// (0.02 / sqrt(2 * n_layer)) so activations do not blow up with depth.
    fn initWeights(self: *Model, seed: u64) void {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const cfg = self.cfg;
        const c: usize = cfg.n_embd;
        const residual_std = 0.02 / @sqrt(2.0 * @as(f32, @floatFromInt(cfg.n_layer)));

        tensor.normal(r, self.view(self.layout.wte), 0, 0.02);
        tensor.normal(r, self.view(self.layout.wpe), 0, 0.01);
        for (self.layout.layers, 0..) |lo, i| {
            _ = i;
            tensor.fill(self.view(lo.ln1g), 1.0);
            tensor.fill(self.view(lo.ln1b), 0.0);
            tensor.normal(r, self.view(lo.wq), 0, 0.02);
            tensor.normal(r, self.view(lo.wk), 0, 0.02);
            tensor.normal(r, self.view(lo.wv), 0, 0.02);
            tensor.normal(r, self.view(lo.wo), 0, residual_std);
            tensor.fill(self.view(lo.ln2g), 1.0);
            tensor.fill(self.view(lo.ln2b), 0.0);
            tensor.normal(r, self.view(lo.fc), 0, 0.02);
            tensor.normal(r, self.view(lo.proj), 0, residual_std);
            _ = c;
        }
        tensor.fill(self.view(self.layout.lnfg), 1.0);
        tensor.fill(self.view(self.layout.lnfb), 0.0);
    }

    pub fn view(self: *const Model, offset: usize) []f32 {
        return self.blob[offset..];
    }

    pub fn tensorByOffset(self: *const Model, offset: usize) []f32 {
        return self.blob[offset..];
    }

    pub fn find(self: *const Model, name: []const u8) ?TensorInfo {
        for (self.tensors) |t| {
            if (std.mem.eql(u8, t.name, name)) return t;
        }
        return null;
    }

    pub fn paramCount(self: *const Model) usize {
        return self.blob.len;
    }

    pub fn blobBytes(self: *const Model) []const u8 {
        return std.mem.sliceAsBytes(self.blob);
    }

    /// Serialize to `.vzm` bytes (caller owns the returned buffer).
    pub fn serialize(self: *const Model, allocator: std.mem.Allocator) ![]u8 {
        if (@import("builtin").cpu.arch.endian() != .little) return error.BigEndianUnsupported;

        var out = std.ArrayList(u8).empty;
        errdefer out.deinit(allocator);

        try out.appendSlice(allocator, magic);
        try appendU32(&out, allocator, version);
        try appendU32(&out, allocator, self.cfg.n_layer);
        try appendU32(&out, allocator, self.cfg.n_head);
        try appendU32(&out, allocator, self.cfg.n_embd);
        try appendU32(&out, allocator, self.cfg.block_size);
        try appendU32(&out, allocator, self.cfg.vocab_size);
        try appendU32(&out, allocator, @intCast(self.vocab.len));
        try out.appendSlice(allocator, self.vocab);
        try appendU32(&out, allocator, @intCast(self.tensors.len));
        for (self.tensors) |t| {
            try appendU32(&out, allocator, @intCast(t.name.len));
            try out.appendSlice(allocator, t.name);
            try appendU32(&out, allocator, @intCast(t.rows));
            try appendU32(&out, allocator, @intCast(t.cols));
            try appendU32(&out, allocator, 0); // dtype: f32
        }
        const blob_bytes = std.mem.sliceAsBytes(self.blob);
        try out.appendSlice(allocator, blob_bytes);
        return out.toOwnedSlice(allocator);
    }

    /// Parse `.vzm` bytes. The returned model owns all its memory.
    pub fn deserialize(allocator: std.mem.Allocator, bytes: []const u8) !Model {
        var cursor: usize = 0;
        const Reader = struct {
            bytes: []const u8,
            pos: *usize,

            fn take(self: @This(), n: usize) ![]const u8 {
                if (self.pos.* + n > self.bytes.len) return error.TruncatedCheckpoint;
                const s = self.bytes[self.pos.*..][0..n];
                self.pos.* += n;
                return s;
            }
            fn u32le(self: @This()) !u32 {
                const s = try self.take(4);
                return std.mem.readInt(u32, s[0..4], .little);
            }
        };

        var rd = Reader{ .bytes = bytes, .pos = &cursor };
        const got_magic = try rd.take(4);
        if (!std.mem.eql(u8, got_magic, magic)) return error.BadMagic;
        const ver = try rd.u32le();
        if (ver != version) return error.UnsupportedVersion;

        var cfg = Config{
            .n_layer = try rd.u32le(),
            .n_head = try rd.u32le(),
            .n_embd = try rd.u32le(),
            .block_size = try rd.u32le(),
            .vocab_size = try rd.u32le(),
        };
        try cfg.validate();

        const vocab_len = try rd.u32le();
        if (vocab_len != cfg.vocab_size) return error.VocabSizeMismatch;
        const vocab = try rd.take(vocab_len);

        const n_tensors = try rd.u32le();
        const tensors = try allocator.alloc(TensorInfo, n_tensors);
        var total: usize = 0;
        var name_bytes: std.ArrayList(u8) = .empty;
        defer name_bytes.deinit(allocator);

        var i: usize = 0;
        errdefer {
            for (tensors[0..i]) |t| allocator.free(t.name);
            allocator.free(tensors);
        }
        while (i < n_tensors) : (i += 1) {
            const name_len = try rd.u32le();
            const name = try rd.take(name_len);
            const rows = try rd.u32le();
            const cols = try rd.u32le();
            const dtype = try rd.u32le();
            if (dtype != 0) return error.UnsupportedDtype;
            if (rows == 0 or cols == 0) return error.InvalidTensor;
            const owned = try allocator.dupe(u8, name);
            tensors[i] = .{
                .name = owned,
                .offset = total,
                .rows = rows,
                .cols = cols,
            };
            total += @as(usize, rows) * cols;
        }

        const blob_bytes = try rd.take(total * @sizeOf(f32));
        const blob = try allocator.alloc(f32, total);
        @memcpy(std.mem.sliceAsBytes(blob), blob_bytes);

        const vocab_copy = try allocator.dupe(u8, vocab);
        // Registry from the file must match the canonical layout; resolve by name.
        const layout = try resolveLayout(allocator, cfg, tensors);

        return Model{
            .allocator = allocator,
            .cfg = cfg,
            .vocab = vocab_copy,
            .tensors = tensors,
            .blob = blob,
            .layout = layout,
        };
    }

    pub fn save(self: *const Model, io: std.Io, path: []const u8) !void {
        const bytes = try self.serialize(self.allocator);
        defer self.allocator.free(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }

    pub fn load(allocator: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize) !Model {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(limit));
        defer allocator.free(bytes);
        return deserialize(allocator, bytes);
    }
};

fn appendU32(list: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

pub fn freeRegistry(allocator: std.mem.Allocator, tensors: []TensorInfo) void {
    for (tensors) |t| allocator.free(t.name);
    allocator.free(tensors);
}

/// Canonical parameter order. Everything else (gradients, optimizer state,
/// checkpoints) reuses these offsets.
pub fn buildRegistry(allocator: std.mem.Allocator, cfg: Config) ![]TensorInfo {
    const c: usize = cfg.n_embd;
    const m: usize = cfg.mlpDim();
    const v: usize = cfg.vocab_size;
    const t: usize = cfg.block_size;
    // 2 global + per layer: 4 ln vectors (2*g,2*b as [c,1]) + 4 attn mats + 2 mlp mats
    const per_layer = 10;
    const n = 2 + @as(usize, cfg.n_layer) * per_layer + 2;
    const tensors = try allocator.alloc(TensorInfo, n);
    var idx: usize = 0;
    var offset: usize = 0;

    tensors[idx] = .{ .name = try allocator.dupe(u8, "wte"), .offset = 0, .rows = v, .cols = c };
    offset += v * c;
    idx += 1;
    tensors[idx] = .{ .name = try allocator.dupe(u8, "wpe"), .offset = offset, .rows = t, .cols = c };
    offset += t * c;
    idx += 1;

    for (0..cfg.n_layer) |l| {
        const dims = [_][2]usize{
            .{ c, 1 }, // ln1.g
            .{ c, 1 }, // ln1.b
            .{ c, c }, // wq
            .{ c, c }, // wk
            .{ c, c }, // wv
            .{ c, c }, // wo
            .{ c, 1 }, // ln2.g
            .{ c, 1 }, // ln2.b
            .{ c, m }, // fc
            .{ m, c }, // proj
        };
        const suffixes = [_][]const u8{
            "ln1.g", "ln1.b", "attn.wq", "attn.wk", "attn.wv", "attn.wo", "ln2.g", "ln2.b", "mlp.fc", "mlp.proj",
        };
        for (dims, suffixes) |d, suffix| {
            const name = try std.fmt.allocPrint(allocator, "h{d}.{s}", .{ l, suffix });
            tensors[idx] = .{ .name = name, .offset = offset, .rows = d[0], .cols = d[1] };
            offset += d[0] * d[1];
            idx += 1;
        }
    }

    tensors[idx] = .{ .name = try allocator.dupe(u8, "lnf.g"), .offset = offset, .rows = c, .cols = 1 };
    offset += c;
    idx += 1;
    tensors[idx] = .{ .name = try allocator.dupe(u8, "lnf.b"), .offset = offset, .rows = c, .cols = 1 };
    idx += 1;

    return tensors;
}

fn resolveLayout(allocator: std.mem.Allocator, cfg: Config, tensors: []const TensorInfo) !Layout {
    const layers = try allocator.alloc(LayerOffsets, cfg.n_layer);
    errdefer allocator.free(layers);

    var layout = Layout{
        .wte = 0,
        .wpe = 0,
        .lnfg = 0,
        .lnfb = 0,
        .layers = layers,
    };

    for (tensors) |t| {
        if (std.mem.eql(u8, t.name, "wte")) layout.wte = t.offset;
        if (std.mem.eql(u8, t.name, "wpe")) layout.wpe = t.offset;
        if (std.mem.eql(u8, t.name, "lnf.g")) layout.lnfg = t.offset;
        if (std.mem.eql(u8, t.name, "lnf.b")) layout.lnfb = t.offset;
    }

    for (layers, 0..) |*lo, l| {
        lo.* = .{
            .ln1g = try need(tensors, l, "ln1.g"),
            .ln1b = try need(tensors, l, "ln1.b"),
            .wq = try need(tensors, l, "attn.wq"),
            .wk = try need(tensors, l, "attn.wk"),
            .wv = try need(tensors, l, "attn.wv"),
            .wo = try need(tensors, l, "attn.wo"),
            .ln2g = try need(tensors, l, "ln2.g"),
            .ln2b = try need(tensors, l, "ln2.b"),
            .fc = try need(tensors, l, "mlp.fc"),
            .proj = try need(tensors, l, "mlp.proj"),
        };
    }
    return layout;
}

fn need(tensors: []const TensorInfo, layer: usize, suffix: []const u8) !usize {
    var buf: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&buf, "h{d}.{s}", .{ layer, suffix }) catch return error.NameTooLong;
    for (tensors) |t| {
        if (std.mem.eql(u8, t.name, name)) return t.offset;
    }
    return error.MissingTensor;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "registry offsets are contiguous and cover the blob" {
    const cfg = Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 8, .vocab_size = 32 };
    const tensors = try buildRegistry(testing.allocator, cfg);
    defer freeRegistry(testing.allocator, tensors);

    var expected_offset: usize = 0;
    for (tensors) |t| {
        try testing.expectEqual(expected_offset, t.offset);
        expected_offset += t.len();
    }
    try testing.expectEqual(@as(usize, 2 + 2 * 10 + 2), tensors.len);

    // wte + wpe + layers*(4*C*C + 2*C*4C + 4*C) + 2*C
    const c: usize = 16;
    const want = cfg.vocab_size * c + cfg.block_size * c +
        cfg.n_layer * (4 * c * c + 2 * c * 4 * c + 4 * c) + 2 * c;
    try testing.expectEqual(want, expected_offset);
}

test "checkpoint round-trip preserves config, vocab and weights" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = Config{ .n_layer = 2, .n_head = 2, .n_embd = 16, .block_size = 8, .vocab_size = 10 };
    const vocab = "abcdefghij";
    var m = try Model.initRandom(allocator, cfg, vocab, 1234);
    defer m.deinit();

    const bytes = try m.serialize(allocator);
    defer allocator.free(bytes);

    var loaded = try Model.deserialize(allocator, bytes);
    defer loaded.deinit();

    try testing.expectEqual(cfg.n_layer, loaded.cfg.n_layer);
    try testing.expectEqual(cfg.n_embd, loaded.cfg.n_embd);
    try testing.expectEqualStrings(vocab, loaded.vocab);
    try testing.expectEqualSlices(f32, m.blob, loaded.blob);
    try testing.expectEqual(m.layout.layers[1].proj, loaded.layout.layers[1].proj);
}

test "deserialize rejects a corrupted magic" {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const cfg = Config{ .n_layer = 1, .n_head = 1, .n_embd = 8, .block_size = 4, .vocab_size = 4 };
    var m = try Model.initRandom(allocator, cfg, "abcd", 7);
    defer m.deinit();
    const bytes = try m.serialize(allocator);
    defer allocator.free(bytes);
    bytes[0] = 'X';
    try testing.expectError(error.BadMagic, Model.deserialize(allocator, bytes));
}

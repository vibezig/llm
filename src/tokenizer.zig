//! Tokenizer facade.
//!
//! Two serialized kinds share one API:
//!   * version 1 — byte-level char vocab (legacy `.vzm` checkpoints);
//!   * version 2 — corpus-trained byte-level BPE (compact blob from
//!     `src/bpe/train.zig`, parsed by `src/bpe/tokenizer.zig`).
//!
//! `encode` always returns `u32` ids; the BPE layer works in `u21` internally.
const std = @import("std");
const bpe = @import("bpe/tokenizer.zig");
const bpe_train = @import("bpe/train.zig");

pub const max_token_bytes = bpe.max_token_bytes;
pub const default_vocab_size = bpe_train.default_vocab_size;
pub const max_vocab_size = bpe_train.max_vocab_size;
pub const train = bpe_train.train;

pub const Tokenizer = union(enum) {
    char: Char,
    bpe: Bpe,

    pub const Kind = enum { char, bpe };

    const missing_id = std.math.maxInt(u32);

    pub const Char = struct {
        allocator: std.mem.Allocator,
        vocab: []u8,
        map: [256]u32,
    };

    pub const Bpe = struct {
        allocator: std.mem.Allocator,
        inner: *bpe.Tokenizer,
    };

    pub fn initChar(allocator: std.mem.Allocator, vocab: []const u8) !Tokenizer {
        const copy = try allocator.dupe(u8, vocab);
        errdefer allocator.free(copy);
        var map = [_]u32{missing_id} ** 256;
        for (copy, 0..) |b, i| map[b] = @intCast(i);
        return .{ .char = .{ .allocator = allocator, .vocab = copy, .map = map } };
    }

    pub fn initBpeFromBlob(allocator: std.mem.Allocator, blob: []const u8) !Tokenizer {
        const inner = try allocator.create(bpe.Tokenizer);
        errdefer allocator.destroy(inner);
        inner.* = try bpe.Tokenizer.initFromBlob(allocator, blob);
        return .{ .bpe = .{ .allocator = allocator, .inner = inner } };
    }

    pub fn fromSerialized(allocator: std.mem.Allocator, ver: u32, blob: []const u8) !Tokenizer {
        return switch (ver) {
            1 => initChar(allocator, blob),
            2 => initBpeFromBlob(allocator, blob),
            else => error.UnsupportedVersion,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        switch (self.*) {
            .char => |c| c.allocator.free(c.vocab),
            .bpe => |b| {
                b.inner.deinit();
                b.allocator.destroy(b.inner);
            },
        }
        self.* = undefined;
    }

    pub fn version(self: *const Tokenizer) u32 {
        return switch (self.*) {
            .char => 1,
            .bpe => 2,
        };
    }

    pub fn kind(self: *const Tokenizer) Kind {
        return switch (self.*) {
            .char => .char,
            .bpe => .bpe,
        };
    }

    pub fn vocabSize(self: *const Tokenizer) usize {
        return switch (self.*) {
            .char => |c| c.vocab.len,
            .bpe => |b| b.inner.vocabSize(),
        };
    }

    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        switch (self.*) {
            .char => |c| {
                const out = try allocator.alloc(u32, text.len);
                errdefer allocator.free(out);
                for (text, 0..) |b, i| {
                    const id = c.map[b];
                    if (id == missing_id) return error.CharacterNotInVocab;
                    out[i] = id;
                }
                return out;
            },
            .bpe => |b| {
                const ids = try b.inner.encode(text);
                defer b.inner.allocator.free(ids);
                const out = try allocator.alloc(u32, ids.len);
                for (ids, 0..) |id, i| out[i] = id;
                return out;
            },
        }
    }

    pub fn decode(self: *const Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        switch (self.*) {
            .char => |c| {
                const out = try allocator.alloc(u8, ids.len);
                errdefer allocator.free(out);
                for (ids, 0..) |id, i| {
                    if (id >= c.vocab.len) return error.InvalidTokenId;
                    out[i] = c.vocab[id];
                }
                return out;
            },
            .bpe => |b| {
                const ids21 = try allocator.alloc(u21, ids.len);
                defer allocator.free(ids21);
                for (ids, 0..) |id, i| {
                    if (id > std.math.maxInt(u21)) return error.InvalidTokenId;
                    ids21[i] = @intCast(id);
                }
                return b.inner.decode(ids21);
            },
        }
    }

    /// Decode one token into `out` without allocating (streaming path).
    pub fn decodeTokenInto(self: *const Tokenizer, id: u32, out: []u8) ![]u8 {
        switch (self.*) {
            .char => |c| {
                if (id >= c.vocab.len) return error.InvalidTokenId;
                if (out.len == 0) return error.NoSpaceLeft;
                out[0] = c.vocab[id];
                return out[0..1];
            },
            .bpe => |b| {
                if (id > std.math.maxInt(u21)) return error.InvalidTokenId;
                return b.inner.decodeTokenInto(@intCast(id), out);
            },
        }
    }
};

test "char tokenizer round-trip" {
    const allocator = std.testing.allocator;
    var tk = try Tokenizer.initChar(allocator, "abcd");
    defer tk.deinit();

    const ids = try tk.encode(allocator, "dcba");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 3, 2, 1, 0 }, ids);

    const text = try tk.decode(allocator, ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("dcba", text);
}

test "trained BPE tokenizer round-trip" {
    const allocator = std.testing.allocator;
    const corpus = "the quick brown fox jumps over the lazy dog\n";
    const blob = try bpe_train.train(allocator, corpus, 300);
    defer allocator.free(blob);

    var tk = try Tokenizer.initBpeFromBlob(allocator, blob);
    defer tk.deinit();
    try std.testing.expectEqual(Tokenizer.Kind.bpe, tk.kind());
    try std.testing.expect(tk.vocabSize() > 256 and tk.vocabSize() <= 300);

    const ids = try tk.encode(allocator, corpus);
    defer allocator.free(ids);
    const text = try tk.decode(allocator, ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings(corpus, text);
}

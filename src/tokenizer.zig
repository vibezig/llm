//! Character-level tokenizer: one token per byte, vocabulary stored in the
//! checkpoint so a model and its tokenizer can never drift apart.
const std = @import("std");

pub const Tokenizer = struct {
    vocab: []const u8,

    pub fn init(vocab: []const u8) Tokenizer {
        return .{ .vocab = vocab };
    }

    pub fn size(self: Tokenizer) usize {
        return self.vocab.len;
    }

    /// Map a byte to its token id, falling back to id 0 for unseen bytes.
    pub fn idForByte(self: Tokenizer, b: u8) u32 {
        for (self.vocab, 0..) |v, i| {
            if (v == b) return @intCast(i);
        }
        return 0;
    }

    pub fn encode(self: Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try allocator.alloc(u32, text.len);
        for (text, 0..) |b, i| out[i] = self.idForByte(b);
        return out;
    }

    /// Encode into a caller-provided buffer; returns the used slice.
    pub fn encodeInto(self: Tokenizer, text: []const u8, out: []u32) []u32 {
        const n = @min(text.len, out.len);
        for (text[0..n], 0..) |b, i| out[i] = self.idForByte(b);
        return out[0..n];
    }

    pub fn decode(self: Tokenizer, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        const out = try allocator.alloc(u8, ids.len);
        for (ids, 0..) |id, i| {
            out[i] = if (id < self.vocab.len) self.vocab[id] else '?';
        }
        return out;
    }

    /// Decode into a caller-provided buffer; returns the used slice.
    pub fn decodeInto(self: Tokenizer, ids: []const u32, out: []u8) []u8 {
        const n = @min(ids.len, out.len);
        for (ids[0..n], 0..) |id, i| {
            out[i] = if (id < self.vocab.len) self.vocab[id] else '?';
        }
        return out[0..n];
    }
};

/// Build a sorted, unique byte vocabulary from a training corpus.
pub fn buildVocab(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var seen: [256]bool = @splat(false);
    for (text) |b| seen[b] = true;
    var count: usize = 0;
    for (seen) |s| {
        if (s) count += 1;
    }
    const vocab = try allocator.alloc(u8, count);
    var i: usize = 0;
    for (seen, 0..) |s, b| {
        if (s) {
            vocab[i] = @intCast(b);
            i += 1;
        }
    }
    return vocab;
}

test "buildVocab is sorted and unique" {
    const vocab = try buildVocab(std.testing.allocator, "hello world");
    defer std.testing.allocator.free(vocab);
    try std.testing.expectEqualStrings(" dehlorw", vocab);
}

test "encode/decode round trip" {
    const vocab = " abcd";
    const tk = Tokenizer.init(vocab);
    const ids = try tk.encode(std.testing.allocator, "a bad cab");
    defer std.testing.allocator.free(ids);
    const back = try tk.decode(std.testing.allocator, ids);
    defer std.testing.allocator.free(back);
    try std.testing.expectEqualStrings("a bad cab", back);
    // unseen byte maps to id 0 (space here)
    try std.testing.expectEqual(@as(u32, 0), tk.idForByte('z'));
}

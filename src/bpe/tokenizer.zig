//! Byte-level BPE tokenizer.
//!
//! Core algorithms vendored from https://github.com/alvarobartt/bpe.zig
//! (MIT OR Apache-2.0), ported to Zig 0.16, with the POSIX-regex pre-tokenizer
//! replaced by `pre_tokenize.zig`. The JSON loader is replaced by a compact
//! `.vzm` v2 blob (`initFromBlob` / `serializeBlob`) fed by the pure-Zig
//! trainer in `train.zig`.
const std = @import("std");
const Pair = @import("pair.zig").Pair;
const PairContext = @import("pair.zig").PairContext;
const bytesToTokensInto = @import("byte_encoding.zig").bytesToTokensInto;
const tokensToBytes = @import("byte_decoding.zig").tokensToBytes;
const tokensToBytesInto = @import("byte_decoding.zig").tokensToBytesInto;
const pre_tokenize = @import("pre_tokenize.zig");
const findRanges = pre_tokenize.findRanges;

pub const blob_magic = "VZB1";
/// BPE starts from the 256 byte tokens; every merge adds one token.
pub const base_vocab_size: usize = 256;
/// Upper bound on a decoded token in bytes (streaming stack buffer).
pub const max_token_bytes: usize = 512;

pub const Merge = [2]u32;

pub const Error = error{
    InvalidTokenizerBlob,
    InvalidVocab,
    InvalidTokenId,
    InvalidUtf8,
    TokenNotInVocab,
    NoSpaceLeft,
    TooManyTokens,
};

pub const Tokenizer = struct {
    const Self = @This();

    vocab: std.StringHashMap(u21),
    vocab_r: std.AutoHashMap(u21, []const u8),
    merges: std.ArrayList(Pair),
    merges_map: std.HashMap(Pair, u21, PairContext, std.hash_map.default_max_load_percentage),
    allocator: std.mem.Allocator,

    /// Build from an ordered token table (IDs 0..n-1) and merge pairs.
    /// Requires `vocab.len == 256 + merges.len` and every merge to reference
    /// an existing token.
    pub fn initFromParts(allocator: std.mem.Allocator, vocab: []const []const u8, merges: []const Merge) !Self {
        if (vocab.len < base_vocab_size) return Error.InvalidVocab;
        if (vocab.len != base_vocab_size + merges.len) return Error.InvalidVocab;

        var vocab_map = std.StringHashMap(u21).init(allocator);
        errdefer {
            var it = vocab_map.iterator();
            while (it.next()) |entry| allocator.free(entry.key_ptr.*);
            vocab_map.deinit();
        }
        var vocab_r = std.AutoHashMap(u21, []const u8).init(allocator);
        errdefer vocab_r.deinit();

        for (vocab, 0..) |token, i| {
            if (i > std.math.maxInt(u21)) return Error.TooManyTokens;
            const owned = try allocator.dupe(u8, token);
            try vocab_map.put(owned, @intCast(i));
            try vocab_r.put(@intCast(i), owned);
        }

        var merge_list: std.ArrayList(Pair) = .empty;
        errdefer {
            for (merge_list.items) |m| {
                allocator.free(m.left);
                allocator.free(m.right);
            }
            merge_list.deinit(allocator);
        }
        var merges_map = std.HashMap(Pair, u21, PairContext, std.hash_map.default_max_load_percentage).initContext(allocator, PairContext{});
        errdefer merges_map.deinit();

        for (merges, 0..) |m, i| {
            if (m[0] >= vocab.len or m[1] >= vocab.len) return Error.InvalidVocab;
            const left_src = vocab_r.get(@intCast(m[0])) orelse return Error.InvalidVocab;
            const right_src = vocab_r.get(@intCast(m[1])) orelse return Error.InvalidVocab;

            // The merged token must be exactly left ++ right.
            const expected = vocab[base_vocab_size + i];
            if (expected.len != left_src.len + right_src.len or
                !std.mem.eql(u8, expected[0..left_src.len], left_src) or
                !std.mem.eql(u8, expected[left_src.len..], right_src))
            {
                return Error.InvalidVocab;
            }

            const left = try allocator.dupe(u8, left_src);
            const right = allocator.dupe(u8, right_src) catch |e| {
                allocator.free(left);
                return e;
            };
            try merge_list.append(allocator, .{ .left = left, .right = right });
            try merges_map.put(.{ .left = left, .right = right }, @intCast(i));
        }

        return .{
            .vocab = vocab_map,
            .vocab_r = vocab_r,
            .merges = merge_list,
            .merges_map = merges_map,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.vocab.iterator();
        while (it.next()) |entry| self.allocator.free(entry.key_ptr.*);
        self.vocab.deinit();
        self.vocab_r.deinit();

        for (self.merges.items) |m| {
            self.allocator.free(m.left);
            self.allocator.free(m.right);
        }
        self.merges.deinit(self.allocator);
        self.merges_map.deinit();
    }

    pub fn vocabSize(self: *const Self) usize {
        return self.vocab.count();
    }

    pub fn tokenText(self: *const Self, id: u21) ?[]const u8 {
        return self.vocab_r.get(id);
    }

    fn preTokenize(self: *const Self, text: []const u8) ![]pre_tokenize.Range {
        return findRanges(self.allocator, text);
    }

    pub fn encode(self: *const Self, text: []const u8) ![]u21 {
        const allocator = self.allocator;
        const Symbol = struct { start: usize, end: usize };

        // Piece -> byte-level token string, then merge each piece. The scratch
        // buffers are reused; no per-symbol or per-merge allocations.
        const matches = try self.preTokenize(text);
        defer allocator.free(matches);

        var text_encoding: std.ArrayList(u21) = .empty;
        errdefer text_encoding.deinit(allocator);

        var encoding: std.ArrayList(u8) = .empty;
        defer encoding.deinit(allocator);
        var symbols: std.ArrayList(Symbol) = .empty;
        defer symbols.deinit(allocator);

        for (matches) |range| {
            const match = text[range.start..range.end];
            encoding.clearRetainingCapacity();
            try bytesToTokensInto(allocator, match, &encoding);
            const enc = encoding.items;

            if (self.vocab.get(enc)) |id| {
                try text_encoding.append(allocator, id);
                continue;
            }

            symbols.clearRetainingCapacity();
            var pos: usize = 0;
            while (pos < enc.len) {
                const len = std.unicode.utf8ByteSequenceLength(enc[pos]) catch return Error.InvalidUtf8;
                if (pos + len > enc.len) return Error.InvalidUtf8;
                try symbols.append(allocator, .{ .start = pos, .end = pos + len });
                pos += len;
            }

            while (symbols.items.len > 1) {
                var best_idx: ?usize = null;
                var best_rank: u21 = std.math.maxInt(u21);

                for (0..symbols.items.len - 1) |i| {
                    const left = symbols.items[i];
                    const right = symbols.items[i + 1];
                    const pair = Pair{ .left = enc[left.start..left.end], .right = enc[right.start..right.end] };
                    if (self.merges_map.get(pair)) |rank| {
                        if (rank < best_rank) {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }

                const merge_idx = best_idx orelse break;
                symbols.items[merge_idx].end = symbols.items[merge_idx + 1].end;
                _ = symbols.orderedRemove(merge_idx + 1);
            }

            for (symbols.items) |s| {
                try text_encoding.append(allocator, self.vocab.get(enc[s.start..s.end]) orelse return Error.TokenNotInVocab);
            }
        }

        return text_encoding.toOwnedSlice(allocator);
    }

    pub fn decode(self: *const Self, input_ids: []const u21) ![]u8 {
        var buffer: std.ArrayList(u8) = .empty;
        errdefer buffer.deinit(self.allocator);

        for (input_ids) |id| {
            const token = self.vocab_r.get(id) orelse return Error.InvalidTokenId;
            const decoded = try tokensToBytes(self.allocator, token);
            defer self.allocator.free(decoded);
            try buffer.appendSlice(self.allocator, decoded);
        }
        return buffer.toOwnedSlice(self.allocator);
    }

    /// Decode into a caller buffer without allocating (token-streaming path).
    pub fn decodeInto(self: *const Self, input_ids: []const u21, out: []u8) ![]u8 {
        var n: usize = 0;
        for (input_ids) |id| {
            const token = self.vocab_r.get(id) orelse return Error.InvalidTokenId;
            const piece = try tokensToBytesInto(token, out[n..]);
            n += piece.len;
        }
        return out[0..n];
    }

    pub fn decodeTokenInto(self: *const Self, id: u21, out: []u8) ![]u8 {
        return self.decodeInto(&.{id}, out);
    }

    /// Serialize the ordered vocab and merge pairs to the `.vzm` v2 blob.
    pub fn serializeBlob(allocator: std.mem.Allocator, vocab: []const []const u8, merges: []const Merge) ![]u8 {
        if (vocab.len != base_vocab_size + merges.len) return Error.InvalidVocab;
        if (vocab.len > std.math.maxInt(u32)) return Error.TooManyTokens;

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);

        try out.appendSlice(allocator, blob_magic);
        try appendU32(&out, allocator, @intCast(vocab.len));
        for (vocab) |token| {
            if (token.len > std.math.maxInt(u16)) return Error.InvalidVocab;
            try appendU16(&out, allocator, @intCast(token.len));
            try out.appendSlice(allocator, token);
        }
        try appendU32(&out, allocator, @intCast(merges.len));
        for (merges) |m| {
            try appendU32(&out, allocator, m[0]);
            try appendU32(&out, allocator, m[1]);
        }
        return out.toOwnedSlice(allocator);
    }

    pub fn initFromBlob(allocator: std.mem.Allocator, blob: []const u8) !Self {
        var cursor: usize = 0;
        const Reader = struct {
            bytes: []const u8,
            pos: *usize,
            fn take(self: @This(), n: usize) ![]const u8 {
                if (self.pos.* + n > self.bytes.len) return Error.InvalidTokenizerBlob;
                const s = self.bytes[self.pos.*..][0..n];
                self.pos.* += n;
                return s;
            }
            fn u16le(self: @This()) !u16 {
                const s = try self.take(2);
                return std.mem.readInt(u16, s[0..2], .little);
            }
            fn u32le(self: @This()) !u32 {
                const s = try self.take(4);
                return std.mem.readInt(u32, s[0..4], .little);
            }
        };

        var rd = Reader{ .bytes = blob, .pos = &cursor };
        if (!std.mem.eql(u8, try rd.take(4), blob_magic)) return Error.InvalidTokenizerBlob;
        const n_tokens = try rd.u32le();
        if (n_tokens < base_vocab_size) return Error.InvalidTokenizerBlob;

        var vocab: std.ArrayList([]u8) = .empty;
        defer {
            for (vocab.items) |s| allocator.free(s);
            vocab.deinit(allocator);
        }
        try vocab.ensureTotalCapacity(allocator, n_tokens);
        for (0..n_tokens) |_| {
            const len = try rd.u16le();
            const bytes = try rd.take(len);
            try vocab.append(allocator, try allocator.dupe(u8, bytes));
        }

        const n_merges = try rd.u32le();
        if (n_tokens != base_vocab_size + n_merges) return Error.InvalidTokenizerBlob;
        var merges: std.ArrayList(Merge) = .empty;
        defer merges.deinit(allocator);
        try merges.ensureTotalCapacity(allocator, n_merges);
        for (0..n_merges) |_| {
            const left = try rd.u32le();
            const right = try rd.u32le();
            try merges.append(allocator, .{ left, right });
        }
        if (cursor != blob.len) return Error.InvalidTokenizerBlob;

        return initFromParts(allocator, vocab.items, merges.items);
    }
};

fn appendU16(list: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u16) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

fn appendU32(list: *std.ArrayList(u8), allocator: std.mem.Allocator, v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

test "ordered vocab + merges round-trip through the blob" {
    const allocator = std.testing.allocator;
    const byte_encoding = @import("byte_encoding.zig");

    var vocab: std.ArrayList([]u8) = .empty;
    defer {
        for (vocab.items) |s| allocator.free(s);
        vocab.deinit(allocator);
    }
    for (0..256) |b| try vocab.append(allocator, try byte_encoding.byteToken(allocator, @intCast(b)));
    // 'a' = 97, 'b' = 98: merge them, then merge "ab" + "a".
    try vocab.append(allocator, try allocator.dupe(u8, "ab"));
    try vocab.append(allocator, try allocator.dupe(u8, "aba"));
    const merges = [_]Merge{ .{ 97, 98 }, .{ 256, 97 } };

    const blob = try Tokenizer.serializeBlob(allocator, vocab.items, &merges);
    defer allocator.free(blob);

    var tk = try Tokenizer.initFromBlob(allocator, blob);
    defer tk.deinit();
    try std.testing.expectEqual(@as(usize, 258), tk.vocabSize());

    const ids = try tk.encode("abab");
    defer allocator.free(ids);
    // The encoder always applies the globally lowest-rank merge first, so the
    // two "ab" pairs merge before the ("ab","a") merge can apply.
    try std.testing.expectEqualSlices(u21, &.{ 256, 256 }, ids);

    const text = try tk.decode(ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings("abab", text);
}

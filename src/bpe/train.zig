//! Corpus-trained byte-level BPE.
//!
//! This is not part of upstream bpe.zig (which only loads a tokenizer). It
//! learns merges on a corpus with the same pre-tokenization the encoder uses,
//! then emits the compact `.vzm` v3 tokenizer blob consumed by `Tokenizer.initFromBlob`.
//!
//! The trainer is incremental: symbols live in a doubly linked list, every
//! adjacent pair is in an intrusive position list, and a max-heap over pair
//! counts selects the next merge. A merge only touches the occurrences of the
//! chosen pair and their neighbours, instead of rescanning the whole corpus.
const std = @import("std");
const pre_tokenize = @import("pre_tokenize.zig");
const byte_encoding = @import("byte_encoding.zig");
const tokenizer = @import("tokenizer.zig");

const Merge = tokenizer.Merge;
const base_vocab_size = tokenizer.base_vocab_size;
pub const max_vocab_size: usize = 4096;
pub const default_vocab_size: usize = 1024;

const none = std.math.maxInt(u32);

pub const Error = error{
    InvalidVocabSize,
    CorpusEncodingMismatch,
};

const Item = struct {
    count: u32,
    pair: u64,
};

/// Max-heap order: highest count first, then the smaller pair key. This is the
/// same deterministic tie-break the reference trainer used.
fn itemOrder(_: void, a: Item, b: Item) std.math.Order {
    if (a.count != b.count) return std.math.order(b.count, a.count);
    return std.math.order(a.pair, b.pair);
}

/// Symbol list + pair-position index + count heap.
const PairIndex = struct {
    allocator: std.mem.Allocator,
    sym: []u32,
    next: []u32,
    prev: []u32,
    pair_next: []u32,
    pair_prev: []u32,
    pair_head: std.AutoHashMap(u64, u32),
    counts: std.AutoHashMap(u64, u32),
    heap: std.PriorityQueue(Item, void, itemOrder),
    current: u64 = 0,
    in_merge: bool = false,

    fn init(allocator: std.mem.Allocator, sym: []u32, next: []u32, prev: []u32) !PairIndex {
        const pair_next = try allocator.alloc(u32, sym.len);
        const pair_prev = allocator.alloc(u32, sym.len) catch |e| {
            allocator.free(pair_next);
            return e;
        };
        @memset(pair_next, none);
        @memset(pair_prev, none);

        var self = PairIndex{
            .allocator = allocator,
            .sym = sym,
            .next = next,
            .prev = prev,
            .pair_next = pair_next,
            .pair_prev = pair_prev,
            .pair_head = std.AutoHashMap(u64, u32).init(allocator),
            .counts = std.AutoHashMap(u64, u32).init(allocator),
            .heap = std.PriorityQueue(Item, void, itemOrder).initContext({}),
        };
        errdefer self.deinit();

        for (0..sym.len) |i| {
            if (self.next[i] != none) try self.addPair(@intCast(i));
        }
        return self;
    }

    fn deinit(self: *PairIndex) void {
        self.allocator.free(self.pair_next);
        self.allocator.free(self.pair_prev);
        self.pair_head.deinit();
        self.counts.deinit();
        self.heap.deinit(self.allocator);
    }

    fn key(self: *const PairIndex, i: u32) u64 {
        return (@as(u64, self.sym[i]) << 32) | self.sym[self.next[i]];
    }

    fn addPair(self: *PairIndex, i: u32) !void {
        const k = self.key(i);
        if (self.in_merge and k == self.current) return;

        const old = self.pair_head.get(k) orelse none;
        self.pair_prev[i] = none;
        self.pair_next[i] = old;
        if (old != none) self.pair_prev[old] = i;
        try self.pair_head.put(k, i);
        try self.updateCount(k, 1);
    }

    fn removePair(self: *PairIndex, i: u32) !void {
        const k = self.key(i);
        if (self.in_merge and k == self.current) return;

        const pv = self.pair_prev[i];
        const nx = self.pair_next[i];
        if (pv != none) {
            self.pair_next[pv] = nx;
        } else if (nx == none) {
            _ = self.pair_head.remove(k);
        } else {
            try self.pair_head.put(k, nx);
        }
        if (nx != none) self.pair_prev[nx] = pv;
        self.pair_prev[i] = none;
        self.pair_next[i] = none;
        try self.updateCount(k, -1);
    }

    fn updateCount(self: *PairIndex, k: u64, delta: i32) !void {
        const old = self.counts.get(k) orelse 0;
        if (delta > 0) {
            const new = old + 1;
            try self.counts.put(k, new);
            if (new >= 2) try self.heap.push(self.allocator, .{ .count = new, .pair = k });
        } else {
            if (old == 0) return;
            const new = old - 1;
            if (new == 0) {
                _ = self.counts.remove(k);
                _ = self.pair_head.remove(k);
            } else {
                try self.counts.put(k, new);
                if (new >= 2) try self.heap.push(self.allocator, .{ .count = new, .pair = k });
            }
        }
    }

    /// Pop the highest-count pair whose heap entry is still current.
    fn nextBest(self: *PairIndex) ?Item {
        while (self.heap.pop()) |item| {
            const c = self.counts.get(item.pair) orelse continue;
            if (c == item.count and c >= 2) return item;
        }
        return null;
    }
};

pub fn train(allocator: std.mem.Allocator, corpus: []const u8, target_vocab: usize) ![]u8 {
    if (target_vocab < base_vocab_size or target_vocab > max_vocab_size) return Error.InvalidVocabSize;

    const ranges = try pre_tokenize.findRanges(allocator, corpus);
    defer allocator.free(ranges);

    const sym = try allocator.alloc(u32, corpus.len);
    defer allocator.free(sym);
    const next = try allocator.alloc(u32, corpus.len);
    defer allocator.free(next);
    const prev = try allocator.alloc(u32, corpus.len);
    defer allocator.free(prev);

    var pos: usize = 0;
    for (ranges) |range| {
        const piece = corpus[range.start..range.end];
        for (piece, 0..) |b, k| {
            if (pos >= sym.len) return Error.CorpusEncodingMismatch;
            sym[pos] = b;
            prev[pos] = if (k == 0) none else @intCast(pos - 1);
            next[pos] = if (k + 1 < piece.len) @intCast(pos + 1) else none;
            pos += 1;
        }
    }
    if (pos != corpus.len) return Error.CorpusEncodingMismatch;

    var vocab: std.ArrayList([]u8) = .empty;
    defer {
        for (vocab.items) |token| allocator.free(token);
        vocab.deinit(allocator);
    }
    try vocab.ensureTotalCapacity(allocator, target_vocab);
    for (0..base_vocab_size) |b| {
        try vocab.append(allocator, try byte_encoding.byteToken(allocator, @intCast(b)));
    }

    var merges: std.ArrayList(Merge) = .empty;
    defer merges.deinit(allocator);

    var index = try PairIndex.init(allocator, sym, next, prev);
    defer index.deinit();

    var positions: std.ArrayList(u32) = .empty;
    defer positions.deinit(allocator);

    while (vocab.items.len < target_vocab) {
        const item = index.nextBest() orelse break;
        const left: u32 = @intCast(item.pair >> 32);
        const right: u32 = @as(u32, @truncate(item.pair));

        positions.clearRetainingCapacity();
        var p = index.pair_head.get(item.pair) orelse none;
        while (p != none) {
            try positions.append(allocator, p);
            p = index.pair_next[p];
        }
        std.mem.sort(u32, positions.items, {}, std.sort.asc(u32));

        index.in_merge = true;
        index.current = item.pair;
        _ = index.counts.remove(item.pair);

        const new_id: u32 = @intCast(vocab.items.len);
        const merged = try concat(allocator, vocab.items[left], vocab.items[right]);
        try vocab.append(allocator, merged);
        try merges.append(allocator, .{ left, right });

        for (positions.items) |i| {
            if (index.sym[i] != left) continue;
            const j = index.next[i];
            if (j == none or index.sym[j] != right) continue;
            const pv = index.prev[i];
            const nn = index.next[j];

            if (pv != none) try index.removePair(pv);
            try index.removePair(i);
            if (nn != none) try index.removePair(j);

            index.sym[i] = new_id;
            index.next[i] = nn;
            if (nn != none) index.prev[nn] = i;
            index.next[j] = none;
            index.prev[j] = none;

            if (pv != none) try index.addPair(pv);
            if (nn != none) try index.addPair(i);
        }

        index.in_merge = false;
        _ = index.pair_head.remove(item.pair);
    }

    return tokenizer.Tokenizer.serializeBlob(allocator, vocab.items, merges.items);
}

fn concat(allocator: std.mem.Allocator, left: []const u8, right: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, left.len + right.len);
    @memcpy(out[0..left.len], left);
    @memcpy(out[left.len..], right);
    return out;
}

/// The original full-rescan trainer, kept as the correctness oracle for tests.
/// It is dead code in release builds.
fn trainReference(allocator: std.mem.Allocator, corpus: []const u8, target_vocab: usize) ![]u8 {
    if (target_vocab < base_vocab_size or target_vocab > max_vocab_size) return Error.InvalidVocabSize;

    const pieces = try pre_tokenize.findAll(allocator, corpus);
    defer {
        for (pieces) |p| allocator.free(p);
        allocator.free(pieces);
    }

    const symbols = try allocator.alloc(u32, corpus.len);
    defer allocator.free(symbols);
    const starts = try allocator.alloc(usize, pieces.len);
    defer allocator.free(starts);
    const lens = try allocator.alloc(usize, pieces.len);
    defer allocator.free(lens);

    var pos: usize = 0;
    for (pieces, 0..) |piece, i| {
        starts[i] = pos;
        for (piece) |b| {
            symbols[pos] = b;
            pos += 1;
        }
        lens[i] = piece.len;
    }

    var vocab: std.ArrayList([]u8) = .empty;
    defer {
        for (vocab.items) |token| allocator.free(token);
        vocab.deinit(allocator);
    }
    for (0..base_vocab_size) |b| {
        try vocab.append(allocator, try byte_encoding.byteToken(allocator, @intCast(b)));
    }

    var merges: std.ArrayList(Merge) = .empty;
    defer merges.deinit(allocator);

    const counts = try allocator.alloc(u32, target_vocab * target_vocab);
    defer allocator.free(counts);
    @memset(counts, 0);
    var touched: std.ArrayList(usize) = .empty;
    defer touched.deinit(allocator);

    while (vocab.items.len < target_vocab) {
        touched.clearRetainingCapacity();

        for (pieces, 0..) |_, i| {
            var r = starts[i];
            const end = r + lens[i];
            while (r + 1 < end) : (r += 1) {
                const key = @as(usize, symbols[r]) * target_vocab + symbols[r + 1];
                if (counts[key] == 0) try touched.append(allocator, key);
                counts[key] += 1;
            }
        }

        var best_key: usize = 0;
        var best_count: u32 = 0;
        for (touched.items) |key| {
            const c = counts[key];
            if (c > best_count or (c == best_count and c > 0 and key < best_key)) {
                best_count = c;
                best_key = key;
            }
        }
        if (best_count < 2) break;

        const left: u32 = @intCast(best_key / target_vocab);
        const right: u32 = @intCast(best_key % target_vocab);
        const new_id: u32 = @intCast(vocab.items.len);

        const merged = try concat(allocator, vocab.items[left], vocab.items[right]);
        try vocab.append(allocator, merged);
        try merges.append(allocator, .{ left, right });

        var write: usize = 0;
        for (0..pieces.len) |i| {
            const new_start = write;
            var r = starts[i];
            const end = r + lens[i];
            while (r < end) {
                if (r + 1 < end and symbols[r] == left and symbols[r + 1] == right) {
                    symbols[write] = new_id;
                    write += 1;
                    r += 2;
                } else {
                    symbols[write] = symbols[r];
                    write += 1;
                    r += 1;
                }
            }
            starts[i] = new_start;
            lens[i] = write - new_start;
        }

        for (touched.items) |key| counts[key] = 0;
    }

    return tokenizer.Tokenizer.serializeBlob(allocator, vocab.items, merges.items);
}

test "trains merges and round-trips through encode/decode" {
    const allocator = std.testing.allocator;
    const corpus = "abab abab abab\nbanana bandana\n";
    const blob = try train(allocator, corpus, 258);
    defer allocator.free(blob);

    var tk = try tokenizer.Tokenizer.initFromBlob(allocator, blob);
    defer tk.deinit();
    try std.testing.expect(tk.vocabSize() > base_vocab_size);
    try std.testing.expect(tk.vocabSize() <= 258);

    const ids = try tk.encode(corpus);
    defer allocator.free(ids);
    const text = try tk.decode(ids);
    defer allocator.free(text);
    try std.testing.expectEqualStrings(corpus, text);
}

test "optimized trainer matches the full-rescan reference" {
    const allocator = std.testing.allocator;
    const corpus =
        "aaaa aaaa abab abab\n\nbanana bandana the quick brown fox\n" ++
        "the the the quick quick brown fox jumps over the lazy dog\n" ++
        "abab abab abab aa aa aa\n";
    const target = 300;

    const fast = try train(allocator, corpus, target);
    defer allocator.free(fast);
    const reference = try trainReference(allocator, corpus, target);
    defer allocator.free(reference);

    try std.testing.expectEqualSlices(u8, reference, fast);
}

test "a 256-token target emits the byte alphabet only" {
    const allocator = std.testing.allocator;
    const blob = try train(allocator, "abc", base_vocab_size);
    defer allocator.free(blob);

    var tk = try tokenizer.Tokenizer.initFromBlob(allocator, blob);
    defer tk.deinit();
    try std.testing.expectEqual(base_vocab_size, tk.vocabSize());

    const ids = try tk.encode("abc");
    defer allocator.free(ids);
    try std.testing.expectEqualSlices(u21, &.{ 97, 98, 99 }, ids);
}

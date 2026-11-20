//! Pure-Zig replacement for bpe.zig's POSIX-regex pre-tokenizer.
//!
//! Mirrors GPT-2's pattern for ASCII text:
//!   's|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+|
//!   ?[^[:alnum:][:space:]]+| +[[:space:]]*| +
//!
//! The upstream implementation calls libc `regcomp`/`regexec` via `@cImport`;
//! this keeps the tokenizer pure Zig and portable.
const std = @import("std");

pub const Range = struct { start: usize, end: usize };

/// Split `text` into the pre-token pieces the BPE merges operate on, as
/// offsets into `text`. The caller frees the returned array; the text is not
/// copied.
pub fn findRanges(allocator: std.mem.Allocator, text: []const u8) ![]Range {
    var ranges: std.ArrayList(Range) = .empty;
    errdefer ranges.deinit(allocator);

    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        const c = text[i];
        const next = if (i + 1 < text.len) text[i + 1] else 0;

        if (c == '\'') {
            if (contractionLen(text[i..])) |n| {
                i += n;
            } else {
                i = consumePunct(text, i);
            }
        } else if (c == ' ' and isAlpha(next)) {
            i += 1;
            while (i < text.len and isAlpha(text[i])) i += 1;
        } else if (isAlpha(c)) {
            while (i < text.len and isAlpha(text[i])) i += 1;
        } else if (c == ' ' and isDigit(next)) {
            i += 1;
            while (i < text.len and isDigit(text[i])) i += 1;
        } else if (isDigit(c)) {
            while (i < text.len and isDigit(text[i])) i += 1;
        } else if (c == ' ' and isPunct(next)) {
            i += 1;
            i = consumePunct(text, i);
        } else if (isPunct(c)) {
            i = consumePunct(text, i);
        } else if (isSpace(c)) {
            var j = i;
            while (j < text.len and isSpace(text[j])) j += 1;
            // GPT-2's `\s+(?!\S)|\s+`: a whitespace run that continues into
            // non-space leaves its LAST whitespace for the next match (so a
            // following " word" keeps its leading space).
            i = if (j < text.len and j - i > 1) j - 1 else j;
        } else {
            // Non-ASCII UTF-8 leads here; emit one byte. The byte-level BPE
            // alphabet covers all 256 values, so this still round-trips.
            i += 1;
        }

        if (i == start) i += 1;
        try ranges.append(allocator, .{ .start = start, .end = i });
    }

    return ranges.toOwnedSlice(allocator);
}

/// Copying wrapper: one owned slice per piece (tests and legacy callers).
pub fn findAll(allocator: std.mem.Allocator, text: []const u8) ![][]const u8 {
    const ranges = try findRanges(allocator, text);
    defer allocator.free(ranges);

    var matches: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (matches.items) |m| allocator.free(m);
        matches.deinit(allocator);
    }
    try matches.ensureTotalCapacity(allocator, ranges.len);
    for (ranges) |r| {
        try matches.append(allocator, try allocator.dupe(u8, text[r.start..r.end]));
    }
    return matches.toOwnedSlice(allocator);
}

fn contractionLen(s: []const u8) ?usize {
    if (s.len < 2 or s[0] != '\'') return null;
    if (std.mem.startsWith(u8, s, "'re")) return 3;
    if (std.mem.startsWith(u8, s, "'ve")) return 3;
    if (std.mem.startsWith(u8, s, "'ll")) return 3;
    return switch (s[1]) {
        's', 't', 'm', 'd' => 2,
        else => null,
    };
}

fn consumePunct(text: []const u8, start: usize) usize {
    var i = start;
    while (i < text.len and isPunct(text[i])) i += 1;
    return i;
}

fn isAlpha(c: u8) bool {
    return std.ascii.isAlphabetic(c);
}

fn isDigit(c: u8) bool {
    return std.ascii.isDigit(c);
}

fn isSpace(c: u8) bool {
    return std.ascii.isWhitespace(c);
}

fn isPunct(c: u8) bool {
    return !isAlpha(c) and !isDigit(c) and !isSpace(c);
}

test "pre-tokenizes the GPT-2 example" {
    const allocator = std.testing.allocator;
    const text = "Hello, I'm a test string with numbers 123 and symbols @#$!";
    const matches = try findAll(allocator, text);
    defer {
        for (matches) |m| allocator.free(m);
        allocator.free(matches);
    }

    try std.testing.expectEqual(@as(usize, 13), matches.len);
    try std.testing.expectEqualStrings("Hello", matches[0]);
    try std.testing.expectEqualStrings(",", matches[1]);
    try std.testing.expectEqualStrings(" I", matches[2]);
    try std.testing.expectEqualStrings("'m", matches[3]);
    try std.testing.expectEqualStrings(" a", matches[4]);
    try std.testing.expectEqualStrings(" 123", matches[9]);
    try std.testing.expectEqualStrings(" @#$!", matches[12]);
}

test "keeps newline runs in streams" {
    const allocator = std.testing.allocator;
    const text = "A\n\nB";
    const matches = try findAll(allocator, text);
    defer {
        for (matches) |m| allocator.free(m);
        allocator.free(matches);
    }

    try std.testing.expectEqual(@as(usize, 4), matches.len);
    try std.testing.expectEqualStrings("A", matches[0]);
    try std.testing.expectEqualStrings("\n", matches[1]);
    try std.testing.expectEqualStrings("\n", matches[2]);
    try std.testing.expectEqualStrings("B", matches[3]);
}

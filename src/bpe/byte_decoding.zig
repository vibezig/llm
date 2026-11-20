//! Reverse of `byte_encoding.zig`: token strings back to raw bytes.
//! Vendored from alvarobartt/bpe.zig (MIT OR Apache-2.0), ported to Zig 0.16.
const std = @import("std");
const BYTE_TO_TOKEN_MAPPING = @import("byte_encoding.zig").BYTE_TO_TOKEN_MAPPING;

const TokenToByteMapping = struct {
    token_to_byte: [324]u8,
};

fn initializeTokenToByteMapping() TokenToByteMapping {
    var token_to_byte: [324]u8 = undefined;

    for (0..256) |byte_value| {
        const byte = @as(u8, @intCast(byte_value));
        const token_id = BYTE_TO_TOKEN_MAPPING.byte_to_token_map[byte];
        token_to_byte[token_id] = byte;
    }

    return TokenToByteMapping{ .token_to_byte = token_to_byte };
}

const TOKEN_TO_BYTE_MAPPING = initializeTokenToByteMapping();

pub fn tokensToBytes(allocator: std.mem.Allocator, encoded: []const u8) ![]u8 {
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(allocator);

    var view = try std.unicode.Utf8View.init(encoded);
    var it = view.iterator();
    while (it.nextCodepoint()) |codepoint| {
        const cp = @as(usize, codepoint);
        if (cp >= TOKEN_TO_BYTE_MAPPING.token_to_byte.len) return error.InvalidToken;
        try output.append(allocator, TOKEN_TO_BYTE_MAPPING.token_to_byte[cp]);
    }

    return output.toOwnedSlice(allocator);
}

/// Zero-allocation decode into `out`; returns the written prefix.
pub fn tokensToBytesInto(encoded: []const u8, out: []u8) ![]u8 {
    var view = try std.unicode.Utf8View.init(encoded);
    var it = view.iterator();
    var n: usize = 0;
    while (it.nextCodepoint()) |codepoint| {
        const cp = @as(usize, codepoint);
        if (cp >= TOKEN_TO_BYTE_MAPPING.token_to_byte.len) return error.InvalidToken;
        if (n >= out.len) return error.NoSpaceLeft;
        out[n] = TOKEN_TO_BYTE_MAPPING.token_to_byte[cp];
        n += 1;
    }
    return out[0..n];
}

test "tokensToBytes" {
    const allocator = std.testing.allocator;

    const input = try tokensToBytes(allocator, "helloĠworld");
    defer allocator.free(input);
    try std.testing.expectEqualStrings("hello world", input);
}

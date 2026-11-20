//! Byte-level token alphabet shared with GPT-2's BPE convention.
//! Vendored from alvarobartt/bpe.zig (MIT OR Apache-2.0), ported to Zig 0.16.
const std = @import("std");

const ByteToTokenMapping = struct {
    byte_to_token_map: [256]u21,
};

fn initializeByteToTokenMapping() ByteToTokenMapping {
    var byte_to_token_map: [256]u21 = undefined;
    var is_byte_mapped = [_]bool{false} ** 256;

    for (33..127) |byte_value| {
        const byte = @as(u8, @intCast(byte_value));
        is_byte_mapped[byte] = true;
        byte_to_token_map[byte] = byte;
    }
    for (161..173) |byte_value| {
        const byte = @as(u8, @intCast(byte_value));
        is_byte_mapped[byte] = true;
        byte_to_token_map[byte] = byte;
    }
    for (174..256) |byte_value| {
        const byte = @as(u8, @intCast(byte_value));
        is_byte_mapped[byte] = true;
        byte_to_token_map[byte] = byte;
    }

    var next_token_id: u16 = 0;
    for (0..256) |byte_value| {
        const byte = @as(u8, @intCast(byte_value));
        if (!is_byte_mapped[byte]) {
            byte_to_token_map[byte] = 256 + next_token_id;
            next_token_id += 1;
        }
    }

    return ByteToTokenMapping{ .byte_to_token_map = byte_to_token_map };
}

pub const BYTE_TO_TOKEN_MAPPING = initializeByteToTokenMapping();

pub fn bytesToTokens(allocator: std.mem.Allocator, utf8_input: []const u8) ![]u8 {
    var output_buffer: std.ArrayList(u8) = .empty;
    errdefer output_buffer.deinit(allocator);
    try bytesToTokensInto(allocator, utf8_input, &output_buffer);
    return output_buffer.toOwnedSlice(allocator);
}

/// Append the byte-level token string for `utf8_input` to `out` (the caller
/// clears it). Avoids a per-piece allocation in the encoder's hot loop.
pub fn bytesToTokensInto(allocator: std.mem.Allocator, utf8_input: []const u8, out: *std.ArrayList(u8)) !void {
    for (utf8_input) |utf8_byte| {
        const token_id = BYTE_TO_TOKEN_MAPPING.byte_to_token_map[utf8_byte];
        var encoded_utf8: [4]u8 = undefined;
        const encoded_length = try std.unicode.utf8Encode(token_id, &encoded_utf8);
        try out.appendSlice(allocator, encoded_utf8[0..@as(usize, encoded_length)]);
    }
}

/// The token string for one raw byte (used by the BPE trainer's initial vocab).
pub fn byteToken(allocator: std.mem.Allocator, byte: u8) ![]u8 {
    const token_id = BYTE_TO_TOKEN_MAPPING.byte_to_token_map[byte];
    var encoded: [4]u8 = undefined;
    const len = try std.unicode.utf8Encode(token_id, &encoded);
    return allocator.dupe(u8, encoded[0..@as(usize, len)]);
}

test "bytesToTokens" {
    const allocator = std.testing.allocator;

    const output = try bytesToTokens(allocator, "hello world");
    defer allocator.free(output);
    try std.testing.expectEqualStrings("helloĠworld", output);
}

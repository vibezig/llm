//! Small deterministic RNG helpers (xoshiro256++), used for reproducible
//! weight initialisation and token sampling.
const std = @import("std");

pub const Prng = std.Random.DefaultPrng;

pub fn init(seed: u64) Prng {
    return Prng.init(seed);
}

/// Fisher-Yates shuffle, deterministic for a given RNG.
pub fn shuffle(rng: std.Random, items: []u32) void {
    var i: usize = items.len;
    while (i > 1) {
        i -= 1;
        const j = rng.intRangeAtMost(usize, 0, i);
        const tmp = items[i];
        items[i] = items[j];
        items[j] = tmp;
    }
}

test "shuffle is deterministic per seed and keeps the multiset" {
    var a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var b = a;
    var p1 = init(42);
    var p2 = init(42);
    shuffle(p1.random(), &a);
    shuffle(p2.random(), &b);
    try std.testing.expectEqualSlices(u32, &a, &b);
    std.mem.sort(u32, &a, {}, std.sort.asc(u32));
    for (a, 1..) |v, i| try std.testing.expectEqual(@as(u32, @intCast(i)), v);
}

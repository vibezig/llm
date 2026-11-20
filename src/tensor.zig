//! Core tensor math for the vibezig LLM.
//!
//! Everything is a row-major `f32` buffer. A `Buf` carries its 2-D shape so the
//! transformer code never has to juggle naked slices, while the hot kernels
//! (`matmul`, `matmulNt`, `softmax`...) take plain slices so they can operate on
//! views without copying.
//!
//! Conventions:
//!   * `matmul`   : c[m,n] = a[m,k] @ b[k,n]         (b is "input-major")
//!   * `matmulNt` : c[m,n] = a[m,k] @ b[n,k]^T      (b is "output-major")
//!
//! Both kernels accumulate, so callers can chain attention heads or split a
//! matmul over several weight shards.
const std = @import("std");
const assert = std.debug.assert;

pub const eps: f32 = 1e-5;

/// A 2-D tensor (row-major).
pub const Buf = struct {
    data: []f32,
    rows: usize,
    cols: usize,
    allocator: ?std.mem.Allocator = null,

    pub fn init(allocator: std.mem.Allocator, rows: usize, cols: usize) !Buf {
        const data = try allocator.alloc(f32, rows * cols);
        @memset(data, 0);
        return .{ .data = data, .rows = rows, .cols = cols, .allocator = allocator };
    }

    pub fn deinit(self: *Buf) void {
        if (self.allocator) |a| a.free(self.data);
        self.* = undefined;
    }

    pub fn at(self: *const Buf, r: usize, c: usize) *f32 {
        return &self.data[r * self.cols + c];
    }

    pub fn get(self: *const Buf, r: usize, c: usize) f32 {
        return self.data[r * self.cols + c];
    }

    pub fn row(self: *const Buf, r: usize) []f32 {
        return self.data[r * self.cols ..][0..self.cols];
    }

    pub fn zero(self: *Buf) void {
        @memset(self.data, 0);
    }
};

pub fn zero(s: []f32) void {
    @memset(s, 0);
}

pub fn fill(s: []f32, v: f32) void {
    @memset(s, v);
}

pub fn copy(dst: []f32, src: []const f32) void {
    assert(dst.len >= src.len);
    @memcpy(dst[0..src.len], src);
}

/// c[m,n] = a[m,k] @ b[k,n]
///
/// 4x8 register-blocked kernel: four rows of `a` share one vector of `b`, which
/// cuts `b` traffic by 4x and lets the FMA loop stay in registers. This is the
/// single hottest function in training, so the blocking is worth the extra code.
pub fn matmul(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    assert(a.len >= m * k);
    assert(b.len >= k * n);
    assert(c.len >= m * n);

    const V = 8;
    var i: usize = 0;
    while (i + 4 <= m) : (i += 4) {
        var j: usize = 0;
        while (j + V <= n) : (j += V) {
            var acc0: @Vector(V, f32) = @splat(0);
            var acc1: @Vector(V, f32) = @splat(0);
            var acc2: @Vector(V, f32) = @splat(0);
            var acc3: @Vector(V, f32) = @splat(0);
            for (0..k) |p| {
                const bv: @Vector(V, f32) = b[p * n + j ..][0..V].*;
                acc0 += @as(@Vector(V, f32), @splat(a[(i + 0) * k + p])) * bv;
                acc1 += @as(@Vector(V, f32), @splat(a[(i + 1) * k + p])) * bv;
                acc2 += @as(@Vector(V, f32), @splat(a[(i + 2) * k + p])) * bv;
                acc3 += @as(@Vector(V, f32), @splat(a[(i + 3) * k + p])) * bv;
            }
            c[(i + 0) * n + j ..][0..V].* = acc0;
            c[(i + 1) * n + j ..][0..V].* = acc1;
            c[(i + 2) * n + j ..][0..V].* = acc2;
            c[(i + 3) * n + j ..][0..V].* = acc3;
        }
        while (j < n) : (j += 1) {
            var acc: [4]f32 = .{ 0, 0, 0, 0 };
            for (0..k) |p| {
                const bv = b[p * n + j];
                acc[0] += a[(i + 0) * k + p] * bv;
                acc[1] += a[(i + 1) * k + p] * bv;
                acc[2] += a[(i + 2) * k + p] * bv;
                acc[3] += a[(i + 3) * k + p] * bv;
            }
            c[(i + 0) * n + j] = acc[0];
            c[(i + 1) * n + j] = acc[1];
            c[(i + 2) * n + j] = acc[2];
            c[(i + 3) * n + j] = acc[3];
        }
    }
    // leftover rows: single-row axpy
    while (i < m) : (i += 1) {
        const crow = c[i * n ..][0..n];
        @memset(crow, 0);
        for (0..k) |p| {
            const av = a[i * k + p];
            const brow = b[p * n ..][0..n];
            for (crow, brow) |*cv, bv| cv.* += av * bv;
        }
    }
}

/// c[m,n] += a[m,k] @ b[k,n]
pub fn matmulAcc(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    assert(a.len >= m * k);
    assert(b.len >= k * n);
    assert(c.len >= m * n);
    for (0..m) |i| {
        const crow = c[i * n ..][0..n];
        const arow = a[i * k ..][0..k];
        for (0..k) |p| {
            const av = arow[p];
            const brow = b[p * n ..][0..n];
            for (crow, brow) |*cv, bv| cv.* += av * bv;
        }
    }
}

/// c[m,n] = a[m,k] @ b[n,k]^T  (i.e. dot products against transposed b)
pub fn matmulNt(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    assert(a.len >= m * k);
    assert(b.len >= n * k);
    assert(c.len >= m * n);
    const V = 8;
    for (0..m) |i| {
        const arow = a[i * k ..][0..k];
        for (0..n) |j| {
            const brow = b[j * k ..][0..k];
            var acc: @Vector(V, f32) = @splat(0);
            var p: usize = 0;
            while (p + V <= k) : (p += V) {
                const av: @Vector(V, f32) = arow[p..][0..V].*;
                const bv: @Vector(V, f32) = brow[p..][0..V].*;
                acc += av * bv;
            }
            var sum: f32 = @reduce(.Add, acc);
            while (p < k) : (p += 1) sum += arow[p] * brow[p];
            c[i * n + j] = sum;
        }
    }
}

/// c[m,n] += a[k,m]^T @ b[k,n]  — gradient of a weight matrix.
///
/// `a` is the activation matrix (rows = tokens) and `b` the upstream gradient;
/// both are read row-wise, which keeps the 4x8 blocking cache friendly.
pub fn matmulTnAcc(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    assert(a.len >= k * m);
    assert(b.len >= k * n);
    assert(c.len >= m * n);
    const V = 8;
    var i: usize = 0;
    while (i + 4 <= m) : (i += 4) {
        var j: usize = 0;
        while (j + V <= n) : (j += V) {
            var acc0: @Vector(V, f32) = @splat(0);
            var acc1: @Vector(V, f32) = @splat(0);
            var acc2: @Vector(V, f32) = @splat(0);
            var acc3: @Vector(V, f32) = @splat(0);
            for (0..k) |r| {
                const bv: @Vector(V, f32) = b[r * n + j ..][0..V].*;
                acc0 += @as(@Vector(V, f32), @splat(a[r * m + i + 0])) * bv;
                acc1 += @as(@Vector(V, f32), @splat(a[r * m + i + 1])) * bv;
                acc2 += @as(@Vector(V, f32), @splat(a[r * m + i + 2])) * bv;
                acc3 += @as(@Vector(V, f32), @splat(a[r * m + i + 3])) * bv;
            }
            inline for (0..4) |d| {
                const idx = (i + d) * n + j;
                const prev: @Vector(V, f32) = c[idx..][0..V].*;
                c[idx..][0..V].* = prev + switch (d) {
                    0 => acc0,
                    1 => acc1,
                    2 => acc2,
                    else => acc3,
                };
            }
        }
        while (j < n) : (j += 1) {
            var acc: [4]f32 = .{ 0, 0, 0, 0 };
            for (0..k) |r| {
                const bv = b[r * n + j];
                acc[0] += a[r * m + i + 0] * bv;
                acc[1] += a[r * m + i + 1] * bv;
                acc[2] += a[r * m + i + 2] * bv;
                acc[3] += a[r * m + i + 3] * bv;
            }
            inline for (0..4) |d| c[(i + d) * n + j] += acc[d];
        }
    }
    while (i < m) : (i += 1) {
        const crow = c[i * n ..][0..n];
        for (0..k) |r| {
            const av = a[r * m + i];
            const brow = b[r * n ..][0..n];
            for (crow, brow) |*cv, bv| cv.* += av * bv;
        }
    }
}

/// c[m,n] += a[m,k] @ b[n,k]^T
pub fn matmulNtAcc(m: usize, n: usize, k: usize, a: []const f32, b: []const f32, c: []f32) void {
    for (0..m) |i| {
        const arow = a[i * k ..][0..k];
        for (0..n) |j| {
            const brow = b[j * k ..][0..k];
            var acc: f32 = 0;
            for (arow, brow) |av, bv| acc += av * bv;
            c[i * n + j] += acc;
        }
    }
}

/// out = x @ w where x is [m,k], w is [k,n], and out is [m,n] (thin kernel wrapper).
pub fn linear(m: usize, n: usize, k: usize, x: []const f32, w: []const f32, out: []f32) void {
    matmul(m, n, k, x, w, out);
}

/// y = a*x + y
pub fn axpy(n: usize, a: f32, x: []const f32, y: []f32) void {
    for (0..n) |i| y[i] += a * x[i];
}

pub fn addInPlace(dst: []f32, src: []const f32) void {
    assert(dst.len == src.len);
    for (dst, src) |*d, s| d.* += s;
}

pub fn scaleInPlace(s: []f32, v: f32) void {
    for (s) |*x| x.* *= v;
}

pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    var acc: f32 = 0;
    for (a, b) |av, bv| acc += av * bv;
    return acc;
}

/// In-place numerically stable softmax over `s`.
pub fn softmax(s: []f32) void {
    var max: f32 = s[0];
    for (s) |v| max = @max(max, v);
    var sum: f32 = 0;
    for (s) |*v| {
        v.* = @exp(v.* - max);
        sum += v.*;
    }
    const inv = 1.0 / sum;
    for (s) |*v| v.* *= inv;
}

/// Row-wise layer norm. Writes normalized output plus per-row mean / rstd so the
/// backward pass can reuse them.
pub fn layerNorm(
    x: []const f32,
    g: []const f32,
    b: []const f32,
    y: []f32,
    mean: []f32,
    rstd: []f32,
    rows: usize,
    cols: usize,
) void {
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(cols));
    for (0..rows) |r| {
        const xr = x[r * cols ..][0..cols];
        var mu: f32 = 0;
        for (xr) |v| mu += v;
        mu *= inv_n;
        var var_: f32 = 0;
        for (xr) |v| {
            const d = v - mu;
            var_ += d * d;
        }
        var_ *= inv_n;
        const rs = 1.0 / @sqrt(var_ + eps);
        mean[r] = mu;
        rstd[r] = rs;
        const yr = y[r * cols ..][0..cols];
        for (xr, yr, 0..) |v, *o, i| o.* = (v - mu) * rs * g[i] + b[i];
    }
}

/// Backward for `layerNorm`. Accumulates parameter grads, writes dx.
pub fn layerNormBwd(
    dy: []const f32,
    x: []const f32,
    g: []const f32,
    mean: []const f32,
    rstd: []const f32,
    dx: []f32,
    dg: []f32,
    db: []f32,
    rows: usize,
    cols: usize,
) void {
    const inv_n: f32 = 1.0 / @as(f32, @floatFromInt(cols));
    for (0..rows) |r| {
        const mu = mean[r];
        const rs = rstd[r];
        const xr = x[r * cols ..][0..cols];
        const dyr = dy[r * cols ..][0..cols];
        const dxr = dx[r * cols ..][0..cols];

        var sum_dy_g: f32 = 0;
        var sum_dy_g_xhat: f32 = 0;
        for (0..cols) |i| {
            const xhat = (xr[i] - mu) * rs;
            const dyg = dyr[i] * g[i];
            dg[i] += dyr[i] * xhat;
            db[i] += dyr[i];
            sum_dy_g += dyg;
            sum_dy_g_xhat += dyg * xhat;
        }
        for (0..cols) |i| {
            const xhat = (xr[i] - mu) * rs;
            dxr[i] += rs * (dyr[i] * g[i] - sum_dy_g * inv_n - xhat * sum_dy_g_xhat * inv_n);
        }
    }
}

const gelu_c: f32 = 0.7978845608028654; // sqrt(2/pi)
const gelu_k: f32 = 0.044715;

/// tanh-approximated GELU, in place.
pub fn geluInPlace(x: []f32) void {
    for (x) |*v| {
        const xi = v.*;
        const inner = gelu_c * (xi + gelu_k * xi * xi * xi);
        v.* = 0.5 * xi * (1.0 + std.math.tanh(inner));
    }
}

/// Backward of tanh-approximated GELU: dx += dy * dgelu(x).
pub fn geluBwdAcc(dy: []const f32, x: []const f32, dx: []f32) void {
    for (dy, x, dx) |dyv, xv, *dxv| {
        const t = std.math.tanh(gelu_c * (xv + gelu_k * xv * xv * xv));
        const d = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * gelu_c * (1.0 + 3.0 * gelu_k * xv * xv);
        dxv.* += dyv * d;
    }
}

/// Softmax cross entropy over `rows` rows of `logits` [rows,vocab].
/// Writes `dlogits` (already divided by `rows`, i.e. mean-loss gradient) and
/// returns the mean loss.
pub fn softmaxCrossEntropy(logits: []const f32, targets: []const u32, dlogits: []f32, rows: usize, vocab: usize) f32 {
    var loss: f32 = 0;
    const inv_rows: f32 = 1.0 / @as(f32, @floatFromInt(rows));
    for (0..rows) |r| {
        const lr = logits[r * vocab ..][0..vocab];
        const dr = dlogits[r * vocab ..][0..vocab];
        var max: f32 = lr[0];
        for (lr) |v| max = @max(max, v);
        var sum: f32 = 0;
        for (lr) |v| sum += @exp(v - max);
        const logsum = @log(sum) + max;
        loss += logsum - lr[targets[r]];
        const inv = 1.0 / sum;
        for (lr, dr) |v, *d| d.* = @exp(v - max) * inv;
        dr[targets[r]] -= 1.0;
        for (dr) |*d| d.* *= inv_rows;
    }
    return loss * inv_rows;
}

/// Uniform random fill in [lo, hi).
pub fn uniform(rng: std.Random, s: []f32, lo: f32, hi: f32) void {
    for (s) |*v| v.* = rng.float(f32) * (hi - lo) + lo;
}

/// Normal(0, std) fill.
pub fn normal(rng: std.Random, s: []f32, mean: f32, stddev: f32) void {
    for (s) |*v| v.* = mean + stddev * rng.floatNorm(f32);
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "matmul matches naive reference" {
    const m, const n, const k = .{ 3, 4, 5 };
    var a: [15]f32 = undefined;
    var b: [20]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(7);
    uniform(rng.random(), &a, -1, 1);
    uniform(rng.random(), &b, -1, 1);

    var c: [12]f32 = undefined;
    matmul(m, n, k, &a, &b, &c);

    var want: [12]f32 = undefined;
    for (0..m) |i| for (0..n) |j| {
        var acc: f32 = 0;
        for (0..k) |p| acc += a[i * k + p] * b[p * n + j];
        want[i * n + j] = acc;
    };
    for (c, want) |got, exp| try testing.expectApproxEqAbs(exp, got, 1e-5);
}

test "matmulNt matches naive reference" {
    const m, const n, const k = .{ 3, 4, 5 };
    var a: [15]f32 = undefined;
    var b: [20]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(11);
    uniform(rng.random(), &a, -1, 1);
    uniform(rng.random(), &b, -1, 1);

    var c: [12]f32 = undefined;
    matmulNt(m, n, k, &a, &b, &c);

    var want: [12]f32 = undefined;
    for (0..m) |i| for (0..n) |j| {
        var acc: f32 = 0;
        for (0..k) |p| acc += a[i * k + p] * b[j * k + p];
        want[i * n + j] = acc;
    };
    for (c, want) |got, exp| try testing.expectApproxEqAbs(exp, got, 1e-5);
}

test "matmulTnAcc matches naive reference" {
    const m, const n, const k = .{ 6, 5, 3 };
    var a: [18]f32 = undefined;
    var b: [15]f32 = undefined;
    var c: [30]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(17);
    uniform(rng.random(), &a, -1, 1);
    uniform(rng.random(), &b, -1, 1);
    @memset(&c, 0.25);

    matmulTnAcc(m, n, k, &a, &b, &c);

    var want: [30]f32 = @splat(0.25);
    for (0..m) |i| for (0..n) |j| {
        for (0..k) |r| want[i * n + j] += a[r * m + i] * b[r * n + j];
    };
    for (c, want) |got, exp| try testing.expectApproxEqAbs(exp, got, 1e-5);
}

test "softmax sums to one and is order preserving" {
    var s = [_]f32{ 1.0, 2.0, 3.0, -100.0, 0.0 };
    softmax(&s);
    var sum: f32 = 0;
    for (s) |v| sum += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
    try testing.expect(s[2] > s[1] and s[1] > s[0]);
    try testing.expect(s[3] < 1e-6);
}

test "layerNorm has zero mean and unit variance" {
    const rows, const cols = .{ 2, 40 };
    var x: [80]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(3);
    uniform(rng.random(), &x, -3, 3);
    var g: [40]f32 = @splat(1.0);
    var b: [40]f32 = @splat(0.0);
    var y: [80]f32 = @splat(0.0);
    var mean: [2]f32 = @splat(0.0);
    var rstd: [2]f32 = @splat(0.0);
    layerNorm(&x, &g, &b, &y, &mean, &rstd, rows, cols);

    for (0..rows) |r| {
        var mu: f32 = 0;
        for (y[r * cols ..][0..cols]) |v| mu += v;
        mu /= @floatFromInt(cols);
        var va: f32 = 0;
        for (y[r * cols ..][0..cols]) |v| va += (v - mu) * (v - mu);
        va /= @floatFromInt(cols);
        try testing.expectApproxEqAbs(@as(f32, 0.0), mu, 1e-4);
        try testing.expectApproxEqAbs(@as(f32, 1.0), va, 1e-3);
    }
}

test "gelu matches reference value at x=1" {
    var x = [_]f32{1.0};
    geluInPlace(&x);
    try testing.expectApproxEqAbs(@as(f32, 0.841192), x[0], 1e-3);
}

test "layerNorm backward matches finite differences" {
    const rows, const cols = .{ 2, 6 };
    var rng = std.Random.DefaultPrng.init(21);
    var x: [12]f32 = undefined;
    var dy: [12]f32 = undefined;
    var g: [6]f32 = undefined;
    var bw: [6]f32 = undefined;
    uniform(rng.random(), &x, -2, 2);
    uniform(rng.random(), &dy, -1, 1);
    uniform(rng.random(), &g, 0.5, 1.5);
    @memset(&bw, 0);

    var y: [12]f32 = undefined;
    var mean: [2]f32 = undefined;
    var rstd: [2]f32 = undefined;
    layerNorm(&x, &g, &bw, &y, &mean, &rstd, rows, cols);

    var dx: [12]f32 = @splat(0.0);
    var dg: [6]f32 = @splat(0.0);
    var db: [6]f32 = @splat(0.0);
    layerNormBwd(&dy, &x, &g, &mean, &rstd, &dx, &dg, &db, rows, cols);

    // numeric grad wrt x[0][2]
    const h: f32 = 1e-3;
    const f = struct {
        fn loss(xx: []const f32, d: []const f32, gg: []const f32) f32 {
            var y2: [12]f32 = undefined;
            var m2: [2]f32 = @splat(0.0);
            var r2: [2]f32 = @splat(0.0);
            layerNorm(xx, gg, &@as([6]f32, @splat(0.0)), &y2, &m2, &r2, 2, 6);
            var acc: f32 = 0;
            for (y2, d) |v, dv| acc += v * dv;
            return acc;
        }
    };
    const plus = blk: {
        var xp = x;
        xp[2] += h;
        break :blk f.loss(&xp, &dy, &g);
    };
    const minus = blk: {
        var xm = x;
        xm[2] -= h;
        break :blk f.loss(&xm, &dy, &g);
    };
    try testing.expectApproxEqAbs((plus - minus) / (2 * h), dx[2], 2e-3);
}

test "softmaxCrossEntropy gradient matches finite differences" {
    const rows, const vocab = .{ 2, 5 };
    var rng = std.Random.DefaultPrng.init(5);
    var logits: [10]f32 = undefined;
    uniform(rng.random(), &logits, -1, 1);
    const targets = [_]u32{ 1, 4 };
    var d: [10]f32 = @splat(0.0);
    _ = softmaxCrossEntropy(&logits, &targets, &d, rows, vocab);

    const h: f32 = 1e-3;
    var lp = logits;
    lp[3] += h;
    var lm = logits;
    lm[3] -= h;
    var tmp: [10]f32 = undefined;
    const loss_p = softmaxCrossEntropy(&lp, &targets, &tmp, rows, vocab);
    const loss_m = softmaxCrossEntropy(&lm, &targets, &tmp, rows, vocab);
    try testing.expectApproxEqAbs((loss_p - loss_m) / (2 * h), d[3], 2e-3);
}

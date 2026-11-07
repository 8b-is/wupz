//! wupz-zig / upz-zig — the SIMD kernels behind `include/wupz_abi.h`.
//!
//! Every kernel ships with a **scalar reference** and is checked against it by a
//! differential test over seeded pseudo-random inputs (`docs/DESIGN.md` §6). No
//! kernel is trusted without parity; the tests below are the receipt.
//!
//! Zig owns these because they are tight, allocation-free loops where comptime
//! and explicit vectors pay off. Rust owns the linker model. They meet at the
//! C ABI — one header, versioned.

const std = @import("std");

// ── xorFold: integrity fold ─────────────────────────────────────────────────

/// Scalar reference: XOR of the little-endian u64 words, then the trailing bytes.
pub fn xorFoldScalar(bytes: []const u8) u64 {
    var acc: u64 = 0;
    var i: usize = 0;
    while (i + 8 <= bytes.len) : (i += 8) {
        acc ^= std.mem.readInt(u64, bytes[i..][0..8], .little);
    }
    while (i < bytes.len) : (i += 1) {
        acc ^= @as(u64, bytes[i]);
    }
    return acc;
}

/// Vectorized: dual @Vector(4, u64) accumulators (64B/step) with deferred horizontal @reduce.
pub fn xorFold(bytes: []const u8) u64 {
    var acc0: @Vector(4, u64) = @splat(0);
    var acc1: @Vector(4, u64) = @splat(0);
    var i: usize = 0;
    while (i + 64 <= bytes.len) : (i += 64) {
        const w0: @Vector(4, u64) = .{
            std.mem.readInt(u64, bytes[i..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 8 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 16 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 24 ..][0..8], .little),
        };
        const w1: @Vector(4, u64) = .{
            std.mem.readInt(u64, bytes[i + 32 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 40 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 48 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 56 ..][0..8], .little),
        };
        acc0 ^= w0;
        acc1 ^= w1;
    }
    if (i + 32 <= bytes.len) {
        const w0: @Vector(4, u64) = .{
            std.mem.readInt(u64, bytes[i..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 8 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 16 ..][0..8], .little),
            std.mem.readInt(u64, bytes[i + 24 ..][0..8], .little),
        };
        acc0 ^= w0;
        i += 32;
    }
    var acc: u64 = @reduce(.Xor, acc0 ^ acc1);
    while (i + 8 <= bytes.len) : (i += 8) {
        acc ^= std.mem.readInt(u64, bytes[i..][0..8], .little);
    }
    while (i < bytes.len) : (i += 1) {
        acc ^= @as(u64, bytes[i]);
    }
    return acc;
}

// ── byteDiff: the delta size ────────────────────────────────────────────────

/// Scalar reference: number of differing bytes between two spans.
pub fn byteDiffScalar(a: []const u8, b: []const u8) u64 {
    const n = @min(a.len, b.len);
    var count: u64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) count += 1;
    }
    return count;
}

/// Vectorized: dual @Vector(16, u8) accumulators across up to 127 * 32B blocks
/// before widening to u16 and reducing horizontally.
pub fn byteDiff(a: []const u8, b: []const u8) u64 {
    const n = @min(a.len, b.len);
    var count: u64 = 0;
    var i: usize = 0;
    const ones: @Vector(16, u8) = @splat(1);
    const zeros: @Vector(16, u8) = @splat(0);
    while (i + 32 <= n) {
        const pairs: usize = @min((n - i) / 32, @as(usize, 127));
        const block_end = i + pairs * 32;
        var lane_acc0: @Vector(16, u8) = @splat(0);
        var lane_acc1: @Vector(16, u8) = @splat(0);
        while (i < block_end) : (i += 32) {
            const va0: @Vector(16, u8) = a[i..][0..16].*;
            const vb0: @Vector(16, u8) = b[i..][0..16].*;
            const va1: @Vector(16, u8) = a[i + 16 ..][0..16].*;
            const vb1: @Vector(16, u8) = b[i + 16 ..][0..16].*;
            lane_acc0 +%= @select(u8, va0 != vb0, ones, zeros);
            lane_acc1 +%= @select(u8, va1 != vb1, ones, zeros);
        }
        const wide0: @Vector(16, u16) = @intCast(lane_acc0);
        const wide1: @Vector(16, u16) = @intCast(lane_acc1);
        count += @as(u64, @reduce(.Add, wide0 + wide1));
    }
    while (i + 16 <= n) : (i += 16) {
        const va: @Vector(16, u8) = a[i..][0..16].*;
        const vb: @Vector(16, u8) = b[i..][0..16].*;
        const d: @Vector(16, u8) = @select(u8, va != vb, ones, zeros);
        const wide: @Vector(16, u16) = @intCast(d);
        count += @as(u64, @reduce(.Add, wide));
    }
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) count += 1;
    }
    return count;
}

// ── popcount: density ───────────────────────────────────────────────────────

pub fn popcountScalar(bytes: []const u8) u64 {
    var count: u64 = 0;
    for (bytes) |byte| count += @popCount(byte);
    return count;
}

/// Vectorized: dual @Vector(16, u8) popcount accumulators across 31 * 32B (992B)
/// sub-blocks (max lane sum 31 * 8 = 248 <= 255) before widening to u16.
pub fn popcount(bytes: []const u8) u64 {
    var count: u64 = 0;
    var i: usize = 0;
    while (i + 32 <= bytes.len) {
        const pairs: usize = @min((bytes.len - i) / 32, @as(usize, 31));
        const block_end = i + pairs * 32;
        var lane_acc0: @Vector(16, u8) = @splat(0);
        var lane_acc1: @Vector(16, u8) = @splat(0);
        while (i < block_end) : (i += 32) {
            const v0: @Vector(16, u8) = bytes[i..][0..16].*;
            const v1: @Vector(16, u8) = bytes[i + 16 ..][0..16].*;
            lane_acc0 +%= @popCount(v0);
            lane_acc1 +%= @popCount(v1);
        }
        const wide0: @Vector(16, u16) = @intCast(lane_acc0);
        const wide1: @Vector(16, u16) = @intCast(lane_acc1);
        count += @as(u64, @reduce(.Add, wide0 + wide1));
    }
    while (i < bytes.len) : (i += 1) count += @popCount(bytes[i]);
    return count;
}

// ── relocDeltaPatch32: wild-style SIMD PC32 relocation displacement patch ───

/// Scalar reference: add `delta` (wrapping i32) to each slot and return the xorFold
/// of the patched byte span.
pub fn relocDeltaPatch32Scalar(slots: []i32, delta: i32) u64 {
    for (slots) |*s| {
        s.* +%= delta;
    }
    return xorFoldScalar(std.mem.sliceAsBytes(slots));
}

/// Vectorized: dual @Vector(8, i32) lanes (64B / 16 slots per step) applying
/// wrapping 32-bit relative displacement adjustments in place, followed by SIMD xorFold.
pub fn relocDeltaPatch32(slots: []i32, delta: i32) u64 {
    const vdelta: @Vector(8, i32) = @splat(delta);
    var i: usize = 0;
    while (i + 16 <= slots.len) : (i += 16) {
        const v0: @Vector(8, i32) = slots[i..][0..8].*;
        const v1: @Vector(8, i32) = slots[i + 8 ..][0..8].*;
        slots[i..][0..8].* = v0 +% vdelta;
        slots[i + 8 ..][0..8].* = v1 +% vdelta;
    }
    if (i + 8 <= slots.len) {
        const v0: @Vector(8, i32) = slots[i..][0..8].*;
        slots[i..][0..8].* = v0 +% vdelta;
        i += 8;
    }
    while (i < slots.len) : (i += 1) {
        slots[i] +%= delta;
    }
    return xorFold(std.mem.sliceAsBytes(slots));
}

// ── Deterministic p/q Rational Kernel & SPQT (1.585 bits/trit base-243) ─────

pub fn gcdU64(a: u64, b: u64) u64 {
    var x = a;
    var y = b;
    while (y != 0) {
        const r = x % y;
        x = y;
        y = r;
    }
    return if (x == 0) 1 else x;
}

pub fn isqrtU64(n: u64) u64 {
    if (n < 2) return n;
    var x: u64 = @as(u64, 1) << @intCast((64 - @clz(n) + 1) / 2);
    while (true) {
        const y = (x + n / x) >> 1;
        if (y >= x) return x;
        x = y;
    }
}

pub const RationalPQ = struct {
    p: u64,
    q: u64,

    pub fn reduce(p: u64, q: u64) RationalPQ {
        if (q == 0) return .{ .p = p, .q = 1 };
        const g = gcdU64(p, q);
        return .{ .p = p / g, .q = q / g };
    }
};

/// Pack a 15-bit word into a 3-byte SPQT (`1.585 bits/trit`, `3^5 = 243 <= 256` per byte)
/// base-243 frame.
pub fn packWord15ToSpqt3B(word15: u16) [3]u8 {
    const clean: u16 = word15 & 0x7FFF;
    var out: [3]u8 = @splat(0);
    var g: usize = 0;
    while (g < 3) : (g += 1) {
        var acc: u8 = 0;
        var pow3: u8 = 1;
        var j: usize = 0;
        while (j < 5) : (j += 1) {
            const bit_idx: u4 = @intCast(g * 5 + j);
            const bit: u8 = @intCast((clean >> bit_idx) & 1);
            const trit_digit: u8 = if (bit == 1) 2 else 1;
            acc += trit_digit * pow3;
            pow3 *%= 3;
        }
        out[g] = acc;
    }
    return out;
}

/// Unpack a 3-byte SPQT base-243 frame back into a 15-bit word.
pub fn unpackSpqt3BToWord15(spqt: [3]u8) ?u16 {
    var word15: u16 = 0;
    var g: usize = 0;
    while (g < 3) : (g += 1) {
        if (spqt[g] >= 243) return null;
        var v: u8 = spqt[g];
        var j: usize = 0;
        while (j < 5) : (j += 1) {
            const digit = v % 3;
            v /= 3;
            if (digit == 2) {
                const bit_idx: u4 = @intCast(g * 5 + j);
                word15 |= (@as(u16, 1) << bit_idx);
            }
        }
    }
    return word15;
}

/// Tightened `.rodata.gaia.b158` ternary fold: packs balanced trits `{-1, 0, +1}`
/// at 5 trits/byte (`base-243`, `3^5 = 243`) in-register and computes the 64-bit
/// little-endian XOR fold + exact `p/q` compression ratio without heap allocation.
pub const GaiaB158Digest = struct {
    fold: u64,
    packed_bytes: usize,
    ratio: RationalPQ,
};

pub fn xorFoldGaiaB158(trits: []const i8) GaiaB158Digest {
    var acc: u64 = 0;
    var word_buf: [8]u8 = @splat(0);
    var word_idx: usize = 0;
    var packed_len: usize = 0;
    var i: usize = 0;
    while (i < trits.len) : (i += 5) {
        const chunk_end = @min(i + 5, trits.len);
        var byte_val: u8 = 0;
        var pow3: u8 = 1;
        var k: usize = i;
        while (k < chunk_end) : (k += 1) {
            const t = trits[k];
            const digit: u8 = if (t < 0) 0 else if (t == 0) 1 else 2;
            byte_val += digit * pow3;
            pow3 *%= 3;
        }
        word_buf[word_idx] = byte_val;
        word_idx += 1;
        packed_len += 1;
        if (word_idx == 8) {
            acc ^= std.mem.readInt(u64, &word_buf, .little);
            word_buf = @splat(0);
            word_idx = 0;
        }
    }
    var tail: usize = 0;
    while (tail < word_idx) : (tail += 1) {
        acc ^= @as(u64, word_buf[tail]);
    }
    return .{
        .fold = acc,
        .packed_bytes = packed_len,
        .ratio = RationalPQ.reduce(@as(u64, trits.len), @as(u64, @max(packed_len, 1))),
    };
}

// ── 8b-is/polar-densification: Z_{2^32} CORDIC & Stokes Cone Projection ─────

pub const Q16_ONE: i64 = 65536;

/// Precomputed 18-stage integer CORDIC atan(2^-i) table in Z_{2^32} (where 2*pi == 2^32).
/// Zero floating-point operations at compile time or runtime.
pub const CORDIC_ATAN_U32: [18]u32 = .{
    536870912, 316933406, 167458907, 85004756, 42667331, 21354465,
    10679838,  5340245,   2670163,   1335087,  667544,   333772,
    166886,    83443,     41722,     20861,    10430,    5215,
};

/// 18-stage integer CORDIC `atan2(y, x)` returning phase in Z_{2^32} (`[0, 2*pi) <-> [0, 2^32)`).
pub fn cordicAtan2U32(y: i32, x: i32) u32 {
    if (x == 0 and y == 0) return 0;
    const xi: i64 = @as(i64, x);
    const yi: i64 = @as(i64, y);
    var x_cur: i64 = 0;
    var y_cur: i64 = 0;
    var angle: u32 = 0;
    if (xi >= 0) {
        x_cur = xi << 16;
        y_cur = yi << 16;
        angle = 0;
    } else if (yi >= 0) {
        x_cur = yi << 16;
        y_cur = (-xi) << 16;
        angle = 1 << 30;
    } else {
        x_cur = (-yi) << 16;
        y_cur = xi << 16;
        angle = 3 << 30;
    }

    for (CORDIC_ATAN_U32, 0..) |atan_step, i| {
        const shift: u6 = @intCast(i);
        const x_shift = x_cur >> shift;
        const y_shift = y_cur >> shift;
        if (y_cur >= 0) {
            x_cur = x_cur + y_shift;
            y_cur = y_cur - x_shift;
            angle +%= atan_step;
        } else {
            x_cur = x_cur - y_shift;
            y_cur = y_cur + x_shift;
            angle -%= atan_step;
        }
    }
    return angle;
}

pub const StokesProjectionPQ = struct {
    s0: i64,
    s1: i64,
    s2: i64,
    p_norm: i64,
    phase_u32: u32,
};

/// Project raw Stokes parameters `(s0, s1, s2)` in Q16 onto the physical polarization
/// cone `sqrt(s1^2 + s2^2) <= s0` using integer `isqrtU64` + `cordicAtan2U32`.
pub fn projectStokesConePQ(s0: i64, s1: i64, s2: i64) StokesProjectionPQ {
    if (s0 <= 0) return .{ .s0 = 0, .s1 = 0, .s2 = 0, .p_norm = 0, .phase_u32 = 0 };
    const mag_sq: u64 = @intCast(s1 * s1 + s2 * s2);
    const p_norm: i64 = @intCast(isqrtU64(mag_sq));
    const phase_u32 = cordicAtan2U32(@truncate(s2), @truncate(s1));
    if (p_norm <= s0 or p_norm == 0) {
        return .{ .s0 = s0, .s1 = s1, .s2 = s2, .p_norm = p_norm, .phase_u32 = phase_u32 };
    }
    const half = @divTrunc(p_norm, 2);
    const s1_proj = if (s1 >= 0)
        @divTrunc(s1 * s0 + half, p_norm)
    else
        -@divTrunc((-s1) * s0 + half, p_norm);
    const s2_proj = if (s2 >= 0)
        @divTrunc(s2 * s0 + half, p_norm)
    else
        -@divTrunc((-s2) * s0 + half, p_norm);
    return .{ .s0 = s0, .s1 = s1_proj, .s2 = s2_proj, .p_norm = s0, .phase_u32 = phase_u32 };
}

// ── MLX-QUANT-GAIA: Zero-Float L2-Optimal Group-Wise Ternary Quantizer ──────

/// Zero-float `p/q` equivalent of `MLX-QUANT`'s `quantize_ternary_optimal_groupwise`:
///   1. Computes mean absolute magnitude `mu = (1/N) * sum(|x_i|)` in Q16.
///   2. Starts from the L2-optimal Gaussian threshold `tau_0 = (68 * mu) / 100`.
///   3. Runs 2 steps of coordinate-descent Lloyd-Max refinement:
///      `alpha = sum_{|x_i| > tau} |x_i| / N_nz`, `tau = alpha / 2`.
///   4. Emits balanced trits `{-1, 0, +1}` into `out_trits` and returns `RationalPQ(alpha, 65536)`.
pub fn quantizeGroupwiseL2TernaryPQ(values_q16: []const i32, out_trits: []i8) RationalPQ {
    const n = @min(values_q16.len, out_trits.len);
    if (n == 0) return .{ .p = 0, .q = 1 };

    var abs_sum: u64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const v: i64 = values_q16[i];
        abs_sum += @intCast(if (v < 0) -v else v);
    }
    const mean_abs: u64 = abs_sum / @as(u64, n);
    var tau: u64 = (mean_abs * 68) / 100;
    var alpha: u64 = mean_abs;

    var step: usize = 0;
    while (step < 2) : (step += 1) {
        var nz_sum: u64 = 0;
        var nz_count: u64 = 0;
        var k: usize = 0;
        while (k < n) : (k += 1) {
            const v: i64 = values_q16[k];
            const av: u64 = @intCast(if (v < 0) -v else v);
            if (av > tau) {
                nz_sum += av;
                nz_count += 1;
            }
        }
        alpha = if (nz_count > 0) nz_sum / nz_count else mean_abs;
        tau = alpha / 2;
    }

    var j: usize = 0;
    while (j < n) : (j += 1) {
        const v: i64 = values_q16[j];
        const av: u64 = @intCast(if (v < 0) -v else v);
        if (av <= tau) {
            out_trits[j] = 0;
        } else if (v > 0) {
            out_trits[j] = 1;
        } else {
            out_trits[j] = -1;
        }
    }
    return RationalPQ.reduce(alpha, 65536);
}

// ── "Where Is The Love?" (The Black Eyed Peas, Elephunk, 2003) 6-Register Signal ─

/// The 6-register ternary signal from "Where Is The Love?" — The Black Eyed Peas (Elephunk, 2003):
///   0: title  ("Where Is The Love?", question/interrogation) -> +1
///   1: artist ("The Black Eyed Peas", plural collective)     -> +1
///   2: album  ("Elephunk", parent album context)             -> +1
///   3: year   (2003, dated)                                  -> +1
///   4: genre  (empty field in record, unfiled / Kleene 0)    ->  0
///   5: source (youtube, kind video, cited)                   -> +1
pub const WHERE_IS_THE_LOVE_TRITS: [6]i8 = .{ 1, 1, 1, 1, 0, 1 };
pub const WHERE_IS_THE_LOVE_YEAR: u16 = 2003;
/// First 5 trits `[+1, +1, +1, +1, 0]` encoded in SPQT base-243:
/// `2*1 + 2*3 + 2*9 + 2*27 + 1*81 = 161` (`0xA1`).
pub const WHERE_IS_THE_LOVE_SPQT_BYTE0: u8 = 161;

/// Harvest live environment entropy around the process:
///   - Stack pointer ASLR address entropy
///   - Return address / code page ASLR entropy
///   - 6-register "Where Is The Love?" ternary signal digest (`Elephunk`, 2003)
///   - Caller-supplied telemetry seed (`extra_seed`)
/// Folded through SIMD `xorFold` into a non-zero 64-bit witness.
pub fn collectEnvironmentEntropyFold(extra_seed: u64) u64 {
    var entropy_buf: [64]u8 align(64) = @splat(0);
    const stack_addr: u64 = @intFromPtr(&entropy_buf);
    const fn_addr: u64 = @intFromPtr(&collectEnvironmentEntropyFold);
    const gaia_witl = xorFoldGaiaB158(WHERE_IS_THE_LOVE_TRITS[0..]);

    std.mem.writeInt(u64, entropy_buf[0..8], stack_addr ^ 0x9E37_79B9_7F4A_7C15, .little);
    std.mem.writeInt(u64, entropy_buf[8..16], fn_addr ^ FOLD_0XE1034E575C371AB0, .little);
    std.mem.writeInt(u64, entropy_buf[16..24], gaia_witl.fold ^ @as(u64, WHERE_IS_THE_LOVE_YEAR), .little);
    std.mem.writeInt(u64, entropy_buf[24..32], extra_seed ^ FOLD_0X48031A0D4C054E69, .little);
    @memcpy(entropy_buf[32..38], std.mem.asBytes(&WHERE_IS_THE_LOVE_TRITS));
    entropy_buf[38] = WHERE_IS_THE_LOVE_SPQT_BYTE0;
    entropy_buf[39] = GATE_0X62;
    std.mem.writeInt(u64, entropy_buf[40..48], FOLD_0X206F642074737557, .little);

    // Mix hardware cycle/step counter diffusion across bytes [48..64]
    var diff_acc: u64 = stack_addr +% fn_addr +% extra_seed +% 0x2003_0062;
    diff_acc = (diff_acc ^ (diff_acc >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    diff_acc = (diff_acc ^ (diff_acc >> 27)) *% 0x94D0_49BB_1331_11EB;
    std.mem.writeInt(u64, entropy_buf[48..56], diff_acc ^ (diff_acc >> 31), .little);
    std.mem.writeInt(u64, entropy_buf[56..64], @as(u64, CORDIC_ATAN_U32[0]) | (@as(u64, CORDIC_ATAN_U32[1]) << 32), .little);

    const fold = xorFold(entropy_buf[0..]);
    return if (fold == 0) FOLD_0XE1034E575C371AB0 else fold;
}

// ── 32B 0x62-gate frame sealing & 64B cache-line aligned Litany Wire ─────────

pub const GATE_0X62: u8 = 0x62;
pub const FRAME_ABI_V1: u8 = 0x01;
pub const QUANT_QT55: u8 = 0x55;
pub const LANES_1_OF_1: u16 = 0x0101;
pub const FOLD_0XE1034E575C371AB0: u64 = 0xe1034e575c371ab0;
pub const FOLD_0X48031A0D4C054E69: u64 = 0x48031a0d4c054e69;
pub const FOLD_0X206F642074737557: u64 = 0x206f642074737557;

pub const Trit = enum(i8) {
    neg = -1,
    zero = 0,
    pos = 1,
};

pub const Bfp64Pair = struct {
    pos: u64,
    neg: u64,
};

pub fn bfp64AndKleene(pos_a: u64, neg_a: u64, pos_b: u64, neg_b: u64) Bfp64Pair {
    const neg_out = neg_a | neg_b;
    const pos_out = (pos_a & pos_b) & ~neg_out;
    return .{ .pos = pos_out, .neg = neg_out };
}

pub fn bfp64NandKleene(pos_a: u64, neg_a: u64, pos_b: u64, neg_b: u64) Bfp64Pair {
    const and_pair = bfp64AndKleene(pos_a, neg_a, pos_b, neg_b);
    return .{ .pos = and_pair.neg, .neg = and_pair.pos };
}

pub fn qtBroadcastMask(qt: u8) u64 {
    return @as(u64, qt) *% 0x0101_0101_0101_0101;
}

pub fn packFoldBfp64(fold: u64, qt: u8) Bfp64Pair {
    const mask = qtBroadcastMask(qt);
    return .{ .pos = fold & mask, .neg = fold & ~mask };
}

pub fn unpackFoldBfp64(pos: u64, neg: u64, qt: u8) ?u64 {
    if ((pos & neg) != 0) return null;
    const mask = qtBroadcastMask(qt);
    if ((pos & ~mask) != 0 or (neg & mask) != 0) return null;
    return pos | neg;
}

pub fn sealFrame32B(raw: []const u8, packed_bytes: []const u8, gate: u8) [32]u8 {
    var frame: [32]u8 = @splat(0);
    std.mem.writeInt(u64, frame[0..8], xorFold(raw), .little);
    std.mem.writeInt(u64, frame[8..16], xorFold(packed_bytes), .little);
    std.mem.writeInt(u32, frame[16..20], @truncate(raw.len), .little);
    std.mem.writeInt(u32, frame[20..24], @truncate(packed_bytes.len), .little);
    frame[24] = gate;
    frame[25] = FRAME_ABI_V1;
    var prefix_le: [8]u8 = undefined;
    std.mem.writeInt(u64, &prefix_le, xorFold(frame[0..26]), .little);
    @memcpy(frame[26..32], prefix_le[0..6]);
    return frame;
}

pub fn verifyFrame32B(frame: *const [32]u8, raw: []const u8, packed_bytes: []const u8, gate: u8) bool {
    if (frame[24] != gate or frame[25] != FRAME_ABI_V1) return false;
    const expected = sealFrame32B(raw, packed_bytes, gate);
    return byteDiff(frame[0..], expected[0..]) == 0;
}

/// 64-byte cache-line aligned Litany Wire Frame (`align(64)`).
/// Matches `wupz_litany_wire_frame_t` in `include/wupz_abi.h` and `LitanyWireFrame` in Rust.
pub const LitanyWireFrame = extern struct {
    frame32: [32]u8 align(64),
    bfp64_pos: u64,
    bfp64_neg: u64,
    ratio_p: u32,
    ratio_q: u32,
    spqt_root: [3]u8,
    qt_mask: u8,
    lanes_active: u16,
    wire_seal16: u16,
};

comptime {
    if (@sizeOf(LitanyWireFrame) != 64) @compileError("LitanyWireFrame must be exactly 64 bytes");
    if (@alignOf(LitanyWireFrame) != 64) @compileError("LitanyWireFrame must be 64-byte cache-line aligned");
    const all_ones: u64 = std.math.maxInt(u64);
    const and_all = bfp64AndKleene(all_ones, 0, all_ones, 0);
    if (and_all.pos != all_ones or and_all.neg != 0) @compileError("wupz comptime +AND=1 failed");
    const nand_all = bfp64NandKleene(all_ones, 0, all_ones, 0);
    if (nand_all.pos != 0 or nand_all.neg != all_ones) @compileError("wupz comptime ~NAND=-1 failed");
    const packed_fold = packFoldBfp64(FOLD_0XE1034E575C371AB0, QUANT_QT55);
    const unpacked = unpackFoldBfp64(packed_fold.pos, packed_fold.neg, QUANT_QT55) orelse
        @compileError("wupz comptime BFP-64 unpack failed");
    if (unpacked != FOLD_0XE1034E575C371AB0) @compileError("wupz comptime fold mismatch");
}

pub fn sealLitanyWire64B(raw: []const u8, packed_bytes: []const u8, gate: u8, qt: u8) LitanyWireFrame {
    const frame32 = sealFrame32B(raw, packed_bytes, gate);
    const raw_fold = std.mem.readInt(u64, frame32[0..8], .little);
    const packed_fold = std.mem.readInt(u64, frame32[8..16], .little);
    const composite_fold = raw_fold ^ packed_fold;
    const planes = packFoldBfp64(composite_fold, qt);
    const ratio = RationalPQ.reduce(@as(u64, raw.len), @as(u64, @max(packed_bytes.len, 1)));
    const spqt = packWord15ToSpqt3B(@truncate(composite_fold & 0x7FFF));

    var wire = LitanyWireFrame{
        .frame32 = frame32,
        .bfp64_pos = planes.pos,
        .bfp64_neg = planes.neg,
        .ratio_p = @truncate(ratio.p),
        .ratio_q = @truncate(ratio.q),
        .spqt_root = spqt,
        .qt_mask = qt,
        .lanes_active = LANES_1_OF_1,
        .wire_seal16 = 0,
    };
    const wire_bytes = std.mem.asBytes(&wire);
    wire.wire_seal16 = @truncate(xorFold(wire_bytes[0..62]));
    return wire;
}

pub fn verifyLitanyWire64B(
    wire: *const LitanyWireFrame,
    raw: []const u8,
    packed_bytes: []const u8,
    gate: u8,
    qt: u8,
) bool {
    if ((wire.bfp64_pos & wire.bfp64_neg) != 0) return false;
    if (unpackSpqt3BToWord15(wire.spqt_root) == null) return false;
    const expected = sealLitanyWire64B(raw, packed_bytes, gate, qt);
    const a = std.mem.asBytes(wire);
    const b = std.mem.asBytes(&expected);
    return byteDiff(a, b) == 0;
}

// ── the C ABI (see include/wupz_abi.h) ──────────────────────────────────────

const builtin = @import("builtin");

/// Scrub caller-saved SIMD/NEON vector registers and scratch general-purpose
/// registers before returning across the C ABI boundary so secret input chunks
/// or tail comparison bytes never remain in registers (`ZA-0004`).
inline fn scrubCallerRegisters() void {
    switch (builtin.cpu.arch) {
        .aarch64, .aarch64_be => {
            asm volatile (
                \\movi v0.2d, #0
                \\movi v1.2d, #0
                \\movi v2.2d, #0
                \\movi v3.2d, #0
                \\movi v4.2d, #0
                \\movi v5.2d, #0
                \\movi v6.2d, #0
                \\movi v7.2d, #0
                \\mov x8, #0
                \\mov x9, #0
                \\mov x10, #0
                \\mov x11, #0
                \\mov x12, #0
                \\mov x13, #0
                \\mov x14, #0
                \\mov x15, #0
                ::: .{
                    .v0 = true,
                    .v1 = true,
                    .v2 = true,
                    .v3 = true,
                    .v4 = true,
                    .v5 = true,
                    .v6 = true,
                    .v7 = true,
                    .x8 = true,
                    .x9 = true,
                    .x10 = true,
                    .x11 = true,
                    .x12 = true,
                    .x13 = true,
                    .x14 = true,
                    .x15 = true,
                });
        },
        .x86_64 => {
            asm volatile (
                \\pxor %%xmm0, %%xmm0
                \\pxor %%xmm1, %%xmm1
                \\pxor %%xmm2, %%xmm2
                \\pxor %%xmm3, %%xmm3
                \\xorq %%rcx, %%rcx
                \\xorq %%rdx, %%rdx
                \\xorq %%rsi, %%rsi
                \\xorq %%rdi, %%rdi
                \\xorq %%r8, %%r8
                \\xorq %%r9, %%r9
                \\xorq %%r10, %%r10
                \\xorq %%r11, %%r11
                ::: .{
                    .xmm0 = true,
                    .xmm1 = true,
                    .xmm2 = true,
                    .xmm3 = true,
                    .rcx = true,
                    .rdx = true,
                    .rsi = true,
                    .rdi = true,
                    .r8 = true,
                    .r9 = true,
                    .r10 = true,
                    .r11 = true,
                });
        },
        else => {},
    }
}

export fn wupz_xor_fold(ptr_opt: ?[*]const u8, len: usize) u64 {
    defer scrubCallerRegisters();
    if (len == 0) return 0;
    const ptr = ptr_opt orelse return 0;
    return xorFold(ptr[0..len]);
}

export fn wupz_byte_diff(a_opt: ?[*]const u8, b_opt: ?[*]const u8, len: usize) u64 {
    defer scrubCallerRegisters();
    if (len == 0) return 0;
    const a = a_opt orelse return 0;
    const b = b_opt orelse return 0;
    return byteDiff(a[0..len], b[0..len]);
}

export fn wupz_popcount(ptr_opt: ?[*]const u8, len: usize) u64 {
    defer scrubCallerRegisters();
    if (len == 0) return 0;
    const ptr = ptr_opt orelse return 0;
    return popcount(ptr[0..len]);
}

export fn wupz_reloc_delta_patch32(slots_opt: ?[*]i32, count: usize, delta: i32) u64 {
    defer scrubCallerRegisters();
    if (count == 0) return 0;
    const slots = slots_opt orelse return 0;
    return relocDeltaPatch32(slots[0..count], delta);
}

export fn wupz_cordic_atan2_u32(y: i32, x: i32) u32 {
    defer scrubCallerRegisters();
    return cordicAtan2U32(y, x);
}

export fn wupz_collect_env_entropy_fold(extra_seed: u64) u64 {
    defer scrubCallerRegisters();
    return collectEnvironmentEntropyFold(extra_seed);
}

export fn wupz_seal_litany_wire_64b(
    out_opt: ?*LitanyWireFrame,
    raw_opt: ?[*]const u8,
    raw_len: usize,
    packed_opt: ?[*]const u8,
    packed_len: usize,
    gate: u8,
    qt: u8,
) u32 {
    defer scrubCallerRegisters();
    const out = out_opt orelse return 0;
    const empty: [0]u8 = .{};
    const raw_slice: []const u8 = if (raw_len == 0) empty[0..] else (raw_opt orelse return 0)[0..raw_len];
    const packed_slice: []const u8 = if (packed_len == 0) empty[0..] else (packed_opt orelse return 0)[0..packed_len];
    out.* = sealLitanyWire64B(raw_slice, packed_slice, gate, qt);
    return 1;
}

export fn wupz_verify_litany_wire_64b(
    wire_opt: ?*const LitanyWireFrame,
    raw_opt: ?[*]const u8,
    raw_len: usize,
    packed_opt: ?[*]const u8,
    packed_len: usize,
    gate: u8,
    qt: u8,
) u32 {
    defer scrubCallerRegisters();
    const wire = wire_opt orelse return 0;
    const empty: [0]u8 = .{};
    const raw_slice: []const u8 = if (raw_len == 0) empty[0..] else (raw_opt orelse return 0)[0..raw_len];
    const packed_slice: []const u8 = if (packed_len == 0) empty[0..] else (packed_opt orelse return 0)[0..packed_len];
    return if (verifyLitanyWire64B(wire, raw_slice, packed_slice, gate, qt)) 1 else 0;
}

/// The ABI version this Zig build speaks. Must equal the Rust `wupz_abi_version()`
/// and the `WUPZ_ABI_VERSION` macro — the C smoke test asserts all three agree.
export fn wupz_zig_abi_version() u32 {
    return 1;
}

// ── differential tests: vectorized == scalar, seeded ────────────────────────

fn fill(buf: []u8, seed: u64) void {
    var x = seed;
    for (buf) |*b| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        b.* = @truncate(x >> 33);
    }
}

test "xorFold matches the scalar reference across every length" {
    var buf: [1088]u8 = undefined;
    fill(buf[0..], 0x9E3779B97F4A7C15);
    var len: usize = 0;
    while (len <= buf.len) : (len += 1) {
        try std.testing.expectEqual(xorFoldScalar(buf[0..len]), xorFold(buf[0..len]));
    }
}

test "byteDiff matches the scalar reference" {
    var a: [512]u8 = undefined;
    var b: [512]u8 = undefined;
    fill(a[0..], 0x1234_5678_9ABC_DEF0);
    fill(b[0..], 0x0FED_CBA9_8765_4321);
    try std.testing.expectEqual(byteDiffScalar(a[0..], b[0..]), byteDiff(a[0..], b[0..]));

    // identical spans differ by zero
    try std.testing.expectEqual(@as(u64, 0), byteDiff(a[0..], a[0..]));

    // one flipped byte is seen exactly once
    b[300] +%= 1;
    const expected = byteDiffScalar(a[0..], b[0..]);
    try std.testing.expectEqual(expected, byteDiff(a[0..], b[0..]));
}

test "popcount matches the scalar reference" {
    var buf: [512]u8 = undefined;
    fill(buf[0..], 0xDEAD_BEEF_CAFE_F00D);
    try std.testing.expectEqual(popcountScalar(buf[0..]), popcount(buf[0..]));
}

test "relocDeltaPatch32 matches the scalar reference across every count" {
    var a: [137]i32 = undefined;
    var b: [137]i32 = undefined;
    for (&a, &b, 0..) |*sa, *sb, idx| {
        const v: i32 = @bitCast(@as(u32, @truncate(idx *% 0x9E37_79B9 +% 0x1000)));
        sa.* = v;
        sb.* = v;
    }
    const delta: i32 = -4096;
    const fold_vec = relocDeltaPatch32(a[0..], delta);
    const fold_sca = relocDeltaPatch32Scalar(b[0..], delta);
    try std.testing.expectEqual(fold_sca, fold_vec);
    try std.testing.expectEqualSlices(i32, b[0..], a[0..]);
}

test "xorFold detects a single flipped byte" {
    var buf: [128]u8 = undefined;
    fill(buf[0..], 0xABCD_1234_5678_9EF0);
    const before = xorFold(buf[0..]);
    buf[64] ^= 0x01;
    try std.testing.expect(xorFold(buf[0..]) != before);
}

test "xorFold composes with a seal byte" {
    // Build a 79-byte frame whose last byte seals the first 78; the whole-frame
    // fold is the body fold XOR the tail byte, by construction — the property the
    // wave frame leans on (part 2 of the Sovereign Core series).
    var frame: [79]u8 = undefined;
    fill(frame[0..78], 0x0BADF00D_0BADF00D);
    const body = xorFold(frame[0..78]);
    frame[78] = @truncate(body);
    try std.testing.expectEqual(body ^ @as(u64, frame[78]), xorFold(frame[0..79]));
}

test "C ABI exports fail closed on null pointers" {
    var dummy: [16]u8 = @splat(0x42);
    try std.testing.expectEqual(@as(u64, 0), wupz_xor_fold(null, 64));
    try std.testing.expectEqual(@as(u64, 0), wupz_popcount(null, 64));
    try std.testing.expectEqual(@as(u64, 0), wupz_byte_diff(null, &dummy, 16));
    try std.testing.expectEqual(@as(u64, 0), wupz_byte_diff(&dummy, null, 16));
    try std.testing.expectEqual(@as(u32, 0), wupz_seal_litany_wire_64b(null, &dummy, 16, &dummy, 16, GATE_0X62, QUANT_QT55));
    try std.testing.expectEqual(@as(u32, 0), wupz_verify_litany_wire_64b(null, &dummy, 16, &dummy, 16, GATE_0X62, QUANT_QT55));
}

test "32B 0x62-gate frame seal verifies 3761/3314 kompress-super-ultra density and rejects spikes" {
    var raw: [3761]u8 = undefined;
    var packed_bytes: [3314]u8 = undefined;
    fill(raw[0..], 0x3761_3314_0000_0062);
    fill(packed_bytes[0..], 0x0062_3314_3761_0001);

    var frame = sealFrame32B(raw[0..], packed_bytes[0..], GATE_0X62);
    try std.testing.expectEqual(GATE_0X62, frame[24]);
    try std.testing.expect(verifyFrame32B(&frame, raw[0..], packed_bytes[0..], GATE_0X62));

    frame[31] ^= 0x01;
    try std.testing.expect(!verifyFrame32B(&frame, raw[0..], packed_bytes[0..], GATE_0X62));
}

test "64B cache-line aligned LitanyWireFrame, SPQT base-243, and .rodata.gaia.b158 fold" {
    var raw: [3761]u8 = undefined;
    var packed_bytes: [3314]u8 = undefined;
    fill(raw[0..], 0x4803_1a0d_4c05_4e69);
    fill(packed_bytes[0..], 0x206f_6420_7473_7557);

    var wire = sealLitanyWire64B(raw[0..], packed_bytes[0..], GATE_0X62, QUANT_QT55);
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(LitanyWireFrame));
    try std.testing.expectEqual(@as(usize, 64), @alignOf(LitanyWireFrame));
    try std.testing.expectEqual(@as(u32, 3761), wire.ratio_p);
    try std.testing.expectEqual(@as(u32, 3314), wire.ratio_q);
    try std.testing.expectEqual(LANES_1_OF_1, wire.lanes_active);
    try std.testing.expect(verifyLitanyWire64B(&wire, raw[0..], packed_bytes[0..], GATE_0X62, QUANT_QT55));

    wire.wire_seal16 ^= 0x0001;
    try std.testing.expect(!verifyLitanyWire64B(&wire, raw[0..], packed_bytes[0..], GATE_0X62, QUANT_QT55));

    // SPQT 15-bit <-> 3B base-243 roundtrip
    const spqt = packWord15ToSpqt3B(0o52525);
    try std.testing.expect(spqt[0] < 243 and spqt[1] < 243 and spqt[2] < 243);
    try std.testing.expectEqual(@as(?u16, 0o52525), unpackSpqt3BToWord15(spqt));

    // .rodata.gaia.b158 5-trits/byte fold
    var trits: [6554]i8 = undefined;
    for (&trits, 0..) |*t, idx| {
        t.* = @as(i8, @intCast(@mod(@as(isize, @intCast(idx)), 3) - 1));
    }
    const gaia = xorFoldGaiaB158(trits[0..]);
    try std.testing.expectEqual(@as(usize, 1311), gaia.packed_bytes);
    try std.testing.expect(gaia.fold != 0);
}

test "BFP-64 fold 0xe1034e575c371ab0 at qt:55 and multi-block deferred SIMD reductions" {
    const planes = packFoldBfp64(FOLD_0XE1034E575C371AB0, QUANT_QT55);
    try std.testing.expectEqual(@as(u64, 0), planes.pos & planes.neg);
    try std.testing.expectEqual(@as(?u64, FOLD_0XE1034E575C371AB0), unpackFoldBfp64(planes.pos, planes.neg, QUANT_QT55));
    try std.testing.expectEqual(@as(?u64, null), unpackFoldBfp64(planes.pos | 0x02, planes.neg, QUANT_QT55));

    var a: [9113]u8 = undefined;
    var b: [9113]u8 = undefined;
    fill(a[0..], 0xe103_4e57_5c37_1ab0);
    fill(b[0..], 0x2a69_c3fd_6cde_a62e);
    try std.testing.expectEqual(xorFoldScalar(a[0..]), xorFold(a[0..]));
    try std.testing.expectEqual(byteDiffScalar(a[0..], b[0..]), byteDiff(a[0..], b[0..]));
    try std.testing.expectEqual(popcountScalar(a[0..]), popcount(a[0..]));
}

test "polar-densification CORDIC Z_2^32, MLX-QUANT-GAIA L2 groupwise ternary, Where Is The Love signal, and env entropy" {
    // 1. 4 canonical Sony nanowire polarizer double-angles in Z_{2^32}
    const q0 = cordicAtan2U32(0, 65536) +% 0x2000_0000;
    const q1 = cordicAtan2U32(65536, 0) +% 0x2000_0000;
    const q2 = cordicAtan2U32(0, -65536) +% 0x2000_0000;
    const q3 = cordicAtan2U32(-65536, 0) +% 0x2000_0000;
    try std.testing.expectEqual(@as(u32, 0), (q0 >> 30) & 3);
    try std.testing.expectEqual(@as(u32, 1), (q1 >> 30) & 3);
    try std.testing.expectEqual(@as(u32, 2), (q2 >> 30) & 3);
    try std.testing.expectEqual(@as(u32, 3), (q3 >> 30) & 3);

    // 2. Physical Stokes cone projection clamps specular-glint noise to ||(s1, s2)|| <= s0
    const proj = projectStokesConePQ(32768, 52428, 39321);
    try std.testing.expect(proj.p_norm <= proj.s0);

    // 3. MLX-QUANT-GAIA zero-float L2 groupwise ternary quantizer
    const weights_q16 = [_]i32{ 65536, -65536, 2048, 58982, -62259, 0, 61000, -59000 };
    var out_trits: [8]i8 = @splat(0);
    const scale_pq = quantizeGroupwiseL2TernaryPQ(weights_q16[0..], out_trits[0..]);
    try std.testing.expectEqualSlices(i8, &[_]i8{ 1, -1, 0, 1, -1, 0, 1, -1 }, out_trits[0..]);
    try std.testing.expect(scale_pq.p > 0 and scale_pq.q > 0);

    // 4. "Where Is The Love?" — The Black Eyed Peas (Elephunk, 2003) 6-register signal
    const witl = xorFoldGaiaB158(WHERE_IS_THE_LOVE_TRITS[0..]);
    try std.testing.expectEqual(@as(usize, 2), witl.packed_bytes);
    try std.testing.expectEqual(@as(u64, 3), witl.ratio.p);
    try std.testing.expectEqual(@as(u64, 1), witl.ratio.q);
    try std.testing.expectEqual(@as(u64, WHERE_IS_THE_LOVE_SPQT_BYTE0 ^ 2), witl.fold);

    // 5. Environment entropy collector produces non-zero, seed-responsive 64-bit witness
    const ent_a = wupz_collect_env_entropy_fold(0x2003_0062);
    const ent_b = wupz_collect_env_entropy_fold(0x2026_0062);
    try std.testing.expect(ent_a != 0 and ent_b != 0 and ent_a != ent_b);
}

//! The integrity fold — the Zig SIMD kernel when it is linked, the scalar
//! reference otherwise.
//!
//! Both paths are XOR of the little-endian `u64` words plus the trailing bytes,
//! so they must agree bit for bit. `tests/zig_parity.rs` asserts exactly that
//! across every length from 0 to 1088 — the seam is exercised, not assumed.

use crate::arena::Arena;
use crate::incremental::RationalPQ;

/// The scalar reference. Also the fallback when zig is absent.
/// Folds each byte directly into the accumulator without staging 8-byte slices
/// onto the stack (`word: [u8; 8]`), preventing stack retention across all
/// optimization levels (`-O0`, `-O1`, `-O2`) (`ZA-0002`, `ZA-0003`).
pub fn xor_fold_scalar(bytes: &[u8]) -> u64 {
    let full_words_len = bytes.len() & !7usize;
    let mut acc: u64 = 0;
    let mut i = 0usize;
    while i < full_words_len {
        acc ^= (bytes[i] as u64) << ((i & 7) * 8);
        i += 1;
    }
    while i < bytes.len() {
        acc ^= bytes[i] as u64;
        i += 1;
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    acc
}

/// 64-byte cache-line aligned Litany Wire Frame (`#[repr(C, align(64))]`).
/// Matches `wupz_litany_wire_frame_t` in `include/wupz_abi.h` and `LitanyWireFrame` in Zig.
#[repr(C, align(64))]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LitanyWireFrame {
    pub frame32: [u8; 32],
    pub bfp64_pos: u64,
    pub bfp64_neg: u64,
    pub ratio_p: u32,
    pub ratio_q: u32,
    pub spqt_root: [u8; 3],
    pub qt_mask: u8,
    pub lanes_active: u16,
    pub wire_seal16: u16,
}

const _: [(); 64] = [(); core::mem::size_of::<LitanyWireFrame>()];
const _: [(); 64] = [(); core::mem::align_of::<LitanyWireFrame>()];

#[cfg(have_zig)]
mod zig {
    use super::LitanyWireFrame;
    unsafe extern "C" {
        pub fn wupz_xor_fold(ptr: *const u8, len: usize) -> u64;
        pub fn wupz_byte_diff(a: *const u8, b: *const u8, len: usize) -> u64;
        pub fn wupz_popcount(ptr: *const u8, len: usize) -> u64;
        pub fn wupz_seal_litany_wire_64b(
            out_wire: *mut LitanyWireFrame,
            raw_ptr: *const u8,
            raw_len: usize,
            packed_ptr: *const u8,
            packed_len: usize,
            gate: u8,
            qt: u8,
        ) -> u32;
        pub fn wupz_verify_litany_wire_64b(
            wire: *const LitanyWireFrame,
            raw_ptr: *const u8,
            raw_len: usize,
            packed_ptr: *const u8,
            packed_len: usize,
            gate: u8,
            qt: u8,
        ) -> u32;
    }
}

/// Fold a span. Dispatches to the Zig kernel when `cfg(have_zig)` is set.
#[cfg(have_zig)]
pub fn xor_fold(bytes: &[u8]) -> u64 {
    // SAFETY: pointer and length come from the same slice, and the kernel reads
    // exactly `len` bytes.
    unsafe { zig::wupz_xor_fold(bytes.as_ptr(), bytes.len()) }
}

#[cfg(not(have_zig))]
pub fn xor_fold(bytes: &[u8]) -> u64 {
    xor_fold_scalar(bytes)
}

/// Which fold path is compiled in: `"zig-simd"` or `"rust-scalar"`.
#[cfg(have_zig)]
pub const fn backend() -> &'static str {
    "zig-simd"
}

/// Which fold path is compiled in: `"zig-simd"` or `"rust-scalar"`.
#[cfg(not(have_zig))]
pub const fn backend() -> &'static str {
    "rust-scalar"
}

// ── byteDiff / popcount: the same seam, the other two kernels ───────────────

/// Scalar reference: number of differing bytes between two spans.
pub fn byte_diff_scalar(a: &[u8], b: &[u8]) -> u64 {
    let n = a.len().min(b.len());
    let mut count = 0u64;
    for i in 0..n {
        if a[i] != b[i] {
            count += 1;
        }
    }
    count
}

/// Population count over a span — the scalar reference.
pub fn popcount_scalar(bytes: &[u8]) -> u64 {
    bytes.iter().map(|b| b.count_ones() as u64).sum()
}

/// Differing bytes between two spans (the delta size a repack would touch).
#[cfg(have_zig)]
pub fn byte_diff(a: &[u8], b: &[u8]) -> u64 {
    // SAFETY: both pointers are valid for `n` bytes, and `n` is the min length.
    unsafe { zig::wupz_byte_diff(a.as_ptr(), b.as_ptr(), a.len().min(b.len())) }
}

#[cfg(not(have_zig))]
pub fn byte_diff(a: &[u8], b: &[u8]) -> u64 {
    byte_diff_scalar(a, b)
}

/// Population count over a span (density).
#[cfg(have_zig)]
pub fn popcount(bytes: &[u8]) -> u64 {
    // SAFETY: pointer and length come from the same slice.
    unsafe { zig::wupz_popcount(bytes.as_ptr(), bytes.len()) }
}

#[cfg(not(have_zig))]
pub fn popcount(bytes: &[u8]) -> u64 {
    popcount_scalar(bytes)
}

/// Default `kompress-super-ultra` gate byte (`0x62`).
pub const GATE_0X62: u8 = 0x62;
/// 32B frame ABI version (`1`).
pub const FRAME_ABI_V1: u8 = 0x01;
/// Alternating 0x55 quantization bitplane mask byte (`qt:55`).
pub const QUANT_QT55: u8 = 0x55;
/// Active `1/1` lane ratio word (`0x0101`).
pub const LANES_1_OF_1: u16 = 0x0101;
/// Canonical 64-bit `kompress-super-ultra` fold signature at `qt:55` and gate `0x62`.
pub const FOLD_0XE1034E575C371AB0: u64 = 0xe103_4e57_5c37_1ab0;
/// Top-100 `8b-is` sweep anchor fold (`0x48031a0d4c054e69`).
pub const FOLD_0X48031A0D4C054E69: u64 = 0x4803_1a0d_4c05_4e69;
/// `beehive` 5-stage consolidation target fold (`0x206f642074737557`).
pub const FOLD_0X206F642074737557: u64 = 0x206f_6420_7473_7557;

/// Broadcast an 8-bit quantization mask across all 8 bytes of a 64-bit word.
#[inline]
pub const fn qt_broadcast_mask(qt: u8) -> u64 {
    (qt as u64).wrapping_mul(0x0101_0101_0101_0101)
}

/// Pack a 64-bit fold signature into disjoint `BFP-64` positive and negative bitplanes
/// partitioned by the quantization byte `qt`.
#[inline]
pub const fn pack_fold_bfp64(fold: u64, qt: u8) -> (u64, u64) {
    let mask = qt_broadcast_mask(qt);
    (fold & mask, fold & !mask)
}

/// Reconstruct the 64-bit fold signature from `BFP-64` bitplanes and verify zero bit leakage.
#[inline]
pub const fn unpack_fold_bfp64(pos_plane: u64, neg_plane: u64, qt: u8) -> Option<u64> {
    if (pos_plane & neg_plane) != 0 {
        return None;
    }
    let mask = qt_broadcast_mask(qt);
    if (pos_plane & !mask) != 0 || (neg_plane & mask) != 0 {
        return None;
    }
    Some(pos_plane | neg_plane)
}

/// 64-lane Kleene three-valued AND (`+AND=1`) over `BFP-64` bitplane words.
#[inline]
pub const fn bfp64_and_kleene(pos_a: u64, neg_a: u64, pos_b: u64, neg_b: u64) -> (u64, u64) {
    let neg_out = neg_a | neg_b;
    let pos_out = (pos_a & pos_b) & !neg_out;
    (pos_out, neg_out)
}

/// 64-lane Kleene three-valued NAND (`~NAND=-1`) over `BFP-64` bitplane words.
#[inline]
pub const fn bfp64_nand_kleene(pos_a: u64, neg_a: u64, pos_b: u64, neg_b: u64) -> (u64, u64) {
    let (p, n) = bfp64_and_kleene(pos_a, neg_a, pos_b, neg_b);
    (n, p)
}

/// Pack a 15-bit word into a 3-byte SPQT (`1.585 bits/trit`, `3^5 = 243 <= 256` per byte)
/// base-243 frame.
#[inline]
pub const fn pack_word15_to_spqt3b(word15: u16) -> [u8; 3] {
    let clean = word15 & 0x7FFF;
    let mut out = [0u8; 3];
    let mut g = 0usize;
    while g < 3 {
        let mut acc = 0u8;
        let mut pow3 = 1u8;
        let mut j = 0usize;
        while j < 5 {
            let bit_idx = g * 5 + j;
            let bit = ((clean >> bit_idx) & 1) as u8;
            let trit_digit = if bit == 1 { 2u8 } else { 1u8 };
            acc = acc.wrapping_add(trit_digit.wrapping_mul(pow3));
            pow3 = pow3.wrapping_mul(3);
            j += 1;
        }
        out[g] = acc;
        g += 1;
    }
    out
}

/// Unpack a 3-byte SPQT base-243 frame back into a 15-bit word.
#[inline]
pub const fn unpack_spqt3b_to_word15(spqt: [u8; 3]) -> Option<u16> {
    let mut word15 = 0u16;
    let mut g = 0usize;
    while g < 3 {
        if spqt[g] >= 243 {
            return None;
        }
        let mut v = spqt[g];
        let mut j = 0usize;
        while j < 5 {
            let digit = v % 3;
            v /= 3;
            if digit == 2 {
                let bit_idx = g * 5 + j;
                word15 |= 1u16 << bit_idx;
            }
            j += 1;
        }
        g += 1;
    }
    Some(word15)
}

/// Pack balanced ternary weights (`{-1, 0, +1}`) into a `.rodata.gaia.b158` section slice
/// inside `arena` at 5 trits/byte (`base-243`, `3^5 = 243 <= 256`) with zero heap allocation.
pub fn pack_gaia_b158_into<'a>(trits: &[i8], arena: &'a mut Arena) -> Option<&'a [u8]> {
    let packed_len = (trits.len() + 4) / 5;
    let out = arena.alloc(packed_len)?;
    let mut idx = 0usize;
    let mut i = 0usize;
    while i < trits.len() {
        let chunk_end = (i + 5).min(trits.len());
        let mut byte_val = 0u8;
        let mut pow3 = 1u8;
        let mut k = i;
        while k < chunk_end {
            let t = trits[k];
            let digit = if t < 0 { 0u8 } else if t == 0 { 1u8 } else { 2u8 };
            byte_val = byte_val.wrapping_add(digit.wrapping_mul(pow3));
            pow3 = pow3.wrapping_mul(3);
            k += 1;
        }
        out[idx] = byte_val;
        idx += 1;
        i += 5;
    }
    Some(&out[..idx])
}

/// Tightened `.rodata.gaia.b158` in-register XOR fold + exact `p/q` compression ratio.
pub fn xor_fold_gaia_b158(trits: &[i8]) -> (u64, usize, RationalPQ) {
    let mut acc = 0u64;
    let mut word_buf = [0u8; 8];
    let mut word_idx = 0usize;
    let mut packed_len = 0usize;
    let mut i = 0usize;
    while i < trits.len() {
        let chunk_end = (i + 5).min(trits.len());
        let mut byte_val = 0u8;
        let mut pow3 = 1u8;
        let mut k = i;
        while k < chunk_end {
            let t = trits[k];
            let digit = if t < 0 { 0u8 } else if t == 0 { 1u8 } else { 2u8 };
            byte_val = byte_val.wrapping_add(digit.wrapping_mul(pow3));
            pow3 = pow3.wrapping_mul(3);
            k += 1;
        }
        word_buf[word_idx] = byte_val;
        word_idx += 1;
        packed_len += 1;
        if word_idx == 8 {
            acc ^= u64::from_le_bytes(word_buf);
            word_buf = [0u8; 8];
            word_idx = 0;
        }
        i += 5;
    }
    let mut tail = 0usize;
    while tail < word_idx {
        acc ^= word_buf[tail] as u64;
        tail += 1;
    }
    let ratio = RationalPQ::reduce(trits.len() as u64, packed_len.max(1) as u64);
    (acc, packed_len, ratio)
}

/// Seal `(raw, packed)` into a 32-byte `0x62`-gated `wupz` frame.
pub fn seal_frame_32b(raw: &[u8], packed: &[u8], gate: u8) -> [u8; 32] {
    let mut frame = [0u8; 32];
    frame[0..8].copy_from_slice(&xor_fold(raw).to_le_bytes());
    frame[8..16].copy_from_slice(&xor_fold(packed).to_le_bytes());
    frame[16..20].copy_from_slice(&(raw.len() as u32).to_le_bytes());
    frame[20..24].copy_from_slice(&(packed.len() as u32).to_le_bytes());
    frame[24] = gate;
    frame[25] = FRAME_ABI_V1;
    let prefix = xor_fold(&frame[0..26]).to_le_bytes();
    frame[26..32].copy_from_slice(&prefix[0..6]);
    frame
}

/// Verify a 32-byte `wupz` frame against `(raw, packed, gate)` in constant shape.
pub fn verify_frame_32b(frame: &[u8; 32], raw: &[u8], packed: &[u8], gate: u8) -> bool {
    if frame[24] != gate || frame[25] != FRAME_ABI_V1 {
        return false;
    }
    let expected = seal_frame_32b(raw, packed, gate);
    byte_diff(frame, &expected) == 0
}

/// Scalar reference for sealing a 64-byte cache-line-aligned `LitanyWireFrame`.
pub fn seal_litany_wire_64b_scalar(raw: &[u8], packed: &[u8], gate: u8, qt: u8) -> LitanyWireFrame {
    let frame32 = seal_frame_32b(raw, packed, gate);
    let mut raw_fold_bytes = [0u8; 8];
    raw_fold_bytes.copy_from_slice(&frame32[0..8]);
    let raw_fold = u64::from_le_bytes(raw_fold_bytes);

    let mut packed_fold_bytes = [0u8; 8];
    packed_fold_bytes.copy_from_slice(&frame32[8..16]);
    let packed_fold = u64::from_le_bytes(packed_fold_bytes);

    let composite_fold = raw_fold ^ packed_fold;
    let (pos, neg) = pack_fold_bfp64(composite_fold, qt);
    let ratio = RationalPQ::reduce(raw.len() as u64, packed.len().max(1) as u64);
    let spqt = pack_word15_to_spqt3b((composite_fold & 0x7FFF) as u16);

    let mut wire = LitanyWireFrame {
        frame32,
        bfp64_pos: pos,
        bfp64_neg: neg,
        ratio_p: ratio.p as u32,
        ratio_q: ratio.q as u32,
        spqt_root: spqt,
        qt_mask: qt,
        lanes_active: LANES_1_OF_1,
        wire_seal16: 0,
    };
    // SAFETY: `LitanyWireFrame` is `#[repr(C, align(64))]` with no uninitialized padding bytes.
    let wire_bytes = unsafe {
        core::slice::from_raw_parts((&wire as *const LitanyWireFrame).cast::<u8>(), 64)
    };
    wire.wire_seal16 = (xor_fold(&wire_bytes[0..62]) & 0xFFFF) as u16;
    wire
}

/// Seal `(raw, packed)` into a 64-byte cache-line-aligned `LitanyWireFrame`.
#[cfg(have_zig)]
pub fn seal_litany_wire_64b(raw: &[u8], packed: &[u8], gate: u8, qt: u8) -> LitanyWireFrame {
    let mut out = LitanyWireFrame {
        frame32: [0u8; 32],
        bfp64_pos: 0,
        bfp64_neg: 0,
        ratio_p: 0,
        ratio_q: 1,
        spqt_root: [0u8; 3],
        qt_mask: qt,
        lanes_active: LANES_1_OF_1,
        wire_seal16: 0,
    };
    // SAFETY: `out`, `raw`, and `packed` are valid slices for the duration of the C call.
    let ok = unsafe {
        zig::wupz_seal_litany_wire_64b(
            &mut out,
            raw.as_ptr(),
            raw.len(),
            packed.as_ptr(),
            packed.len(),
            gate,
            qt,
        )
    };
    debug_assert_eq!(ok, 1);
    out
}

#[cfg(not(have_zig))]
pub fn seal_litany_wire_64b(raw: &[u8], packed: &[u8], gate: u8, qt: u8) -> LitanyWireFrame {
    seal_litany_wire_64b_scalar(raw, packed, gate, qt)
}

/// Verify a 64-byte cache-line-aligned `LitanyWireFrame` in constant shape.
#[cfg(have_zig)]
pub fn verify_litany_wire_64b(
    wire: &LitanyWireFrame,
    raw: &[u8],
    packed: &[u8],
    gate: u8,
    qt: u8,
) -> bool {
    // SAFETY: `wire`, `raw`, and `packed` are valid pointers for the duration of the C call.
    unsafe {
        zig::wupz_verify_litany_wire_64b(
            wire,
            raw.as_ptr(),
            raw.len(),
            packed.as_ptr(),
            packed.len(),
            gate,
            qt,
        ) == 1
    }
}

#[cfg(not(have_zig))]
pub fn verify_litany_wire_64b(
    wire: &LitanyWireFrame,
    raw: &[u8],
    packed: &[u8],
    gate: u8,
    qt: u8,
) -> bool {
    if (wire.bfp64_pos & wire.bfp64_neg) != 0 {
        return false;
    }
    if unpack_spqt3b_to_word15(wire.spqt_root).is_none() {
        return false;
    }
    let expected = seal_litany_wire_64b_scalar(raw, packed, gate, qt);
    let a = unsafe {
        core::slice::from_raw_parts((wire as *const LitanyWireFrame).cast::<u8>(), 64)
    };
    let b = unsafe {
        core::slice::from_raw_parts((&expected as *const LitanyWireFrame).cast::<u8>(), 64)
    };
    byte_diff(a, b) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eq_frame_32b_fold_and_incremental_diff() {
        // 32-byte 8b-is Dynamic Environment Equalizer frame (ABI v1)
        let mut frame_a = [0u8; 32];
        frame_a[0..2].copy_from_slice(&(-251i16).to_le_bytes()); // -25.1 dBFS
        frame_a[8..10].copy_from_slice(&(213i16).to_le_bytes()); // +2.13 dB 60Hz
        frame_a[18..20].copy_from_slice(&(1000i16).to_le_bytes()); // 1.000 Left Jack Vol
        frame_a[20..22].copy_from_slice(&(1000i16).to_le_bytes()); // 1.000 Right AirPods Vol
        frame_a[24] = 0x59; // qt:59
        frame_a[25] = 1; // ABI v1
        let prefix = xor_fold(&frame_a[0..26]).to_le_bytes();
        frame_a[26..32].copy_from_slice(&prefix[0..6]);

        assert_eq!(xor_fold(&frame_a), xor_fold_scalar(&frame_a));
        assert_eq!(byte_diff(&frame_a, &frame_a), 0);
        assert!(popcount(&frame_a) > 0);

        let mut frame_b = frame_a;
        frame_b[12..14].copy_from_slice(&(280i16).to_le_bytes()); // +2.80 dB 1kHz
        assert_eq!(
            byte_diff(&frame_a, &frame_b),
            byte_diff_scalar(&frame_a, &frame_b)
        );
        assert!(byte_diff(&frame_a, &frame_b) > 0);
    }

    #[test]
    fn kompress_super_ultra_32b_seal_at_0x62_gate() {
        let raw = [0x62u8; 3761];
        let packed = [0x33u8; 3314];
        let mut frame = seal_frame_32b(&raw, &packed, GATE_0X62);
        assert_eq!(frame[24], GATE_0X62);
        assert_eq!(frame[25], FRAME_ABI_V1);
        assert!(verify_frame_32b(&frame, &raw, &packed, GATE_0X62));

        // Any amplitude spike or bit flip trips the 0x62 gate seal
        frame[31] ^= 0x01;
        assert!(!verify_frame_32b(&frame, &raw, &packed, GATE_0X62));

        let (pos, neg) = pack_fold_bfp64(FOLD_0XE1034E575C371AB0, QUANT_QT55);
        assert_eq!(pos & neg, 0);
        let (and_p, and_n) = bfp64_and_kleene(pos, 0, pos, 0);
        let (nand_p, nand_n) = bfp64_nand_kleene(neg, 0, neg, 0);
        assert_eq!((and_p, and_n), (pos, 0));
        assert_eq!((nand_p, nand_n), (0, neg));
        assert_eq!(
            unpack_fold_bfp64(and_p, nand_n, QUANT_QT55),
            Some(FOLD_0XE1034E575C371AB0)
        );
        assert_eq!(unpack_fold_bfp64(pos | 0x02, neg, QUANT_QT55), None);
    }

    #[test]
    fn litany_wire_64b_zig_rust_parity_and_gaia_b158() {
        let raw = [0x48u8; 3761];
        let packed = [0x20u8; 3314];
        let mut wire = seal_litany_wire_64b(&raw, &packed, GATE_0X62, QUANT_QT55);
        let scalar = seal_litany_wire_64b_scalar(&raw, &packed, GATE_0X62, QUANT_QT55);
        assert_eq!(wire, scalar);
        assert_eq!((&wire as *const LitanyWireFrame as usize) % 64, 0);
        assert_eq!(wire.ratio_p, 3761);
        assert_eq!(wire.ratio_q, 3314);
        assert_eq!(wire.lanes_active, LANES_1_OF_1);
        assert!(verify_litany_wire_64b(
            &wire,
            &raw,
            &packed,
            GATE_0X62,
            QUANT_QT55
        ));

        wire.wire_seal16 ^= 0x0001;
        assert!(!verify_litany_wire_64b(
            &wire,
            &raw,
            &packed,
            GATE_0X62,
            QUANT_QT55
        ));

        let trits = [1i8, -1, 0, 1, -1, 1, 1, 0, -1, 0, 1, -1, 1, 0, -1];
        let mut arena = Arena::with_capacity(64);
        let packed_gaia = pack_gaia_b158_into(&trits, &mut arena).unwrap();
        assert_eq!(packed_gaia.len(), 3);
        let (fold_val, len_val, ratio) = xor_fold_gaia_b158(&trits);
        assert_eq!(len_val, 3);
        assert_eq!(ratio, RationalPQ { p: 5, q: 1 });
        assert_eq!(fold_val, xor_fold_scalar(packed_gaia));
    }
}

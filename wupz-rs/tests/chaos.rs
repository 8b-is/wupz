//! Chaos harness — throw randomized inputs at every Ω₃ primitive and assert the
//! invariants never break. Deterministic LCG so failures reproduce exactly.
//!
//! Invariants under test:
//!   * pack_gaia_b158_into and xor_fold_gaia_b158 agree byte-for-byte.
//!   * xor_fold == xor_fold_scalar for every length.
//!   * byte_diff is symmetric and bounded; popcount is bounded.
//!   * pack_fold_bfp64 → unpack_fold_bfp64 round-trips; pos & neg == 0.
//!   * seal_frame_32b → verify_frame_32b true; any single-byte flip → false.
//!   * seal_litany_wire_64b → verify_litany_wire_64b true; tamper → false.

use wupz_core::arena::Arena;
use wupz_core::fold::*;

/// Deterministic 64-bit LCG (Numerical Recipes constants). No external deps.
struct Chaos(u64);

impl Chaos {
    fn new(seed: u64) -> Self {
        Chaos(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
    fn range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
    fn trit(&mut self) -> i8 {
        match self.next_u64() % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        }
    }
    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u8()).collect()
    }
}

#[test]
fn chaos_gaia_packers_agree() {
    let mut c = Chaos::new(0xA11CE);
    for _ in 0..2000 {
        let n = c.range(64);
        let trits: Vec<i8> = (0..n).map(|_| c.trit()).collect();

        let mut arena = Arena::with_capacity(64);
        let packed = pack_gaia_b158_into(&trits, &mut arena).unwrap();
        let (fold, len, _ratio) = xor_fold_gaia_b158(&trits);

        assert_eq!(len, packed.len(), "packed length mismatch for n={n}");
        assert_eq!(
            fold,
            xor_fold_scalar(packed),
            "fold != scalar fold for n={n}"
        );
    }
}

#[test]
fn chaos_fold_matches_scalar_every_length() {
    let mut c = Chaos::new(0xF01D);
    for len in 0..=300 {
        let buf = c.bytes(len);
        assert_eq!(
            xor_fold(&buf),
            xor_fold_scalar(&buf),
            "fold mismatch at len={len}"
        );
    }
}

#[test]
fn chaos_byte_diff_and_popcount_bounds() {
    let mut c = Chaos::new(0xBEEF);
    for _ in 0..1000 {
        let n = c.range(128);
        let a = c.bytes(n);
        let b = c.bytes(n);
        let d = byte_diff(&a, &b);
        assert_eq!(d, byte_diff(&b, &a), "byte_diff not symmetric");
        assert!(d <= n as u64, "byte_diff exceeds length");
        assert_eq!(byte_diff(&a, &a), 0, "self diff nonzero");
        let pc = popcount(&a);
        assert!(pc <= (n as u64) * 8, "popcount exceeds bit count");
    }
}

#[test]
fn chaos_bfp64_roundtrip() {
    let mut c = Chaos::new(0x0B1E);
    for _ in 0..5000 {
        let fold = c.next_u64();
        let qt = c.next_u8();
        let (pos, neg) = pack_fold_bfp64(fold, qt);
        assert_eq!(pos & neg, 0, "bitplanes overlap");
        assert_eq!(
            unpack_fold_bfp64(pos, neg, qt),
            Some(fold),
            "bfp64 roundtrip failed"
        );
    }
}

#[test]
fn chaos_frame32b_seal_and_tamper() {
    let mut c = Chaos::new(0x5EA1);
    for _ in 0..1000 {
        let rl = c.range(200);
        let raw = c.bytes(rl);
        let pl = c.range(200);
        let packed = c.bytes(pl);
        let gate = c.next_u8();
        let frame = seal_frame_32b(&raw, &packed, gate);
        assert!(
            verify_frame_32b(&frame, &raw, &packed, gate),
            "sealed frame failed verify"
        );

        // Flip one byte: verification must fail.
        let mut bad = frame;
        let i = c.range(32);
        bad[i] ^= 1 << (c.range(8));
        assert!(
            !verify_frame_32b(&bad, &raw, &packed, gate),
            "tampered frame passed verify"
        );
    }
}

#[test]
fn chaos_litany_wire_seal_and_tamper() {
    let mut c = Chaos::new(0x117A);
    for _ in 0..1000 {
        let rl = c.range(200);
        let raw = c.bytes(rl);
        let pl = c.range(200);
        let packed = c.bytes(pl);
        let mut wire = seal_litany_wire_64b(&raw, &packed, GATE_0X62, QUANT_QT55);
        assert!(
            verify_litany_wire_64b(&wire, &raw, &packed, GATE_0X62, QUANT_QT55),
            "wire failed verify"
        );
        assert_eq!(wire.bfp64_pos & wire.bfp64_neg, 0, "wire bitplanes overlap");

        wire.wire_seal16 ^= 1 << (c.range(16));
        assert!(
            !verify_litany_wire_64b(&wire, &raw, &packed, GATE_0X62, QUANT_QT55),
            "tampered wire passed verify"
        );
    }
}

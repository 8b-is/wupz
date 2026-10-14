//! The seam, exercised: the linked Zig kernel must match the Rust scalar
//! reference on every length. If `zig` was unavailable at build time this runs
//! against the fallback and still passes — it reports which backend it got.
//! CI sets WUPZ_EXPECT_BACKEND=zig-simd to reject an unintended fallback.

use wupz_core::fold;

fn pseudo_random(len: usize) -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        })
        .collect()
}

#[test]
fn reports_the_compiled_backend() {
    let backend = fold::backend();
    assert!(
        backend == "zig-simd" || backend == "rust-scalar",
        "unexpected backend: {backend}"
    );
    println!("wupz fold backend = {backend}");
    if let Some(expected) = std::env::var_os("WUPZ_EXPECT_BACKEND") {
        assert_eq!(
            backend,
            expected
                .to_str()
                .expect("WUPZ_EXPECT_BACKEND must be UTF-8"),
            "compiled fold backend does not match WUPZ_EXPECT_BACKEND"
        );
    }
}

#[test]
fn fold_matches_the_scalar_reference_for_every_length() {
    let buf = pseudo_random(1088);
    let mut mismatches = 0usize;
    for len in 0..=buf.len() {
        if fold::xor_fold(&buf[..len]) != fold::xor_fold_scalar(&buf[..len]) {
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "fold disagrees with the scalar reference");
    println!(
        "fold parity ok across 0..=1088 (backend = {})",
        fold::backend()
    );
}

#[test]
fn the_fold_is_its_own_witness() {
    // A body whose length is a whole number of words, plus its own fold as an
    // 8-byte seal: XOR cancels, so the sealed span folds to zero. (The body
    // length must stay word-aligned — appending a word shifts the grouping.)
    let buf = pseudo_random(256);
    let seal = fold::xor_fold(&buf);
    let mut sealed = buf.clone();
    sealed.extend_from_slice(&seal.to_le_bytes());
    assert_eq!(fold::xor_fold(&sealed), 0);
}

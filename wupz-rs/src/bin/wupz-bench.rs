//! wupz-bench — the kernel throughput harness. Dependency-free.
//!
//! Measures `xor_fold`, `byte_diff` and `popcount` over a fixed working set and
//! reports GiB/s using exact integer arithmetic (centi-GiB/s and integer µs/ms).
//! Run it twice to compare the seam:
//!
//! ```text
//! cargo run --release --bin wupz-bench              # zig-simd (when zig is linked)
//! WUPZ_NO_ZIG=1 cargo run --release --bin wupz-bench # rust-scalar fallback
//! ```
//!
//! Nothing is inferred: the result of every pass is folded into a sink and
//! black-boxed, so the optimiser cannot delete the work.

use std::hint::black_box;
use std::time::{Duration, Instant};

use wupz_core::fold;

const MIB: usize = 1024 * 1024;
const GIB_U128: u128 = 1024 * 1024 * 1024;
const WORKING_SET: usize = 64 * MIB; // 64 MiB
const RUNS: usize = 9;

/// Deterministic fill — an LCG, no RNG dependency.
fn fill(buf: &mut [u8], seed: u64) {
    let mut x = seed;
    for b in buf.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (x >> 33) as u8;
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

/// Throughput in centi-GiB/s (100 = 1 GiB/s), computed with exact u128 integer arithmetic.
fn gib_per_sec_x100(bytes: usize, d: Duration) -> u64 {
    let nanos = d.as_nanos();
    if nanos == 0 {
        return 0;
    }
    let num = (bytes as u128) * 100 * 1_000_000_000;
    let den = GIB_U128 * nanos;
    (num / den) as u64
}

fn fmt_rate(rate_x100: u64) -> String {
    format!("{}.{:02}", rate_x100 / 100, rate_x100 % 100)
}

fn bench<F: FnMut() -> u64>(mut f: F) -> (u64, Duration) {
    // warm-up
    let _ = black_box(f());
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let start = Instant::now();
        let out = black_box(f());
        let elapsed = start.elapsed();
        black_box(out);
        samples.push(elapsed);
    }
    let med = median(samples);
    (gib_per_sec_x100(WORKING_SET, med), med)
}

fn main() {
    let mut a = vec![0u8; WORKING_SET];
    let mut b = vec![0u8; WORKING_SET];
    fill(&mut a, 0x9E37_79B9_7F4A_7C15);
    fill(&mut b, 0x0BAD_F00D_DEAD_BEEF);
    // a single flipped byte, so byte_diff has a real (non-zero) answer
    b[WORKING_SET / 2] ^= 0xFF;

    println!("wupz-bench · backend = {}", fold::backend());
    println!(
        "working set = {} MiB · runs = {RUNS} (median)",
        WORKING_SET / MIB
    );
    println!();
    println!("{:<24} {:>10} {:>12}", "kernel", "GiB/s", "median");
    println!("{:-<48}", "");

    let (fold_rate, fold_med) = bench(|| fold::xor_fold(black_box(&a)));
    println!(
        "{:<24} {:>10} {:>12}",
        "xor_fold",
        fmt_rate(fold_rate),
        fmt(fold_med)
    );

    let (diff_rate, diff_med) = bench(|| fold::byte_diff(black_box(&a), black_box(&b)));
    println!(
        "{:<24} {:>10} {:>12}",
        "byte_diff",
        fmt_rate(diff_rate),
        fmt(diff_med)
    );

    let (pop_rate, pop_med) = bench(|| fold::popcount(black_box(&a)));
    println!("{:<24} {:>10} {:>12}", "popcount", fmt_rate(pop_rate), fmt(pop_med));

    println!();
    println!(
        "sink: fold={:016x} diff={} pop={}",
        fold::xor_fold(&a),
        fold::byte_diff(&a, &b),
        fold::popcount(&a)
    );
}

fn fmt(d: Duration) -> String {
    let micros = d.as_micros();
    let ms_whole = micros / 1000;
    let ms_hundredths = (micros % 1000) / 10;
    format!("{ms_whole}.{ms_hundredths:02} ms")
}

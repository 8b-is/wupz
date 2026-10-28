//! wupz-bench — the kernel throughput harness. Dependency-free.
//!
//! Measures `xor_fold`, `byte_diff` and `popcount` over a fixed working set and
//! reports GiB/s. Run it twice to compare the seam:
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

fn gib_per_sec(bytes: usize, d: Duration) -> f64 {
    let secs = d.as_secs_f64();
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / secs
}

fn bench<F: FnMut() -> u64>(mut f: F) -> (f64, Duration) {
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
    (gib_per_sec(WORKING_SET, med), med)
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
        "{:<24} {:>10.2} {:>12}",
        "xor_fold",
        fold_rate,
        fmt(fold_med)
    );

    let (diff_rate, diff_med) = bench(|| fold::byte_diff(black_box(&a), black_box(&b)));
    println!(
        "{:<24} {:>10.2} {:>12}",
        "byte_diff",
        diff_rate,
        fmt(diff_med)
    );

    let (pop_rate, pop_med) = bench(|| fold::popcount(black_box(&a)));
    println!("{:<24} {:>10.2} {:>12}", "popcount", pop_rate, fmt(pop_med));

    println!();
    println!(
        "sink: fold={:016x} diff={} pop={}",
        fold::xor_fold(&a),
        fold::byte_diff(&a, &b),
        fold::popcount(&a)
    );
}

fn fmt(d: Duration) -> String {
    format!("{:.2} ms", d.as_secs_f64() * 1000.0)
}

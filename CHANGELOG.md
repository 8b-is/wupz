# Changelog

All notable changes to wupz are documented here.
The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org).

## [Unreleased]

### Changed
- The `base-243` trit packer is now a single shared `pack_trit_chunk` helper
  with a branchless digit map, used by both `pack_gaia_b158_into` and
  `xor_fold_gaia_b158`. Encoding is unchanged; the two packers can no longer
  drift. Length idioms use `div_ceil`, clearing the remaining
  `clippy::manual_div_ceil` warnings.
- `wupz` is now the canonical superset of the former `upz` crate. The shared
  core was synced up (`Arena::alloc_aligned`, `LayoutPlan::prefix_sum_section_addrs`
  and resolve fast paths, `incremental::SectionBitSet`, `beehive::isqrt_u64` and
  `seal_wigle_databomb_v3_custodian`), and the `upz` extras (CORDIC, the
  group-wise ternary quantizer, reloc patching, env-entropy, song signals) now
  live behind a default-on `extras` feature with matching Zig kernels.
  `--no-default-features --features std` yields the bare core. See
  `docs/UNIFICATION-PLAN.md`.

### Fixed
- Cross-target Cargo builds select the Rust scalar backend instead of linking
  a Zig object compiled for the build host. Native Zig builds are unchanged.
- Benchmark comparisons reject unexpected backends and missing, duplicate or
  unequal result sinks instead of treating two fallback runs as a Zig comparison.
- Empty sections no longer claim an address window.
- Window indices retain all 64 bits instead of wrapping or disappearing at the
  `u32` boundary (16 TiB for 4096-byte windows).
- Regular CI installs the official Zig 0.16.0 archive with a pinned SHA-256,
  avoiding a disallowed third-party setup action while preserving the test gates.
- Nightly CI resolves Zig master from the official index and verifies its
  archive and reported version without a third-party setup action. The master
  kernel and Rust seam checks remain; both workflows exercise setup failures.

### Changed
- Prepare crate version `0.4.0`: `LayoutPlan::window_of` now returns `u64` and
  `incremental::windows_touched` returns `Vec<u64>`. Callers with explicit `u32`
  result types must migrate. The C ABI, block indices and plan image are unchanged.

## [0.3.0] — 2026-10-10

### Added
- **The benchmark lane.** `wupz-bench` (a dependency-free harness) measures
  `xor_fold`, `byte_diff` and `popcount` over a 64 MiB working set — median of 9,
  black-boxed — for both backends; `scripts/bench.sh` runs the pair.
- `fold::byte_diff` and `fold::popcount` join the seam: Zig when linked, scalar
  reference otherwise.
- `BENCHMARKS.md` — the measured top 5, the method, and the findings.
- The **nightly Zig lane** (`nightly-zig.yml`): the kernels and the seam on Zig
  master. `0.18.0-dev.131` passes 5/5 with zero source changes.

### Findings (recorded, not swept up)
- `byte_diff` **3.2×** faster on zig-simd (18.44 vs 5.70 GiB/s).
- `xor_fold` **~9% slower** on zig-simd (45.65 vs 49.72 GiB/s) — the seam wins
  per kernel, not in general. Candidate: drop `@reduce` or fall back to scalar.
- `popcount` is a tie.
- **UPX does not support macOS** (`CantPackException`); `--force-macos` packs
  (252,512 → 36,880 B, 14.61%) but is unofficial. The P0 corpus must be Linux/ELF.

### Changed
- Crate version `0.2.0` → `0.3.0`.

## [0.2.0] — 2026-10-10

### Added
- **The Zig path, linked into the Rust core.** `wupz-rs/build.rs` builds
  `wupz-zig/src/kernels.zig` when `zig` is on `PATH` and links the static
  library, so `wupz_core::fold::xor_fold` runs the SIMD kernel. Without zig the
  crate still builds and tests: it falls back to the scalar reference and reports
  `backend() == "rust-scalar"`. Disable explicitly with `WUPZ_NO_ZIG=1`.
- **Backend parity test** (`wupz-rs/tests/zig_parity.rs`) — the linked Zig kernel
  and the Rust scalar reference must agree on every length from 0 to 1088.
- `fold::backend()` — which path is compiled in (`zig-simd` or `rust-scalar`).
- Repo furniture: `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS`,
  `.editorconfig`, `.gitattributes`, `rust-toolchain.toml`.
- README badges (CI, license, ABI, toolchains).

### Changed
- Crate version `0.1.0` → `0.2.0`.
- `wupz-rs` gains a build script; the ABI version it reports is unchanged (`1`).

## [0.1.0] — 2026-10-10

### Added
- The layout plan, the incremental rebuild rule, the pre-sized arena and the
  deterministic plan image (`wupz-rs/`).
- The first Zig SIMD kernels — `xorFold`, `byteDiff`, `popcount` — each with a
  scalar reference and seeded differential tests.
- The frozen C ABI (`include/wupz_abi.h`) and the parity smoke test that links
  both static libraries.
- `scripts/ci.sh` and the CI workflow.
- The docs: `docs/DESIGN.md`, `docs/ROADMAP.md`.

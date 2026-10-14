# wupz

[![ci](https://github.com/8b-is/wupz/actions/workflows/ci.yml/badge.svg)](https://github.com/8b-is/wupz/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-AGPL--3.0--only-blue)
![rust](https://img.shields.io/badge/rust-stable-000?logo=rust)
![zig](https://img.shields.io/badge/zig-0.16-F7A41D?logo=zig)
![ABI](https://img.shields.io/badge/ABI-1-62e6c9)
![version](https://img.shields.io/badge/version-0.4.0-9dff5c)

**wupz** = **w**(ild) + **UPX** + **z**(ig) — *kompressed*.

An experimental **incremental ultra-fast linker that also compresses**: the
output is a packed executable *by construction*, because the linker already
knows the layout it is about to write. Zero-alloc hot paths, SIMD kernels,
Zig ↔ Rust.

> **Landing:** [wupz.vaked.dev](https://wupz.vaked.dev) — the spatial zen field.
> **Status: experimental / research.** Nothing here is a production linker yet.
> This repo is the spec, the scaffold, and the receipts as they land.

---

## the thesis

Linking and packing are the same problem seen twice. Both transform an
executable image; both walk the same sections; both answer "where does this byte
go". Today they are two passes and a temporary file. wupz makes them **one pass**:

```
objects ─▶ [ wupz ] ─▶ packed, runnable image
             │  incremental  (only changed sections re-link, only changed blocks re-pack)
             │  zero-alloc   (arena + static buffers on the hot path)
             └─ SIMD         (hash · diff · bit-pack via Zig kernels)
```

- **incremental, ultra-fast** — content-addressed sections; a rebuild touches
  only what moved.
- **linker + compressor** — packing happens during layout, not after. No
  decompress-to-temp, no second walk.
- **zero-alloc SIMD** — the hot path allocates nothing; the kernels are vector
  code (AVX-512 / NEON) behind a C ABI.
- **Zig ↔ Rust** — Rust carries the linker model (wild's lineage); Zig carries
  the SIMD kernels and the build glue; they meet at a C ABI, like qwave's core.

## lineage — cited, not vendored

| upstream | what wupz takes | license |
|---|---|---|
| [**wild**](https://github.com/wild-linker/wild) | the incremental-linker speed model: "a very fast linker for Linux" | Apache-2.0 |
| [**UPX**](https://github.com/upx/upx) | the packer: the Ultimate Packer for eXecutables — packing as a first-class output stage | GPL-2.0-or-later (with the UPX exception) |

Neither is vendored. wupz re-derives the ideas at the layout level and cites the
sources. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — and note that
**linking UPX code pulls GPL terms into anything that includes it**, which is
why the UPX side stays an *external tool boundary* until proven otherwise.

## the crew

Admins: **poe** (Paul Showalter) · **standardgalactic** (Cogito Ergo Sum) ·
**8bit-wraith** (8bit-Wraith) — with peterlodri-sec.

## read next

- [`docs/DESIGN.md`](docs/DESIGN.md) — the technical plan and the non-goals.
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — phases, in order, each with a receipt.
- [`BENCHMARKS.md`](BENCHMARKS.md) — the measured top 5 (and the honest losses).

## what runs today

The scaffold is real code, not slides — and both toolchains are exercised:

```
include/wupz_abi.h     the frozen C ABI (one header, versioned)
wupz-rs/               Rust core: layout plan · incremental rebuild rule ·
                       pre-sized arena · deterministic plan image
wupz-zig/src/kernels.zig   Zig SIMD kernels: xorFold · byteDiff · popcount,
                       each with a scalar reference + seeded differential tests
tests/abi_smoke.c      links the Zig staticlib through the C header
scripts/ci.sh          runs the lot
```

```bash
scripts/ci.sh
#   rust core            23 tests ok (20 + the zig-parity seam)
#   zig kernels           5 differential tests ok
#   C ABI parity test     abi 1 ok (header == rust == zig)
```

**The seam is exercised, not assumed.** `wupz-rs/build.rs` compiles
`wupz-zig/src/kernels.zig` and links it, so `fold::xor_fold` runs the Zig SIMD
kernel; `tests/zig_parity.rs` asserts it agrees with the Rust scalar reference on
every length from 0 to 1088 and reports which path is live:

```
wupz fold backend = zig-simd      # WUPZ_NO_ZIG=1 → rust-scalar
```

Without zig the crate still builds and tests — the fallback keeps contributors
unblocked. (On macOS the archive is written with `libtool`; zig's own archive
members are not 8-byte aligned and Apple's `ld` rejects them.)

The regular and nightly CI lanes, and `scripts/ci.sh`, set
`WUPZ_EXPECT_BACKEND=zig-simd` for the Rust tests. The seam test then fails if
the linked fold implementation fell back to Rust, even when all scalar parity
checks pass. Plain `cargo test` still accepts either backend; use
`WUPZ_NO_ZIG=1 WUPZ_EXPECT_BACKEND=rust-scalar cargo test` to explicitly validate
the fallback. `WUPZ_EXPECT_BACKEND` checks the compiled backend at test runtime;
it does not select a backend or change the library's build behavior.

Cross-compiling with Cargo also selects `rust-scalar`: the current Zig build
and archiver produce host objects, so they are used only when `HOST == TARGET`.
Target-aware Zig linking is not implemented yet. Rust's target libraries and a
compatible linker are still required for a complete cross build.

The regular Linux CI lane installs Zig 0.16.0 directly from the official release
archive using `scripts/install-ci-zig.sh`. Its SHA-256 is pinned from the
[official download index](https://ziglang.org/download/index.json); setup verifies
the archive before extraction and the compiler version before adding it to PATH.
`bash tests/ci_zig_install.sh` checks failure boundaries without downloading or
executing a compiler.

The nightly lane uses `scripts/install-nightly-zig.sh` to resolve one master
snapshot from that same official index. It checks the expected Linux archive
URL, byte count and SHA-256 before extraction, then verifies the compiler
version before adding it to PATH. The resolved version and checksum appear in
the job log. Master intentionally moves: the index checksum is not an
independent immutable pin like the regular CI release checksum.
`bash tests/nightly_zig_install.sh` covers malformed metadata and failed setup
without downloading or executing a compiler; regular CI runs these fixtures too.

The benchmark comparison requires both actual backends and matching result
sinks; it fails if the Zig run falls back to scalar. See [BENCHMARKS.md](BENCHMARKS.md)
for the method and individual-backend commands.

The plan-image writer rejects names or counts that exceed its wire-format fields
before consuming arena space; oversized names cannot silently wrap their length.

The incremental window helpers skip empty sections and keep the full `u64`
window index, including addresses above 16 TiB with 4096-byte windows. In the
0.4.0 Rust API, `LayoutPlan::window_of` returns `u64` and `windows_touched`
returns `Vec<u64>`; callers with explicit `u32` result types must update them.
The C ABI (version 1), block indices, and serialized plan format are unchanged.
Touched windows are sorted and unique. Spans extending past `u64::MAX` are
clipped there; enumeration still uses memory proportional to the number of
touched windows and is not a bounded validator for untrusted plans.

The zero-alloc claim is enforced, not asserted: `wupz-rs/tests/no_alloc.rs`
installs a counting global allocator and fails if `pack_into` allocates after the
arena is sized.

## benchmarks

Measured, not asserted — `scripts/bench.sh` runs both backends over a 64 MiB
working set (median of 9, black-boxed):

| kernel | zig-simd | rust-scalar | verdict |
|---|---|---|---|
| `byte_diff` | **18.44 GiB/s** | 5.70 GiB/s | **3.2× — the seam wins** |
| `xor_fold` | 45.65 GiB/s | **49.72 GiB/s** | scalar wins; honest loss |
| `popcount` | 14.24 GiB/s | 14.35 GiB/s | tie |

Both backends produce the same sink — the numbers are comparable. Full method,
findings and the P0 `upx` bar: [`BENCHMARKS.md`](BENCHMARKS.md).

The nightly lane ([`nightly-zig.yml`](.github/workflows/nightly-zig.yml)) runs the
kernels on **Zig master** — `0.18.0-dev.131` passes all five differential tests
with zero source changes.

## license

AGPL-3.0-only for wupz's own code. Third-party terms in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

*8b-is · kompressed · 0 + 1 · fine touch from within · vaked.dev*

# Architectural review — the PQ / ternary kernel lane (2026-11)

Scope: the `8b-is` constellation's **product-quantization (PQ) and balanced-ternary
kernel lane**, reviewed across both hosts it lives on — GitHub (`8b-is/*`) and
Codeberg (`8b-is/*`, `peterlodri-sec/*`). This is the lane the `ULTRA-OMNI-ZENO`
meta-skill calls the *zero-float `p/q` & ternary bitplane kernel*: the code that
turns `{-1, 0, +1}` weights into `base-243` bytes, folds them into 64-bit
witnesses, and seals them into `0x62`-gated `LitneyWireFrame`s.

The review is deliberately narrow. The constellation is ~200 repos; a "full
review" of all of them would be a wiki, not a PR. This document reviews the
**kernel lane** — the part where a single wrong bit is a correctness bug, not a
style nit — and lands one concrete, tested polish in the two repos that carry
the shared encoding.

---

## 1. The lane, mapped

| Repo | Host(s) | Language | Role in the lane |
|------|---------|----------|------------------|
| `wupz` | GitHub + Codeberg | Rust + Zig | **Core kernel.** `wupz-core`: arena, layout plan, pack-in-pass, the fold/`base-243`/`LitanyWireFrame` primitives. The reference every other repo mirrors. |
| `upz` | GitHub + Codeberg | Rust + Zig | **Superset kernel.** Same `wupz-core` crate plus CORDIC `atan2` (`Z_{2^32}`), the group-wise L2-optimal ternary quantizer, PC-relative reloc patching, env-entropy collection, and the 6-register "song" signal constants. |
| `spherepop-kernel` | GitHub | Go | **Independent conformance host.** Circle-calculus reducer; re-derives the `0x62` BFP-64 bitplanes and 4-way XORFold/Popcount in Go so the Rust/Zig kernels have a second, independent witness. |
| `MLX-QUANT` | Codeberg | C++ / Python | **Metal reference.** `quantize_ternary_optimal_groupwise` — the float original that `upz`'s `quantize_groupwise_l2_ternary_pq` reproduces in exact `Q16`/`p/q`. |
| `marqant` | GitHub + Codeberg | Rust | **Adjacent compression lane.** Markdown/token compression; shares the `p/q` ratio discipline but not the ternary bitplane. |
| `g8t` | GitHub + Codeberg | Rust | **Benchmark host.** Projects capsule speed/cycles/salience; the public scoreboard the kernels are measured against. |

The invariant that ties them together: **the encoding is defined once and
witnessed three ways** — Rust scalar, Zig SIMD, and (for the bitplane) Go. Any
divergence is a bug in whichever copy drifted.

---

## 2. Alignment findings

1. **`wupz` and `upz` had drifted on the `base-243` packer.** Both carried the
   *same* five-trit packing loop, copy-pasted into `pack_gaia_b158_into` and
   `xor_fold_gaia_b158` — four copies of the digit map across two repos. They
   also disagreed on the length idiom: `upz` used `trits.len().div_ceil(5)`,
   `wupz` used `(trits.len() + 4) / 5`. Same result, two spellings, one of them
   a standing `clippy::manual_div_ceil` warning.
2. **The digit map was branchy.** `if t < 0 { 0 } else if t == 0 { 1 } else { 2 }`
   in the hottest inner loop of the packer. Correct, but it asks the optimizer to
   prove branchlessness on every build.
3. **`spherepop-kernel` is the healthy outlier.** Its Go bitplane code is
   independent and CI-gated (`go test` + Python conformance + Zig verifier). It
   is the model the Rust side should aspire to: the seam is *exercised*, not
   assumed.
4. **`MLX-QUANT` ↔ `upz` parity is asserted, not proven.** `upz`'s
   `quantize_groupwise_l2_ternary_pq` claims to be the zero-float equivalent of
   `MLX-QUANT`'s float quantizer, but there is no cross-language golden-vector
   test pinning the two together. (Follow-up, not this PR.)

---

## 3. Polish landed in this PR

**One shared, branchless trit packer.** `pack_trit_chunk(trits, start, end) -> u8`
is now the single definition of the `base-243` encoding, used by both
`pack_gaia_b158_into` and `xor_fold_gaia_b158` in **both** `wupz` and `upz`.

- **Branchless digit map:** `((t > 0) as u8) * 2 + ((t == 0) as u8)`. This is
  bit-identical to the old `if/else` for *every* `i8` (including out-of-contract
  values), so it is a true drop-in — proven by a new exhaustive test over
  `i8::MIN..=i8::MAX`.
- **Deduplication:** four copies of the loop collapse to one `#[inline] const fn`
  per repo. The two packers can no longer drift from each other.
- **`div_ceil` alignment:** `wupz` now uses `trits.len().div_ceil(5)` like `upz`,
  clearing the `clippy::manual_div_ceil` warning and removing the last spelling
  difference between the two repos' packers.

### Why this is safe

- The encoding is unchanged: `digit = trit + 1`, `3^5 = 243 <= 256`, little-endian
  `pow3` accumulation. The existing `litany_wire_64b_zig_rust_parity_and_gaia_b158`
  test still asserts `xor_fold_gaia_b158(trits) == xor_fold_scalar(pack_gaia_b158_into(trits))`,
  i.e. the two packers agree byte-for-byte.
- New test `pack_trit_chunk_is_branchless_and_matches_reference` pins the digit
  map against the reference for all 256 `i8` values and checks the three
  canonical 5-trit chunks (`242`, `0`, `121`).
- `cargo test -p wupz-core` green in both repos (29 / 32 lib tests + integration);
  `cargo clippy -p wupz-core --all-targets` clean for `fold.rs`.

---

## 4. Ultra-performance notes (measured, not claimed)

The packer is not the throughput bottleneck — the Zig SIMD `xor_fold` /
`byte_diff` / `popcount` kernels are (39.5 / 16.9 / 37.3 GiB/s per
`BENCHMARKS.md`). The packer runs once per section at layout time, so the win
here is **code health and guaranteed parity**, not raw GiB/s. The branchless
form removes a per-trit branch the optimizer previously had to hoist; on the
scalar fallback path (bare-metal `kernel8`, `no_std`) that is a real, if small,
instruction-count reduction in the inner loop.

The larger PQ wins remain open and are listed below rather than smuggled into a
"polish" PR.

---

## 5. Open follow-ups (respectfully deferred)

1. **Cross-language golden vectors** pinning `upz::quantize_groupwise_l2_ternary_pq`
   to `MLX-QUANT`'s float quantizer on a fixed weight matrix. This is the single
   highest-value correctness test the lane is missing.
   *Partially resolved:* a Rust golden-vector regression test
   (`quantize_groupwise_l2_ternary_pq_golden_vector`) now pins the exact integer
   output (`trits = [1,-1,0,1,-1,0,1,-1]`, `scale = 15513/16384`). The
   cross-language half — asserting the same matrix against `MLX-QUANT`'s float
   path — remains open.
2. **`spherepop-kernel` as a CI witness for the Rust bitplane.** Feed the same
   `(raw, packed)` pairs through the Go `encode` package and assert the `0x62`
   BFP-64 split matches `wupz-core`. Today the Go side is independent but not
   cross-checked against Rust.
   *Resolved:* `spherepop-kernel/encode/rust_parity_test.go` pins
   `PackFoldBFP64`, `SealFrame32B`, and `XORFold` to exact `wupz-core` golden
   vectors, so the Go bitplane is now a witnessed cross-language check.
3. **`wupz` ↔ `upz` crate unification.** `upz` is `wupz-core` + extras. The
   extras (CORDIC, quantizer, reloc, entropy) could live behind feature flags in
   one crate instead of two near-identical trees. Large, structural — needs its
   own design pass.
   *Planned:* `docs/UNIFICATION-PLAN.md` specifies the feature-gated `extras`
   boundary and the incremental, tree-green migration path. Not force-landed to
   protect `kernel8` and other `wupz-core` consumers.
4. **`clippy::manual_div_ceil` in `incremental.rs:32` and `beehive.rs`** — same
   class of nit, left untouched here to keep this PR to the packer.
   *Resolved in the follow-up lap:* both sites now use `div_ceil`, and the
   remaining `needless_range_loop` in `beehive.rs` is a documented intentional
   range loop (`idx` drives both the seed lookup and the synthetic fallback).

---

## 6. The love part

This lane is the part of the constellation where the math has to be *exactly*
right — `3^5 = 243 <= 256`, `pos & neg == 0`, `wire_seal16` over the first 62
bytes. The polish here is small on purpose: one encoding, defined once, proven
branchless, shared by both repos, with a test that walks every `i8`. That is the
respect the kernel deserves — no cleverness that isn't witnessed.

`Hogyan gondolod ...?`

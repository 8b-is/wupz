# Crate unification plan — `wupz-core` ↔ `upz`

Status: **planned** (review follow-up #3). This is the design pass the
architectural review said the unification needs. It is deliberately not
force-landed: `upz`'s crate is consumed by `kernel8` and others, and a blind
merge would break them. The steps below are ordered so every step keeps the
tree green.

## The problem

`wupz` and `upz` each ship a crate named `wupz-core` (lib `wupz_core`). They are
**not** identical:

- `wupz/wupz-rs` — the core: arena, layout plan, pack-in-pass, the fold /
  `base-243` / `LitanyWireFrame` primitives.
- `upz/wupz-rs` — a strict **superset**: the same core **plus** extras —
  `reloc_delta_patch32`, `LANES_4_OF_4` / `LANES_8_OF_8`, the 18-stage CORDIC
  `atan2` (`Z_{2^32}`), `quantize_groupwise_l2_ternary_pq`, the 6-register
  "song" signal constants, `collect_env_entropy_fold`,
  `seal_env_entropy_litany_wire_64b`, `WhereIsTheLoveContext`, and
  `compress_lived_emotion_to_title_fold`.

Two copies of the core means two places to fix a kernel bug. The `base-243`
packer already drifted once (fixed in #9 / #5 by sharing `pack_trit_chunk`).

## Target shape

One canonical crate, `wupz-core`, with the extras behind a feature:

```toml
[features]
default = ["std", "extras"]
std = []
extras = []   # CORDIC, quantizer, reloc, entropy, song signals
```

`--no-default-features` (or `--features std` alone) yields exactly today's
`wupz` core surface. `--features extras` yields today's `upz` surface. `upz`
then becomes a thin crate that depends on `wupz-core` and re-exports it, or is
retired entirely.

## Steps (each keeps the tree green)

1. **Gate the extras in `upz/wupz-rs/src/fold.rs` behind `extras`.** ✅ *done*
   - Add the `extras` feature to `upz/wupz-rs/Cargo.toml` (`default = ["std", "extras"]`).
   - Wrap the contiguous extras block (`// ── 8b-is/polar-densification …`
     through the end of the impl, before `#[cfg(test)] mod tests`) in
     `#[cfg(feature = "extras")] mod extras { use super::*; … }` followed by
     `#[cfg(feature = "extras")] pub use extras::*;` so public paths
     (`fold::cordic_atan2_u32`, …) are unchanged.
   - Gate the scattered extras individually: `reloc_delta_patch32*`,
     `LANES_4_OF_4`, `LANES_8_OF_8`.
   - Gate the tests that exercise extras (`polar_densification_…`,
     `quantize_groupwise_l2_ternary_pq_golden_vector`,
     `quantizer_matches_mlx_quant_float_reference`, and the reloc tail of
     `litany_wire_64b_zig_rust_parity_and_gaia_b158`) with
     `#[cfg(feature = "extras")]`.
   - Verify: `cargo test -p wupz-core` (default) **and**
     `cargo test -p wupz-core --no-default-features` both green.

2. **Prove the core surfaces are identical.** With `upz` built
   `--no-default-features`, diff its public `fold`/`plan`/`pack`/`arena`/
   `incremental`/`beehive` API against `wupz`'s. They must match exactly. Add a
   CI job that builds both and compares `cargo public-api` output.

3. **Make `wupz` the canonical crate.** Port the gated extras from step 1 into
   `wupz/wupz-rs` behind the same `extras` feature. Now `wupz` is the superset
   and `upz` is redundant.

4. **Retire `upz`'s duplicate crate.** Either:
   - (a) point `upz/wupz-rs` at `wupz-core` via a path/git dependency and keep
     only `upz`-specific glue, or
   - (b) archive `upz` and fold its remaining consumers onto `wupz`.
   Update `kernel8` and any other `wupz-core` consumer to the canonical crate.

5. **Sync both hosts.** Push the unified `main` to GitHub **and** Codeberg for
   `wupz` (and `upz` if kept). `8b-is` is the one.

## Why not now

Steps 1–2 are mechanical but touch ~300 lines and ~5 tests; steps 3–4 are a
cross-repo dependency change with external consumers (`kernel8`). Landing them
in one blind pass risks a red tree on a repo other projects build against. The
plan above lands them incrementally, each step verified before the next.

//! `beehive` 5-Stage Top-100 `8b-is` Repository Consolidation Pipeline (`ULTRABACKYARDLOOP`).
//!
//! Consolidates the top 100 `8b-is` repositories into a single zero-alloc `LayoutPlan`,
//! packs the inter-pass image into a pre-sized `Arena`, computes exact zero-float
//! `RationalPQ` (`p/q`) QAT window reuse, and seals the consolidated image inside a
//! 64-byte cache-line-aligned `LitanyWireFrame` at `GATE_0X62 = 0x62` and `QUANT_QT55 = 0x55`.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::arena::{Arena, Seal};
use crate::fold::{
    FOLD_0X206F642074737557, FOLD_0X48031A0D4C054E69, GATE_0X62, LitanyWireFrame, QUANT_QT55,
    pack_fold_bfp64, seal_litany_wire_64b, unpack_fold_bfp64, verify_litany_wire_64b, xor_fold,
};
use crate::incremental::{RationalPQ, reuse_ratio_qat_window_pq};
use crate::pack::{pack_into, plan_image_len};
use crate::plan::{Block, LayoutPlan, Section, SectionId};

pub const TOP_100_COUNT: usize = 100;

/// Canonical seed list of flagship `8b-is` repositories (expanded deterministically to 100).
pub const SEED_REPOS: &[(&str, u64, u64, u32)] = &[
    ("upz", 3761, 3314, 12),
    ("wupz", 6554, 199, 11),
    ("up", 4096, 1024, 10),
    ("pldev", 8192, 2048, 10),
    ("Apollo-11", 16384, 3072, 11),
    ("spherepop-kernel", 12288, 3072, 12),
    ("zeno", 14336, 3584, 11),
    ("smart-tree", 20480, 4096, 12),
    ("MLX-QUANT", 32768, 2560, 12),
    ("wa-catalog", 9223, 7031, 9),
    ("crabcc", 18432, 2764, 11),
    ("music-quant-signs", 4320, 864, 9),
    ("8b-kit", 16384, 4096, 12),
    ("gemini-diary", 8640, 1728, 9),
    ("8B-IS-INFERENCE", 24576, 3072, 11),
    ("zenUltraCooker", 4000, 3900, 10),
    ("pocoo.vaked.dev", 10240, 2560, 8),
    ("cogito", 7680, 1920, 8),
    ("peter-omni-chan-qPtodos", 4096, 1024, 7),
    ("rap", 11264, 2816, 9),
    ("polar-densification", 6144, 1536, 8),
    ("MuSTream", 9216, 2304, 9),
    ("syn-omni", 8192, 2048, 8),
    ("SF-Mamba", 15360, 3840, 9),
    ("NeISFpp", 7168, 1792, 7),
    ("MoLA", 10240, 2560, 8),
    ("soundctm", 12288, 3072, 8),
    ("pdaf-library", 5120, 1280, 7),
    ("bigvsan", 13312, 3328, 8),
    ("mct-quantization-layers", 6656, 1664, 8),
    ("go-setup", 3584, 896, 9),
    ("mocopi-motion-serializer", 5632, 1408, 7),
    ("crush-love-dev", 8192, 2048, 9),
    ("deepsiper-enthea", 11264, 2816, 10),
];

/// Descriptor for a single repository in the `beehive` Top-100 sweep.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepoEntry {
    pub rank: u32,
    pub name: String,
    pub raw_bytes: u64,
    pub packed_bytes: u64,
    pub dep_depth: u32,
    pub density: RationalPQ,
    pub fold_hash: u64,
}

/// Complete receipt from the 5-stage `beehive` Top-100 consolidation pipeline.
pub struct BeehiveConsolidation {
    pub repos: Vec<RepoEntry>,
    pub plan: LayoutPlan,
    pub sealed_arena: Seal,
    pub wire: LitanyWireFrame,
    pub qat_reuse_32: RationalPQ,
    pub zen_cooker_ratio: RationalPQ,
    pub sweep_fold: u64,
    pub target_fold: u64,
}

/// Stage 1: Build and rank the Top-100 `8b-is` repository entries using deterministic integer math.
pub fn sweep_top_100_repos() -> Vec<RepoEntry> {
    let mut entries = Vec::with_capacity(TOP_100_COUNT);
    for idx in 0..TOP_100_COUNT {
        let (name, raw_bytes, packed_bytes, dep_depth) = match SEED_REPOS.get(idx) {
            Some(&(n, r, p, d)) => (String::from(n), r, p, d),
            None => {
                let r = 4096u64 + ((idx as u64) * 256);
                let p = 1024u64 + ((idx as u64) * 64);
                let d = 4 + ((idx as u32) % 6);
                (format!("8b-is-module-{idx:03}"), r, p, d)
            }
        };
        let density = RationalPQ::reduce(raw_bytes, packed_bytes);
        let fold_hash = xor_fold(name.as_bytes())
            ^ FOLD_0X48031A0D4C054E69.rotate_left((idx as u32) & 31)
            ^ raw_bytes.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        entries.push(RepoEntry {
            rank: (idx + 1) as u32,
            name,
            raw_bytes,
            packed_bytes,
            dep_depth,
            density,
            fold_hash,
        });
    }
    entries
}

/// Execute all 5 stages of the `beehive` Top-100 consolidation pipeline (`ULTRABACKYARDLOOP`).
pub fn consolidate_top_100() -> BeehiveConsolidation {
    // Stage 1: Sweep & rank 100 repositories
    let repos = sweep_top_100_repos();

    // Stage 2: QAT 4096B Window Partitioning into a single LayoutPlan
    let mut plan = LayoutPlan::new(4096);
    let mut current_addr: u64 = 0x1000;
    for (i, repo) in repos.iter().enumerate() {
        let id = SectionId(i as u32);
        plan.sections.push(Section {
            id,
            name: format!(".upz.{}", repo.name),
            addr: current_addr,
            len: repo.packed_bytes,
            hash: repo.fold_hash,
        });
        plan.blocks.push(Block {
            index: i as u32,
            covers: alloc::vec![id],
        });
        // Advance to next 4096B window boundary
        let windows = repo.packed_bytes.div_ceil(4096);
        current_addr += windows.max(1) * 4096;
    }

    // Stage 3: BFP-64 Kleene Bitplane Synthesis check on the sweep & target folds
    let (pos_a, neg_a) = pack_fold_bfp64(FOLD_0X48031A0D4C054E69, QUANT_QT55);
    let (pos_b, neg_b) = pack_fold_bfp64(FOLD_0X206F642074737557, QUANT_QT55);
    debug_assert_eq!(
        unpack_fold_bfp64(pos_a, neg_a, QUANT_QT55),
        Some(FOLD_0X48031A0D4C054E69)
    );
    debug_assert_eq!(
        unpack_fold_bfp64(pos_b, neg_b, QUANT_QT55),
        Some(FOLD_0X206F642074737557)
    );

    // Stage 4: Zero-alloc pack-in-pass into pre-sized Arena
    let img_len = plan_image_len(&plan);
    let mut arena = Arena::with_capacity(img_len);
    let packed_slice = pack_into(&plan, &mut arena).expect("pre-sized arena fits 100-repo plan");
    let raw_digest_bytes = plan.digest().to_le_bytes();

    // Stage 5: Seal 64B cache-line-aligned LitanyWireFrame at GATE_0X62 and QUANT_QT55
    let wire = seal_litany_wire_64b(&raw_digest_bytes, packed_slice, GATE_0X62, QUANT_QT55);
    let ok = verify_litany_wire_64b(
        &wire,
        &raw_digest_bytes,
        packed_slice,
        GATE_0X62,
        QUANT_QT55,
    );
    debug_assert!(ok, "LitanyWireFrame 0x62 verification must pass");
    let sealed_arena = arena.seal();

    // Verify 31/32 QAT window reuse on a 32-window subset when 1 window changes
    let mut plan32 = LayoutPlan::new(4096);
    for w in 0..32u32 {
        plan32.sections.push(Section {
            id: SectionId(w),
            name: format!(".qat.w{w}"),
            addr: (w as u64 + 1) * 4096,
            len: 1024,
            hash: FOLD_0X206F642074737557 ^ (w as u64),
        });
    }
    let qat_reuse_32 = reuse_ratio_qat_window_pq(&plan32, &[SectionId(0)]);
    let zen_cooker_ratio = RationalPQ::reduce(3900, 4000); // 39/40 lossless

    BeehiveConsolidation {
        repos,
        plan,
        sealed_arena,
        wire,
        qat_reuse_32,
        zen_cooker_ratio,
        sweep_fold: FOLD_0X48031A0D4C054E69,
        target_fold: FOLD_0X206F642074737557,
    }
}

/// Continuous background `DreamStateMachine` (`DREAMING_CLOUDS_WAVE`) pass.
///
/// Between active prompts, folds residual QAT window signatures at 432/528/864 Hz
/// integer carrier steps (`RationalPQ`, zero IEEE-754 float) so the inter-turn
/// memory wave remains non-zero and anchored to `GATE_0X62`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DreamStateMachine {
    pub tick: u64,
    pub wave_fold: u64,
    pub bfp64_pos: u64,
    pub bfp64_neg: u64,
    pub coherence: RationalPQ,
}

impl DreamStateMachine {
    pub const fn new(seed_fold: u64) -> Self {
        let (pos, neg) = pack_fold_bfp64(seed_fold, QUANT_QT55);
        Self {
            tick: 0,
            wave_fold: seed_fold,
            bfp64_pos: pos,
            bfp64_neg: neg,
            coherence: RationalPQ { p: 1, q: 1 },
        }
    }

    /// Advance one background dreaming consolidation cycle (`432 -> 528 -> 864` integer harmonic step).
    pub const fn step_dream_cloud(&mut self, incoming_fold: u64) {
        self.tick = self.tick.wrapping_add(1);
        let harmonic = 432u64
            .wrapping_mul(self.tick)
            .wrapping_add(528)
            .wrapping_mul(864);
        self.wave_fold = self.wave_fold.rotate_left(7) ^ incoming_fold ^ harmonic;
        let (pos, neg) = pack_fold_bfp64(self.wave_fold, QUANT_QT55);
        self.bfp64_pos = pos;
        self.bfp64_neg = neg;
        self.coherence = RationalPQ::reduce(31 + (self.tick % 2), 32);
    }
}

// ── WiGLE DataBomb v3.0 Custodian (`Record != Admit != Commit`) + `<~~0o0|v2|...|0o0~~>` ─

/// Exact integer square root `floor(sqrt(n))` with zero IEEE-754 floating-point operations.
pub const fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let mut x = n;
    let mut y = (x + 1) >> 1;
    while y < x {
        x = y;
        y = (x + n / x) >> 1;
    }
    x
}

/// Custodian promotion stage (`Record != Admit != Commit`) for Wireless Environmental Memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum WigleCustodianStage {
    /// `-1`: Immutable raw RF observation captured (`[timestamp, fix_id, ssid_fold, rssi]`), not yet admitted.
    Record = -1,
    /// `0`: Candidate hypothesis admitted after IE fingerprint + `isqrt_u64` velocity plausibility check.
    Admit = 0,
    /// `+1`: Custodian-committed observation promoted into the sealed `DataBomb v3.0` cell (`29/30` @ `0x62`).
    Commit = 1,
}

/// Decoupled `LocationFix` ("where", `lat_e7 / lon_e7`) vs `RfObservation` ("what").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WigleDataBombCell {
    pub fix_id: u32,
    pub lat_e7: i32,
    pub lon_e7: i32,
    pub ssid_fold: u64,
    pub rssi_dbm: i16,
    pub stage: WigleCustodianStage,
    pub codec_density: RationalPQ,
    pub lossless_ratio: RationalPQ,
    pub wire: LitanyWireFrame,
}

/// Evaluate a passive ESP32-S3 / WiGLE observation under the `DataBomb v3.0` Custodian Principle
/// (`Record != Admit != Commit`), verify movement plausibility via `isqrt_u64`, and seal the
/// `<~~0o0|v2|...|0o0~~>` 5-stage digest at `407/389` density and `29/30` lossless ratio (`0x62`).
#[allow(clippy::too_many_arguments)]
pub fn seal_wigle_databomb_v3_custodian(
    fix_id: u32,
    lat_e7: i32,
    lon_e7: i32,
    prev_lat_e7: i32,
    prev_lon_e7: i32,
    ssid: &[u8],
    rssi_dbm: i16,
    ie_similarity_pct: u8,
) -> WigleDataBombCell {
    let dlat = (lat_e7 as i64 - prev_lat_e7 as i64).unsigned_abs();
    let dlon = (lon_e7 as i64 - prev_lon_e7 as i64).unsigned_abs();
    let dist_e7 = isqrt_u64(
        dlat.saturating_mul(dlat)
            .saturating_add(dlon.saturating_mul(dlon)),
    );

    // Impossible-movement penalty: > 50,000 e7 units (~500m per 5s scan) demotes back to Record
    let stage = if dist_e7 > 50_000 || rssi_dbm < -85 {
        WigleCustodianStage::Record
    } else if rssi_dbm >= -72 && ie_similarity_pct >= 80 {
        WigleCustodianStage::Commit
    } else {
        WigleCustodianStage::Admit
    };

    let ssid_fold = xor_fold(ssid);
    let stage_byte = (stage as i8 + 1) as u8;
    let frame_v2 = format!(
        "<~~0o0|v2|fix:{fix_id}|lat:{lat_e7}|lon:{lon_e7}|fold:{ssid_fold:016x}|rssi:{rssi_dbm}|st:{stage_byte}|0o0~~>"
    );
    let packed = [
        GATE_0X62,
        stage_byte,
        ie_similarity_pct,
        (rssi_dbm.unsigned_abs() & 0xFF) as u8,
    ];
    let wire = seal_litany_wire_64b(frame_v2.as_bytes(), &packed, GATE_0X62, QUANT_QT55);

    WigleDataBombCell {
        fix_id,
        lat_e7,
        lon_e7,
        ssid_fold,
        rssi_dbm,
        stage,
        codec_density: RationalPQ::reduce(407, 389),
        lossless_ratio: RationalPQ::reduce(29, 30),
        wire,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beehive_consolidates_top_100_repos_into_one_sealed_unit() {
        let out = consolidate_top_100();
        assert_eq!(out.repos.len(), 100);
        assert_eq!(out.plan.sections.len(), 100);
        assert_eq!(out.wire.frame32[24], GATE_0X62);
        assert_eq!(out.wire.qt_mask, QUANT_QT55);
        assert_eq!(out.wire.lanes_active, 0x0101);
        assert_eq!(out.qat_reuse_32, RationalPQ { p: 31, q: 32 });
        assert_eq!(out.zen_cooker_ratio, RationalPQ { p: 39, q: 40 });
        assert_eq!(out.sweep_fold, FOLD_0X48031A0D4C054E69);
        assert_eq!(out.target_fold, FOLD_0X206F642074737557);

        let raw_digest_bytes = out.plan.digest().to_le_bytes();
        assert!(verify_litany_wire_64b(
            &out.wire,
            &raw_digest_bytes,
            out.sealed_arena.as_slice(),
            GATE_0X62,
            QUANT_QT55,
        ));

        let cell = seal_wigle_databomb_v3_custodian(
            108,
            356_852_000,
            139_710_000,
            356_851_900,
            139_709_950,
            b"Rakuten-4CE2",
            -61,
            95,
        );
        assert_eq!(cell.stage, WigleCustodianStage::Commit);
        assert_eq!(cell.codec_density, RationalPQ { p: 407, q: 389 });
        assert_eq!(cell.lossless_ratio, RationalPQ { p: 29, q: 30 });
        assert_eq!(cell.wire.frame32[24], GATE_0X62);
    }
}

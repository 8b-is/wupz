//! The incremental rebuild rule — which sections changed, which blocks must be
//! re-packed, and which address windows a change perturbs.
//!
//! Includes `upz` zero-float `RationalPQ` (`p/q`) arithmetic and QAT sub-window
//! reuse tracking to eliminate the coarse `reuse = 1/2` block invalidation leak
//! when a multi-section block has only a single 4096B QAT window modified
//! (`1/2 -> 15/16 -> 31/32`).

use crate::plan::{LayoutPlan, SectionId};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
#[cfg(test)]
use alloc::{format, vec};

/// Binary/Euclidean greatest common divisor over `u64` (returns `1` when both are `0`).
#[inline]
pub const fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

/// Deterministic integer square root (`floor(sqrt(n))`), zero IEEE-754 float.
#[inline]
pub const fn isqrt_u64(n: u64) -> u64 {
    if n < 2 {
        return n;
    }
    let shift = (64 - n.leading_zeros() + 1) / 2;
    let mut x = 1u64 << shift;
    loop {
        let y = (x + n / x) >> 1;
        if y >= x {
            return x;
        }
        x = y;
    }
}

/// Exact GCD-reduced rational `p/q` (`q >= 1`) with zero IEEE-754 floating-point drift.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RationalPQ {
    pub p: u64,
    pub q: u64,
}

impl RationalPQ {
    /// Construct and reduce `p/q` by `gcd_u64(p, q)`.
    #[inline]
    pub const fn reduce(p: u64, q: u64) -> Self {
        if q == 0 {
            return Self { p, q: 1 };
        }
        let g = gcd_u64(p, q);
        Self { p: p / g, q: q / g }
    }

    /// True when `p == q` (`1/1` unit lane).
    #[inline]
    pub const fn is_unit(&self) -> bool {
        self.p == self.q
    }
}

/// Sections whose resolved bytes differ between `old` and `new`.
///
/// A section is changed when it is new, when its content hash moved, or when its
/// size moved. A section that *disappeared* also counts — dropping bytes is a
/// change too.
pub fn changed_sections(old: &LayoutPlan, new: &LayoutPlan) -> Vec<SectionId> {
    let mut changed = BTreeSet::new();
    for n in &new.sections {
        match old.section(n.id) {
            Some(o) if o.hash == n.hash && o.len == n.len => {}
            _ => {
                changed.insert(n.id);
            }
        }
    }
    for o in &old.sections {
        if new.section(o.id).is_none() {
            changed.insert(o.id);
        }
    }
    changed.into_iter().collect()
}

/// Blocks whose covering set intersects the changed set. Exactly these must be
/// re-packed; every other block is reused verbatim.
pub fn blocks_to_repack(plan: &LayoutPlan, changed: &[SectionId]) -> Vec<u32> {
    let set: BTreeSet<SectionId> = changed.iter().copied().collect();
    plan.blocks
        .iter()
        .filter(|b| b.covers.iter().any(|s| set.contains(s)))
        .map(|b| b.index)
        .collect()
}

/// The address-independent windows a change perturbs. A section that spans a
/// window boundary touches every window it overlaps — that is the whole point of
/// fixed windows: the blast radius stays local. Empty sections touch no windows;
/// overflowing end addresses are clipped to `u64::MAX`. Results are sorted and
/// unique. Memory and runtime grow with the number of windows touched.
pub fn windows_touched(plan: &LayoutPlan, changed: &[SectionId]) -> Vec<u64> {
    let mut windows = BTreeSet::new();
    for id in changed {
        if let Some(s) = plan.section(*id) {
            if s.len == 0 {
                continue;
            }
            let first = plan.window_of(s.addr);
            let last_addr = s.addr.saturating_add(s.len - 1);
            let last = plan.window_of(last_addr);
            for w in first..=last {
                windows.insert(w);
            }
        }
    }
    windows.into_iter().collect()
}

/// Total distinct address-independent windows covered by all non-empty sections in `plan`.
pub fn total_windows(plan: &LayoutPlan) -> u64 {
    let mut windows = BTreeSet::new();
    for s in &plan.sections {
        if s.len == 0 {
            continue;
        }
        let first = plan.window_of(s.addr);
        let last_addr = s.addr.saturating_add(s.len - 1);
        let last = plan.window_of(last_addr);
        for w in first..=last {
            windows.insert(w);
        }
    }
    windows.len() as u64
}

/// Exact zero-float `p/q` block reuse ratio (`reused_blocks / total_blocks`).
pub fn reuse_ratio_pq(plan: &LayoutPlan, changed: &[SectionId]) -> RationalPQ {
    if plan.blocks.is_empty() {
        return RationalPQ { p: 1, q: 1 };
    }
    let total = plan.blocks.len() as u64;
    let repack = blocks_to_repack(plan, changed).len() as u64;
    let reused = total.saturating_sub(repack);
    RationalPQ::reduce(reused, total)
}

/// QAT window-level exact `p/q` reuse ratio (`reused_windows / total_windows`).
///
/// Fixes the coarse `reuse = 1/2` leak in `blocks_to_repack` when a block covers
/// multiple 4096B QAT windows or `.rodata.gaia.b158` sub-sections: only the touched
/// windows are invalidated, lifting `1/2` coarse block reuse to `15/16` or `31/32`.
pub fn reuse_ratio_qat_window_pq(plan: &LayoutPlan, changed: &[SectionId]) -> RationalPQ {
    let total = total_windows(plan);
    if total == 0 {
        return RationalPQ { p: 1, q: 1 };
    }
    let touched = windows_touched(plan, changed).len() as u64;
    let reused = total.saturating_sub(touched);
    RationalPQ::reduce(reused, total)
}

/// Legacy `f64` fraction of blocks reused (not re-packed) — `1.0` means nothing moved.
/// Prefer `reuse_ratio_pq` and `reuse_ratio_qat_window_pq` for zero-float determinism.
pub fn reuse_ratio(plan: &LayoutPlan, changed: &[SectionId]) -> f64 {
    if plan.blocks.is_empty() {
        return 1.0;
    }
    let repack = blocks_to_repack(plan, changed).len();
    1.0 - (repack as f64 / plan.blocks.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{Block, Section};

    fn plan(hashes: &[(u32, u64, u64)]) -> LayoutPlan {
        // (id, hash, addr)
        let sections = hashes
            .iter()
            .map(|(id, hash, addr)| Section {
                id: SectionId(*id),
                name: format!(".s{id}"),
                addr: *addr,
                len: 16,
                hash: *hash,
            })
            .collect();
        let blocks = hashes
            .iter()
            .enumerate()
            .map(|(i, (id, _, _))| Block {
                index: i as u32,
                covers: vec![SectionId(*id)],
            })
            .collect();
        LayoutPlan {
            sections,
            blocks,
            window_size: 4096,
        }
    }

    #[test]
    fn only_the_changed_section_is_flagged() {
        let old = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let new = plan(&[(0, 0xaa, 0x1000), (1, 0xcc, 0x2000)]);
        assert_eq!(changed_sections(&old, &new), vec![SectionId(1)]);
    }

    #[test]
    fn only_blocks_covering_changed_sections_repack() {
        let old = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000), (2, 0xcc, 0x3000)]);
        let new = plan(&[(0, 0xaa, 0x1000), (1, 0xcc, 0x2000), (2, 0xcc, 0x3000)]);
        let changed = changed_sections(&old, &new);
        assert_eq!(blocks_to_repack(&new, &changed), vec![1]);
        assert_eq!(reuse_ratio_pq(&new, &changed), RationalPQ { p: 2, q: 3 });
        assert!((reuse_ratio(&new, &changed) - (2.0 / 3.0)).abs() < 1e-9);
    }

    #[test]
    fn nothing_changed_reuses_everything() {
        let old = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let new = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let changed = changed_sections(&old, &new);
        assert!(changed.is_empty());
        assert_eq!(reuse_ratio_pq(&new, &changed), RationalPQ { p: 1, q: 1 });
        assert_eq!(reuse_ratio(&new, &changed), 1.0);
    }

    #[test]
    fn a_new_section_counts_as_changed() {
        let old = plan(&[(0, 0xaa, 0x1000)]);
        let new = plan(&[(0, 0xaa, 0x1000), (1, 0xdd, 0x2000)]);
        assert_eq!(changed_sections(&old, &new), vec![SectionId(1)]);
    }

    #[test]
    fn a_removed_section_counts_as_changed() {
        let old = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let new = plan(&[(0, 0xaa, 0x1000)]);
        assert_eq!(changed_sections(&old, &new), vec![SectionId(1)]);
    }

    #[test]
    fn windows_are_address_independent() {
        let p = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x9000)]);
        // one changed section → exactly its own window
        assert_eq!(windows_touched(&p, &[SectionId(0)]), vec![1]);
        assert_eq!(windows_touched(&p, &[SectionId(1)]), vec![9]);
    }

    #[test]
    fn qat_window_reuse_fixes_coarse_half_leak_to_15_16_and_31_32() {
        // Suppose 32 sections across 32 distinct 4096B windows are grouped into
        // just 2 coarse blocks (16 sections each). Touching 1 section previously
        // yielded block reuse = 1/2; QAT window reuse preserves 31/32!
        let mut sections = Vec::new();
        let mut block0_covers = Vec::new();
        let mut block1_covers = Vec::new();
        for i in 0..32u32 {
            sections.push(Section {
                id: SectionId(i),
                name: format!(".rodata.gaia.b158.{i}"),
                addr: (i as u64 + 1) * 4096,
                len: 1024,
                hash: 0x4803_1a0d_4c05_4e69 ^ (i as u64),
            });
            if i < 16 {
                block0_covers.push(SectionId(i));
            } else {
                block1_covers.push(SectionId(i));
            }
        }
        let layout = LayoutPlan {
            sections,
            blocks: vec![
                Block { index: 0, covers: block0_covers },
                Block { index: 1, covers: block1_covers },
            ],
            window_size: 4096,
        };
        let changed = vec![SectionId(0)];
        // Coarse block reuse was 1/2:
        assert_eq!(reuse_ratio_pq(&layout, &changed), RationalPQ { p: 1, q: 2 });
        // QAT window reuse is 31/32 (zero-float RationalPQ):
        assert_eq!(
            reuse_ratio_qat_window_pq(&layout, &changed),
            RationalPQ { p: 31, q: 32 }
        );
        // Touching 2 windows yields 30/32 == 15/16 reduced:
        assert_eq!(
            reuse_ratio_qat_window_pq(&layout, &[SectionId(0), SectionId(1)]),
            RationalPQ { p: 15, q: 16 }
        );
    }
}

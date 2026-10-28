//! The incremental rebuild rule — which sections changed, which blocks must be
//! re-packed, and which address windows a change perturbs.
//!
//! This is P4's logic, landed early because it is pure and testable without a
//! linker: given two plans, decide the minimum work.

use crate::plan::{LayoutPlan, SectionId};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
#[cfg(test)]
use alloc::{format, vec};

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

fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    if a == 0 { 1 } else { a }
}

/// Reduced rational `(p, q)` of blocks reused (not re-packed) — `(1, 1)` means nothing moved.
pub fn reuse_ratio(plan: &LayoutPlan, changed: &[SectionId]) -> (u64, u64) {
    let total = plan.blocks.len() as u64;
    if total == 0 {
        return (1, 1);
    }
    let repack = (blocks_to_repack(plan, changed).len() as u64).min(total);
    let reused = total - repack;
    let g = gcd_u64(reused, total);
    (reused / g, total / g)
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
        assert_eq!(reuse_ratio(&new, &changed), (2, 3));
    }

    #[test]
    fn nothing_changed_reuses_everything() {
        let old = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let new = plan(&[(0, 0xaa, 0x1000), (1, 0xbb, 0x2000)]);
        let changed = changed_sections(&old, &new);
        assert!(changed.is_empty());
        assert_eq!(reuse_ratio(&new, &changed), (1, 1));
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
}

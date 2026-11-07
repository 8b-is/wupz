//! The layout plan — the artifact the linker emits and the packer consumes.

#[cfg(test)]
use alloc::vec;
use alloc::{string::String, vec::Vec};

/// A section's identity. Stable across rebuilds: it names a logical input, not
/// a byte offset.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SectionId(pub u32);

/// One resolved section: where it landed and the strong hash of its bytes.
///
/// The hash is over the *resolved* content (post-relocation), so a plan can be
/// diffed without re-reading the object files.
#[derive(PartialEq, Eq, Debug)]
pub struct Section {
    pub id: SectionId,
    pub name: String,
    pub addr: u64,
    pub len: u64,
    pub hash: u64,
}

impl Section {
    /// Volatily scrub all metadata and section hash fields in place.
    pub fn zeroize(&mut self) {
        // SAFETY: `self.name.as_bytes_mut()` is overwritten with valid ASCII NULs (0u8).
        let name_bytes = unsafe { self.name.as_bytes_mut() };
        crate::arena::zeroize_slice(name_bytes);
        self.name.clear();
        unsafe {
            core::ptr::write_volatile(&mut self.id.0, 0u32);
            core::ptr::write_volatile(&mut self.addr, 0u64);
            core::ptr::write_volatile(&mut self.len, 0u64);
            core::ptr::write_volatile(&mut self.hash, 0u64);
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// One packed block: the sections whose bytes it covers.
///
/// The covering set is what makes incremental repack possible — a block is
/// invalidated iff its set intersects the changed set.
#[derive(PartialEq, Eq, Debug)]
pub struct Block {
    pub index: u32,
    pub covers: Vec<SectionId>,
}

impl Block {
    /// Volatily scrub block index and covered section IDs in place.
    pub fn zeroize(&mut self) {
        for cover in self.covers.iter_mut() {
            unsafe { core::ptr::write_volatile(&mut cover.0, 0u32) };
        }
        self.covers.clear();
        unsafe { core::ptr::write_volatile(&mut self.index, 0u32) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// The whole plan. `window_size` is the address-independent packing window
/// (see `docs/DESIGN.md` §4): layout is packed in fixed windows so a shifted
/// section perturbs one window, not the tail.
#[derive(PartialEq, Eq, Debug)]
pub struct LayoutPlan {
    pub sections: Vec<Section>,
    pub blocks: Vec<Block>,
    pub window_size: u64,
}

impl LayoutPlan {
    /// Volatily scrub all sections, blocks, and window metadata in place.
    pub fn zeroize(&mut self) {
        for s in self.sections.iter_mut() {
            s.zeroize();
        }
        self.sections.clear();
        for b in self.blocks.iter_mut() {
            b.zeroize();
        }
        self.blocks.clear();
        unsafe { core::ptr::write_volatile(&mut self.window_size, 0u64) };
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for LayoutPlan {
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl Default for LayoutPlan {
    fn default() -> Self {
        Self {
            sections: Vec::new(),
            blocks: Vec::new(),
            window_size: 4096,
        }
    }
}

impl LayoutPlan {
    pub fn new(window_size: u64) -> Self {
        Self {
            sections: Vec::new(),
            blocks: Vec::new(),
            window_size: window_size.max(1),
        }
    }

    /// Total bytes the plan resolves.
    pub fn total_bytes(&self) -> u64 {
        self.sections.iter().map(|s| s.len).sum()
    }

    /// Resolve a section by `SectionId`.
    ///
    /// Fast path (`wild`-linker dense section table trick): when sections are stored
    /// in dense `SectionId(0..N)` order, `self.sections.get(id.0 as usize)` resolves
    /// in `O(1)` direct array indexing without scanning. Falls back to linear scan
    /// when sections are sparse or reordered.
    #[inline]
    pub fn section(&self, id: SectionId) -> Option<&Section> {
        if let Some(s) = self.sections.get(id.0 as usize).filter(|s| s.id == id) {
            return Some(s);
        }
        self.sections.iter().find(|s| s.id == id)
    }

    /// Resolve a block by `index`.
    ///
    /// Fast path: `O(1)` direct array indexing when `self.blocks[index]` matches `index`.
    #[inline]
    pub fn block(&self, index: u32) -> Option<&Block> {
        if let Some(b) = self.blocks.get(index as usize).filter(|b| b.index == index) {
            return Some(b);
        }
        self.blocks.iter().find(|b| b.index == index)
    }

    /// The first block covering `id`, if any.
    pub fn block_for(&self, id: SectionId) -> Option<&Block> {
        self.blocks.iter().find(|b| b.covers.contains(&id))
    }

    /// Address-independent window index for an address, without narrowing the
    /// address space. A zero window size is treated as one byte.
    #[inline]
    pub fn window_of(&self, addr: u64) -> u64 {
        addr / self.window_size.max(1)
    }

    /// `wild`-style linear prefix-sum section address layout pass: assigns aligned
    /// output addresses starting from `base_addr` with power-of-two `align` (minimum 1)
    /// in a single forward sweep and returns the final end address.
    pub fn prefix_sum_section_addrs(&mut self, base_addr: u64, align: u64) -> u64 {
        let step_align = if align.is_power_of_two() { align } else { 1 };
        let mask = step_align - 1;
        let mut cursor = base_addr;
        for s in &mut self.sections {
            cursor = cursor.saturating_add(mask) & !mask;
            s.addr = cursor;
            cursor = cursor.saturating_add(s.len);
        }
        cursor
    }

    /// Fold the plan into a deterministic 64-bit digest (names + addrs + hashes).
    ///
    /// Uses word-wide `fold::xor_fold` over section names instead of byte-by-byte
    /// loops so long symbol/section names fold at SIMD bandwidth.
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        let mut mix = |x: u64| {
            h ^= x;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        mix(self.window_size);
        for s in &self.sections {
            mix(s.id.0 as u64);
            mix(s.addr);
            mix(s.len);
            mix(s.hash);
            mix(s.name.len() as u64);
            mix(crate::fold::xor_fold(s.name.as_bytes()));
        }
        for b in &self.blocks {
            mix(b.index as u64);
            for c in &b.covers {
                mix(c.0 as u64);
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> LayoutPlan {
        LayoutPlan {
            sections: vec![
                Section {
                    id: SectionId(0),
                    name: ".text".into(),
                    addr: 0x1000,
                    len: 0x200,
                    hash: 0xaa,
                },
                Section {
                    id: SectionId(1),
                    name: ".data".into(),
                    addr: 0x3000,
                    len: 0x40,
                    hash: 0xbb,
                },
            ],
            blocks: vec![
                Block {
                    index: 0,
                    covers: vec![SectionId(0)],
                },
                Block {
                    index: 1,
                    covers: vec![SectionId(1)],
                },
            ],
            window_size: 4096,
        }
    }

    #[test]
    fn total_bytes_sums_sections() {
        assert_eq!(sample().total_bytes(), 0x240);
    }

    #[test]
    fn window_is_address_divided() {
        let p = sample();
        assert_eq!(p.window_of(0x1000), 1);
        assert_eq!(p.window_of(0x3000), 3);
    }

    #[test]
    fn digest_is_deterministic() {
        assert_eq!(sample().digest(), sample().digest());
    }

    #[test]
    fn digest_changes_with_content() {
        let mut other = sample();
        other.sections[0].hash ^= 0xff;
        assert_ne!(sample().digest(), other.digest());
    }

    #[test]
    fn block_for_finds_the_cover() {
        let p = sample();
        assert_eq!(p.block_for(SectionId(1)).map(|b| b.index), Some(1));
        assert!(p.block_for(SectionId(9)).is_none());
    }

    #[test]
    fn prefix_sum_section_addrs_aligns_monotonically() {
        let mut p = sample();
        p.sections[0].addr = 0;
        p.sections[0].len = 500;
        p.sections[1].addr = 0;
        p.sections[1].len = 128;
        let end = p.prefix_sum_section_addrs(0x1000, 64);
        assert_eq!(p.sections[0].addr, 0x1000);
        // 0x1000 + 500 = 4596 -> aligned up to 64B = 4608 (0x1200)
        assert_eq!(p.sections[1].addr, 0x1200);
        assert_eq!(end, 0x1280);
    }

    #[test]
    fn zeroize_scrubs_section_hashes_and_names() {
        let mut s = Section {
            id: SectionId(7),
            name: ".secret_section".into(),
            addr: 0x4000,
            len: 0x100,
            hash: 0xDEAD_BEEF_CAFE_BABE,
        };
        let name_ptr = s.name.as_ptr();
        let name_len = s.name.len();
        s.zeroize();
        assert_eq!(s.id.0, 0);
        assert_eq!(s.addr, 0);
        assert_eq!(s.len, 0);
        assert_eq!(s.hash, 0);
        assert!(s.name.is_empty());
        let scrubbed_name = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
        assert!(scrubbed_name.iter().all(|&b| b == 0));
    }
}

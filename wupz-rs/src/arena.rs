//! The arena — one allocation up front, none on the hot path.
//!
//! `Arena::with_capacity` allocates once. After that, `alloc` and `alloc_aligned`
//! hand out slices and never touch the global allocator. The post-layout invariant
//! (`docs/DESIGN.md` §5) is: once the plan is sealed, packing must not allocate.
//! The integration test `tests/no_alloc.rs` enforces it with a counting
//! allocator.

use alloc::{vec, vec::Vec};

/// Overwrite every byte in `slice` using volatile writes followed by a
/// sequentially-consistent compiler fence so LLVM DSE cannot elide the wipe.
#[inline(never)]
pub fn zeroize_slice(slice: &mut [u8]) {
    for byte in slice.iter_mut() {
        // SAFETY: `byte` is a valid, aligned mutable reference within `slice`.
        unsafe { core::ptr::write_volatile(byte, 0u8) };
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    core::hint::black_box(slice);
}

/// A bump arena with a fixed buffer.
pub struct Arena {
    buf: Vec<u8>,
    used: usize,
}

impl Arena {
    /// Allocate the whole buffer once.
    pub fn with_capacity(bytes: usize) -> Self {
        Self {
            buf: vec![0u8; bytes],
            used: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub fn used(&self) -> usize {
        self.used
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.used
    }

    /// Bump-allocate `len` zeroed bytes. Returns `None` when the arena is full —
    /// it never grows (growing would allocate).
    pub fn alloc(&mut self, len: usize) -> Option<&mut [u8]> {
        if len > self.remaining() {
            return None;
        }
        let start = self.used;
        self.used += len;
        Some(&mut self.buf[start..start + len])
    }

    /// Bump-allocate `len` zeroed bytes whose starting address is aligned to `align`
    /// (must be a non-zero power of two, e.g. `8`, `16`, `32`, `64` cache line).
    /// Returns `None` if `align` is not a power of two or if remaining capacity is
    /// insufficient. Never grows the underlying buffer.
    pub fn alloc_aligned(&mut self, len: usize, align: usize) -> Option<&mut [u8]> {
        if !align.is_power_of_two() {
            return None;
        }
        let base_addr = self.buf.as_ptr() as usize;
        let curr_addr = base_addr.checked_add(self.used)?;
        let mask = align - 1;
        let aligned_addr = curr_addr.checked_add(mask)? & !mask;
        let pad = aligned_addr - curr_addr;
        let total = pad.checked_add(len)?;
        if total > self.remaining() {
            return None;
        }
        let start = self.used + pad;
        self.used += total;
        Some(&mut self.buf[start..start + len])
    }

    /// Explicitly scrub the entire arena buffer in place.
    pub fn zeroize(&mut self) {
        zeroize_slice(&mut self.buf);
        self.used = 0;
    }

    /// Seal the arena: all allocation is done, only reads remain.
    /// Unused tail capacity `[used..]` is volatily scrubbed before sealing.
    pub fn seal(mut self) -> Seal {
        let used = self.used;
        if used < self.buf.len() {
            zeroize_slice(&mut self.buf[used..]);
        }
        let buf = core::mem::take(&mut self.buf);
        self.used = 0;
        Seal { buf, used }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        zeroize_slice(&mut self.buf);
        self.used = 0;
    }
}

/// A sealed arena. Read-only by construction; volatily scrubbed on drop.
pub struct Seal {
    buf: Vec<u8>,
    used: usize,
}

impl Seal {
    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.used]
    }

    pub fn len(&self) -> usize {
        self.used
    }

    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    /// Explicitly scrub the sealed buffer in place.
    pub fn zeroize(&mut self) {
        zeroize_slice(&mut self.buf);
        self.used = 0;
    }
}

impl Drop for Seal {
    fn drop(&mut self) {
        zeroize_slice(&mut self.buf);
        self.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_returns_zeroed_slices() {
        let mut a = Arena::with_capacity(64);
        let s = a.alloc(8).unwrap();
        assert_eq!(s, &[0u8; 8]);
        s[0] = 7;
        assert_eq!(a.used(), 8);
    }

    #[test]
    fn alloc_refuses_to_grow() {
        let mut a = Arena::with_capacity(8);
        assert!(a.alloc(8).is_some());
        assert!(a.alloc(1).is_none());
    }

    #[test]
    fn slices_do_not_overlap() {
        let mut a = Arena::with_capacity(16);
        let first = a.alloc(8).unwrap().as_ptr() as usize;
        let second = a.alloc(8).unwrap().as_ptr() as usize;
        assert_eq!(second, first + 8);
    }

    #[test]
    fn alloc_aligned_honors_64_byte_cache_line_alignment() {
        let mut a = Arena::with_capacity(256);
        // Intentionally misalign cursor by 3 bytes
        let _ = a.alloc(3).unwrap();
        let aligned = a.alloc_aligned(64, 64).unwrap();
        assert_eq!((aligned.as_ptr() as usize) % 64, 0);
        assert_eq!(aligned.len(), 64);
        // Non-power-of-two alignment is rejected without advancing cursor
        let used_before = a.used();
        assert!(a.alloc_aligned(16, 3).is_none());
        assert_eq!(a.used(), used_before);
    }

    #[test]
    fn seal_exposes_exactly_what_was_used() {
        let mut a = Arena::with_capacity(32);
        let _ = a.alloc(5);
        assert_eq!(a.seal().as_slice().len(), 5);
    }

    #[test]
    fn arena_and_seal_zeroize_scrubs_bytes() {
        let mut a = Arena::with_capacity(32);
        let s = a.alloc(8).unwrap();
        s.copy_from_slice(&0xCAFE_BABE_DEAD_BEEFu64.to_le_bytes());
        let mut sealed = a.seal();
        assert_eq!(sealed.as_slice(), &0xCAFE_BABE_DEAD_BEEFu64.to_le_bytes());
        let raw_ptr = sealed.as_slice().as_ptr();
        sealed.zeroize();
        assert!(sealed.is_empty());
        // Verify the underlying buffer bytes were scrubbed to 0 in place before dealloc
        let scrubbed = unsafe { core::slice::from_raw_parts(raw_ptr, 8) };
        assert_eq!(scrubbed, &[0u8; 8]);
    }
}

//! wupz-core — the incremental layout plan and the pack-in-pass scaffold.
//!
//! The thesis (see `docs/DESIGN.md`): a linker already computes, for every byte,
//! its final address and value. A packer then re-derives those facts. wupz keeps
//! them as a first-class **layout plan** and packs *during* layout.
//!
//! This crate is the P1/P4 scaffold: the plan, the incremental rebuild rule, the
//! pre-sized arena, and the plan-image emitter. No ELF reader and no real
//! compressor yet — those are P1/P3 in `docs/ROADMAP.md`.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;
#[cfg(any(test, feature = "std"))]
extern crate std;

pub mod arena;
pub mod beehive;
pub mod fold;
pub mod incremental;
pub mod pack;
pub mod plan;

/// ABI version mirrored in `include/wupz_abi.h` (`WUPZ_ABI_VERSION`).
pub const ABI_VERSION: u32 = 1;

/// C ABI: the ABI version this build speaks. Keeps the Zig/Rust seam honest.
#[unsafe(no_mangle)]
pub extern "C" fn wupz_abi_version() -> u32 {
    ABI_VERSION
}

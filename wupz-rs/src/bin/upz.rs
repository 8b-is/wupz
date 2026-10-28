//! `upz` — Sovereign Incremental Linker, `kompress-super-ultra` Packer & Top-100 Beehive Consolidator.
//!
//! Usage:
//!   cargo run --release --bin upz               # run full beehive Top-100 consolidation + 64B LitanyWire seal
//!   cargo run --release --bin upz -- beehive    # display Top-100 repo consolidation manifest
//!   cargo run --release --bin upz -- gaia       # run .rodata.gaia.b158 5-trit/byte SPQT fold check

use wupz_core::beehive;
use wupz_core::fold::{self, GATE_0X62, QUANT_QT55};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    println!(
        "upz · backend = {} · gate = 0x{:02x} · qt = 0x{:02x}",
        fold::backend(),
        GATE_0X62,
        QUANT_QT55
    );

    if cmd == "all" || cmd == "beehive" {
        let out = beehive::consolidate_top_100();
        println!(
            "beehive Top-100 consolidation: {} repos -> {} bytes plan image (1/1 lanes = 0x{:04x})",
            out.repos.len(),
            out.sealed_arena.len(),
            out.wire.lanes_active
        );
        println!(
            "  sweep_fold      = 0x{:016x} · target_fold = 0x{:016x}",
            out.sweep_fold, out.target_fold
        );
        println!(
            "  qat_reuse_32    = {}/{} · zenUltraCooker = {}/{} · upz = 3761/3314",
            out.qat_reuse_32.p, out.qat_reuse_32.q, out.zen_cooker_ratio.p, out.zen_cooker_ratio.q
        );
        println!(
            "  bfp64_pos       = 0x{:016x} · bfp64_neg   = 0x{:016x}",
            out.wire.bfp64_pos, out.wire.bfp64_neg
        );
        println!(
            "  spqt_root (243) = [{}, {}, {}] · wire_seal16 = 0x{:04x}",
            out.wire.spqt_root[0],
            out.wire.spqt_root[1],
            out.wire.spqt_root[2],
            out.wire.wire_seal16
        );
        println!("  top 12 ranked repos:");
        for repo in out.repos.iter().take(12) {
            println!(
                "    #{:02} {:<24} raw={:>6}B packed={:>6}B ratio={}/{} depth={} fold=0x{:016x}",
                repo.rank,
                repo.name,
                repo.raw_bytes,
                repo.packed_bytes,
                repo.density.p,
                repo.density.q,
                repo.dep_depth,
                repo.fold_hash
            );
        }
    }

    if cmd == "all" || cmd == "gaia" {
        let mut trits = vec![0i8; 6554];
        for (i, t) in trits.iter_mut().enumerate() {
            *t = ((i % 3) as i8) - 1;
        }
        let (gaia_fold, packed_len, ratio) = fold::xor_fold_gaia_b158(&trits);
        println!(
            ".rodata.gaia.b158 · trits={} -> packed={}B (base-243 5 trits/B, ratio={}/{}) · fold=0x{:016x}",
            trits.len(),
            packed_len,
            ratio.p,
            ratio.q,
            gaia_fold
        );
    }
}

//! Build script — link the Zig kernels into the Rust core when zig is available.
//!
//! The seam is not a diagram: this builds `../wupz-zig/src/kernels.zig`, links
//! the resulting static library, and sets `cfg(have_zig)` so `src/fold.rs` can
//! dispatch to it. Without zig the crate still builds and tests — the fold falls
//! back to the scalar reference — so a contributor without the toolchain is not
//! blocked. `WUPZ_NO_ZIG=1` forces the fallback on purpose.

use std::path::PathBuf;
use std::process::Command;

fn cargo_target_to_zig_target(target: &str) -> Option<&'static str> {
    match target {
        "aarch64-apple-darwin" => Some("aarch64-macos"),
        "x86_64-apple-darwin" => Some("x86_64-macos"),
        "aarch64-unknown-linux-gnu" => Some("aarch64-linux-gnu"),
        "aarch64-unknown-linux-musl" => Some("aarch64-linux-musl"),
        "x86_64-unknown-linux-gnu" => Some("x86_64-linux-gnu"),
        "x86_64-unknown-linux-musl" => Some("x86_64-linux-musl"),
        "aarch64-linux-android" => Some("aarch64-linux-android"),
        _ => None,
    }
}

fn main() {
    println!("cargo:rerun-if-changed=../wupz-zig/src/kernels.zig");
    println!("cargo:rerun-if-env-changed=WUPZ_NO_ZIG");
    println!("cargo:rerun-if-env-changed=WUPZ_CROSS_ZIG");
    println!("cargo:rustc-check-cfg=cfg(have_zig)");

    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let manifest = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"),
    );
    let src = manifest.join("../wupz-zig/src/kernels.zig");
    let lib = out.join("libwupz_zig.a");

    if std::env::var_os("WUPZ_NO_ZIG").is_some() {
        return; // explicit opt-out: no warning, no cfg
    }

    // By default, keep cross builds on the scalar path so host toolchains are
    // never invoked for a mismatched target unless `WUPZ_CROSS_ZIG=1` opts in
    // to target-aware `zig build-obj -target <zig_target>`.
    let host = std::env::var("HOST").expect("HOST is set by cargo");
    let target = std::env::var("TARGET").expect("TARGET is set by cargo");
    let cross_zig_target = if host != target {
        if std::env::var_os("WUPZ_CROSS_ZIG").is_none() {
            println!(
                "cargo:warning=wupz: cross-compiling {host} -> {target} — using the Rust scalar fold"
            );
            return;
        }
        let Some(zig_target) = cargo_target_to_zig_target(&target) else {
            println!(
                "cargo:warning=wupz: unsupported Zig cross target {target} — using the Rust scalar fold"
            );
            return;
        };
        Some(zig_target)
    } else {
        None
    };

    let zig_present = Command::new("zig")
        .arg("version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !zig_present || !src.exists() {
        println!(
            "cargo:warning=wupz: zig not available — using the Rust scalar fold (install zig for the SIMD path)"
        );
        return;
    }

    // Compile to an object, then archive it with the platform archiver. Zig's
    // own `build-lib` emits members Apple's ld rejects ("not 8-byte aligned"),
    // so on macOS host targets we route through `libtool`, which writes a
    // well-formed archive. (The kernels are one object; there is nothing to merge.)
    let obj = out.join("wupz_kernels.o");
    let mut cmd = Command::new("zig");
    cmd.arg("build-obj").arg(&src).args(["-O", "ReleaseFast"]);
    if let Some(zt) = cross_zig_target {
        cmd.args(["-target", zt]);
    }
    let compiled = cmd
        .arg(format!("-femit-bin={}", obj.display()))
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !compiled {
        println!("cargo:warning=wupz: zig build-obj failed — using the Rust scalar fold");
        return;
    }

    let archived = if cfg!(target_os = "macos") {
        Command::new("libtool")
            .args(["-static", "-o"])
            .arg(&lib)
            .arg(&obj)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    } else {
        Command::new("ar")
            .arg("crs")
            .arg(&lib)
            .arg(&obj)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };

    if !archived {
        println!(
            "cargo:warning=wupz: could not archive the Zig object — using the Rust scalar fold"
        );
        return;
    }

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=wupz_zig");
    println!("cargo:rustc-cfg=have_zig");
}

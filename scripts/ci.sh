#!/usr/bin/env bash
# wupz — build + test both toolchains, then prove the C ABI end to end.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s ==\n' "$1"; }

step "rust core (plan · incremental · arena · pack-image)"
WUPZ_EXPECT_BACKEND=zig-simd cargo test --quiet

step "benchmark comparison contract (synthetic command outputs)"
bash tests/bench_smoke.sh

step "zig kernels (differential: vectorized vs scalar, seeded)"
zig test wupz-zig/src/kernels.zig

step "static libs + C ABI parity test (header == rust == zig)"
mkdir -p build
zig build-lib wupz-zig/src/kernels.zig -O ReleaseFast -femit-bin=build/libwupz_zig.a
cargo build --release --quiet
zig cc tests/abi_smoke.c -Iinclude build/libwupz_zig.a target/release/libwupz_core.a -o build/abi_smoke
./build/abi_smoke

printf '\nok — both sides built, all tests green, ABI linked.\n'

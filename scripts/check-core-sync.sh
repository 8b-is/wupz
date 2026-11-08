#!/usr/bin/env bash
# Drift guard: assert the wupz and upz crates stay byte-identical.
#
# wupz is the canonical superset; upz is a strict subset. When both repos are
# checked out as siblings, this fails loudly if any shared file diverges — the
# exact regression that bit us twice (core drift, then fold.rs drift).
#
# Usage:
#   ./scripts/check-core-sync.sh            # look for ../upz
#   ./scripts/check-core-sync.sh /path/to/upz
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
sibling="${1:-$here/../upz}"

if [ ! -d "$sibling/wupz-rs/src" ]; then
  echo "skip: no upz checkout at $sibling" >&2
  exit 0
fi

status=0
check() {
  local rel="$1"
  if ! diff -q "$here/$rel" "$sibling/$rel" >/dev/null 2>&1; then
    echo "DRIFT: $rel differs between wupz and upz" >&2
    diff -u "$sibling/$rel" "$here/$rel" | head -40 >&2 || true
    status=1
  fi
}

for f in wupz-rs/src/arena.rs wupz-rs/src/plan.rs wupz-rs/src/pack.rs \
         wupz-rs/src/incremental.rs wupz-rs/src/beehive.rs wupz-rs/src/lib.rs \
         wupz-rs/src/fold.rs wupz-rs/tests/chaos.rs \
         wupz-zig/src/kernels.zig wupz-rs/Cargo.toml; do
  check "$f"
done

if [ "$status" -eq 0 ]; then
  echo "ok: wupz and upz crates are in sync"
fi
exit "$status"

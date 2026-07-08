#!/usr/bin/env bash
# Re-materialize the vendored pi agent from upstream at the pinned commit, and regenerate the
# integrity ledger. Idempotent: on a correct vendoring, a second run leaves `git status` clean.
#
#   wmh/harness/vendor/vendor_pi.sh            # vendor + write manifest
#   wmh/harness/vendor/vendor_pi.sh --check    # verify only: fail if the committed tree drifts from upstream
#
# This is the single source of truth for HOW the vendoring is produced. If you bump the pin, edit
# PIN/TAG here, run it, and commit the result (including the regenerated manifest and VENDOR.md).
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
UPSTREAM="https://github.com/earendil-works/pi"
TAG="v0.80.3"
PIN="a23abe4a695df8b69b613f73e9fdda2a8af894d4"
SUBTREE="packages/agent"
DEST="$REPO/wmh/harness/vendor/pi-agent"
MANIFEST="$REPO/wmh/harness/vendor/manifest.sha256"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "vendor_pi: cloning $UPSTREAM @ $PIN ($TAG)…"
git clone --no-checkout --quiet "$UPSTREAM" "$TMP/pi"
git -C "$TMP/pi" checkout --quiet "$PIN"
HEAD="$(git -C "$TMP/pi" rev-parse HEAD)"
[ "$HEAD" = "$PIN" ] || { echo "FAIL: upstream HEAD $HEAD != pin $PIN"; exit 1; }

# Byte-exact extract of the subtree at the pinned commit (git archive uses blob bytes, no
# working-tree normalization).
STAGE="$TMP/stage"
mkdir -p "$STAGE"
git -C "$TMP/pi" archive "$PIN" "$SUBTREE" | tar -x -C "$STAGE" --strip-components=2 -f -

if [ "$CHECK_ONLY" = 1 ]; then
  diff -r "$STAGE" "$DEST" \
    --exclude=VENDOR.md --exclude=LICENSE \
    && echo "OK: committed tree is byte-identical to upstream@$PIN" \
    || { echo "FAIL: committed tree drifts from upstream@$PIN"; exit 1; }
  exit 0
fi

# Materialize: replace the tree with upstream bytes, then restore the two repo-authored artifacts
# (VENDOR.md is ours; LICENSE is copied from upstream's ROOT, which is outside the subtree).
mkdir -p "$DEST"
find "$DEST" -mindepth 1 -not -name VENDOR.md -delete
cp -R "$STAGE"/. "$DEST"/
cp "$TMP/pi/LICENSE" "$DEST/LICENSE"

# Regenerate the ledger over every vendored file (sorted, repo-relative to the vendor root).
( cd "$REPO/wmh/harness/vendor" && find pi-agent -type f | LC_ALL=C sort | xargs sha256 ) > "$MANIFEST"

echo "vendor_pi: wrote $(find "$DEST" -type f | wc -l | tr -d ' ') files + manifest.sha256"
echo "vendor_pi: done. Review 'git status' and commit."

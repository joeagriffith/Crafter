#!/usr/bin/env bash
# Reproduce the r2dreamer DreamerV3 baseline environment from scratch.
#
# Idempotent: safe to re-run. Clones the vendored impl, pins it to the exact
# commit this baseline was built against, creates an ISOLATED venv (the main
# project .venv with SB3/torch-cu130 is never touched), and installs the
# framework + the `crafter` extra into that venv.
#
# Usage:  bash experiments/20260625-dreamerv3/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

VENDOR_DIR="third_party/r2dreamer"
REPO_URL="https://github.com/NM512/r2dreamer"
# Exact HEAD this baseline was pinned to (record from `git rev-parse HEAD`).
PINNED_SHA="546e4fab8146ea4b14e1d7726bbc1a8a1d50322f"
# r2dreamer pyproject pins `requires-python = ">=3.11,<3.12"`, so 3.11 is REQUIRED.
PYVER="3.11"

# 1. Vendor + pin -------------------------------------------------------------
if [ ! -d "$VENDOR_DIR/.git" ]; then
  echo "[setup] cloning $REPO_URL -> $VENDOR_DIR"
  git clone "$REPO_URL" "$VENDOR_DIR"
fi
echo "[setup] pinning $VENDOR_DIR to $PINNED_SHA"
git -C "$VENDOR_DIR" fetch --quiet origin "$PINNED_SHA" 2>/dev/null || true
git -C "$VENDOR_DIR" checkout --quiet "$PINNED_SHA"
echo "[setup] HEAD now: $(git -C "$VENDOR_DIR" rev-parse HEAD)"

# 2. Isolated venv + install --------------------------------------------------
if [ ! -x "$VENDOR_DIR/.venv/bin/python" ]; then
  echo "[setup] creating isolated venv ($VENDOR_DIR/.venv, python $PYVER)"
  uv venv "$VENDOR_DIR/.venv" --python "$PYVER"
fi
echo "[setup] installing r2dreamer[crafter] into the isolated venv"
uv pip install --python "$VENDOR_DIR/.venv" -e "${VENDOR_DIR}[crafter]"

echo "[setup] done. python: $("$VENDOR_DIR/.venv/bin/python" --version)"
echo "[setup] torch:  $("$VENDOR_DIR/.venv/bin/python" -c 'import torch; print(torch.__version__)')"

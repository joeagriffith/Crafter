#!/usr/bin/env bash
# =============================================================================
#  DreamerV3 strong baseline on Crafter  (r2dreamer PyTorch reproduction)
#
#  >>> THIS IS THE REAL ~1M-STEP RUN. IT IS *NOT* AUTO-EXECUTED. <<<
#  Expected wall-clock ~6-12h and ~6-10GB VRAM on a single RTX 4090.
#  Launch it deliberately:   bash experiments/20260625-dreamerv3/run_dreamerv3.sh
#  (optionally pin GPU / seed / sweep id via env vars below).
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/20260625-dreamerv3"
VENDOR_DIR="$REPO_ROOT/third_party/r2dreamer"
PY="$VENDOR_DIR/.venv/bin/python"

# ---- knobs ------------------------------------------------------------------
SEED=${SEED:-0}
SWEEP_ID=${SWEEP_ID:-"dv3$(date +%m%d)"}
GPU_ID=${GPU_ID:-0}
# DreamerV3 reproduction selected via model.rep_loss=dreamer (r2dreamer's own
# default is rep_loss=r2dreamer). size200M is the Crafter recipe from
# third_party/r2dreamer/runs/crafter.sh.
MODEL=${MODEL:-size200M}
# Replay-buffer storage device. Default cuda:0 (fast). Set BUFFER_DEVICE=cpu to
# keep the replay buffer in system RAM -- this frees several GB of VRAM so the
# larger models (size200M/size400M) can fit on a 24GB card, at the cost of
# slower CPU<->GPU transfers each train step.
BUFFER_DEVICE=${BUFFER_DEVICE:-cuda:0}
# torch.compile (Inductor/Triton) needs a system C compiler (gcc/cc) to JIT
# kernels. This machine has none installed, so compile defaults to False (eager
# mode -- correct, just slower). Install one (`sudo apt install build-essential`)
# and set COMPILE=True for a meaningful per-step speedup.
COMPILE=${COMPILE:-False}

# Include the model size in the out dir so different sizes do not collide.
OUT_DIR="$EXP_DIR/trials/${MODEL}/out/s${SEED}"
# Start each run from a clean out dir so a re-run's metrics.jsonl / logs / tb
# can't mix with a prior attempt's (keeps report_results + monitoring unambiguous).
rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR"

if [ ! -x "$PY" ]; then
  echo "isolated venv missing; run: bash $EXP_DIR/setup.sh" >&2
  exit 1
fi

echo "[run] DreamerV3 baseline  sweep_id=$SWEEP_ID seed=$SEED model=$MODEL buffer=$BUFFER_DEVICE"
echo "[run] logdir -> $OUT_DIR"

# r2dreamer uses relative imports + Hydra config_path="configs", so it must be
# launched from the repo dir. Hydra (version_base=None) does NOT chdir, so the
# absolute logdir below is honoured while imports keep working.
cd "$VENDOR_DIR"
CUDA_VISIBLE_DEVICES=$GPU_ID "$PY" -u train.py \
    env=crafter \
    model="$MODEL" \
    model.rep_loss=dreamer \
    model.compile=${COMPILE} \
    device=cuda:0 \
    buffer.storage_device="$BUFFER_DEVICE" \
    seed="$SEED" \
    logdir="$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/train.log"

# After training completes, fold results into our schema (metrics.json + ledger
# + W&B). report_results.py reads $OUT_DIR/metrics.jsonl (r2dreamer's eval log).
echo "[run] training finished; writing metrics + ledger + W&B"
"$REPO_ROOT/.venv/bin/python" "$EXP_DIR/report_results.py" \
    --out-dir "$OUT_DIR" --seed "$SEED" --sweep-id "$SWEEP_ID" \
    --model-size "$MODEL" \
    --pinned-sha "546e4fab8146ea4b14e1d7726bbc1a8a1d50322f" --wandb

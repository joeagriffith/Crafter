#!/usr/bin/env bash
# =============================================================================
#  DreamerV3 size-ladder sweep orchestrator  (r2dreamer on Crafter)
#
#  Runs a SEQUENTIAL sweep over model sizes ASCENDING (12M -> 400M) x seeds,
#  ~1M Crafter steps each. Each run is the real single-run launcher
#  (run_dreamerv3.sh). The ladder STOPS CLIMBING the moment a size proves too
#  big for VRAM: if a size's SEED 0 hits a CUDA OOM, that size and every larger
#  size are abandoned and the sweep exits.
#
#  >>> THIS LAUNCHES REAL ~1M-STEP RUNS (hours each). NOT AUTO-EXECUTED. <<<
#  Dry-run the plan first:   DRYRUN=1 bash run_sweep_dreamerv3.sh
#  Launch for real (detached):
#      nohup setsid bash experiments/20260625-dreamerv3/run_sweep_dreamerv3.sh &
#
#  Env knobs:
#    SIZES          space-separated, ascending (default 12M..400M)
#    SEEDS          space-separated (default "0 1 2")
#    SWEEP_ID       default dv3MMDD
#    BUFFER_DEVICE  cuda:0 (default, fast) or cpu (frees VRAM for big models)
#    DRYRUN=1       print the full planned matrix and exit without running
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_DIR="$REPO_ROOT/experiments/20260625-dreamerv3"
VENDOR_DIR="$REPO_ROOT/third_party/r2dreamer"
PY="$VENDOR_DIR/.venv/bin/python"

# ---- knobs ------------------------------------------------------------------
SIZES=${SIZES:-"size12M size25M size50M size100M size200M size400M"}
SEEDS=${SEEDS:-"0 1 2"}
SWEEP_ID=${SWEEP_ID:-"dv3$(date +%m%d)"}
BUFFER_DEVICE=${BUFFER_DEVICE:-cuda:0}
GPU_ID=${GPU_ID:-0}
STEPS="1.01e6"   # Crafter default budget (configs/env/crafter.yaml steps)

SUMMARY="$EXP_DIR/sweep_summary.txt"

# OOM markers we scan for in a run's train.log.
OOM_RE="out of memory|CUDA error: out of memory|torch.OutOfMemoryError|CUDA out of memory"

out_dir_for() { echo "$EXP_DIR/trials/$1/out/s$2"; }

train_cmd_for() {
  # $1=size $2=seed -> the exact train.py command run_dreamerv3.sh will execute.
  local size="$1" seed="$2" out
  out="$(out_dir_for "$size" "$seed")"
  echo "CUDA_VISIBLE_DEVICES=$GPU_ID $PY train.py env=crafter model=$size" \
       "model.rep_loss=dreamer model.compile=True device=cuda:0" \
       "buffer.storage_device=$BUFFER_DEVICE seed=$seed logdir=$out"
}

# ---- DRYRUN: print the planned matrix and exit -------------------------------
if [ "${DRYRUN:-0}" = "1" ]; then
  echo "================ DREAMERV3 SIZE-LADDER SWEEP (DRYRUN) ================"
  echo "sweep_id      : $SWEEP_ID"
  echo "sizes (asc)   : $SIZES"
  echo "seeds         : $SEEDS"
  echo "steps/run     : $STEPS"
  echo "buffer device : $BUFFER_DEVICE   (cpu frees VRAM for big models)"
  echo "gpu id        : $GPU_ID"
  n=0
  for size in $SIZES; do
    for seed in $SEEDS; do
      n=$((n + 1))
      printf -- "--- run %02d : model=%-9s seed=%s\n" "$n" "$size" "$seed"
      echo "    OUT_DIR : $(out_dir_for "$size" "$seed")"
      echo "    (cd $VENDOR_DIR && $(train_cmd_for "$size" "$seed"))"
    done
  done
  echo "---------------------------------------------------------------------"
  echo "TOTAL planned runs: $n (ladder may stop early on seed-0 OOM)"
  echo "No training launched (DRYRUN=1)."
  exit 0
fi

# ---- real sweep -------------------------------------------------------------
if [ ! -x "$PY" ]; then
  echo "isolated venv missing; run: bash $EXP_DIR/setup.sh" >&2
  exit 1
fi

{
  echo "===================================================================="
  echo "[HEADER] DreamerV3 size-ladder sweep"
  echo "[HEADER] sweep_id=$SWEEP_ID"
  echo "[HEADER] sizes=$SIZES"
  echo "[HEADER] seeds=$SEEDS"
  echo "[HEADER] steps=$STEPS"
  echo "[HEADER] buffer_device=$BUFFER_DEVICE gpu_id=$GPU_ID"
  echo "[HEADER] start=$(date -Is)"
  echo "===================================================================="
} | tee -a "$SUMMARY"

for size in $SIZES; do
  for seed in $SEEDS; do
    out="$(out_dir_for "$size" "$seed")"
    log="$out/train.log"
    t0=$(date +%s)
    echo "[START] $(date -Is) model=$size seed=$seed out=$out" | tee -a "$SUMMARY"

    MODEL="$size" SEED="$seed" SWEEP_ID="$SWEEP_ID" \
      BUFFER_DEVICE="$BUFFER_DEVICE" GPU_ID="$GPU_ID" \
      bash "$EXP_DIR/run_dreamerv3.sh"
    rc=$?

    t1=$(date +%s); dur=$((t1 - t0))

    # Detect CUDA OOM from the train log (and as a fallback the exit code path).
    oom=0
    if [ -f "$log" ] && grep -qiE "$OOM_RE" "$log"; then
      oom=1
    fi

    if [ "$rc" -eq 0 ]; then
      echo "[DONE rc=0] $(date -Is) model=$size seed=$seed dur=${dur}s" | tee -a "$SUMMARY"
    elif [ "$oom" -eq 1 ]; then
      echo "[OOM] $(date -Is) model=$size seed=$seed rc=$rc dur=${dur}s" | tee -a "$SUMMARY"
      if [ "$seed" = "0" ]; then
        echo "[LADDER-STOP at $size: insufficient VRAM] seed 0 OOMed; skipping remaining seeds and all larger sizes." | tee -a "$SUMMARY"
        echo "[HEADER] end=$(date -Is) (ladder stopped at $size)" | tee -a "$SUMMARY"
        exit 0
      else
        echo "[OOM-NONSEED0] model=$size seed=$seed OOMed but seed!=0; continuing seeds for this size." | tee -a "$SUMMARY"
      fi
    else
      # Non-OOM crash: log and keep going (do not stop the ladder).
      echo "[FAIL rc=$rc] $(date -Is) model=$size seed=$seed dur=${dur}s (non-OOM crash; continuing)" | tee -a "$SUMMARY"
    fi
  done
done

echo "[HEADER] end=$(date -Is) (sweep complete)" | tee -a "$SUMMARY"

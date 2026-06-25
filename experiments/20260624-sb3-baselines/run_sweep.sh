#!/usr/bin/env bash
# Unattended Crafter baseline sweep: PPO / A2C / DQN across seeds.
# Round-robin by seed so all three methods appear early; deadline-bounded so it
# stops launching new runs before the time budget is exhausted. Each finished
# run is logged to W&B, writes its trial out dir (metrics.json + checkpoint +
# tensorboard), appends to the global experiments/ledger.jsonl, and is also
# summarised in summary.txt. Failures are isolated (one crashed run never blocks
# the rest).
set -u

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXP_DIR/../.." && pwd)"
cd "$REPO_ROOT" || exit 1
export PATH="$HOME/.local/bin:$PATH"

STEPS=${STEPS:-1000000}
SEEDS=${SEEDS:-"0 1 2 3"}
ALGOS=${ALGOS:-"ppo a2c dqn"}
# Shared identity for every run in this sweep (override to resume/extend).
SWEEP_ID=${SWEEP_ID:-"bl$(date +%m%d)"}
# Stop STARTING new runs after this many seconds (leaves buffer for the last
# in-flight run + its eval to finish within the ~10h window).
BUDGET=${BUDGET:-32400}   # 9 hours

START=$(date +%s)
DEADLINE=$(( START + BUDGET ))
SUMMARY="$EXP_DIR/summary.txt"
{
  echo "=============================================="
  echo "Crafter baseline sweep"
  echo "experiment: 20260624-sb3-baselines"
  echo "sweep_id  : $SWEEP_ID"
  echo "started   : $(date)"
  echo "steps/run : $STEPS"
  echo "algos     : $ALGOS"
  echo "seeds     : $SEEDS"
  echo "budget    : ${BUDGET}s (stop launching new runs after this)"
  echo "=============================================="
} > "$SUMMARY"

run_job () {
  local algo=$1 seed=$2 tag="$1-s$2"
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[SKIP ] $tag (past launch deadline)" | tee -a "$SUMMARY"
    return
  fi
  local out_dir="$EXP_DIR/trials/${algo}/out/s${seed}"
  mkdir -p "$out_dir"
  echo "[START] $tag @ $(date '+%H:%M:%S')" | tee -a "$SUMMARY"
  local t0=$(date +%s)
  uv run python "$EXP_DIR/train.py" --algo "$algo" --seed "$seed" \
      --steps "$STEPS" --out-dir "$out_dir" --name "$tag" \
      --sweep-id "$SWEEP_ID" --exp-type baseline > "$out_dir/train.log" 2>&1
  local rc=$? mins=$(( ($(date +%s) - t0) / 60 ))
  local res
  res=$(grep -E 'eval mean reward|distinct achievements' "$out_dir/train.log" | tr '\n' ' ')
  echo "[DONE ] $tag rc=$rc (${mins}m) ${res:-<no eval — check $out_dir/train.log>}" | tee -a "$SUMMARY"
}

for seed in $SEEDS; do
  for algo in $ALGOS; do
    run_job "$algo" "$seed"
  done
done

echo "==============================================" | tee -a "$SUMMARY"
echo "sweep finished : $(date)" | tee -a "$SUMMARY"

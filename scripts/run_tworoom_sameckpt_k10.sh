#!/usr/bin/env bash
# Same-checkpoint k=10 control for TwoRoom.
# Purpose: hold the trained EB head checkpoint constant while comparing fixed-k vs event-bottleneck
# modes on untouched seeds 400-409 at offset 75 and budget 70.
# Required: CKPT and a k=10, cem_horizon=25 heads checkpoint.
# Outputs: <OUT>/sameckpt_<mode>_k10_b70.txt and a paired comparison report.
# Usage: DRY=1 bash scripts/run_tworoom_sameckpt_k10.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CKPT=${CKPT:-tworoom_lewm/weights.pt}
OUT=${OUT:-$HOME/.stable_worldmodel/tworoom_lewm}
HEADS=${HEADS:-$OUT/eb_heads_unconfounded_k10.pt}
SEEDS=${SEEDS:-"400 401 402 403 404 405 406 407 408 409"}
OFFSET=${OFFSET:-75}
BUDGET=${BUDGET:-70}
DRY=${DRY:-0}

run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }

[ "$DRY" = "1" ] || [ -f "$HEADS" ] || run python train_eb_heads.py \
  --policy "$CKPT" --dataset tworoom --num_train_episodes 300 \
  --num_val_episodes 50 --cem_horizon 25 --fixed_k 10 \
  --max_horizon 75 --steps 2500 --seed 0 --out "$HEADS"

for MODE in ebjewm_fixedk ebjewm; do
  for S in $SEEDS; do
    run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
      eval.goal_offset_steps="$OFFSET" eval.eval_budget="$BUDGET" \
      +solver_mode="$MODE" +eb_heads_path="$HEADS" \
      output.filename="sameckpt_${MODE}_k10_b${BUDGET}.txt"
  done
done

run python benchmark-protocol/compare_runs.py \
  "$OUT/sameckpt_ebjewm_fixedk_k10_b${BUDGET}.txt" \
  "$OUT/sameckpt_ebjewm_k10_b${BUDGET}.txt"

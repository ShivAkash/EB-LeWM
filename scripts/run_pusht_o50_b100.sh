#!/usr/bin/env bash
# Frozen Push-T protocol (offset=50, budget=100, fresh seeds 100-109, 500 held-out eps/arm).
# Strict split: Train (ep>=2000), Val (1000..1999), Test (ep<1000).
# Purpose: frozen Push-T cross-benchmark generalization.
# Outputs: <OUT>/pusht_final_<arm>_o50_b100.txt.
# Usage: DRY=1 bash scripts/run_pusht_o50_b100.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CKPT=${CKPT:-pusht_lewm/weights.pt}
DATASET=${DATASET:-pusht_expert_train}
OUT=${OUT:-$HOME/.stable_worldmodel/pusht_lewm}
HEADS=${HEADS:-$OUT/eb_heads_pusht_final.pt}
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109"}
OFFSET=${OFFSET:-50}
BUDGET=${BUDGET:-100}
DRY=${DRY:-0}

run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }

echo "=================================================================="
echo "STEP 1: Train & Calibrate Push-T EBHeads (Train>=2000, Val 1000-1999)"
echo "=================================================================="
[ -f "$HEADS" ] || run python train_eb_heads.py \
  --policy "$CKPT" --dataset "$DATASET" \
  --num_train_episodes 300 --num_val_episodes 50 \
  --cem_horizon 25 --fixed_k 25 --max_horizon 75 \
  --steps 2500 --seed 0 --out "$HEADS"

echo "=================================================================="
echo "STEP 2: Frozen Eval on Held-Out Test Split (ep<1000, Seeds 100-109)"
echo "=================================================================="
for MODE in vanilla tempdist_min ebjewm_noflush ebjewm ebjewm_fixedk; do
  for S in $SEEDS; do
    run python eval.py --config-name=pusht.yaml policy="$CKPT" \
      eval.dataset_name="$DATASET" seed="$S" \
      eval.goal_offset_steps="$OFFSET" eval.eval_budget="$BUDGET" \
      +solver_mode="$MODE" +eb_heads_path="$HEADS" \
      output.filename="pusht_final_${MODE}_o${OFFSET}_b${BUDGET}.txt"
  done
done

echo "=================================================================="
echo "STEP 3: Paired Comparisons & Overhead Summary"
echo "=================================================================="
for CTRL in vanilla tempdist_min ebjewm_noflush ebjewm_fixedk; do
  echo; echo "##### Push-T Control=$CTRL (base) vs ebjewm (ours) [o=$OFFSET, b=$BUDGET]"
  run python benchmark-protocol/compare_runs.py \
    "$OUT/pusht_final_${CTRL}_o${OFFSET}_b${BUDGET}.txt" \
    "$OUT/pusht_final_ebjewm_o${OFFSET}_b${BUDGET}.txt"
done

echo; echo "##### Push-T Control=tempdist_min (base) vs ebjewm_noflush (ours)"
run python benchmark-protocol/compare_runs.py \
  "$OUT/pusht_final_tempdist_min_o${OFFSET}_b${BUDGET}.txt" \
  "$OUT/pusht_final_ebjewm_noflush_o${OFFSET}_b${BUDGET}.txt"

run python analysis/summarize_overhead.py "$OUT"/pusht_final_*_o${OFFSET}_b${BUDGET}.txt
#!/usr/bin/env bash
# Pre-declared follow-up protocol implementing Reviewer Items 1-5 in order:
#   Item 1: Remaining ablations (ebjewm_noterm, ebjewm_noprune) on seeds 100-109 (b=90) to complete the frozen table.
#   Item 2: Head-seed variance (head seeds 1 and 2 evaluated across 5 fresh eval seeds 200-204).
#   Item 3: Tighter budgets (budgets 60 and 70 on fresh eval seeds 200-209, offset 75) to get off the 98% ceiling.
#   Item 4: Events vs Fixed-k sensitivity (k=10 and k=40 heads evaluated on fresh eval seeds 200-209 at budget 70).
#   Item 5: Overhead report (plans, sub-goal plans, flushes, and wall-clock time per episode).
# Publication ablation/budget protocol.
# Purpose: head-seed variance, b=70/60 scaling, and k=10/40 sensitivity on TwoRoom seeds 200-209.
# Outputs: <OUT>/tight_*.txt, <OUT>/headseed*.txt, <OUT>/ksweep*.txt.
# Usage: DRY=1 bash scripts/run_tworoom_ablations_budgets.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CKPT=${CKPT:-tworoom_lewm/weights.pt}
OUT=${OUT:-$HOME/.stable_worldmodel/tworoom_lewm}
HEADS_S0=${HEADS_S0:-$OUT/eb_heads_final.pt}
HEADS_S1="$OUT/eb_heads_final_s1.pt"
HEADS_S2="$OUT/eb_heads_final_s2.pt"
HEADS_K10="$OUT/eb_heads_final_k10.pt"
HEADS_K40="$OUT/eb_heads_final_k40.pt"
SEEDS_PRIMARY=${SEEDS_PRIMARY:-"100 101 102 103 104 105 106 107 108 109"}
SEEDS_HEADS=${SEEDS_HEADS:-"200 201 202 203 204"}
SEEDS_TIGHT=${SEEDS_TIGHT:-"200 201 202 203 204 205 206 207 208 209"}
DRY=${DRY:-0}

run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }

echo "=================================================================="
echo "STEP 0: Verify vanilla paired comparison & overhead on seeds 100-109"
echo "=================================================================="
run python benchmark-protocol/compare_runs.py "$OUT/final_vanilla_b90.txt" "$OUT/final_ebjewm_b90.txt"
run python analysis/summarize_overhead.py "$OUT"/final_*_b90.txt

echo "=================================================================="
echo "ITEM 1: Remaining Ablations (noterm, noprune) on Seeds 100-109 (b=90)"
echo "=================================================================="
for MODE in ebjewm_noterm ebjewm_noprune; do
  for S in $SEEDS_PRIMARY; do
    run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
      eval.goal_offset_steps=75 eval.eval_budget=90 \
      +solver_mode="$MODE" +eb_heads_path="$HEADS_S0" \
      output.filename="final_${MODE}_b90.txt"
  done
done

for CTRL in ebjewm_noterm ebjewm_noprune; do
  echo; echo "##### Item 1 Ablation: $CTRL (base) vs ebjewm (ours) [Seeds 100-109, b=90]"
  run python benchmark-protocol/compare_runs.py "$OUT/final_${CTRL}_b90.txt" "$OUT/final_ebjewm_b90.txt"
done

echo "=================================================================="
echo "ITEM 2: Head-Seed Variance (Train Head Seeds 1 & 2, Eval Seeds 200-204)"
echo "=================================================================="
[ -f "$HEADS_S1" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --seed 1 --out "$HEADS_S1"
[ -f "$HEADS_S2" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --seed 2 --out "$HEADS_S2"

for H_IDX in 1 2; do
  H_FILE="$OUT/eb_heads_final_s${H_IDX}.pt"
  for MODE in tempdist_min ebjewm ebjewm_fixedk; do
    for S in $SEEDS_HEADS; do
      run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
        eval.goal_offset_steps=75 eval.eval_budget=90 \
        +solver_mode="$MODE" +eb_heads_path="$H_FILE" \
        output.filename="headseed${H_IDX}_${MODE}_b90.txt"
    done
  done
  echo; echo "##### Item 2 (Head Seed $H_IDX): tempdist_min vs ebjewm"
  run python benchmark-protocol/compare_runs.py "$OUT/headseed${H_IDX}_tempdist_min_b90.txt" "$OUT/headseed${H_IDX}_ebjewm_b90.txt"
  echo; echo "##### Item 2 (Head Seed $H_IDX): ebjewm_fixedk vs ebjewm"
  run python benchmark-protocol/compare_runs.py "$OUT/headseed${H_IDX}_ebjewm_fixedk_b90.txt" "$OUT/headseed${H_IDX}_ebjewm_b90.txt"
done

echo "=================================================================="
echo "ITEM 3: Tighter Budgets (b=60 and b=70 on Fresh Seeds 200-209)"
echo "=================================================================="
for B in 70 60; do
  for MODE in tempdist_min ebjewm ebjewm_fixedk ebjewm_noterm; do
    for S in $SEEDS_TIGHT; do
      run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
        eval.goal_offset_steps=75 eval.eval_budget="$B" \
        +solver_mode="$MODE" +eb_heads_path="$HEADS_S0" \
        output.filename="tight_${MODE}_b${B}.txt"
    done
  done
  for CTRL in tempdist_min ebjewm_fixedk ebjewm_noterm; do
    echo; echo "##### Item 3 Tight Budget (b=$B): $CTRL (base) vs ebjewm (ours)"
    run python benchmark-protocol/compare_runs.py "$OUT/tight_${CTRL}_b${B}.txt" "$OUT/tight_ebjewm_b${B}.txt"
  done
done

echo "=================================================================="
echo "ITEM 4: Events vs Fixed-k Sensitivity (k=10 and k=40 at b=70, Seeds 200-209)"
echo "=================================================================="
[ -f "$HEADS_K10" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --fixed_k 10 --seed 0 --out "$HEADS_K10"
[ -f "$HEADS_K40" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --fixed_k 40 --seed 0 --out "$HEADS_K40"

for K in 10 40; do
  H_K="$OUT/eb_heads_final_k${K}.pt"
  for MODE in ebjewm ebjewm_fixedk; do
    for S in $SEEDS_TIGHT; do
      run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
        eval.goal_offset_steps=75 eval.eval_budget=70 \
        +solver_mode="$MODE" +eb_heads_path="$H_K" \
        output.filename="ksweep_k${K}_${MODE}_b70.txt"
    done
  done
  echo; echo "##### Item 4 (k=$K at b=70): ebjewm_fixedk (base) vs ebjewm (ours)"
  run python benchmark-protocol/compare_runs.py "$OUT/ksweep_k${K}_ebjewm_fixedk_b70.txt" "$OUT/ksweep_k${K}_ebjewm_b70.txt"
done

echo "=================================================================="
echo "ITEM 5: Complete Overhead Summary Across All Runs"
echo "=================================================================="
run python analysis/summarize_overhead.py "$OUT"/final_*_b90.txt "$OUT"/tight_*_b70.txt "$OUT"/tight_*_b60.txt "$OUT"/ksweep_*_b70.txt
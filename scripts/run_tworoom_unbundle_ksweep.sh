#!/usr/bin/env bash
# Pre-declared protocol on untouched seeds 300-309 (offset 75, budget 70) to:
#   1. Unbundle the 3 components of Termination:
#        - Runtime Mid-Plan Flush ('ebjewm' vs 'ebjewm_noflush')
#        - Rollout Beta Bonus     ('ebjewm' vs 'ebjewm_nobonus')
#        - Plan-Time Gate Handoff ('ebjewm_noflush' vs 'ebjewm_costonly' vs 'ebjewm_noterm')
#   2. Run an unconfounded k=10 and k=40 fixed-k sweep where tau_far and far_mask are fixed at cem_horizon=25.
# Publication termination-unbundling protocol.
# Purpose: isolate gate, runtime flush, and beta-bonus components; sweep k with cem_horizon fixed at 25.
# Outputs: <OUT>/unbundle_*.txt and paired comparison reports.
# Usage: DRY=1 bash scripts/run_tworoom_unbundle_ksweep.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CKPT=${CKPT:-tworoom_lewm/weights.pt}
OUT=${OUT:-$HOME/.stable_worldmodel/tworoom_lewm}
HEADS=${HEADS:-$OUT/eb_heads_final.pt}
SEEDS=${SEEDS:-"300 301 302 303 304 305 306 307 308 309"}
OFFSET=${OFFSET:-75}
BUDGET=${BUDGET:-70}
DRY=${DRY:-0}

run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }

echo "=================================================================="
echo "PART 1: Unbundling Termination on Fresh Seeds 300-309 (b=70)"
echo "=================================================================="
for MODE in tempdist_min ebjewm_noterm ebjewm_costonly ebjewm_noflush ebjewm_nobonus ebjewm; do
  for S in $SEEDS; do
    run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
      eval.goal_offset_steps="$OFFSET" eval.eval_budget="$BUDGET" \
      +solver_mode="$MODE" +eb_heads_path="$HEADS" \
      output.filename="unbundle_${MODE}_b${BUDGET}.txt"
  done
done

for CTRL in tempdist_min ebjewm_noterm ebjewm_costonly ebjewm_noflush ebjewm_nobonus; do
  echo; echo "##### Unbundle Comparison: $CTRL (base) vs ebjewm (ours) [Seeds 300-309, b=$BUDGET]"
  run python benchmark-protocol/compare_runs.py "$OUT/unbundle_${CTRL}_b${BUDGET}.txt" "$OUT/unbundle_ebjewm_b${BUDGET}.txt"
done

echo; echo "##### Direct Test of Plan-Time Gate Alone: ebjewm_costonly (no gate) vs ebjewm_noflush (with gate)"
run python benchmark-protocol/compare_runs.py "$OUT/unbundle_ebjewm_costonly_b${BUDGET}.txt" "$OUT/unbundle_ebjewm_noflush_b${BUDGET}.txt"

echo; echo "##### Direct Test of Runtime Flush Alone (without beta bonus): ebjewm_noterm vs ebjewm_nobonus"
run python benchmark-protocol/compare_runs.py "$OUT/unbundle_ebjewm_noterm_b${BUDGET}.txt" "$OUT/unbundle_ebjewm_nobonus_b${BUDGET}.txt"

echo "=================================================================="
echo "PART 2: Unconfounded Fixed-k Sweep (k=10, k=40 with cem_horizon=25)"
echo "=================================================================="
HEADS_CLEAN_K10="$OUT/eb_heads_unconfounded_k10.pt"
HEADS_CLEAN_K40="$OUT/eb_heads_unconfounded_k40.pt"

[ -f "$HEADS_CLEAN_K10" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --cem_horizon 25 --fixed_k 10 --seed 0 --out "$HEADS_CLEAN_K10"
[ -f "$HEADS_CLEAN_K40" ] || run python train_eb_heads.py --policy "$CKPT" --num_train_episodes 300 --cem_horizon 25 --fixed_k 40 --seed 0 --out "$HEADS_CLEAN_K40"

for K in 10 40; do
  H_K="$OUT/eb_heads_unconfounded_k${K}.pt"
  for S in $SEEDS; do
    run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
      eval.goal_offset_steps="$OFFSET" eval.eval_budget="$BUDGET" \
      +solver_mode=ebjewm_fixedk +eb_heads_path="$H_K" \
      output.filename="unbundle_fixedk${K}_b${BUDGET}.txt"
  done
  echo; echo "##### Unconfounded k=$K (base) vs ebjewm (ours) [Seeds 300-309, b=$BUDGET]"
  run python benchmark-protocol/compare_runs.py "$OUT/unbundle_fixedk${K}_b${BUDGET}.txt" "$OUT/unbundle_ebjewm_b${BUDGET}.txt"
done

echo "=================================================================="
echo "PART 3: Complete Overhead Table for Unbundled Arms (Seeds 300-309)"
echo "=================================================================="
run python analysis/summarize_overhead.py "$OUT"/unbundle_*_b${BUDGET}.txt
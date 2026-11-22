#!/usr/bin/env bash
# Frozen final protocol for EB-JeWM vs. simple controls on TwoRoom (offset 75). Run ONCE, then read the verdict.
#   - Do NOT edit eb_planner.py / the heads / thresholds after starting: the point is no selection on these seeds.
#   - Seeds 100-109 were never used while developing (dev used 0-2).
#   - Primary:   ebjewm  vs. the best of {tempdist, tempdist_min, tempdist_rh2}   (paired by seed)
#   - Secondary: ebjewm  vs. ebjewm_fixedk   (do EVENTS matter, or any learned sub-goal + replanning?)
# Publication primary protocol.
# Purpose: compare frozen EB-JeWM heads against TwoRoom controls on seeds 100-109.
# Required: CKPT and HEADS paths below. The default HEADS file must contain calibrated EBHeads.
# Outputs: <OUT>/final_<arm>_b<budget>.txt and paired compare_runs.py reports.
# Usage: CKPT=tworoom_lewm/weights.pt HEADS=$HOME/.stable_worldmodel/tworoom_lewm/eb_heads_final.pt bash scripts/run_tworoom_primary_b90.sh
#        DRY=1 bash scripts/run_tworoom_primary_b90.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CKPT=${CKPT:-tworoom_lewm/weights.pt}
OUT=${OUT:-$HOME/.stable_worldmodel/tworoom_lewm}
HEADS=${HEADS:-$OUT/eb_heads_final.pt}     # pin ONE heads file; never rely on the default eb_heads.pt
SEEDS=${SEEDS:-"100 101 102 103 104 105 106 107 108 109"}
OFFSET=${OFFSET:-75}
BUDGETS=${BUDGETS:-"90"}                   # add 70 for a tighter, efficiency-sensitive budget
DRY=${DRY:-0}

# name | solver_mode | extra hydra overrides
ARMS=(
  "vanilla|vanilla|"
  "tempdist|tempdist|"
  "tempdist_min|tempdist_min|"
  "tempdist_rh2|tempdist|plan_config.receding_horizon=2"   # replans every 10 env steps: matches EB's extra replanning
  "ebjewm|ebjewm|"
  "ebjewm_fixedk|ebjewm_fixedk|"
)

run() { if [ "$DRY" = "1" ]; then echo "$*"; else "$@"; fi; }

[ "$DRY" = "1" ] || [ -f "$HEADS" ] || { echo "heads file not found: $HEADS"; exit 1; }
for B in $BUDGETS; do
  for ARM in "${ARMS[@]}"; do
    IFS='|' read -r NAME MODE EXTRA <<< "$ARM"
    for S in $SEEDS; do
      # shellcheck disable=SC2086
      run python eval.py --config-name=tworoom.yaml policy="$CKPT" seed="$S" \
        eval.goal_offset_steps="$OFFSET" eval.eval_budget="$B" \
        +solver_mode="$MODE" +eb_heads_path="$HEADS" $EXTRA \
        output.filename="final_${NAME}_b${B}.txt"
    done
  done
done

for B in $BUDGETS; do
  F="$OUT/final_%s_b${B}.txt"
  for CTRL in tempdist tempdist_min tempdist_rh2 ebjewm_fixedk; do
    echo; echo "##### control=$CTRL (base)  vs  ebjewm (ours)   budget=$B"
    # shellcheck disable=SC2059
    run python benchmark-protocol/compare_runs.py "$(printf "$F" "$CTRL")" "$(printf "$F" ebjewm)"
  done
done
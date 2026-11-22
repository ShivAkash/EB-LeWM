# LeWM: Event-Bottleneck Sub-Goal Planning with Termination Gating

This repository extends the upstream **LeWorldModel (LeWM)** codebase with **Event-Bottleneck sub-goal planning** for frozen world-model backbones. It accompanies the publication:

> Event-Bottleneck Sub-Goal Planning with Termination Gating for Latent World Models (LeWM)

The upstream LeWM implementation and pretrained models are preserved in this repository. We thank the original authors and maintain attribution to the upstream project throughout.

## Overview and key contributions

The method adds four tightly scoped mechanisms while keeping the LeWM backbone frozen:

- a temporal-distance head for latent goal and sub-goal costs;
- a peak-latent-velocity bottleneck head that predicts an event-aligned sub-goal;
- a plan-time termination handoff gate that decides whether rollout cost should be handed to the sub-goal or final goal; and
- an optional runtime plan flush that replans when the latent termination probability changes.

The rollout cost combines terminal temporal distance with minimum-over-horizon distance and a termination bonus. Fixed-\(k\) and no-gate variants isolate whether event-aligned generation or fixed replanning explains performance.

**Epistemic LoRA pruning:** an epistemic LoRA branch was tested on TwoRoom. It was inert: EB-JeWM and EB-JeWM no-prune both reached \(98.8\%\) success, and the paired exact McNemar test gave \(p=1.0\). The branch was removed from the released method.

## Repository map

```text
le-wm/
├── README.md
├── REPRODUCIBILITY.md
├── LICENSE
├── .gitignore
├── requirements-eb.txt
├── eb_planner.py                  # EB heads, criterion, policy wrapper
├── train_eb_heads.py              # frozen-backbone head training/calibration
├── eval.py / train.py / jepa.py   # upstream LeWM evaluation and training
├── module.py / utils.py
├── config/ / assets/
├── benchmark-protocol/
│   └── compare_runs.py            # Wilson, paired bootstrap, exact McNemar
├── scripts/                       # content-named frozen experiment protocols
├── analysis/
│   ├── check_*_events.py          # physical event-alignment checks
│   ├── diagnose_tworoom_failures.py
│   ├── summarize_overhead.py
│   └── generate_figures.py
├── results/
│   ├── all_benchmarks_summary.txt
│   └── environment_manifest.txt
└── logs/                          # local raw logs; gitignored
```

Preserved upstream root files are intentionally kept at the repository root for Hydra and import compatibility.

## Environment and data setup

Create Python 3.10 environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-eb.txt
python -m pip install "stable-worldmodel[train,env]"
```

The release manifest is [results/environment_manifest.txt](results/environment_manifest.txt). The recorded image stack uses EGL/MuJoCo. Set `MUJOCO_GL=egl` for evaluation. CUDA is optional for the code but hardware numbers were collected on a configured GPU environment.

Download each official LeWM checkpoint with the Hugging Face CLI:

```bash
huggingface-cli download quentinll/lewm-tworooms --include "weights.pt" "config.json" --local-dir ~/.stable_worldmodel/tworoom_lewm
huggingface-cli download quentinll/lewm-pusht --include "weights.pt" "config.json" --local-dir ~/.stable_worldmodel/pusht_lewm
huggingface-cli download quentinll/lewm-cube --include "weights.pt" "config.json" --local-dir ~/.stable_worldmodel/cube_lewm
huggingface-cli download quentinll/lewm-reacher --include "weights.pt" "config.json" --local-dir ~/.stable_worldmodel/reacher_lewm
```

Place official HDF5 datasets in `~/.stable_worldmodel/datasets/`:

```text
tworoom.h5
pusht_expert_train.h5
cube_single_expert.h5
reacher.h5
```

Dataset names in Hydra are the filenames without `.h5`.

## Step-by-step replication

1. **Verify physical event alignment**

```bash
python analysis/check_tworoom_events.py --policy tworoom_lewm/weights.pt
python analysis/check_pusht_events.py --policy pusht_lewm/weights.pt
python analysis/check_cube_events.py --policy cube_lewm/weights.pt
```

2. **Train and calibrate EB heads**

```bash
python train_eb_heads.py --policy tworoom_lewm/weights.pt --dataset tworoom \
  --num_train_episodes 300 --num_val_episodes 50 \
  --cem_horizon 25 --fixed_k 25 --max_horizon 75 \
  --steps 2500 --seed 0 --out ~/.stable_worldmodel/tworoom_lewm/eb_heads_final.pt
```

Episodes with `ep >= 2000` are training, `1000..1999` validation, and `ep < 1000` held-out test. `--cem_horizon` controls far-mask and `tau_far`; `--fixed_k` only targets the fixed-\(k\) ablation. This decoupling prevents k-sweeps from changing event-gate geometry.

3. **Run frozen protocols**

```bash
bash scripts/run_tworoom_primary_b90.sh
bash scripts/run_tworoom_ablations_budgets.sh
bash scripts/run_tworoom_unbundle_ksweep.sh
bash scripts/run_tworoom_sameckpt_k10.sh
bash scripts/run_pusht_o50_b100.sh
bash scripts/run_cube_o50_b100.sh
bash scripts/run_reacher_o50_b100.sh
```

Every script supports `DRY=1` to print commands without executing:

```bash
DRY=1 bash scripts/run_tworoom_primary_b90.sh
```

Output variables include `CKPT`, `OUT`, `HEADS`, `SEEDS`, `OFFSET`, `BUDGET`, and `DRY` where applicable.

4. **Paired evaluation and diagnostics**

```bash
python benchmark-protocol/compare_runs.py BASE.txt OURS.txt
python analysis/diagnose_tworoom_failures.py BASE.txt
python analysis/summarize_overhead.py ~/.stable_worldmodel/tworoom_lewm/*.txt
```

`compare_runs.py` reports Wilson 95% confidence intervals, paired episode bootstrap 95% intervals, and exact McNemar tests.

## Complete results summary

All values below are extracted from the retained protocol summaries/logs. See [results/all_benchmarks_summary.txt](results/all_benchmarks_summary.txt) for the compact table.

### TwoRoom primary and budget scaling

| Arm | b=90 | b=70 | b=60 |
|---|---:|---:|---:|
| vanilla | 26.0±6.6 | — | — |
| temporal distance | 93.2±3.1 | 91.6±3.7 | 91.0±4.0 |
| fixed-k EB-JeWM | 98.4±2.0 | 98.6±1.8 | 96.8±2.2 |
| EB-JeWM | 98.8±1.3 | 98.6±1.3 | 97.8±1.7 |
| no termination | 88.8±3.8 | 83.6±4.8 | 78.0±5.1 |

At b=90, EB-JeWM versus fixed-k had a paired difference of `+0.4` points, bootstrap 95% `[-0.8,+1.6]`, and exact McNemar `p=0.754`.

### TwoRoom termination unbundling

| Arm | Success |
|---|---:|
| noterm | 81.8±6.4 |
| costonly | 94.2±2.1 |
| noflush | 96.6±1.8 |
| nobonus | 97.4±1.8 |
| EB-JeWM | 98.2±1.9 |

No numeric Same-Room/Cross-Room breakdown was retained in this checkout. The diagnostic tool classifies held-out episodes using the true environment wall and should be rerun on fresh result files.

### Unconfounded fixed-k sweep

| k | Fixed-k | EB-JeWM |
|---:|---:|---:|
| 10 | 91.2±3.1 | 98.8±1.3 |
| 25 | 98.6±1.8 at b=70 | 98.6±1.3 |
| 40 | 97.8±1.7 | 98.2±1.9 |

The same-checkpoint k=10 control is implemented for Seeds 400-409 in `scripts/run_tworoom_sameckpt_k10.sh`. Historical table entries: EB-JeWM k=10 \(97.4\pm2.5\%\), fixed-k \(91.2\pm4.5\%\); a raw reproducibility log was not present.

### Cross-benchmark generalization

| Benchmark | Vanilla | Temp-dist min | EB-JeWM fixed-k | EB-JeWM |
|---|---:|---:|---:|---:|
| TwoRoom | 26.0±6.6 | 93.2±3.1 | 98.4±2.0 | 98.8±1.3 |
| Push-T | 40.4±6.1 | 43.4±6.2 | 45.0±6.3 | 46.8±8.5 |
| OGBench-Cube | 48.2±6.2 | 41.0±8.1 | 38.8±6.0 | 40.8±7.1 |
| Reacher | 67.4±8.2 | 69.0±6.6 | 66.2±8.6 | 65.0±8.0 |

TwoRoom is positive. Push-T shows a modest paired gain versus vanilla (`+6.4` points, bootstrap 95% `[+0.8,+12.0]`). Cube is negative versus vanilla (`-7.4` points, exact McNemar \(p=1.97e-06\)); EB-JeWM is not distinguishable from temp-dist or fixed-k controls. Reacher is negative/null relative to vanilla/temp-dist controls.

## Citation

```bibtex
@inproceedings{placeholder_lewm_eb2026,
  title={Event-Bottleneck Sub-Goal Planning with Termination Gating for Latent World Models},
  author={Shiv Akash S M},
  year={2026}
}
```

Please also cite the upstream LeWM and stable-worldmodel projects:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

- Upstream LeWM: https://github.com/lucas-maes/le-wm
- stable-worldmodel: https://github.com/galilai-group/stable-worldmodel
- stable-pretraining: https://github.com/galilai-group/stable-pretraining

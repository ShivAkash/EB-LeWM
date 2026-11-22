# Reproducibility Guide

## Environment

- Python: 3.10.22
- NumPy: 2.5.3
- SciPy: 1.18.1
- Recorded package versions are also listed in [environment_manifest.txt](results/environment_manifest.txt).
- Install direct dependencies with `python -m pip install -r requirements-eb.txt`.
- Evaluation sets `MUJOCO_GL=egl`.

## Data splits

For datasets with at least 2500 episode indices:

- held-out test: `ep < 1000`;
- validation/calibration: `1000 <= ep < 2000`;
- head training: `ep >= 2000`.

For smaller proportional datasets, the implementation uses first 20% test, next 10% validation, and remaining 70% training. Evaluation never uses training or calibration episodes.

## Pre-declared seed registry

| Purpose | Environment | Seeds |
|---|---|---|
| Development/debugging | TwoRoom | 0-2 |
| Primary frozen protocol | TwoRoom, o=75, b=90 | 100-109 |
| Tight budgets/head variance | TwoRoom, b=70/60 | 200-209 |
| Extra head-seed evaluations | TwoRoom, Head Seeds 1/2 | 200-204 |
| Termination unbundling | TwoRoom, b=70 | 300-309 |
| Unconfounded fixed-k sweep | TwoRoom, b=70 | 300-309 |
| Same-checkpoint k=10 control | TwoRoom, b=70 | 400-409 |
| Cross-benchmark generalization | Push-T, OGBench-Cube, Reacher, o=50, b=100 | 100-109 |

No frozen evaluation seed overlaps development seeds.

## Exact replication sequence

1. Download checkpoints and datasets as shown in `README.md`.
2. Verify physical event alignment:
   - `python analysis/check_tworoom_events.py`
   - `python analysis/check_pusht_events.py`
   - `python analysis/check_cube_events.py`
3. Train heads with frozen backbone:
   - use only `ep >= 2000` for training;
   - use `1000..1999` only for threshold calibration;
   - never use `ep < 1000` in `train_eb_heads.py`;
   - keep `--cem_horizon` separate from `--fixed_k`.
4. Run the frozen scripts in `scripts/` with no changed threshold or head code.
5. Compare paired files with `benchmark-protocol/compare_runs.py`.
6. Diagnose same-room/cross-room failures with `analysis/diagnose_tworoom_failures.py`.
7. Summarize overhead with `analysis/summarize_overhead.py`.

## Hardware and storage

Raw console logs are retained locally in `logs/` and ignored by Git. Checkpoints, datasets, `.pt`, `.ckpt`, `.h5`, and Hydra outputs are also ignored or remain outside the repository cache.

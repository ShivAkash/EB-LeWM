#!/usr/bin/env python
"""
Physical event alignment check for Push-T (pusht_expert_train.h5).

Tests whether peak latent velocity frames (argmax ||z_{k+1} - z_k||^2) align with:
  1. Block translation / contact (||diff(block_xy)|| > 1.0 px)
  2. Block rotation (|diff(block_angle)| > 0.02 rad)
  3. Any block motion (translation OR rotation)
compared against CHANCE (random frame from the same segment), split by moved-block vs static-block segments.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import stable_worldmodel as swm
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eb_planner import load_lewm_model
from train_eb_heads import encode_episode_list, get_img_transform


def dilate(mask: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return mask.astype(bool)
    return np.convolve(mask.astype(float), np.ones(2 * window + 1), mode="same") > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="pusht_lewm/weights.pt")
    ap.add_argument("--dataset", default="pusht_expert_train")
    ap.add_argument("--ep_lo", type=int, default=1000)
    ap.add_argument("--ep_hi", type=int, default=2000)
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--segs_per_ep", type=int, default=10)
    ap.add_argument("--min_delta", type=int, default=25)
    ap.add_argument("--max_delta", type=int, default=80)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = load_lewm_model(a.policy, device=device)

    cache_dir = Path(swm.data.utils.get_cache_dir())
    ds = swm.data.HDF5Dataset(a.dataset, keys_to_cache=["state"], cache_dir=cache_dir)
    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_col = ds.get_col_data(col)
    state = ds.get_col_data("state")  # (N, 7): [agent_x, agent_y, block_x, block_y, block_angle, ...]

    pool = np.unique(ep_col)
    pool = pool[(pool >= a.ep_lo) & (pool < a.ep_hi)]
    if len(pool) == 0:
        pool = np.unique(ep_col)

    eps = rng.choice(pool, size=min(a.n_eps, len(pool)), replace=False)
    print(f"Encoding {len(eps)} held-out Push-T episodes from [{a.ep_lo}, {a.ep_hi})...")
    lat = encode_episode_list(jepa, ds, ep_col, eps, get_img_transform(224), device)

    def empty_bucket():
        return {"n": 0, "t_hit": 0, "t_ch": 0.0, "r_hit": 0, "r_ch": 0.0, "m_hit": 0, "m_ch": 0.0}

    stats = {"moved": empty_bucket(), "static": empty_bucket()}

    for ep_id, z in zip(eps, lat):
        rows = np.nonzero(ep_col == ep_id)[0]
        s_ep = state[rows]
        L = len(z)
        if L < a.min_delta + 6:
            continue

        block_xy = s_ep[:, 2:4]
        block_ang = s_ep[:, 4]

        b_trans_vel = np.linalg.norm(np.diff(block_xy, axis=0, prepend=block_xy[:1]), axis=-1)
        b_rot_vel = np.abs(np.diff(np.unwrap(block_ang), prepend=block_ang[0]))

        trans_win = dilate(b_trans_vel > 1.0, a.window)
        rot_win = dilate(b_rot_vel > 0.02, a.window)
        move_win = trans_win | rot_win

        for _ in range(a.segs_per_ep):
            t0 = int(rng.integers(0, L - a.min_delta - 1))
            tg = t0 + int(rng.integers(a.min_delta, min(a.max_delta, L - 1 - t0) + 1))
            seg = z[t0 : tg + 1]
            t_star = int((seg[1:] - seg[:-1]).pow(2).sum(-1).argmax()) + 1

            disp = float(np.linalg.norm(block_xy[tg] - block_xy[t0]))
            ang_disp = float(abs(block_ang[tg] - block_ang[t0]))
            kind = "moved" if (disp > 5.0 or ang_disp > 0.1) else "static"
            s = stats[kind]

            t_seg, r_seg, m_seg = trans_win[t0 : tg + 1], rot_win[t0 : tg + 1], move_win[t0 : tg + 1]
            s["n"] += 1
            s["t_hit"] += int(t_seg[t_star])
            s["t_ch"] += float(t_seg.mean())
            s["r_hit"] += int(r_seg[t_star])
            s["r_ch"] += float(r_seg.mean())
            s["m_hit"] += int(m_seg[t_star])
            s["m_ch"] += float(m_seg.mean())

    print(f"\nevent frame = argmax latent velocity; tolerance = +-{a.window} frames")
    for kind, s in stats.items():
        n = s["n"]
        if n == 0:
            continue
        print(
            f"  {kind:6s}-block segments (n={n:4d}):\n"
            f"    block translation  : {100*s['t_hit']/n:5.1f}% (chance {100*s['t_ch']/n:5.1f}%)\n"
            f"    block rotation     : {100*s['r_hit']/n:5.1f}% (chance {100*s['r_ch']/n:5.1f}%)\n"
            f"    any block motion   : {100*s['m_hit']/n:5.1f}% (chance {100*s['m_ch']/n:5.1f}%)"
        )


if __name__ == "__main__":
    main()
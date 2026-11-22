#!/usr/bin/env python
"""
Physical event alignment check for OGBench-Cube (cube_single_expert.h5).

Tests whether peak latent velocity frames (argmax ||z_{k+1} - z_k||^2) align with:
  1. Gripper contact transitions (changes in `proprio_gripper_contact`, i.e. grasp/release)
  2. Gripper aperture transitions (top-15% |diff(proprio_gripper_opening)|)
  3. Block motion / lift events (block velocity ||diff(privileged_block_0_pos)|| > threshold)
compared against CHANCE (random frame from the same segment), split by segments where
the block actually moves vs. static-block segments.
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
    ap.add_argument("--policy", default="cube_lewm/weights.pt")
    ap.add_argument("--dataset", default="cube_single_expert")
    ap.add_argument("--ep_lo", type=int, default=1000)
    ap.add_argument("--ep_hi", type=int, default=2000)
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--segs_per_ep", type=int, default=10)
    ap.add_argument("--min_delta", type=int, default=25)
    ap.add_argument("--max_delta", type=int, default=80)
    ap.add_argument("--window", type=int, default=3, help="+-frames tolerance around physical transition")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    jepa = load_lewm_model(a.policy, device=device)

    cache_dir = Path(swm.data.utils.get_cache_dir())
    keys = ["proprio_gripper_contact", "proprio_gripper_opening", "privileged_block_0_pos"]
    ds = swm.data.HDF5Dataset(a.dataset, keys_to_cache=keys, cache_dir=cache_dir)
    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_col = ds.get_col_data(col)
    contact = ds.get_col_data("proprio_gripper_contact").reshape(-1)
    opening = ds.get_col_data("proprio_gripper_opening").reshape(-1)
    block_pos = ds.get_col_data("privileged_block_0_pos")

    pool = np.unique(ep_col)
    pool = pool[(pool >= a.ep_lo) & (pool < a.ep_hi)]
    if len(pool) == 0:
        pool = np.unique(ep_col)

    eps = rng.choice(pool, size=min(a.n_eps, len(pool)), replace=False)
    print(f"Encoding {len(eps)} held-out Cube episodes from [{a.ep_lo}, {a.ep_hi})...")
    lat = encode_episode_list(jepa, ds, ep_col, eps, get_img_transform(224), device)

    def empty_bucket():
        return {"n": 0, "c_hit": 0, "c_ch": 0.0, "o_hit": 0, "o_ch": 0.0, "b_hit": 0, "b_ch": 0.0}

    stats = {"moved": empty_bucket(), "static": empty_bucket()}

    for ep_id, z in zip(eps, lat):
        rows = np.nonzero(ep_col == ep_id)[0]
        c_ep = contact[rows]
        o_ep = opening[rows]
        b_ep = block_pos[rows]
        L = len(z)
        if L < a.min_delta + 6:
            continue

        # 1. Contact state transitions (grasp/release edges)
        c_edge = np.abs(np.diff(c_ep, prepend=c_ep[0])) > 1e-3
        c_win = dilate(c_edge, a.window)

        # 2. Gripper opening/closing transitions
        o_vel = np.abs(np.diff(o_ep, prepend=o_ep[0]))
        o_edge = o_vel >= max(np.percentile(o_vel, 85), 1e-4)
        o_win = dilate(o_edge, a.window)

        # 3. Block motion / lift frames (block moving > 2mm per step)
        b_vel = np.linalg.norm(np.diff(b_ep, axis=0, prepend=b_ep[:1]), axis=-1)
        b_moving = b_vel > 0.002

        for _ in range(a.segs_per_ep):
            t0 = int(rng.integers(0, L - a.min_delta - 1))
            tg = t0 + int(rng.integers(a.min_delta, min(a.max_delta, L - 1 - t0) + 1))
            seg = z[t0 : tg + 1]
            t_star = int((seg[1:] - seg[:-1]).pow(2).sum(-1).argmax()) + 1

            block_disp = float(np.linalg.norm(b_ep[tg] - b_ep[t0]))
            kind = "moved" if block_disp > 0.02 else "static"
            s = stats[kind]

            c_seg, o_seg, b_seg = c_win[t0 : tg + 1], o_win[t0 : tg + 1], b_moving[t0 : tg + 1]
            s["n"] += 1
            s["c_hit"] += int(c_seg[t_star])
            s["c_ch"] += float(c_seg.mean())
            s["o_hit"] += int(o_seg[t_star])
            s["o_ch"] += float(o_seg.mean())
            s["b_hit"] += int(b_seg[t_star])
            s["b_ch"] += float(b_seg.mean())

    print(f"\nevent frame = argmax latent velocity; tolerance = +-{a.window} frames")
    for kind, s in stats.items():
        n = s["n"]
        if n == 0:
            continue
        print(
            f"  {kind:6s}-block segments (n={n:4d}):\n"
            f"    contact transition : {100*s['c_hit']/n:5.1f}% (chance {100*s['c_ch']/n:5.1f}%)\n"
            f"    gripper open/close : {100*s['o_hit']/n:5.1f}% (chance {100*s['o_ch']/n:5.1f}%)\n"
            f"    block in motion    : {100*s['b_hit']/n:5.1f}% (chance {100*s['b_ch']/n:5.1f}%)"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Do the "event" frames used to train the bottleneck jumper land at the TwoRoom doorway?

train_eb_heads.py defines the event of a segment [t0, tg] as the frame with peak latent velocity
||z_{k+1} - z_k||^2. This script samples segments from held-out episodes, finds that frame, and reports how
often the agent is (a) next to the wall and (b) inside the door box at that frame, against CHANCE (the same
statistic over all frames of the same segments). Cross-room segments are the ones that must pass the door.

    python analysis/check_tworoom_events.py --policy tworoom_lewm/weights.pt            # HF-style checkpoint
    python analysis/check_tworoom_events.py --policy run/weights_epoch_N.pt --loader swm # checkpoint written by train.py

Defaults match the default TwoRoom variation (wall at x=112, door centre y=49, half-extent 14).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="tworoom_lewm/weights.pt")
    ap.add_argument("--loader", choices=["hf", "swm"], default="hf")
    ap.add_argument("--dataset", default="tworoom")
    ap.add_argument("--ep_lo", type=int, default=1000)  # validation split used by train_eb_heads.py
    ap.add_argument("--ep_hi", type=int, default=2000)
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--segs_per_ep", type=int, default=10)
    ap.add_argument("--min_delta", type=int, default=25)
    ap.add_argument("--max_delta", type=int, default=80)
    ap.add_argument("--wall_x", type=float, default=112.0)
    ap.add_argument("--door_y", type=float, default=49.0)
    ap.add_argument("--door_half", type=float, default=14.0)
    ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.loader == "swm":
        jepa = swm.wm.utils.load_pretrained(a.policy).to(device).eval()
        jepa.requires_grad_(False)
        jepa.interpolate_pos_encoding = True
    else:
        jepa = load_lewm_model(a.policy, device=device)

    cache_dir = Path(swm.data.utils.get_cache_dir())
    ds = swm.data.HDF5Dataset(a.dataset, keys_to_cache=["proprio"], cache_dir=cache_dir)
    col = "episode_idx" if "episode_idx" in ds.column_names else "ep_idx"
    ep_col, proprio = ds.get_col_data(col), ds.get_col_data("proprio")
    pool = np.unique(ep_col)
    pool = pool[(pool >= a.ep_lo) & (pool < a.ep_hi)]
    if len(pool) == 0:
        print(f"[warn] no episodes in [{a.ep_lo}, {a.ep_hi}); using all episodes")
        pool = np.unique(ep_col)
    eps = rng.choice(pool, size=min(a.n_eps, len(pool)), replace=False)
    lat = encode_episode_list(jepa, ds, ep_col, eps, get_img_transform(224), device)

    def in_door(p):
        return (np.abs(p[..., 0] - a.wall_x) <= a.margin) & (np.abs(p[..., 1] - a.door_y) <= a.door_half + a.margin)

    def near_wall(p):
        return np.abs(p[..., 0] - a.wall_x) <= a.margin

    stats = {
        "cross": {"n": 0, "wall": 0, "door": 0, "wall_c": 0.0, "door_c": 0.0},
        "same": {"n": 0, "wall": 0, "door": 0, "wall_c": 0.0, "door_c": 0.0},
    }
    for ep_id, z in zip(eps, lat):
        pos = proprio[np.nonzero(ep_col == ep_id)[0]]
        L = len(z)
        if L < a.min_delta + 6:
            continue
        for _ in range(a.segs_per_ep):
            t0 = int(rng.integers(0, L - a.min_delta - 1))
            tg = t0 + int(rng.integers(a.min_delta, min(a.max_delta, L - 1 - t0) + 1))
            seg = z[t0 : tg + 1]
            t_star = int((seg[1:] - seg[:-1]).pow(2).sum(-1).argmax()) + 1  # same rule as train_eb_heads.py
            p_seg = pos[t0 : tg + 1]
            kind = "cross" if (pos[t0, 0] < a.wall_x) != (pos[tg, 0] < a.wall_x) else "same"
            s = stats[kind]
            s["n"] += 1
            s["wall"] += bool(near_wall(p_seg[t_star]))
            s["door"] += bool(in_door(p_seg[t_star]))
            s["wall_c"] += float(near_wall(p_seg).mean())  # chance: a random frame of the same segment
            s["door_c"] += float(in_door(p_seg).mean())

    print(f"\nevent frame = argmax latent velocity; wall x={a.wall_x}, door y={a.door_y}+-{a.door_half}, margin {a.margin}px")
    for kind, s in stats.items():
        if s["n"] == 0:
            continue
        n = s["n"]
        print(
            f"  {kind:5s}-room segments (n={n:4d}): next to wall {100*s['wall']/n:5.1f}% (chance {100*s['wall_c']/n:5.1f}%) | "
            f"inside door box {100*s['door']/n:5.1f}% (chance {100*s['door_c']/n:5.1f}%)"
        )
    print("Reading: events only support the 'doorway' story if cross-room rates clearly beat chance.")


if __name__ == "__main__":
    main()
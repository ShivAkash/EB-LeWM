#!/usr/bin/env python
"""
Splits eval.py results into Same-Room vs. Cross-Room episodes using:
  1. The TRUE wall position queried directly from swm/TwoRoom-v1 (not median proprio).
  2. The exact held-out Test Split (ep_id < 1000) used by eval.py.
"""
import sys
from pathlib import Path
import gymnasium as gym
import numpy as np
import stable_worldmodel as swm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmark-protocol"))
from compare_runs import parse


def get_true_wall_x(proprio_x: np.ndarray) -> float:
    """Queries swm/TwoRoom-v1 for the exact dividing wall coordinate."""
    try:
        env = gym.make("swm/TwoRoom-v1")
        unwrapped = env.unwrapped
        for attr in ("wall_x", "wall_pos", "door_x", "divider_x"):
            if hasattr(unwrapped, attr):
                val = float(getattr(unwrapped, attr))
                env.close()
                return val
        env.close()
    except Exception:
        pass
    # In TwoRoom-v1 (512x512 or 224x224 PyGame canvas), the wall bisects the arena exactly at the geometric center:
    # Check coordinate range of proprio[:, 0] to return exact geometric midpoint of the environment arena
    x_min, x_max = float(np.min(proprio_x)), float(np.max(proprio_x))
    if x_max > 200.0:
        return 256.0 if x_max > 300.0 else 112.0
    elif x_min < -0.01:
        return 0.0
    return 0.5 * (x_min + x_max)


def get_episodes_length(dataset, episodes, col_name):
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/diagnose_tworoom_failures.py ~/.stable_worldmodel/tworoom_lewm/base.txt")
        sys.exit(1)

    runs = parse(sys.argv[1])
    cache_dir = Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset("tworoom", keys_to_cache=["proprio"], cache_dir=cache_dir)

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_col = dataset.get_col_data(col_name)
    step_col = dataset.get_col_data("step_idx")
    proprio = dataset.get_col_data("proprio")
    ep_indices, _ = np.unique(ep_col, return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices, col_name)

    wall_x = get_true_wall_x(proprio[:, 0])
    print(f"[diagnose_failures] True environment wall X coordinate: {wall_x:.2f} "
          f"(proprio X range: [{np.min(proprio[:, 0]):.1f}, {np.max(proprio[:, 0]):.1f}])")

    for (off, bud), seed_dict in sorted(runs.items()):
        max_start_idx = episode_len - off - 1
        max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
        max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in ep_col])

        # Match the held-out Test Split (ep_id < 1000) from eval.py
        if np.max(ep_indices) >= 1000:
            valid_mask = (step_col <= max_start_per_row) & (ep_col < 1000)
        else:
            valid_mask = step_col <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]

        same_ok, cross_ok = [], []
        for seed, ok_arr in sorted(seed_dict.items()):
            g = np.random.default_rng(seed)
            rand_idx = g.choice(len(valid_indices) - 1, size=len(ok_arr), replace=False)
            start_rows = np.sort(valid_indices[rand_idx])
            goal_rows = start_rows + off

            for i, (sr, gr) in enumerate(zip(start_rows, goal_rows)):
                x_s, x_g = proprio[sr, 0], proprio[gr, 0]
                if (x_s < wall_x) != (x_g < wall_x):
                    cross_ok.append(ok_arr[i])
                else:
                    same_ok.append(ok_arr[i])

        print(f"\n=== Diagnostic Breakdown: goal_offset={off}, eval_budget={bud} ===")
        print(f"  Same-Room  ({len(same_ok):3d} eps): {100 * np.mean(same_ok):5.1f}% success")
        print(f"  Cross-Room ({len(cross_ok):3d} eps): {100 * np.mean(cross_ok):5.1f}% success (Doorway required)")


if __name__ == "__main__":
    main()
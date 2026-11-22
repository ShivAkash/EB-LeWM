#!/usr/bin/env python
"""Paired comparison of two LeWM-style eval result files (baseline vs. your method).

eval.py appends one "==== CONFIG ==== / ==== RESULTS ====" block per run to <policy dir>/<output.filename>
(that is $STABLEWM_HOME/<policy folder>/<filename>). Run the SAME seeds / goal offsets for both methods:
eval.py draws the start/goal pairs with np.random.default_rng(seed), so equal seed + offset = identical
episodes, which makes the comparison paired (much tighter than comparing two independent success rates).

    python benchmark-protocol/compare_runs.py base.txt ours.txt [--boot 10000]

Reports, per goal offset: pooled success rate with Wilson 95% CI, the per-seed spread, and the paired
difference (ours - base) with a bootstrap 95% CI and an exact McNemar test on discordant episodes.
Needs only numpy.
"""
import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse(path):
    """-> {(goal_offset, eval_budget): {seed: np.ndarray[bool]}}   (later runs of a seed overwrite earlier ones)"""
    runs = defaultdict(dict)
    for chunk in Path(path).read_text().split("==== CONFIG ====")[1:]:
        cfg, _, res = chunk.partition("==== RESULTS ====")
        m = re.search(r"episode_successes'?:\s*array\(\[(.*?)\]", res, re.S)
        seed = re.search(r"^seed:\s*(-?\d+)", cfg, re.M)
        off = re.search(r"goal_offset_steps:\s*(\d+)", cfg)
        bud = re.search(r"eval_budget:\s*(\d+)", cfg)
        if not (m and seed and off and bud):
            print(f"[warn] skipped a block in {path} (no per-episode successes / seed / offset found)")
            continue
        ok = np.array([t == "True" for t in re.findall(r"True|False", m.group(1))])
        runs[(int(off.group(1)), int(bud.group(1)))][int(seed.group(1))] = ok
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def pct(x):
    return f"{100 * x:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("ours")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    A, B = parse(a.base), parse(a.ours)
    rng = np.random.default_rng(0)
    for key in sorted(set(A) | set(B)):
        off, bud = key
        print(f"\n=== goal_offset={off}  eval_budget={bud} ===")
        if key not in A or key not in B:
            print("  only present in one file -> nothing to compare")
            continue
        for name, R in (("base", A[key]), ("ours", B[key])):
            allk = np.concatenate(list(R.values()))
            lo, hi = wilson(int(allk.sum()), allk.size)
            per = [100 * r.mean() for r in R.values()]
            print(
                f"  {name}: {len(R)} seeds, {allk.size} episodes, success {pct(allk.mean())}%  Wilson95 [{pct(lo)}, {pct(hi)}]"
                f"   per-seed mean {np.mean(per):.1f} sd {np.std(per):.1f}"
            )
        seeds = sorted(set(A[key]) & set(B[key]))
        pairs = [(A[key][s], B[key][s]) for s in seeds if A[key][s].size == B[key][s].size]
        if len(pairs) < len(seeds):
            print("  [warn] some seeds have different episode counts and were dropped from the paired test")
        if not pairs:
            print("  no common seeds -> run both methods with the same seed list to get a paired test")
            continue
        x = np.concatenate([p[0] for p in pairs]).astype(float)
        y = np.concatenate([p[1] for p in pairs]).astype(float)
        d = y - x
        idx = rng.integers(0, d.size, size=(a.boot, d.size))
        boots = d[idx].mean(1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        only_b, only_a = int(((y == 1) & (x == 0)).sum()), int(((x == 1) & (y == 0)).sum())
        p = mcnemar_exact(only_b, only_a)
        print(
            f"  paired ({len(pairs)} common seeds, {d.size} episodes): ours-base = {100 * d.mean():+.1f} pts, bootstrap95 [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        )
        print(f"  discordant episodes: ours-only {only_b}, base-only {only_a}, McNemar exact p = {p:.3g}")
        verdict = (
            "CI excludes 0"
            if (lo > 0 or hi < 0)
            else "CI includes 0 -> not distinguishable from noise at this sample size"
        )
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
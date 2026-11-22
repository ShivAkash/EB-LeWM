#!/usr/bin/env python
"""
Summarizes planning overhead (plans/ep, subgoal_plans/ep, flushes/ep, wall-clock sec/ep)
and success rates across eval.py output files.
Usage:
    python analysis/summarize_overhead.py ~/.stable_worldmodel/tworoom_lewm/final_*_b90.txt
"""
import re
import sys
from pathlib import Path
import numpy as np


def summarize_file(path: Path):
    text = path.read_text()
    blocks = text.split("==== CONFIG ====")[1:]
    if not blocks:
        return None

    succs, times, ticks, plans, sub_plans, flushes = [], [], [], [], [], []
    n_eps_total = 0

    for b in blocks:
        cfg, _, res = b.partition("==== RESULTS ====")
        m_ok = re.search(r"episode_successes'?:\s*array\(\[(.*?)\]", res, re.S)
        m_t = re.search(r"evaluation_time:\s*([\d.]+)", res)
        m_w = re.search(
            r"wrapper_stats:\s*\{'ticks':\s*(\d+),\s*'plans':\s*(\d+),\s*'subgoal_plans':\s*(\d+),\s*'flushes':\s*(\d+)\}",
            res,
        )
        if not m_ok:
            continue
        ok = np.array([t == "True" for t in re.findall(r"True|False", m_ok.group(1))])
        n_eps = len(ok)
        n_eps_total += n_eps
        succs.append(ok.mean() * 100.0)
        if m_t:
            times.append(float(m_t.group(1)) / n_eps)
        if m_w:
            ticks.append(int(m_w.group(1)) / n_eps)
            plans.append(int(m_w.group(2)) / n_eps)
            sub_plans.append(int(m_w.group(3)) / n_eps)
            flushes.append(int(m_w.group(4)) / n_eps)

    return {
        "file": path.name,
        "seeds": len(succs),
        "eps": n_eps_total,
        "succ_mean": np.mean(succs),
        "succ_sd": np.std(succs),
        "sec_per_ep": np.mean(times) if times else float("nan"),
        "ticks_per_ep": np.mean(ticks) if ticks else 0.0,
        "plans_per_ep": np.mean(plans) if plans else 0.0,
        "sub_per_ep": np.mean(sub_plans) if sub_plans else 0.0,
        "flush_per_ep": np.mean(flushes) if flushes else 0.0,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analysis/summarize_overhead.py <eval_file1.txt> <eval_file2.txt> ...")
        sys.exit(1)

    print(
        f"{'File':34s} | {'Seeds':>5s} | {'Succ (%)':>11s} | {'Plans/Ep':>8s} | "
        f"{'SubPl/Ep':>8s} | {'Flush/Ep':>8s} | {'Sec/Ep':>7s}"
    )
    print("-" * 98)
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            continue
        s = summarize_file(p)
        if s:
            print(
                f"{s['file']:34s} | {s['seeds']:5d} | {s['succ_mean']:5.1f}±{s['succ_sd']:4.1f}% | "
                f"{s['plans_per_ep']:8.2f} | {s['sub_per_ep']:8.2f} | {s['flush_per_ep']:8.2f} | "
                f"{s['sec_per_ep']:6.2f}s"
            )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Inspect a dump produced by NeroRobot.dump_ik_diagnostics().

Reads the JSON file written automatically on disconnect (default
`/tmp/nero_ik_failures_<timestamp>.json`, or whatever you set
`NERO_IK_DIAG_DUMP=` to) and prints summary statistics + clusters of
failures so you can tell at a glance:

  • Are failures concentrated at a specific time / call index?
  • Are they at a specific region of the workspace?
  • How big is the convergence residual (just over tol vs. far off)?

Usage
-----
    python3 scripts/analyze_ik_failures.py /tmp/nero_ik_failures_*.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fmt(arr) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in arr) + "]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="JSON dump file from NeroRobot")
    p.add_argument("--top", type=int, default=10,
                   help="show the top-N failures by err (default 10)")
    args = p.parse_args()

    data = load(args.path)

    print("=" * 72)
    print(f"IK diagnostics: {args.path}")
    print("=" * 72)
    print(f"  total IK calls       : {data['total_ik_calls']}")
    print(f"  total IK failures    : {data['total_ik_failures']}")
    print(f"  failure rate         : {data['failure_rate'] * 100:.2f} %")
    print(f"  buffer size          : {data['buffered_failures']} / {data['buffer_capacity']}")

    failures = data.get("failures", [])
    if not failures:
        print("\nNo failures recorded — clean run.")
        return

    # --- Time-since-connect distribution ---
    ts = [f["t_since_connect_s"] for f in failures if f.get("t_since_connect_s") is not None]
    if ts:
        print("\nTime-since-connect (seconds):")
        print(f"  min={min(ts):.2f}  median={statistics.median(ts):.2f}  max={max(ts):.2f}")

    # --- err distribution ---
    errs = [f["err"] for f in failures]
    print("\nIK err residual:")
    print(f"  min={min(errs):.4f}  median={statistics.median(errs):.4f}  max={max(errs):.4f}")
    print(f"  (tol = 5e-3 = 0.005, so anything close to that is a boundary case)")

    # --- AIRBOT-frame target distribution ---
    pos_airbot = np.array([
        f["target_pose7_airbot"][:3] for f in failures if f.get("target_pose7_airbot")
    ])
    if len(pos_airbot):
        print("\nAIRBOT-frame target position (m):")
        print(f"  x range = [{pos_airbot[:,0].min():.3f}, {pos_airbot[:,0].max():.3f}]   "
              f"mean = {pos_airbot[:,0].mean():.3f}")
        print(f"  y range = [{pos_airbot[:,1].min():.3f}, {pos_airbot[:,1].max():.3f}]   "
              f"mean = {pos_airbot[:,1].mean():.3f}")
        print(f"  z range = [{pos_airbot[:,2].min():.3f}, {pos_airbot[:,2].max():.3f}]   "
              f"mean = {pos_airbot[:,2].mean():.3f}")
        print("  (training-data action ranges from normaliser stats:")
        print("     x ∈ [0.272, 0.676], y ∈ [-0.198, 0.305], z ∈ [-0.076, 0.573])")

    # --- Worst N failures ---
    print(f"\nTop {min(args.top, len(failures))} failures by err:")
    sorted_failures = sorted(failures, key=lambda f: -f["err"])[: args.top]
    for f in sorted_failures:
        ab = f.get("target_pose7_airbot")
        ab_str = fmt(ab[:3]) if ab else "N/A"
        print(
            f"  call#{f['ik_call_index']:5d}  t={f['t_since_connect_s']:.2f}s  "
            f"err={f['err']:.4f}  airbot_pos={ab_str}"
        )


if __name__ == "__main__":
    main()

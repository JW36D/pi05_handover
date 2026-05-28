#!/usr/bin/env python3
"""Move the Nero arm to a known starting joint configuration, then disconnect.

Typical use: run this BEFORE starting `policy_server` + `robot_client` for
pi05 inference, so the arm starts inside the training distribution and the
first IK call has a sensible warm-start.

Sources of starting joints (in priority order):
    --joints "j1,j2,j3,j4,j5,j6,j7"     explicit 7-tuple
    --parquet ... --episode K           episode K's first observation.state
    (default)                            training episode-start q50

Usage
-----
    # Move to the median training episode-start pose (safe default)
    python3 scripts/preposition.py

    # Move to episode 0's recorded start
    python3 scripts/preposition.py --parquet ../chunk-000/file-000.parquet --episode 0

    # Move to an explicit joint config
    python3 scripts/preposition.py --joints "0.0,0.0,0.0,1.5708,0.0,0.0,0.0"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot_robot_nero import NeroRobot, NeroRobotConfig


def state_array_from_obs(obs: dict) -> np.ndarray:
    """Repack a flat hardware observation dict into the canonical (8,) state.

    NeroRobot.get_observation() returns hardware-level keys (one float per
    joint + gripper), not the model-level `observation.state` blob. This
    helper re-assembles them in the order expected by training data
    (STATE_FEATURE_NAMES).
    """
    return np.asarray(
        [float(obs[name]) for name in NeroRobot.STATE_FEATURE_NAMES],
        dtype=np.float64,
    )

# Median first-frame joint configuration from the converted LeRobot v3
# training data (`chunk-000/file-000.parquet`). This is a better rollout
# start than the full-trajectory normaliser q50, which is closer to the middle
# of demonstrations and can make the first chunk out-of-distribution.
TRAINING_START_Q50: list[float] = [
    0.0105, 0.1168, -0.0392, 1.5489, 0.0434, 0.0406, 0.3477,
]


def parse_joints(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 7:
        raise argparse.ArgumentTypeError(f"--joints needs 7 comma-separated values, got {len(parts)}")
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--joints", type=parse_joints, default=None,
                   help="Explicit 7-tuple, e.g. '0,0,0,1.57,0,0,0'")
    p.add_argument("--parquet", type=Path, default=None,
                   help="Training parquet — read first frame of --episode")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--channel", default="can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--bitrate", type=int, default=1_000_000)
    p.add_argument("--firmware", default="v1.11")
    p.add_argument("--use-gripper", action="store_true")
    p.add_argument("--speed-percent", type=int, default=15)
    p.add_argument("--settle-tol", type=float, default=0.02,
                   help="Per-joint tolerance for settle (rad), default 0.02")
    p.add_argument("--settle-timeout", type=float, default=20.0)
    return p.parse_args()


def resolve_target(args: argparse.Namespace) -> list[float]:
    if args.joints is not None:
        return args.joints
    if args.parquet is not None:
        import pandas as pd
        df = pd.read_parquet(args.parquet)
        ep = df[df["episode_index"] == args.episode].sort_values("frame_index")
        if ep.empty:
            raise SystemExit(f"No rows for episode_index={args.episode}")
        row = ep.iloc[0]
        if "observation.state" in ep.columns:
            state = np.asarray(row["observation.state"], dtype=np.float64)
            if state.shape != (8,):
                raise SystemExit(f"observation.state shape {state.shape}, expected (8,)")
            return state[:7].tolist()
        v21_key = "observation.state.agx_nero__arm.joint_position"
        if v21_key in ep.columns:
            joints = np.asarray(row[v21_key], dtype=np.float64)
            if joints.shape != (7,):
                raise SystemExit(f"{v21_key} shape {joints.shape}, expected (7,)")
            return joints.tolist()
        raise SystemExit(
            "Parquet does not contain observation.state or "
            f"{v21_key}; columns={list(ep.columns)}"
        )
    return list(TRAINING_START_Q50)


def main() -> None:
    args = parse_args()
    target = resolve_target(args)

    print("=" * 60)
    print("Nero pre-position")
    print("=" * 60)
    print(f"  target joints (rad): {target}")
    print()

    cfg = NeroRobotConfig(
        interface=args.interface,
        channel=args.channel,
        bitrate=args.bitrate,
        firmware=args.firmware,
        use_gripper=args.use_gripper,
        use_camera=False,
        speed_percent=args.speed_percent,
    )
    robot = NeroRobot(cfg)
    try:
        print("Connecting...")
        robot.connect()

        cur = state_array_from_obs(robot.get_observation())[:7]
        max_delta = float(np.max(np.abs(np.asarray(target) - cur)))
        print(f"  current  joints (rad): {cur.tolist()}")
        print(f"  max delta to target  : {max_delta:.3f} rad")
        if max_delta > 1.5:
            print("  WARNING: large delta — review carefully before continuing.")

        input("\nPress ENTER to move (Ctrl-C to abort)...")
        robot.move_to_joints(target)

        t0 = time.monotonic()
        while True:
            measured = state_array_from_obs(robot.get_observation())[:7]
            err = float(np.max(np.abs(measured - np.asarray(target))))
            if err < args.settle_tol:
                print(f"\nSettled at: {measured.tolist()}  (err={err:.4f} rad)")
                break
            if time.monotonic() - t0 > args.settle_timeout:
                print(f"\nWARNING: settle timeout ({args.settle_timeout}s).")
                print(f"  current measured: {measured.tolist()}  err={err:.4f} rad")
                break
            time.sleep(0.1)
    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()

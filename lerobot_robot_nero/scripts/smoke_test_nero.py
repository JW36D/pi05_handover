#!/usr/bin/env python3
"""Minimal smoke test for AgileX Nero: connect, read observation, optionally move.

This test verifies the adapter is wired correctly before running pi05 inference:

  1. Connects to the arm (and optionally gripper + camera).
  2. Reads get_observation() and prints keys / shapes / dtypes.
  3. Optionally sends a single move_p action via --execute.

Usage
-----
    # Dry run (read observation, print planned action, no motion)
    python3 scripts/smoke_test_nero.py

    # Actually send motion (small relative Cartesian step on X axis)
    python3 scripts/smoke_test_nero.py --execute

    # With gripper
    python3 scripts/smoke_test_nero.py --use-gripper --execute

    # With camera
    python3 scripts/smoke_test_nero.py --use-camera --execute
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from pprint import pprint

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot_robot_nero import NeroRobot, NeroRobotConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal smoke test for AgileX Nero.")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--firmware", default="default")
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--use-camera", action="store_true")
    parser.add_argument("--camera-serial", default="")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument(
        "--dx", type=float, default=0.01,
        help="Relative X-axis step (m) for the test action. Default: 0.01",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually send motion command. Omit for dry-run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = NeroRobotConfig(
        interface=args.interface,
        channel=args.channel,
        bitrate=args.bitrate,
        firmware=args.firmware,
        use_gripper=args.use_gripper,
        gripper_force=args.gripper_force,
        use_camera=args.use_camera,
        camera_serial=args.camera_serial,
        speed_percent=args.speed_percent,
    )

    robot = NeroRobot(cfg)
    try:
        print("Connecting to Nero...")
        robot.connect()

        print("\nCurrent observation (flat hardware dict):")
        obs = robot.get_observation()
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                print(f"  {key!r}  shape={val.shape}  dtype={val.dtype}")
            else:
                print(f"  {key!r}: {val!r}")

        # Re-pack flat dict into the canonical (8,) state for printing/use.
        from lerobot_robot_nero import NeroRobot as _NR
        state = np.asarray(
            [float(obs[name]) for name in _NR.STATE_FEATURE_NAMES],
            dtype=np.float32,
        )
        print(f"\n    joints (rad) = {state[:7].tolist()}")
        print(f"    gripper  (m) = {state[7]:.4f}")

        # Build a test action: trivial identity-quat pose slightly displaced
        # in X. For a real sanity test, set the pose to the arm's current EE.
        print(
            "\nNote: smoke test uses a fixed test pose [0.3, 0.0, 0.4] + identity quat.\n"
            "      For a proper end-to-end test, set the pose to the arm's current EE pose."
        )
        pos = np.array([0.3 + args.dx, 0.0, 0.4], dtype=np.float32)
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        gripper_cmd = float(state[7])
        action = np.concatenate([pos, quat_xyzw, [gripper_cmd]])

        print(f"\nPlanned action (shape={action.shape}, dtype={action.dtype}):")
        print(f"  pos      = {action[:3]}")
        print(f"  quat     = {action[3:7]}")
        print(f"  gripper  = {action[7]:.4f} m")

        if args.execute:
            print("\nSending action...")
            robot.send_action(action)
            print("Action sent. Reading observation again...")
            obs2 = robot.get_observation()
            state2 = np.asarray(
                [float(obs2[name]) for name in _NR.STATE_FEATURE_NAMES],
                dtype=np.float32,
            )
            print(f"  joints after (rad) = {state2[:7].tolist()}")
            print(f"  gripper after  (m) = {state2[7]:.4f}")
        else:
            print("\nDry run. Use --execute to send the motion command.")

    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()

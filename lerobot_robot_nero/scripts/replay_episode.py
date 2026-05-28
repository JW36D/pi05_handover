#!/usr/bin/env python3
"""Ground-truth replay: feed a training episode's action sequence to the real Nero arm.

Usage
-----
    python3 scripts/replay_episode.py \\
        --parquet /path/to/data/chunk-000/file-000.parquet \\
        --episode 0 \\
        [--fps 30] \\
        [--channel can0] \\
        [--interface socketcan] \\
        [--firmware default] \\
        [--speed-percent 10] \\
        [--use-gripper] \\
        [--dry-run]

Pass --dry-run to print actions without sending them to hardware.

What this validates
-------------------
If the arm reproduces the training trajectory faithfully:
  → observation.state, send_action, euler/unit/coordinate conventions are all correct.

If the arm moves to wrong poses:
  → Check the euler_seq ('xyz' vs 'zyx'), or whether apply_command_pose_fix
    from rollio_device_nero.airbot_aligned_pose needs to be applied first.
    See the comment in nero.py::send_action() for details.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from lerobot_robot_nero import NeroRobot, NeroRobotConfig


def state_array_from_obs(obs: dict) -> np.ndarray:
    """Repack a flat hardware observation dict into the canonical (8,) state.

    NeroRobot.get_observation() returns hardware-level keys (joint scalars +
    optional camera frame), not a model-level `observation.state` blob.
    """
    return np.asarray(
        [float(obs[name]) for name in NeroRobot.STATE_FEATURE_NAMES],
        dtype=np.float64,
    )


def load_episode_actions(parquet_path: str, episode_index: int) -> np.ndarray:
    """Load the action column for a single episode from a parquet file.

    Returns
    -------
    np.ndarray shape=(N, 8) float32
        Each row is [x, y, z, qx, qy, qz, qw, gripper].
    """
    actions, _ = load_episode_actions_and_initial_state(parquet_path, episode_index)
    return actions


def load_episode_actions_and_initial_state(
    parquet_path: str, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load both action sequence and the FIRST observation.state of the episode.

    The initial state is used to pre-position the arm so IK has a sensible
    seed and the first cartesian command is reachable from the warm-start.
    """
    df = pd.read_parquet(parquet_path)

    if "episode_index" not in df.columns or "action" not in df.columns:
        raise KeyError(f"Parquet missing required columns. Got: {list(df.columns)}")

    ep = df[df["episode_index"] == episode_index].sort_values("frame_index")
    if ep.empty:
        available = sorted(df["episode_index"].unique().tolist())
        raise ValueError(
            f"No rows for episode_index={episode_index}. Available: {available}"
        )

    actions = np.array(ep["action"].tolist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(
            f"Expected action shape (N, 8), got {actions.shape}. "
            "Action layout must be [x, y, z, qx, qy, qz, qw, gripper]."
        )

    if "observation.state" not in ep.columns:
        # Some parquet variants don't carry state; caller must handle None.
        return actions, np.array([])
    initial_state = np.asarray(ep.iloc[0]["observation.state"], dtype=np.float32)
    if initial_state.shape != (8,):
        raise ValueError(
            f"observation.state must be shape (8,), got {initial_state.shape}"
        )
    return actions, initial_state


def print_action_summary(actions: np.ndarray) -> None:
    pos_min = actions[:, :3].min(axis=0)
    pos_max = actions[:, :3].max(axis=0)
    gripper_min = actions[:, 7].min()
    gripper_max = actions[:, 7].max()
    print(f"  frames       : {len(actions)}")
    print(f"  pos range    : x=[{pos_min[0]:.3f}, {pos_max[0]:.3f}]  "
          f"y=[{pos_min[1]:.3f}, {pos_max[1]:.3f}]  "
          f"z=[{pos_min[2]:.3f}, {pos_max[2]:.3f}]  (m)")
    print(f"  gripper range: [{gripper_min:.4f}, {gripper_max:.4f}] m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth replay for Nero arm.")
    parser.add_argument(
        "--parquet", required=True,
        help="Path to training parquet file, e.g. data/chunk-000/file-000.parquet",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Replay framerate (default: 30)")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--firmware", default="v1.11")
    parser.add_argument("--use-gripper", action="store_true")
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument(
        "--no-airbot-fix", action="store_true",
        help="Skip apply_command_pose_fix (use when Nero physical orientation "
             "matches AIRBOT data-collection rig, i.e. NOT rotated 180°).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without sending to hardware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Nero ground-truth replay")
    print("=" * 60)
    print(f"  parquet  : {args.parquet}")
    print(f"  episode  : {args.episode}")
    print(f"  fps      : {args.fps}")
    print(f"  dry-run  : {args.dry_run}")
    print()

    print("Loading episode actions...")
    actions, initial_state = load_episode_actions_and_initial_state(
        args.parquet, args.episode
    )
    print_action_summary(actions)
    if initial_state.size:
        print(f"  initial joints : {initial_state[:7].tolist()}")
        print(f"  initial gripper: {float(initial_state[7]):.4f} m")
    print()

    if args.dry_run:
        print("[DRY RUN] First 3 actions:")
        for i, a in enumerate(actions[:3]):
            print(f"  [{i}] pos=({a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f})  "
                  f"quat=({a[3]:.4f}, {a[4]:.4f}, {a[5]:.4f}, {a[6]:.4f})  "
                  f"gripper={a[7]:.4f}")
        print("[DRY RUN] Skipping hardware connection. Remove --dry-run to execute.")
        return

    cfg = NeroRobotConfig(
        interface=args.interface,
        channel=args.channel,
        bitrate=args.bitrate,
        firmware=args.firmware,
        use_gripper=args.use_gripper,
        gripper_force=args.gripper_force,
        use_camera=False,
        speed_percent=args.speed_percent,
        airbot_aligned_action=not args.no_airbot_fix,
    )
    print(f"  airbot frame fix: {'OFF (--no-airbot-fix)' if args.no_airbot_fix else 'ON'}")

    robot = NeroRobot(cfg)
    period = 1.0 / args.fps

    try:
        print("Connecting to Nero arm...")
        robot.connect()

        print("\nReading initial observation:")
        state = state_array_from_obs(robot.get_observation())
        print(f"  joint_angles (rad)      : {state[:7].tolist()}")
        print(f"  gripper_width (m)       : {state[7]:.4f}")

        # SAFETY: pre-position the arm to the recorded first-frame joint state
        # before kicking off cartesian inference. Without this, IK has to
        # bridge a large gap from the home pose [0,0,0,π/2,0,0,0] to the
        # episode's start in a single tick, which often fails to converge.
        if initial_state.size:
            target_joints = np.asarray(initial_state[:7], dtype=np.float64)
            max_delta = float(np.max(np.abs(target_joints - state[:7])))
            print(f"\nPre-positioning arm to episode start "
                  f"(max joint delta {max_delta:.3f} rad)...")
            input("Press ENTER to move the arm to the start pose (Ctrl-C to abort)...")
            robot.move_to_joints(target_joints.tolist())

            # Poll measured joints until close to target or timeout.
            settle_tol = 0.02  # ~1.1°
            settle_timeout_s = 15.0
            t0 = time.monotonic()
            while True:
                settled = state_array_from_obs(robot.get_observation())[:7]
                settle_err = float(np.max(np.abs(settled - target_joints)))
                if settle_err < settle_tol:
                    break
                if time.monotonic() - t0 > settle_timeout_s:
                    break
                time.sleep(0.1)
            print(f"  settled at: {settled.tolist()}")
            print(f"  settle err: {settle_err:.4f} rad")
            if settle_err > settle_tol:
                print("  WARNING: arm did not reach the start pose within tolerance. "
                      "Consider raising --speed-percent.")
                resp = input("  Continue anyway? [y/N]: ")
                if resp.strip().lower() != "y":
                    raise RuntimeError("Aborted by user — arm did not pre-position correctly.")

        input("\nPress ENTER to start replaying the episode (Ctrl-C to abort)...")
        print()

        t_start = time.monotonic()
        for i, action in enumerate(actions):
            t_frame = t_start + i * period
            a = action.astype(np.float32)
            try:
                robot.send_action(a)
            except RuntimeError as exc:
                print()  # finish the carriage-return line
                print(f"\n[HALT] Frame {i}: send_action failed → {exc}")
                print("Replay aborted to prevent unsafe motion.")
                break

            elapsed = time.monotonic() - t_start
            print(
                f"\r  frame {i+1:4d}/{len(actions)}"
                f"  pos=({a[0]:.3f},{a[1]:.3f},{a[2]:.3f})"
                f"  gripper={a[7]:.3f}"
                f"  elapsed={elapsed:.1f}s",
                end="",
                flush=True,
            )

            sleep_s = t_frame + period - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        print()
        print("\nReplay complete. Reading final observation:")
        state = state_array_from_obs(robot.get_observation())
        print(f"  joint_angles (rad) : {state[:7].tolist()}")
        print(f"  gripper_width (m)  : {state[7]:.4f}")

    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()

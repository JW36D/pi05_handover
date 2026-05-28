#!/usr/bin/env python3
"""End-to-end data-contract sanity check (no robot needed).

Runs assertions against the four artifacts that determine the deployment
behaviour:

  1. rollio v2.1 parquet  — `action` column must equal the AIRBOT leader pose
  2. lerobot v3 parquet   — column shapes match what convert_rollio_to_lerobot
                            produces and pi05 expects
  3. pi05 checkpoint      — action_feature_names match NeroRobot.ACTION_FEATURE_NAMES
  4. rollio runtime       — apply_command_pose_fix is importable and well-defined

If any check fails the contract is broken; fix it before running pi05.
The full contract is documented in lerobot_robot_nero/DATA_CONTRACT.md.

Usage
-----
    python3 scripts/verify_data_contract.py \\
        --v21-parquet  /path/to/output/data/chunk-000/episode_000000.parquet \\
        --v3-parquet   /path/to/chunk-000/file-000.parquet \\
        --checkpoint   /path/to/checkpoints/050000/pretrained_model

Each `--*` argument is optional; checks for missing artifacts are skipped
with a warning instead of failing.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Same as NeroRobot.ACTION_FEATURE_NAMES — duplicated here to avoid pulling
# in the full lerobot dependency chain just to read 8 strings.
EXPECTED_ACTION_NAMES: tuple[str, ...] = (
    "agx_nero__arm.end_pose.0",
    "agx_nero__arm.end_pose.1",
    "agx_nero__arm.end_pose.2",
    "agx_nero__arm.end_pose.3",
    "agx_nero__arm.end_pose.4",
    "agx_nero__arm.end_pose.5",
    "agx_nero__arm.end_pose.6",
    "agx_nero__gripper.parallel_mit.0",
)


# --- minimal safetensors reader (header only is enough for stats) -----------
_SAFETENSORS_DTYPE = {
    "F32": (np.float32, 4),
    "F64": (np.float64, 8),
    "F16": (np.float16, 2),
    "I64": (np.int64, 8),
    "I32": (np.int32, 4),
}


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        raw = f.read()
    out: dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        np_dtype, _ = _SAFETENSORS_DTYPE.get(info["dtype"], (None, None))
        if np_dtype is None:
            continue
        off_lo, off_hi = info["data_offsets"]
        shape = info["shape"] or [1]
        out[name] = np.frombuffer(raw[off_lo:off_hi], dtype=np_dtype).reshape(shape).copy()
    return out


# --- pretty result tracking -------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def ok(self, name: str) -> None:
        print(f"  [PASS] {name}")
        self.passed.append(name)

    def fail(self, name: str, reason: str) -> None:
        print(f"  [FAIL] {name}\n         {reason}")
        self.failed.append((name, reason))

    def skip(self, name: str, reason: str) -> None:
        print(f"  [SKIP] {name}  ({reason})")
        self.skipped.append((name, reason))

    def summarize(self) -> int:
        print()
        print("=" * 60)
        print(f"PASSED:  {len(self.passed)}")
        print(f"FAILED:  {len(self.failed)}")
        print(f"SKIPPED: {len(self.skipped)}")
        print("=" * 60)
        if self.failed:
            print("\nFAILURES:")
            for n, r in self.failed:
                print(f"  - {n}: {r}")
            return 1
        return 0


# --- check 1: rollio v2.1 parquet ------------------------------------------


def check_v21_parquet(path: Path, report: Report) -> None:
    print(f"\n[1] rollio v2.1 parquet — {path}")
    if not path.exists():
        report.skip("rollio v2.1 parquet", f"not found: {path}")
        return

    try:
        import pandas as pd
    except ImportError:
        report.skip("rollio v2.1 parquet", "pandas not installed")
        return

    df = pd.read_parquet(path)

    expected_cols = {
        "action",
        "observation.state.airbot_play__arm.end_effector_pose",
        "observation.state.airbot_play__e2.parallel_position",
        "observation.state.agx_nero__arm.joint_position",
        "observation.state.agx_nero__gripper.parallel_position",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        report.fail("v21.columns", f"missing columns: {missing}")
        return
    report.ok("v21.columns present")

    # CORE invariant — action[:7] is byte-identical to AIRBOT leader pose
    a = np.stack([np.asarray(x) for x in df["action"].values])
    ab = np.stack([np.asarray(x) for x in
                   df["observation.state.airbot_play__arm.end_effector_pose"].values])
    diff = float(np.abs(a[:, :7] - ab).max())
    if diff > 1e-9:
        report.fail("v21.action_equals_airbot_pose", f"max abs diff = {diff}")
    else:
        report.ok(f"v21.action_equals_airbot_pose (max diff = {diff:.1e})")

    # Gripper scale 1.8 should hold (mostly — small precision noise allowed)
    e2 = np.stack([np.asarray(x) for x in
                   df["observation.state.airbot_play__e2.parallel_position"].values])
    nonzero = e2[:, 0] > 1e-3
    if nonzero.any():
        ratios = a[nonzero, 7] / e2[nonzero, 0]
        if abs(ratios.mean() - 1.8) < 0.01:
            report.ok(f"v21.gripper_scale ≈ 1.8 (mean={ratios.mean():.4f})")
        else:
            report.fail("v21.gripper_scale", f"expected 1.8, got {ratios.mean():.4f}")


# --- check 2: lerobot v3 parquet -------------------------------------------


def check_v3_parquet(path: Path, report: Report) -> None:
    print(f"\n[2] lerobot v3 parquet — {path}")
    if not path.exists():
        report.skip("lerobot v3 parquet", f"not found: {path}")
        return

    try:
        import pandas as pd
    except ImportError:
        report.skip("lerobot v3 parquet", "pandas not installed")
        return

    df = pd.read_parquet(path)

    needed = {"action", "observation.state", "episode_index", "frame_index"}
    missing = needed - set(df.columns)
    if missing:
        report.fail("v3.columns", f"missing: {missing}")
        return
    report.ok("v3.columns present")

    a = np.stack([np.asarray(x) for x in df["action"].values])
    s = np.stack([np.asarray(x) for x in df["observation.state"].values])

    if a.shape[1] == 8 and a.dtype == np.float32:
        report.ok(f"v3.action shape=(N,8) dtype=float32  (rows={len(a)})")
    else:
        report.fail("v3.action shape/dtype",
                    f"got shape={a.shape}, dtype={a.dtype}")

    if s.shape[1] == 8 and s.dtype == np.float32:
        report.ok(f"v3.observation.state shape=(N,8) dtype=float32")
    else:
        report.fail("v3.observation.state shape/dtype",
                    f"got shape={s.shape}, dtype={s.dtype}")

    # Range sanity checks
    x_range = (a[:, 0].min(), a[:, 0].max())
    if x_range[0] > 0 and x_range[1] < 1.5:
        report.ok(f"v3.action[x] in plausible range {x_range}")
    else:
        report.fail("v3.action[x] range", f"{x_range} looks unphysical")

    g_state_range = (s[:, 7].min(), s[:, 7].max())
    if 0 <= g_state_range[0] and g_state_range[1] <= 0.15:
        report.ok(f"v3.state[gripper] range {g_state_range} (m)")
    else:
        report.fail("v3.state[gripper] range",
                    f"{g_state_range} unexpected")


# --- check 3: pi05 checkpoint ----------------------------------------------


def check_checkpoint(ckpt_dir: Path, report: Report) -> None:
    print(f"\n[3] pi05 checkpoint — {ckpt_dir}")
    if not ckpt_dir.exists():
        report.skip("pi05 checkpoint", f"not found: {ckpt_dir}")
        return

    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        report.fail("ckpt.config.json present", "config.json missing")
        return
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Type
    if cfg.get("type") == "pi05":
        report.ok("ckpt.type == pi05")
    else:
        report.fail("ckpt.type", f"got {cfg.get('type')!r}, expected 'pi05'")

    # Action feature names
    names = tuple(cfg.get("action_feature_names", ()))
    if names == EXPECTED_ACTION_NAMES:
        report.ok("ckpt.action_feature_names matches NeroRobot.ACTION_FEATURE_NAMES")
    else:
        report.fail(
            "ckpt.action_feature_names",
            f"checkpoint:\n           {names}\n         expected:\n           {EXPECTED_ACTION_NAMES}",
        )

    # Input/output shapes
    inp = cfg.get("input_features", {})
    out = cfg.get("output_features", {})
    if inp.get("observation.state", {}).get("shape") == [8]:
        report.ok("ckpt.input.observation.state shape=[8]")
    else:
        report.fail("ckpt.input.observation.state.shape",
                    str(inp.get("observation.state")))

    img_shape = inp.get("observation.images.realsense_color", {}).get("shape")
    if img_shape == [3, 1080, 1920]:
        report.ok("ckpt.input.observation.images.realsense_color shape=[3,1080,1920]")
    else:
        report.fail("ckpt.input image shape", str(img_shape))

    if out.get("action", {}).get("shape") == [8]:
        report.ok("ckpt.output.action shape=[8]")
    else:
        report.fail("ckpt.output.action.shape", str(out.get("action")))

    # Normalization mapping
    nm = cfg.get("normalization_mapping", {})
    if nm.get("STATE") == "QUANTILES" and nm.get("ACTION") == "QUANTILES":
        report.ok("ckpt.normalization_mapping STATE/ACTION = QUANTILES")
    else:
        report.fail("ckpt.normalization_mapping", str(nm))

    # Normalizer stats consistency
    norm_path = ckpt_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if norm_path.exists():
        stats = _read_safetensors(norm_path)
        for key, dim in (("action.min", 8), ("action.max", 8),
                         ("observation.state.min", 8), ("observation.state.max", 8)):
            if key not in stats:
                report.fail(f"ckpt.norm[{key}]", "missing from normalizer safetensors")
            elif stats[key].size != dim:
                report.fail(f"ckpt.norm[{key}]",
                            f"size {stats[key].size} != {dim}")
        if all(k in stats for k in ("action.min", "action.max")):
            x_min = float(stats["action.min"][0])
            x_max = float(stats["action.max"][0])
            if 0.0 < x_min < x_max < 1.0:
                report.ok(f"ckpt.norm action[x] range = [{x_min:.3f}, {x_max:.3f}]")
            else:
                report.fail("ckpt.norm action[x]",
                            f"unexpected range [{x_min}, {x_max}]")


# --- check 4: rollio runtime importable + transform sane -------------------


def check_rollio_runtime(report: Report) -> None:
    print(f"\n[4] rollio_device_nero runtime")
    try:
        from rollio_device_nero.airbot_aligned_pose import (
            apply_command_pose_fix,
            apply_publish_pose_fix,
        )
    except ImportError as exc:
        report.fail("rollio_runtime.import",
                    f"cannot import apply_command_pose_fix: {exc}")
        return
    report.ok("rollio_runtime.import (apply_command_pose_fix / apply_publish_pose_fix)")

    # Round-trip property: command_fix must invert publish_fix.
    sample = [0.45, -0.05, 0.30, 0.0, 0.0, 0.0, 1.0]
    aligned = apply_publish_pose_fix(sample)
    back = apply_command_pose_fix(aligned)
    err = float(np.max(np.abs(np.asarray(sample) - np.asarray(back))))
    if err < 1e-6:
        report.ok(f"rollio_runtime.round_trip command∘publish ≈ identity (err={err:.1e})")
    else:
        report.fail("rollio_runtime.round_trip", f"max err {err}")

    # Sanity: identity quat + non-zero x must change x sign (180° about z).
    in_pose = [0.5, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0]
    out_pose = apply_command_pose_fix(in_pose)
    if out_pose[0] * in_pose[0] < 0 and abs(out_pose[1] + in_pose[1]) < 1e-6:
        report.ok("rollio_runtime.transform flips x and y (180°z) as expected")
    else:
        report.fail("rollio_runtime.transform",
                    f"x/y flip not observed: in={in_pose[:3]} out={out_pose[:3]}")


# --- main ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v21-parquet", type=Path,
                   default=Path("output/data/chunk-000/episode_000000.parquet"))
    p.add_argument("--v3-parquet", type=Path,
                   default=Path("chunk-000/file-000.parquet"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/050000/pretrained_model"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()
    print("=" * 60)
    print("Nero × pi05 data-contract verification")
    print("=" * 60)
    check_v21_parquet(args.v21_parquet, report)
    check_v3_parquet(args.v3_parquet, report)
    check_checkpoint(args.checkpoint, report)
    check_rollio_runtime(report)
    return report.summarize()


if __name__ == "__main__":
    sys.exit(main())

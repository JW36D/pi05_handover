# PI05 Handover on AgileX Nero

This repository contains a working PI05/OpenPI handover runtime for the AgileX Nero robot arm, built on top of LeRobot async inference.

The project integrates a PI05 policy checkpoint with a real Nero arm by adding:

- a LeRobot external robot adapter for Nero,
- an AIRBOT-to-Nero end-effector pose execution path,
- Pinocchio-based inverse kinematics through `rollio_device_nero`,
- real-time chunking support for async inference,
- action chunk overlap smoothing,
- joint-space safety limiting and low-pass smoothing,
- diagnostics for IK failures and runtime instability.

The main goal of this repository is to preserve the code and engineering notes needed to reproduce the stabilized Nero handover demo.

## Repository Layout

```text
pi05_handover/
├── convert_rollio_to_lerobot.py
├── lerobot/
├── lerobot_robot_nero/
└── pyAgxArm/
```

### `lerobot/`

A local LeRobot snapshot with changes for Nero async inference stability, including:

- async client `must_go` handling to avoid queue starvation,
- `smooth_overlap` action chunk aggregation,
- PI05 async inference compatibility fixes.

### `lerobot_robot_nero/`

The Nero-specific LeRobot robot package.

Important files:

- `lerobot_robot_nero/nero.py`: observation/action interface, AIRBOT pose conversion, IK, safety clamp, joint smoothing
- `scripts/policy_server_rtc.py`: RTC-enabled policy server wrapper
- `scripts/preposition.py`: move Nero to the training-start pose
- `HANDOVER_STABILIZATION_NOTES.md`: detailed debugging and stabilization record
- `DATA_CONTRACT.md`: action/state feature contract

### `pyAgxArm/`

AgileX Python SDK snapshot used for Nero and gripper communication.

### `convert_rollio_to_lerobot.py`

Dataset conversion script for rollio-format handover data.

## What Is Not Included

The repository intentionally excludes runtime artifacts and large/private dependencies:

- PI05 checkpoints and model weights,
- rollio `.deb` packages,
- `rollio_device_nero-*.whl`,
- `rollio-ng/`,
- generated logs and caches,
- LeRobot test binary fixtures such as `.safetensors` and `.bag` files.

These files should be installed or restored separately in the target runtime environment.

## Installation

The runtime was developed in a conda environment named `lerobot`.

```bash
conda activate lerobot
cd /path/to/pi05_handover

pip install -e ./lerobot
pip install -e ./pyAgxArm
pip install -e ./lerobot_robot_nero
```

Install the Nero IK/runtime wheel separately:

```bash
pip install /path/to/rollio_device_nero-1.0.0-py3-none-any.whl
```

Hardware assumptions:

- AgileX Nero connected through `socketcan`,
- CAN interface usually named `can0`,
- Nero firmware `v1.11`,
- RealSense RGB camera at `1920x1080@30`,
- PI05 checkpoint available locally.

## Running Async Inference

Move the robot to the training-start pose before policy execution:

```bash
conda activate lerobot
cd /path/to/pi05_handover

python lerobot_robot_nero/scripts/preposition.py \
  --channel can0 \
  --use-gripper
```

Start the RTC-enabled policy server:

```bash
conda activate lerobot
cd /path/to/pi05_handover

python lerobot_robot_nero/scripts/policy_server_rtc.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

Start the robot client:

```bash
conda activate lerobot
cd /path/to/pi05_handover

python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/path/to/checkpoints/050000/pretrained_model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.6 \
  --aggregate_fn_name=smooth_overlap \
  --fps=30 \
  --task="hand over the object" \
  --robot.type=nero \
  --robot.interface=socketcan \
  --robot.channel=can0 \
  --robot.bitrate=1000000 \
  --robot.firmware=v1.11 \
  --robot.use_gripper=true \
  --robot.gripper_force=1.0 \
  --robot.use_camera=true \
  --robot.camera_width=1920 \
  --robot.camera_height=1080 \
  --robot.camera_fps=30 \
  --robot.speed_percent=10 \
  --robot.airbot_aligned_action=true \
  --robot.joint_smoothing_alpha=0.3
```

## Stabilization Summary

The final stable runtime uses:

- `actions_per_chunk=50`
- `chunk_size_threshold=0.6`
- `aggregate_fn_name=smooth_overlap`
- `robot.joint_smoothing_alpha=0.3`
- RTC-enabled policy serving

Key engineering fixes:

- registered Nero as a LeRobot external robot type,
- converted PI05 end-effector pose actions into Nero joint commands through IK,
- aligned AIRBOT-frame policy actions to Nero's Pinocchio frame,
- changed joint delta limiting to reference the last sent command instead of delayed measured feedback,
- held the last safe joint target on IK failure,
- added IK diagnostics,
- forced threshold-triggered observations through server filtering with `must_go`,
- added smooth action chunk crossfade to reduce chunk-boundary discontinuities,
- added joint-space low-pass filtering to reduce small residual jitter.

The detailed troubleshooting record is available in:

[lerobot_robot_nero/HANDOVER_STABILIZATION_NOTES.md](./lerobot_robot_nero/HANDOVER_STABILIZATION_NOTES.md)

## Notes

This repository is a reproducible project snapshot rather than a clean upstream fork. It is intended to document and preserve the complete handover runtime used during development.

The upstream codebases are included as source snapshots so the Nero-specific changes can be inspected and reproduced without relying on external branches.

## Attribution

This repository includes modified source snapshots from:

- Hugging Face LeRobot, under the license in `lerobot/LICENSE`
- AgileX Robotics pyAgxArm, under the license in `pyAgxArm/LICENSE`

The Nero adapter, RTC server wrapper, stabilization changes, and handover runtime notes are project-specific additions.


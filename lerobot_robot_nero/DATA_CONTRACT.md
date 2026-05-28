# Nero × pi05 Data Contract

This document is the **single source of truth** for the data formats flowing
between rollio data collection → lerobot v3 conversion → pi05 training →
NeroRobot deployment. It was reconstructed from the actual artifacts (rollio
runtime source, conversion script, parquet contents, checkpoint stats); each
claim has a verification path.

If you change anything in any of the four stages — **re-run
`scripts/verify_data_contract.py`** before declaring it works.

---

## TL;DR

| Stage             | What is in `action[:7]`                          | Frame              |
|-------------------|--------------------------------------------------|--------------------|
| rollio IPC bus    | AIRBOT leader end-effector pose                  | AIRBOT-aligned     |
| rollio parquet    | same, recorded verbatim                          | AIRBOT-aligned     |
| lerobot v3        | same, dtype cast float64 → float32               | AIRBOT-aligned     |
| pi05 model output | same distribution, after QUANTILES denormalize   | **AIRBOT-aligned** |
| `arm.move_p`      | needs **Nero native** [x,y,z,roll,pitch,yaw]     | Nero native        |

→ **`send_action()` must call `apply_command_pose_fix()`** to convert
AIRBOT-aligned pose7 to Nero native pose7 before deriving Euler and calling
`move_p`. Without this step the arm flies to mirrored positions.

---

## Stage A — rollio data collection (closed-source `.deb`, open `rollio_device_nero` runtime)

### Frame transform on the IPC bus

`rollio_device_nero/runtime/arm.py`:

```
publish path  (state):
    Nero native pose ← FK(joint angles)
    apply_publish_pose_fix() → AIRBOT-aligned pose
    publish to IPC

subscribe path (command):
    AIRBOT-aligned pose ← IPC (from leader)
    apply_command_pose_fix() → Nero native pose
    arm.move_p(...)
```

**Consequence**: every pose visible on the IPC bus, and therefore every pose
recorded in the parquet, is **AIRBOT-aligned**. The Nero base is rotated 180°
around z relative to AIRBOT, so the transform is roughly a sign flip on x/y
plus the corresponding orientation rotation.

### What rollio writes to parquet

Verified against `output/data/chunk-000/episode_000000.parquet`
(rollio's "v2.1" format, 22 columns):

| Column                                                   | Shape | Meaning                                              |
|----------------------------------------------------------|-------|------------------------------------------------------|
| `action`                                                 | [8]   | AIRBOT leader pose [x,y,z,qx,qy,qz,qw] + gripper(m) |
| `observation.state.airbot_play__arm.end_effector_pose`   | [7]   | identical to `action[:7]` (max abs diff = 0.0)      |
| `observation.state.airbot_play__e2.parallel_position`    | [1]   | AIRBOT E2 stroke (m)                                 |
| `observation.state.agx_nero__arm.joint_position`         | [7]   | Nero joint angles (rad)                              |
| `observation.state.agx_nero__arm.end_effector_pose`      | [7]   | AIRBOT-aligned, derived from Nero FK + transform     |
| `observation.state.agx_nero__gripper.parallel_position`  | [1]   | Nero gripper width feedback (m)                      |
| (+ velocity / effort / depth — dropped during conversion)|       |                                                       |

### Gripper scale

`config.toml` declares `joint_scales = [1.8]` for the AIRBOT-E2 → Nero-gripper
pairing. Empirically, `action[7] / observation.state.airbot_play__e2.parallel_position == 1.8`
exactly. Nero's gripper saturates around 0.07 m physically, so the recorded
state range is [0, 0.07] even when commands go up to ~0.087 m.

---

## Stage B — `convert_rollio_to_lerobot.py`

**This script is a pure pass-through for poses; no coordinate transform happens.**

```
v2.1 column                                            →  v3 column            dtype
-----------------------------------------------------------------------------------------
action                                                 →  action               float32 [8]
observation.state.agx_nero__arm.joint_position[7]      ┐
observation.state.agx_nero__gripper.parallel_position[1]┴→  observation.state    float32 [8]
videos/realsense__color/episode_*.mp4                  →  observation.images.realsense_color
                                                          (uint8 HWC RGB, 1080×1920)
```

Dropped: depth, all `airbot_play__*`, all velocity/effort, Nero
`end_effector_pose` (kept only `joint_position`).

**Implication**: `observation.state` is in **joint space + gripper width**
(coordinate-free), but `action` remains in **AIRBOT-aligned cartesian frame**.
The two halves of the model live in different spaces.

---

## Stage C — pi05 training (`checkpoints/050000/pretrained_model/`)

`config.json`:

```
type: pi05
input_features:
  observation.state                      STATE   shape=[8]
  observation.images.realsense_color     VISUAL  shape=[3, 1080, 1920]
output_features:
  action                                 ACTION  shape=[8]

action_feature_names: [
  agx_nero__arm.end_pose.[0..6],
  agx_nero__gripper.parallel_mit.0,
]
normalization_mapping:
  VISUAL: IDENTITY
  STATE:  QUANTILES
  ACTION: QUANTILES
chunk_size: 50
```

QUANTILES normalization stats are stored in
`policy_preprocessor_step_2_normalizer_processor.safetensors`:

| Tensor                 | min              | max              | q01              | q99              |
|------------------------|------------------|------------------|------------------|------------------|
| `action[0]` (x)        | 0.272            | 0.676            | 0.400            | 0.623            |
| `action[1]` (y)        | -0.198           | 0.305            | -0.059           | 0.087            |
| `action[2]` (z)        | -0.076           | 0.573            | -0.017           | 0.378            |
| `action[6]` (qw)       | 0.706            | 1.000            | 0.901            | 0.999            |
| `action[7]` (gripper)  | -0.001           | 0.0887           | 0.047            | 0.0847           |
| `state[3]` (joint 4)   | 0.889            | 2.142            | 1.250            | 2.017            |
| `state[7]` (gripper)   | 0.000            | 0.0703           | 0.048            | 0.0698           |

Notable: `state[3]` (joint 4) median ≈ π/2, matches rollio's
`DISABLED_HOLD_Q = [0, 0, 0, π/2, 0, 0, 0]` — i.e., the arm rests near that
configuration when not actively driven.

The denormalization is performed automatically by `policy_postprocessor`
(loaded from `policy_postprocessor_step_0_unnormalizer_processor.safetensors`)
inside `policy_server.py`. **You never reverse-normalize manually** in the
adapter.

---

## Stage D — `NeroRobot.send_action()` (this package)

Required pipeline, in order:

```python
# 1. accept dict (from RobotClient) or array (from replay script)
a = unpack_to_8d_float32(action)

# 2. invert rollio's publish transform: AIRBOT-aligned → Nero native
pose7_native = apply_command_pose_fix(a[0:7])

# 3. extract pose7_native into pos + normalized quat
pos       = pose7_native[0:3]
quat_xyzw = pose7_native[3:7] / ||pose7_native[3:7]||

# 4. quat → XYZ extrinsic Euler (the convention move_p expects)
euler_xyz = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)

# 5. dispatch
arm.move_p([*pos, *euler_xyz])
gripper.move_gripper_m(value=a[7], force=cfg.gripper_force)
```

`observation.state` (8,) returned by `get_observation()` must be:
```
[0:7] = arm.get_joint_angles()       # Nero joints (rad)
[7]   = gripper.get_gripper_status() # parallel width (m), 0 if no gripper
dtype = float32
```

`observation.images.realsense_color` must be `uint8 HWC RGB (1080,1920,3)`.

---

## Hardware / firmware contract

* Nero firmware: **v1.11** → `NeroFW.V111` (config_nero.py default)
* CAN: `socketcan` / `can0` / `1_000_000` baud
* `pyAgxArm` API parameter is `firmeware_version` (yes, with the typo)
* RealSense D435i color stream: 1920×1080 RGB @ 30 fps
* Gripper effector: `agx_gripper`, controlled with `move_gripper_m(width, force)`,
  trained against `max_range_config ≈ 0.07 m` — re-calibrate before inference
  if the current physical max differs.

---

## How to verify everything is still consistent

```
cd lerobot_robot_nero
python3 scripts/verify_data_contract.py \
    --v21-parquet ../output/data/chunk-000/episode_000000.parquet \
    --v3-parquet  ../chunk-000/file-000.parquet \
    --checkpoint  ../checkpoints/050000/pretrained_model
```

If any assertion fails the contract is broken — fix it before running pi05.

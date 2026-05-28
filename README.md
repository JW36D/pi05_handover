# pi05_handover

这是一个面向 Nero + PI05 handover 的整理版代码仓库。

仓库名改成 `pi05_handover`，但本地文件名不改，便于你后续继续对照原始目录。

## 包含内容

- `lerobot/`：本地 LeRobot 工作副本，包含 async inference、RTC、chunk overlap 等修改
- `lerobot_robot_nero/`：Nero 机械臂适配包、RTC policy server wrapper、调参记录
- `pyAgxArm/`：AgileX Nero / 夹爪 SDK 副本
- `convert_rollio_to_lerobot.py`：rollio 数据转换脚本

## 不包含内容

为了让 GitHub 仓库更干净，这些内容不上传：

- `rollio-ng/`
- `rollio_*.deb`
- 运行日志、缓存、`__pycache__`
- 大量生成的测试/训练数据与二进制 fixture
- `checkpoints/`、`output/`、`output1/`、`chunk-000/`
- 外部依赖 wheel，例如 `rollio_device_nero-*.whl`（需要时本地单独安装）

## 目录结构

```text
pi05_handover/
├── README.md
├── .gitignore
├── convert_rollio_to_lerobot.py
├── lerobot/
├── lerobot_robot_nero/
└── pyAgxArm/
```

## 推荐安装顺序

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/pi05_handover

pip install -e ./lerobot
pip install -e ./pyAgxArm
pip install -e ./lerobot_robot_nero
```

如果你还要用 Nero 的外部 IK/runtime wheel，请再单独安装本地的：

```bash
pip install /path/to/rollio_device_nero-1.0.0-py3-none-any.whl
```

## 运行方式

详细参数和稳定化过程，见：

- [Nero handover 稳定化记录](./lerobot_robot_nero/HANDOVER_STABILIZATION_NOTES.md)
- [Nero 数据契约](./lerobot_robot_nero/DATA_CONTRACT.md)

最常用的 async inference 流程：

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/pi05_handover

# 1) policy server
python lerobot_robot_nero/scripts/policy_server_rtc.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/pi05_handover

# 2) robot client
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/developer/Documents/Hand_Over/checkpoints/050000/pretrained_model \
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

## 版本说明

这个仓库是一个“整理后的 handover 快照”，不是完整的上游镜像。

我保留了手头真正要用的源代码、适配逻辑和说明文档，同时有意排除了大量运行期/测试期/安装包文件，方便你直接在 GitHub 主页展示和后续维护。

## Attribution

本仓库包含并修改了以下第三方/上游项目的代码快照：

- LeRobot: `lerobot/`，原项目为 Hugging Face LeRobot，许可见 `lerobot/LICENSE`
- pyAgxArm: `pyAgxArm/`，原项目为 AgileX Robotics pyAgxArm，许可见 `pyAgxArm/LICENSE`

本项目的核心工作集中在 Nero robot adapter、PI05 async inference runtime、RTC policy server wrapper、action chunk overlap 稳定化、IK 安全执行链路和 handover 调参记录。

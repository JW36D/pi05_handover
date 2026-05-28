# lerobot_robot_nero

AgileX Nero 的 LeRobot 外部机器人适配包（非 ROS2 版本）。

目标是和 LeRobot 官方硬件扩展机制一致，做到和 `so101` 一样可通过 `--robot.type=nero` 被框架发现并创建。

本次迁移、PI05 推理稳定化、50 action chunk 抖动优化的详细过程见：
[HANDOVER_STABILIZATION_NOTES.md](./HANDOVER_STABILIZATION_NOTES.md)

## 这版做了什么

- 7 关节观测：`joint_1.pos ... joint_7.pos`
- 7 关节动作：`joint_1.pos ... joint_7.pos`
- 基于 `pyAgxArm` 的连接、读角度、`move_j` 下发
- 可选夹爪控制：`gripper.width_m`（走 `move_gripper_m`）
- 安全检查：
  - 必须已连接
  - 关节动作必须提供 7 维（夹爪开启时可额外带 `gripper.width_m`）
  - 每个值必须是有限数

## 与 LeRobot 集成方式（对齐官方文档）

本包遵循官方插件约定（见文档）：
- 包名以 `lerobot_robot_` 开头
- `NeroRobotConfig` 使用 `@RobotConfig.register_subclass("nero")`
- 设备类命名为 `NeroRobot`（与 Config 去掉 `Config` 后缀一致）
- `__init__.py` 导出 Config 和 Robot

官方文档：
- https://huggingface.co/docs/lerobot/integrate_hardware

## 安装

```bash
pip install -e .
```

运行前提：
- 目标机可正常使用 `pyAgxArm` 连接 Nero
- 已安装 LeRobot
- CAN 配置正常（默认 `socketcan` + `can0`）

## 配置项

`NeroRobotConfig`：
- `interface: str = "socketcan"`
- `channel: str = "can0"`
- `bitrate: int = 1000000`
- `firmware: str = "default"`
- `use_gripper: bool = False`
- `gripper_force: float = 1.0`
- `use_camera: bool = False`
- `speed_percent: int = 10`
- `execute: list[float] | None = None`（用于 `calibrate()` 时可选执行目标姿态）

## 在 LeRobot CLI 中使用

安装完成后，按你本地 LeRobot 版本的命令入口，传 `--robot.type=nero` 即可。

示例（命令名可能因版本是 `lerobot-*` 或 `python -m lerobot.*`）：

```bash
# 仅示例参数结构，具体子命令按你的 lerobot 版本替换
... --robot.type=nero \
    --robot.interface=socketcan \
    --robot.channel=can0 \
    --robot.bitrate=1000000 \
    --robot.use_gripper=true \
    --robot.gripper_force=1.0 \
    --robot.use_camera=false
```

## pi05 async 推理运行

当前交接目录里的推荐运行方式是 LeRobot async inference：一个终端跑 policy server，一个终端跑 Nero robot client。下面命令默认使用：

- conda 环境：`lerobot`
- checkpoint：`/home/developer/Documents/Hand_Over/checkpoints/050000/pretrained_model`
- CAN：`socketcan` + `can0`
- Nero 固件：`v1.11`
- RealSense RGB：`1920x1080@30`
- 任务文本：`hand over the object`（当前实测使用的任务指令）

先把机械臂移动到训练 episode 起点附近：

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero

python scripts/preposition.py --channel can0 --use-gripper
```

终端 1：启动 policy server（带 RTC）：

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero

python scripts/policy_server_rtc.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

终端 2：启动 robot client：

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero

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

注意：

- 修改过本地 LeRobot async server 后，必须重启旧的 server/client 进程；已经卡在 `Skipping observation #49 - Timestep predicted already!` 的进程不会自动恢复。
- 建议从 `/home/developer/Documents/Hand_Over/lerobot_robot_nero` 目录运行 client，避免 Python 把外层同名目录当 namespace 包，导致 LeRobot 插件注册不到 `nero`。
- `scripts/preposition.py` 的默认姿态是训练集每个 episode 第一帧的关节中位数；不要用全轨迹 normalizer q50 当启动姿态，它更像演示中途姿态，50 chunk 冷启动时更容易出大首段。
- 当前最佳实测参数是 `actions_per_chunk=50`、`chunk_size_threshold=0.6`、`aggregate_fn_name=smooth_overlap`、`robot.joint_smoothing_alpha=0.3`。这样队列剩约 30 个动作时请求下一段，旧 chunk 和新 chunk 有更长重叠区，可按位置渐变融合，明显减少 chunk 接缝处的硬切抖动。
- 本机 LeRobot client 已把首次低队列 observation 标成 `must_go=True`，避免 server 因 `Observation too similar` 过滤提前请求。用 `actions_per_chunk=50 --chunk_size_threshold=0.6` 时，正常情况下日志应能看到接近 `QUEUE SIZE: 30 (Must go: True)` 的触发点，server 推理 timestep 间隔通常接近 19 帧。
- 不建议把 `chunk_size_threshold` 设成 `0` 作为常规参数。它会等 action 队列空了才请求下一段，基本没有 overlap，`smooth_overlap` 也就无法平滑新旧 chunk 的接缝。
- 看到 `Joint delta clamped` 不等于程序卡死，它表示单帧 IK 目标超过安全限幅。当前实现按上一帧已下发 joint target 逐步限幅，能从较远初始姿态连续爬向目标；如果初始位姿仍明显不稳，先用 `actions_per_chunk=20` 或把 `chunk_size_threshold` 提到 `0.9` 做保守启动。
- 如果仍有小抖，可调 `--robot.joint_smoothing_alpha` 做关节空间低通；值越大越平滑但响应越慢，建议先在 `0.2~0.3` 内试，不要一开始超过 `0.35`。
- 这个 checkpoint 的 `config.json` 里 `compile_model=true`。本机 LeRobot async server 已默认在推理时关闭 `torch.compile`，避免 PI05 第二段推理触发 CUDA graph capture 错误后卡在同一个 observation。只有要专门压测编译性能时才加 `LEROBOT_ASYNC_DISABLE_TORCH_COMPILE=0`。
- 如果要临时关闭 RTC，可在启动 server 时加环境变量：`NERO_RTC_DISABLED=1 python scripts/policy_server_rtc.py ...`
- 上机器人前可先跑 `python scripts/smoke_test_nero.py --channel can0` 做连接 dry-run。

## calibrate() 语义与 execute

在 LeRobot 里，`calibrate()` 的作用通常是：
- 建立/加载电机零位与可运动范围
- 把原始电机读数映射到稳定可复用的动作空间

Nero 这个 v1 适配里，默认没有单独校准流程，因此：
- 不传 `execute` 时，`calibrate()` 只标记已校准（no-op）

如果你希望在 `lerobot-calibrate` 时顺带执行一个 7 关节目标姿态，可传：

```bash
lerobot-calibrate \
  --robot.type=nero \
  --robot.interface=socketcan \
  --robot.channel=can0 \
  --robot.bitrate=1000000 \
  --robot.execute='[0.02, -0.57, 0.013, 2.14, 0.02, -0.06, 0.05]'
```

说明：
- `execute` 会在 `calibrate()` 中执行
- 执行路径会直接调用 `move_j` 到目标姿态
- 当 `use_gripper=true` 且 `execute` 给 8 维（前7维关节 + 第8维夹爪宽度）时，会额外调用一次 `move_gripper_m`

## Smoke Test（硬件连通性）

默认 dry-run（不下发运动）：

```bash
python3 scripts/smoke_test_nero.py --channel can0
```

下发很小动作：

```bash
python3 scripts/smoke_test_nero.py --channel can0 --joint 7 --step 0.01 --execute
```

带夹爪宽度指令：

```bash
python3 scripts/smoke_test_nero.py \
  --channel can0 \
  --use-gripper \
  --gripper-force 1.0 \
  --gripper-width 0.04 \
  --execute
```

也支持直接给 7 关节绝对目标：

```bash
python3 scripts/smoke_test_nero.py \
  --channel can0 \
  --absolute-target 0.0 -0.5 0.0 2.1 0.0 -0.06 0.05 \
  --execute
```

## 当前限制

- camera 目前走 RealSense RGB（`pyrealsense2`），推理 checkpoint 期望 `1920x1080@30`
- 夹爪当前仅支持宽度控制（`move_gripper_m`）
- 不支持 TCP/末端位姿观测与动作
- 不含 ROS2（按需求刻意不引入）

## TODO

- camera 观测特征
- gripper 角度模式/更丰富状态字段
- TCP pose / 速度控制
- 与 LeRobot 处理器链路（processor）联动

## 夹爪 API 参考

- https://github.com/agilexrobotics/pyAgxArm/blob/master/docs/effector/agx_gripper/agx_gripper_api.md

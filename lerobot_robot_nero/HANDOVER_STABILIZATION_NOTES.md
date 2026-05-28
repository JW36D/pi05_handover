# Nero + LeRobot PI05 Handover Stabilization Notes

本文档记录本次将 Nero 机械臂项目从旧工作站迁移到当前工作站后，完成环境恢复、LeRobot 外部机器人接入、PI05 async inference 跑通、50 action chunk 稳定化、抖动优化的完整过程。

目标用途：

- 给后续维护者说明当前工程为什么这样设计。
- 给个人学习复盘留下足够细的技术上下文。
- 给简历/面试准备提供可抽象的工程经历素材。

相关目录：

- Nero 外部机器人适配包：`/home/developer/Documents/Hand_Over/lerobot_robot_nero`
- 本地修改过的 LeRobot：`/home/developer/Documents/Hand_Over/lerobot`
- 运行环境：conda env `lerobot`
- 推理 checkpoint：`/home/developer/Documents/Hand_Over/checkpoints/050000/pretrained_model`
- Nero 私有依赖 wheel：`/home/developer/Documents/Hand_Over/rollio_device_nero-1.0.0-py3-none-any.whl`

## 1. 当前最佳运行方式

先在终端 1 启动 policy server：

```bash
conda activate lerobot
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero

python scripts/policy_server_rtc.py \
  --host=127.0.0.1 \
  --port=8080 \
  --fps=30
```

再在终端 2 启动 robot client：

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

当前实测最佳参数组合：

- `actions_per_chunk=50`
- `chunk_size_threshold=0.6`
- `aggregate_fn_name=smooth_overlap`
- `robot.joint_smoothing_alpha=0.3`
- server 使用 `scripts/policy_server_rtc.py`，保持 RTC 开启

最近一次最佳日志表现：

- 运行窗口约 `79s`
- client 收到 action chunk：`124` 次
- server 推理：`124` 次
- server 推理间隔基本稳定为 `19` 帧一次
- action queue 中位数约 `26`，平均约 `25.93`
- `1004` 个 queue 采样里 `985` 个 `>=20`
- `Joint delta clamped` 只有 `1` 次
- `IK failed` 只有 `31` 次，且集中在最后约 `2s`
- 没有 `Control loop overrun`

这说明主要抖动源已经从“系统性 chunk 接缝/队列掉空”降级为“局部动作段 IK 边界问题”。

## 2. 系统链路总览

本项目实际运行链路如下：

```text
RealSense + Nero joint/gripper state
        |
        v
LeRobot RobotClient.get_observation()
        |
        v
PolicyServer + PI05 policy
        |
        v
model outputs action chunk: shape [50, 8]
        |
        v
client action queue + overlap aggregation
        |
        v
NeroRobot.send_action(action[8])
        |
        v
AIRBOT-frame end pose -> Nero Pinocchio frame
        |
        v
rollio_device_nero IK: pose7 -> 7 joint angles
        |
        v
joint delta safety clamp
        |
        v
joint low-pass smoothing
        |
        v
pyAgxArm move_j + gripper command
```

这里有两个容易混淆的点。

第一，PI05 输出的 action 不是关节角，而是末端位姿加夹爪：

```text
action[0:3] = end-effector position, metre
action[3:7] = quaternion, qx qy qz qw
action[7]   = gripper width / parallel command
```

第二，模型 action 的末端位姿是 AIRBOT 对齐坐标系。Nero 真机 IK 使用的是 Nero Pinocchio 模型坐标系，因此发送前必须经过 `rollio_device_nero.airbot_aligned_pose.apply_command_pose_fix()` 做坐标变换。

关键实现位置：

- Nero robot config：`lerobot_robot_nero/config_nero.py`
- Nero action/IK/hardware dispatch：`lerobot_robot_nero/nero.py`
- async client queue/overlap：`lerobot/src/lerobot/async_inference/robot_client.py`
- async client/server config：`lerobot/src/lerobot/async_inference/configs.py`
- RTC server wrapper：`scripts/policy_server_rtc.py`

## 3. 环境和插件接入问题

### 3.1 从旧工作站迁移后的环境恢复

现象：

- 当前工作站没有原来的 Python/conda 环境。
- LeRobot、pyAgxArm、rollio_device_nero、Nero 外部 robot package 都需要重新装。

处理：

- 创建 conda 环境 `lerobot`。
- 在该环境中安装本地 LeRobot。
- 安装 `pyAgxArm`。
- 安装私有 wheel：`rollio_device_nero-1.0.0-py3-none-any.whl`。
- 以 editable 方式安装 `lerobot_robot_nero`：

```bash
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero
pip install -e .
```

涉及知识：

- conda env 用于隔离 CUDA/PyTorch/机器人依赖。
- editable install 让本地源码改动立刻生效，适合硬件适配包开发。
- LeRobot 的外部机器人扩展机制依赖 Python 包导入和 config subclass 注册。

### 3.2 `--robot.type=nero` invalid choice

现象：

```text
robot_client.py: error: argument --robot.type: invalid choice: 'nero'
(choose from so100_follower, so101_follower, ...)
```

根因：

LeRobot CLI 启动时没有发现并导入 `lerobot_robot_nero` 插件，因此 `NeroRobotConfig` 没有注册到 robot config registry。

解决：

在 `lerobot_robot_nero/config_nero.py` 中：

```python
@RobotConfigBase.register_subclass("nero")
@dataclass
class NeroRobotConfig(RobotConfigBase):
    ...
```

在 `lerobot_robot_nero/__init__.py` 中导出 config 和 robot 类，使插件包被导入后可被 LeRobot 发现。

同时运行时从项目目录启动：

```bash
cd /home/developer/Documents/Hand_Over/lerobot_robot_nero
python -m lerobot.async_inference.robot_client ... --robot.type=nero
```

避免 Python 把外层同名目录当 namespace package，导致导入路径异常。

涉及知识：

- Python package discovery
- editable install
- LeRobot hardware integration plugin convention
- argparse/draccus dataclass config 注册

## 4. policy server 看起来卡住的问题

现象：

```text
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080 --fps=30
INFO ... PolicyServer started on 127.0.0.1:8080
```

之后看起来不动。

判断：

这不是卡死，也不是 tokenizer 下载问题。policy server 本来就是长驻 gRPC 服务，启动后等待 robot client 发送 observation 和 policy setup。没有 client 连接时，它不会继续打印推理日志。

涉及知识：

- server/client async inference 架构
- gRPC server 长驻进程
- policy server 等待 observation 而不是主动推理

## 5. PI05 启动后 “已杀死” 的问题

现象：

```text
The PI05 model is a direct port of the OpenPI implementation.
...
已杀死
```

根因判断：

这是系统 OOM kill 的典型表现，不是 tokenizer 问题。PI05 模型加载和初始化时会占用较多内存/显存，如果物理内存和 swap 不够，Linux OOM killer 会直接杀进程，Python 没机会抛异常。

解决方向：

- 加大 swap。
- 避免不必要的 compile/graph capture。
- 确认只启动一个 server。
- 观察 `dmesg`/系统监控确认 OOM kill。

涉及知识：

- Linux OOM killer
- swap 是磁盘上的虚拟内存后备区，能降低进程被 OOM kill 的概率
- 大模型加载时 CPU RAM、GPU VRAM、page cache 都可能成为瓶颈

## 6. checkpoint `compile_model=true` 和 async inference 的冲突

现象：

PI05 checkpoint 的 `config.json` 中带有 `compile_model=true`。在 async inference 中，早期运行会在第二段或后续推理出现 CUDA graph / torch compile 相关异常或卡住。

解决：

本地 LeRobot async server 里默认在异步推理时关闭 checkpoint 请求的 `torch.compile`，日志中会出现类似：

```text
Checkpoint requests torch.compile (...); disabling it for async inference.
```

只在明确压测编译性能时才用环境变量恢复：

```bash
LEROBOT_ASYNC_DISABLE_TORCH_COMPILE=0
```

涉及知识：

- `torch.compile` 对动态 shape、动态图控制流、异步 server 长跑场景可能不稳定
- CUDA graph capture 对内存地址、执行图稳定性有要求
- 机器人在线控制优先考虑确定性和稳定性，而不是单次 benchmark 性能

## 7. 数据契约：为什么模型 action 要走 IK

### 7.1 action 不是关节角

最开始容易误解为：

```text
模型输出 action -> 直接发送给机械臂关节
```

但本项目 PI05 checkpoint 的 action 实际是：

```text
[end_pose x y z qx qy qz qw, gripper]
```

因此正确链路是：

```text
模型输出末端位姿
    -> 坐标系转换
    -> IK 反解到关节角
    -> move_j 发送到 Nero
```

相关代码在 `NeroRobot.send_action()`：

```python
native7 = apply_command_pose_fix([float(v) for v in a[0:7]])
joint_targets = self._solve_ik_to_joints(native7, airbot_pose7=[float(v) for v in a[0:7]])
self._arm.move_j(joint_targets)
```

### 7.2 AIRBOT pose 和 Nero pose

训练数据里的 action 和 AIRBOT leader 末端位姿一致，而 Nero 真机 base 相对 AIRBOT 数据采集 rig 有坐标对齐差异。rollio runtime 中通过 pose fix 函数处理这个差异。

因此增加配置：

```python
airbot_aligned_action: bool = True
```

默认开启：

```bash
--robot.airbot_aligned_action=true
```

涉及知识：

- 机器人坐标系
- 末端位姿 action
- 四元数表示姿态，当前顺序是 `qx, qy, qz, qw`
- imitation learning checkpoint 的 feature schema 必须和训练数据完全一致
- 同一台机械臂不同 runtime 对 TCP frame 的约定可能不同

## 8. IK 无解和安全处理

### 8.1 为什么会有 IK failed

现象：

运行中看到：

```text
IK failed, holding last joint target: IK did not converge ...
```

可能原因：

- 模型输出的末端位姿接近工作空间边界。
- 姿态四元数对应的末端方向在当前 null-space 分支难以到达。
- 机械臂实际状态和模型预期状态偏离，导致下一段目标 out-of-distribution。
- release、抓取、放置附近视觉/动作分布变化大，模型输出局部不连续。
- IK 是非线性问题，7 自由度冗余机械臂存在多个解分支。

### 8.2 当前 IK 实现

代码位置：`lerobot_robot_nero/nero.py` 的 `_solve_ik_to_joints()`。

使用 rollio 的 Nero Pinocchio 模型：

```python
from rollio_device_nero.gravity import NeroModel
from rollio_device_nero import ik as nero_ik
```

关键点：

- 使用 `NeroModel(with_gripper=True)`，因为训练数据是在带夹爪 TCP 下记录的。
- 使用当前测量关节 `q_meas` 作为 null-space anchor。
- 使用上一帧 IK 目标 `_latest_ik_target` 作为 warm start。
- IK 参数从原来的高精度/短迭代调整为更适合在线策略输出：

```python
q_target, converged, err = nero_ik.solve(
    self._nero_model,
    list(target_pose7),
    q0=q_seed,
    q_anchor=q_meas,
    tol=5e-3,
    max_iter=200,
)
```

这里 `tol=5e-3` 约等于允许 5mm / 5mrad 量级残差。对真实机械臂在线控制来说，这比 `1e-4` 更实际，可以避免很多“数学上没完全收敛但工程上可接受”的误报。

### 8.3 IK 失败时为什么 hold last target

早期如果 IK 失败直接发送 unconverged solution，会有安全风险：非收敛 IK 解可能跳到不可预期关节配置。

现在逻辑是：

```python
try:
    joint_targets = self._solve_ik_to_joints(...)
except RuntimeError:
    if self._last_sent_joint_target is not None:
        joint_targets = self._last_sent_joint_target
    else:
        joint_targets = measured_joints
```

也就是 IK 失败时保持上一条安全发送过的关节目标。

同时设置连续失败保护：

```python
MAX_CONSECUTIVE_IK_FAILURES = 30
```

30Hz 下约等于连续 1 秒 IK 失败。如果超过这个阈值，说明策略已经明显跑偏或机械臂状态异常，应停止而不是无限 hold。

涉及知识：

- IK 非收敛解不能当成安全解
- 在线机器人控制要优先保证 fail-safe
- hold-last-command 是常见的短时异常处理策略
- 连续失败阈值用于区分偶发噪声和系统性失控

### 8.4 IK diagnostics

为了后续分析失败原因，加入了 IK 失败记录：

- 总调用次数
- 总失败次数
- failure rate
- 失败时 target pose
- measured joints
- seed joints
- unconverged q_target

断开连接时会打印：

```text
IK summary: 3507 calls, 116 failures (3.31%).
IK diagnostics dumped to /tmp/nero_ik_failures_xxx.json
```

这为后续判断“目标超工作空间”还是“姿态/分支问题”提供数据。

## 9. 关节限幅：为什么需要，怎么实现

### 9.1 为什么要加关节限幅

模型输出是末端位姿，IK 反解到关节空间时可能出现较大跳变。原因包括：

- 末端位姿小变化在奇异点附近可能对应较大关节变化。
- 7 自由度冗余机械臂有多个关节解分支。
- 模型 chunk 切换时，新 chunk 预测可能和旧 chunk 不完全一致。
- 初始姿态和训练数据第一帧不够接近时，第一段动作可能跨度较大。

如果每一帧都直接把 IK 解发送给硬件，可能导致突然大幅运动。因此加了每 tick 关节变化限幅。

### 9.2 当前限幅实现

代码位置：`NeroRobot._solve_ik_to_joints()`。

```python
MAX_JOINT_DELTA_RAD = 0.20

q_ref = self._last_sent_joint_target if self._last_sent_joint_target is not None else q_meas
delta = np.clip(
    q_target - q_ref,
    -self.MAX_JOINT_DELTA_RAD,
    self.MAX_JOINT_DELTA_RAD,
)
q_clamped = q_ref + delta
```

`0.20 rad` 约等于 `11.5 deg`。在 30Hz 控制下，这是一个安全上限，而不是平滑器。

日志：

```text
Joint delta clamped (max requested 0.203 rad > 0.200 rad limit).
```

表示当前 IK 目标相对上一帧发送命令的最大单关节差值超过了 0.20 rad，所以被截断。

### 9.3 为什么参考上一帧发送命令，而不是测量反馈

早期 `actions_per_chunk=50` 时出现：

```text
机械臂动一下就不动
Joint delta clamped ...
```

根因是：如果每次都用 `q_meas` 作为 clamp reference，而 Nero 的反馈频率/延迟跟 30Hz 命令循环不完全同步，就会反复把目标截到“第一小步”，机械臂还没来得及反馈到新位置，下一帧又从旧测量位置截一遍，表现为长 chunk 冷启动时移动一下就停住。

修复：

使用 `_last_sent_joint_target` 作为下一次 clamp reference：

```python
q_ref = self._last_sent_joint_target if self._last_sent_joint_target is not None else q_meas
```

这样即使硬件反馈滞后，命令轨迹本身仍能连续向目标推进。

涉及知识：

- commanded state 和 measured state 的区别
- 控制回路中反馈延迟会造成 aliasing
- slew-rate limiting 应该基于命令轨迹推进，而不是每帧被慢反馈拉回

## 10. `actions_per_chunk=50` 为什么之前更容易出问题

现象：

- `actions_per_chunk=20` 可以正常跑。
- `actions_per_chunk=50` 从标准初始位姿启动时，机械臂动一下后不动，频繁出现 `Joint delta clamped`。
- 如果从某次中途暂停的关节角启动，50 又可能正常。

原因综合判断：

- 50 chunk 更长，第一段动作覆盖更远未来。
- 如果初始姿态和训练 episode 第一帧不够接近，长 chunk 的前几帧可能已经进入较大末端位姿变化。
- IK 目标相对当前关节分支跨度变大，更容易触发 clamp。
- 使用测量反馈做 clamp reference 时，会放大“动一下就停”的现象。

解决：

- 使用训练 episode 第一帧附近的 preposition，而不是全轨迹中位数。
- clamp reference 改为 `_last_sent_joint_target`。
- 后续再通过 overlap/RTC/smoothing 降低 chunk 接缝跳变。

涉及知识：

- 模仿学习策略对初始状态分布敏感
- action horizon 越长，越依赖稳定的 receding-horizon 执行
- “突然暂停的中途关节角能跑”说明策略在某些轨迹分布内是稳定的，问题主要出在冷启动状态和动作接缝

## 11. async inference 队列和 `chunk_size_threshold`

### 11.1 参数含义

`actions_per_chunk=50` 表示 server 每次返回最多 50 帧未来动作。

`chunk_size_threshold` 决定 client 什么时候请求下一段 chunk：

```python
return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold
```

如果：

```text
actions_per_chunk = 50
chunk_size_threshold = 0.6
```

那么当本地队列剩余动作数小于等于 `50 * 0.6 = 30` 时，就开始发新的 observation 请求下一段动作。

这会带来约 30 帧重叠区，让新旧 chunk 有足够空间做平滑融合。

### 11.2 为什么 `chunk_size_threshold=0` 不推荐

如果 threshold 是 0，client 会等 action queue 接近空了才请求下一段。这样会导致：

- overlap 区几乎没有。
- `smooth_overlap` 没有足够重叠动作可平滑。
- server 推理延迟期间 client 可能无动作可执行。
- 机械臂更容易在 chunk 接缝处停顿或抖动。

实测坏日志：

- `QUEUE SIZE: 0` 出现 `864` 次
- IK failed `242` 次
- clamp `76` 次

### 11.3 为什么 `0.6` 比 `0.2` 更平滑

`0.2` 时：

```text
50 * 0.2 = 10
```

只剩约 10 帧时才请求下一段。考虑 server 推理约 0.21s，30Hz 下约 6 帧，再加网络/序列化/调度，实际 overlap 不算长。

`0.6` 时：

```text
50 * 0.6 = 30
```

更早请求下一段，overlap 区显著变长。`smooth_overlap` 有更多帧把 old chunk 平滑过渡到 new chunk，因此接缝抖动明显降低。

实测对比：

| 参数 | Received chunks | Queue 0 | Queue mean | IK failed | Clamp |
| --- | ---: | ---: | ---: | ---: | ---: |
| threshold=0 | 129 | 864 | 0.51 | 242 | 76 |
| threshold=0.2 + smooth_overlap + alpha=0.2 | 99 | 20 | 6.72 | 116 | 14 |
| threshold=0.6 + smooth_overlap + alpha=0.3 | 124 | 19 | 25.93 | 31 | 1 |

涉及知识：

- receding-horizon control
- action queue buffering
- inference latency compensation
- overlap 越长，轨迹拼接越容易平滑

## 12. server 过滤 observation 和 `must_go`

### 12.1 原问题

早期出现周期性大抖，大约每 1.5 到 1.8 秒一次，同时日志有 `Joint delta clamped`。

分析时发现，client 虽然在 queue 低于 threshold 时发送 observation，但 server 会因为 observation 与上一帧太相似而过滤掉：

```text
Skipping observation #... - Observation too similar to last obs predicted!
```

结果是：

- client 以为自己提前请求了下一段。
- server 实际没有推理新 chunk。
- queue 继续下降甚至掉空。
- 下一段动作变成硬接，导致周期性大抖。

### 12.2 修复：首次低队列 observation 强制 must_go

在 client 收到新 actions 后：

```python
self.must_go.set()
```

下一次 queue 低于 threshold 时发 observation：

```python
observation.must_go = self.must_go.is_set()
...
if observation.must_go:
    self.must_go.clear()
```

这样每收到一段 chunk 后，下一次真正需要补 chunk 的 observation 会带上 `must_go=True`，让 server 不因为“太相似”而过滤它。

效果：

- server 推理 timestep 不再固定卡在不合适的边界。
- action queue 不再长期掉到 0。
- 周期性大抖明显消失。

涉及知识：

- 去重过滤在控制系统里可能和补偿机制冲突
- must-go 是控制流信号，不是模型输入
- 需要区分“节省推理”的过滤逻辑和“保证控制连续性”的硬约束

## 13. overlap aggregation：`weighted_average`、`conservative`、`smooth_overlap`

### 13.1 为什么需要 overlap aggregation

在 async inference 中，server 每次返回一段未来动作。例如：

```text
old chunk: action timestep 100..149
new chunk: action timestep 120..169
```

那么 timestep `120..149` 同时有 old 和 new 两个预测。

如果直接用 new 覆盖 old，轨迹可能在 timestep 120 突然跳变。

如果直接保留 old，又会降低重规划响应。

所以需要 overlap aggregation：对同一 timestep 的 old/new action 做融合。

### 13.2 固定加权策略

已有策略：

```python
weighted_average = 0.3 * old + 0.7 * new
conservative    = 0.7 * old + 0.3 * new
average         = 0.5 * old + 0.5 * new
latest_only     = new
```

这些策略的共同点是权重固定。问题是，在新 chunk 刚接入的第一帧就给 new 很高权重，仍可能产生突变。

### 13.3 `smooth_overlap` 的实现

新增 `smooth_overlap`，代码位置：

- `lerobot/src/lerobot/async_inference/configs.py`
- `lerobot/src/lerobot/async_inference/robot_client.py`

核心思想：

在 overlap 区内，new chunk 的权重从小到大平滑增长。

权重函数：

```python
position = (overlap_index + 1) / (overlap_count + 1)
new_weight = position * position * (3.0 - 2.0 * position)
```

这是 smoothstep 曲线：

```text
f(x) = x^2 * (3 - 2x)
```

特点：

- 开头斜率接近 0，避免刚接入就跳变。
- 中间平滑过渡。
- 结尾斜率接近 0，避免到新 chunk 末尾时出现突兀变化。

融合：

```python
blended_action = (1.0 - new_weight) * old_action + new_weight * new_action
```

所以 `smooth_overlap` 和 `weighted_average=0.3*old+0.7*new` 的区别是：

- `weighted_average` 每个 overlap timestep 都固定 70% new。
- `smooth_overlap` 在 overlap 开头几乎用 old，中间逐渐混合，末尾几乎用 new。

这正好适合 chunk 接缝问题。

涉及知识：

- trajectory blending
- smoothstep interpolation
- fixed-ratio filter vs position-dependent crossfade
- 在 action 层平滑可以减少 IK 前的目标位姿不连续

## 14. RTC：Real-Time Chunking

### 14.1 为什么 upstream policy_server 不够

checkpoint 虽然是 PI05/OpenPI port，但 upstream `policy_server.py` 默认调用：

```python
policy.predict_action_chunk(observation)
```

RTC 需要额外传入：

- `prev_chunk_left_over`
- `inference_delay`

否则即使 config 里启用了 RTC processor，它也拿不到上一段剩余动作和推理延迟信息，实际效果接近 no-op。

### 14.2 当前 RTC wrapper

使用 `scripts/policy_server_rtc.py` 替代 upstream server。

它做两件事：

第一，模型加载后注入 RTC config：

```python
cfg.rtc_config = RTCConfig(
    enabled=True,
    execution_horizon=10,
    max_guidance_weight=10.0,
    prefix_attention_schedule=RTCAttentionSchedule.EXP,
)
self.policy.init_rtc_processor()
```

第二，每次 `_get_action_chunk()` 时估计上一段剩余 chunk：

```python
elapsed = time.monotonic() - self._rtc_chunk_returned_at
consumed = int(round(elapsed * fps))
tail = self._rtc_prev_chunk[consumed:]
kwargs["prev_chunk_left_over"] = tail
```

并估计推理延迟：

```python
kwargs["inference_delay"] = int(round(self._rtc_inference_duration * fps))
```

然后调用：

```python
chunk = self.policy.predict_action_chunk(observation, **kwargs)
```

### 14.3 RTC 和 overlap 的区别

RTC 发生在 model/policy 内部，用上一段剩余动作和推理延迟影响下一段 chunk 的生成，属于“模型推理时的时间一致性约束”。

overlap aggregation 发生在 client action queue，用 old/new chunk 对同一 timestep 的输出做融合，属于“模型输出后的轨迹拼接”。

二者不是重复关系：

```text
RTC: 让新 chunk 本身更考虑旧 chunk 和延迟。
overlap: 即使新旧 chunk 仍有差异，也在执行前做平滑拼接。
```

涉及知识：

- real-time chunking
- inference delay compensation
- receding horizon policy
- server 不能直接知道 client 消耗了多少动作，只能用 wall-clock 估计

## 15. 关节命令低通滤波 `robot.joint_smoothing_alpha`

### 15.1 它是什么

`joint_smoothing_alpha` 是 IK 之后、发送 `move_j` 之前的关节空间低通滤波。

代码位置：

- `config_nero.py`
- `nero.py` 的 `_smooth_joint_targets()`

公式：

```python
smoothed = alpha * previous_smoothed + (1.0 - alpha) * current_target
```

例如：

```text
alpha = 0.3
smoothed = 0.3 * 上一帧平滑命令 + 0.7 * 当前关节目标
```

### 15.2 它不是什么

它不是关节限幅。

关节限幅是：

```python
MAX_JOINT_DELTA_RAD = 0.20
```

限幅是硬安全约束：

```text
单帧最多允许每个关节变化 0.20 rad
```

低通滤波是软平滑：

```text
降低连续命令中的高频抖动，但会引入一点滞后
```

### 15.3 为什么它和 `smooth_overlap` 不重复

`smooth_overlap`：

- 发生在 IK 之前。
- 平滑的是模型输出的末端位姿 action。
- 主要解决 chunk 交界处的轨迹跳变。

`joint_smoothing_alpha`：

- 发生在 IK 之后。
- 平滑的是最终关节角命令。
- 主要解决 IK 非线性、关节空间小跳变、硬件位置控制引起的小抖动。

因此它们互补。

涉及知识：

- low-pass filter
- exponential moving average
- IK 前的 task-space smoothing 和 IK 后的 joint-space smoothing 是两层不同问题
- alpha 越大越平滑，但响应越慢

当前经验：

- `alpha=0.2` 已能改善小抖。
- `alpha=0.3` 配合 threshold=0.6 表现目前最好。
- 不建议一开始超过 `0.35`，否则动作会明显滞后，甚至影响抓取时机。

## 16. 日志诊断方法

### 16.1 关键日志指标

判断是否稳定，主要看：

- `QUEUE SIZE: 0` 次数
- queue 中位数/均值/最大值
- `Received actions on device`
- server `Running inference for observation #...` 的 timestep 间隔
- `Joint delta clamped`
- `IK failed`
- `Control loop overrun`
- server inference time p50/p95/max

### 16.2 典型含义

`QUEUE SIZE: 0` 多：

- client action queue 掉空。
- chunk 接缝没有 overlap。
- 容易停顿/抖动。

server inference timestep 间隔固定接近 chunk 尾部：

- 说明太晚请求下一段。

`Joint delta clamped` 多：

- IK 解相邻帧跨度大。
- 初始姿态不佳、chunk 接缝不连续、模型输出不连续或接近奇异点。

`IK failed` 多：

- 模型输出 pose 局部不可达或 IK 分支不稳定。
- 如果集中在任务末尾/release 附近，可能是该动作段数据分布或策略输出最不稳定。

`Control loop overrun`：

- control loop 超过目标周期。
- 需要检查相机、CAN、IK、日志量、CPU/GPU 负载。

## 17. 本次已解决问题清单

### 17.1 环境迁移

问题：

- 新工作站缺少环境。
- 本地 LeRobot、pyAgxArm、rollio_device_nero、外部 robot package 未安装。

解决：

- 创建 conda env `lerobot`。
- 安装本地 LeRobot 和 Nero plugin。
- 安装 `rollio_device_nero` wheel。

### 17.2 server 启动后不动

问题：

- server 打印 started 后不继续。

解决：

- 确认这是正常等待 client observation，不是卡死。

### 17.3 `nero` 机器人类型无法识别

问题：

- `--robot.type=nero` invalid choice。

解决：

- 安装并导入 `lerobot_robot_nero` 外部插件。
- 在 config 中注册 `@RobotConfig.register_subclass("nero")`。

### 17.4 PI05 进程被杀

问题：

- 打印 OpenPI port 信息后 `已杀死`。

解决：

- 判断为 OOM/swap 问题，不是 tokenizer。
- 加大 swap 并避免不必要 compile。

### 17.5 `actions_per_chunk=50` 冷启动动一下就停

问题：

- 20 chunk 可跑，50 chunk 从初始位姿会触发 clamp 后停止。

解决：

- clamp reference 从 `q_meas` 改为 `_last_sent_joint_target`。
- 使用训练 episode 起点附近 preposition。

### 17.6 IK failed 导致不安全动作风险

问题：

- IK 无解时不能发送 unconverged target。

解决：

- IK failed 时 hold last safe joint target。
- 连续失败超过 30 帧则 abort。
- 添加 IK diagnostics dump。

### 17.7 周期性大抖

问题：

- 约 1.5 到 1.8 秒一次大抖。
- server 过滤 threshold-triggered observation，导致队列补充太晚。

解决：

- client 收到 action 后设置 `must_go`。
- 下一次低队列 observation 强制通过 server filter。

### 17.8 chunk 接缝抖动

问题：

- old/new chunk 切换时模型输出不完全一致。

解决：

- 新增 `smooth_overlap`，按 overlap 位置动态 crossfade。
- 将 `chunk_size_threshold` 调到 `0.6`，扩大 overlap 区。

### 17.9 小抖动

问题：

- 周期性大抖解决后仍有持续小抖。

解决：

- 增加 `robot.joint_smoothing_alpha`，在 IK 后做关节命令低通。
- 当前推荐 `0.3`。

## 18. 当前未完全解决的问题

### 18.1 任务末尾 IK 失败仍会集中出现

最新最佳日志中，IK failed 只剩 `31` 次，集中在最后 `18:49:43-18:49:44`。

可能原因：

- 放置/松爪/release 附近模型输出更不连续。
- 末端位姿接近工作空间边界。
- 某些姿态方向在当前 null-space 分支下难以收敛。
- 夹爪打开前后视觉和动作条件变化较大。

后续方向：

- 分析 `/tmp/nero_ik_failures_*.json` 中的 target pose 分布。
- 对 action pose 做 workspace clamp 或 pose projection。
- 对 release 附近单独做 gripper/action 解耦。
- 使用更接近硬件控制方式的 MIT mode 或 velocity/impedance mode。
- 如果允许重新训练/finetune，补充 release 附近数据或加入 action smoothness 正则。

### 18.2 当前仍是 `move_j` 位置控制

rollio runtime 本身使用更高频、更适合连续控制的方式。当前适配为了兼容 LeRobot 和 pyAgxArm，使用 30Hz 下发 `move_j`。

这能跑通并稳定，但不是最理想的底层控制形式。

后续方向：

- 研究 rollio 的 MIT mode 控制接口。
- 将 PI05 输出转换成更合适的低层控制命令。
- 让硬件侧控制器承担轨迹插值和阻抗控制。

### 18.3 task-space smoothing 尚未加入

现在有：

- chunk overlap action blending
- joint-space smoothing

但还没有对末端位姿做显式 task-space smoothing，例如：

- position EMA
- quaternion slerp
- workspace projection
- velocity/acceleration limit

如果后续仍出现局部末端位姿跳变，可以考虑在 IK 前加入 task-space smoother。但这要非常谨慎，因为平滑四元数和末端轨迹可能改变策略意图，影响抓取精度。

## 19. 可用于简历的工程提炼

可以提炼为：

```text
完成 LeRobot + PI05/OpenPI 模型在 AgileX Nero 机械臂上的迁移部署与稳定化：
实现 Nero 外部机器人插件、末端位姿到关节控制的 IK 执行链路、AIRBOT/Nero 坐标系对齐、IK 失败保护、关节安全限幅、action chunk overlap 平滑、RTC 推理接入和关节空间低通滤波。
通过日志驱动调参将 50-step action chunk 运行中的 queue starvation、周期性抖动和 IK/clamp 频发问题显著降低，最终实现稳定的 handover 推理执行。
```

更技术向的版本：

```text
Built a LeRobot external robot adapter for AgileX Nero and integrated a PI05/OpenPI checkpoint for real-time handover. Implemented the full runtime path from camera/joint observations to action chunk inference, AIRBOT-to-Nero pose conversion, Pinocchio-based IK, safety slew-rate limiting, IK failure hold/abort logic, RTC-enabled policy serving, smooth overlap chunk blending, and joint-space low-pass filtering. Diagnosed async queue starvation and observation filtering via logs, reducing action queue underruns and joint clamp/IK failures under 50-step chunks.
```

## 20. 关键提交

Nero package：

- `429126a Capture Nero handover state`
- `d329513 Stabilize Nero 50-chunk handover`
- `d4d0de8 Document Nero async inference tuning`
- `8e8a3fd Add Nero joint smoothing`

LeRobot：

- `2da1ad2e Stabilize async inference for Nero`
- `7b5c3678 Stabilize async chunk overlap`
- `64f4b6c5 Add smooth overlap chunking`

## 21. 术语速查

Action chunk：

- 模型一次推理输出的一段未来动作序列。

Receding horizon：

- 每次只执行未来动作的一部分，同时不断根据最新 observation 重规划。

Overlap：

- 旧 chunk 尚未执行完，新 chunk 已经回来，两者对同一未来 timestep 都有预测。

`chunk_size_threshold`：

- 本地 action queue 剩余比例低于该阈值时，请求下一段 chunk。

`smooth_overlap`：

- 在 overlap 区用 smoothstep 曲线从 old chunk 平滑过渡到 new chunk。

RTC：

- Real-Time Chunking，在模型推理时利用上一段剩余 chunk 和推理延迟，提升 chunk 间时间一致性。

IK：

- Inverse Kinematics，给定末端位姿求关节角。

Warm start：

- 用上一帧 IK 解作为下一帧 IK 初始值，加速并稳定求解。

Null-space anchor：

- 对冗余机械臂，在满足末端位姿的同时，把关节解拉向某个参考姿态，减少解分支漂移。

Joint delta clamp：

- 对每一帧关节命令的最大变化量做硬限制。

Joint low-pass smoothing：

- 对连续关节目标做指数滑动平均，降低高频抖动。

OOM：

- Out of Memory，内存不足时 Linux 可能直接 kill 进程。


from __future__ import annotations

from dataclasses import dataclass


def _load_robot_config_base():
    # 优先适配新版路径，旧版路径作为兼容分支。
    try:
        from lerobot.robots.config import RobotConfig  # type: ignore

        return RobotConfig
    except Exception:
        pass

    try:
        from lerobot.robots import RobotConfig  # type: ignore

        return RobotConfig
    except Exception:
        pass

    @dataclass
    class _FallbackRobotConfig:
        @classmethod
        def register_subclass(cls, _name: str):
            def decorator(subclass):
                return subclass

            return decorator

    return _FallbackRobotConfig


RobotConfigBase = _load_robot_config_base()


@RobotConfigBase.register_subclass("nero")
@dataclass
class NeroRobotConfig(RobotConfigBase):
    interface: str = "socketcan"
    channel: str = "can0"
    bitrate: int = 1_000_000
    firmware: str = "v1.11"
    use_gripper: bool = False
    gripper_force: float = 1.0
    use_camera: bool = False
    camera_serial: str = ""
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    speed_percent: int = 10
    # Optional joint-space smoothing applied before move_j. 0.0 disables it.
    joint_smoothing_alpha: float = 0.0
    # If True, apply rollio's apply_command_pose_fix (R_z(180°) flip) before
    # move_p. Required ONLY when the physical Nero base is rotated 180° about
    # z relative to the AIRBOT data-collection rig. If your Nero faces the
    # same direction as the AIRBOT did during data collection, set this False.
    airbot_aligned_action: bool = True
    execute: list[float] | None = None

    def __post_init__(self):
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()
        if not 0.0 <= float(self.joint_smoothing_alpha) < 1.0:
            raise ValueError("joint_smoothing_alpha must be in [0.0, 1.0).")

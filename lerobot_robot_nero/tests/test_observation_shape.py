"""Tests for get_observation() returning the pi05-required format.

Expected keys (strict):
    "observation.state"                  np.ndarray shape=(8,) dtype=float32
        [0:7]  arm joint_position (rad)
        [7]    gripper parallel_position (m)
    "observation.images.realsense_color" np.ndarray shape=(H,W,3) dtype=uint8
        only present when use_camera=True
"""
import numpy as np
import pytest

from lerobot_robot_nero import NeroRobot, NeroRobotConfig


class StubGripper:
    def __init__(self, width: float = 0.03):
        self._width = float(width)

    def get_gripper_status(self):
        class Status:
            def __init__(self, value: float):
                self.value = value
                self.mode = "width"

        class Message:
            def __init__(self, msg):
                self.msg = msg

        return Message(Status(self._width))


class StubArm:
    class OPTIONS:
        class EFFECTOR:
            AGX_GRIPPER = "agx_gripper"

    def __init__(self):
        self._connected = False
        self._angles = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
        self._gripper = StubGripper(0.024)
        self.last_move_p: list[float] | None = None
        self.last_move_j: list[float] | None = None

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def get_joint_angles(self):
        class Msg:
            def __init__(self, msg):
                self.msg = msg

        return Msg(list(self._angles))

    def init_effector(self, _effector_name: str):
        return self._gripper

    def enable(self):
        return None

    def set_speed_percent(self, _percent: int):
        return None

    def move_j(self, values):
        self._angles = list(values)
        self.last_move_j = list(values)

    def move_p(self, pose):
        self.last_move_p = list(pose)


# --------------------------------------------------------------------------


def test_observation_returns_flat_dict_matching_observation_features():
    """get_observation() returns the keys declared in observation_features."""
    robot = NeroRobot(NeroRobotConfig(), arm=StubArm())
    robot.connect()
    obs = robot.get_observation()

    for name in NeroRobot.STATE_FEATURE_NAMES:
        assert name in obs, f"missing {name}"
        assert isinstance(obs[name], float), f"{name} should be float, got {type(obs[name])}"

    robot.disconnect()


def test_observation_joint_values_match_arm():
    arm = StubArm()
    arm._angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()
    obs = robot.get_observation()

    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for name, exp in zip(NeroRobot.STATE_FEATURE_NAMES[:7], expected):
        np.testing.assert_allclose(obs[name], exp, atol=1e-6)

    robot.disconnect()


def test_observation_gripper_zero_when_no_gripper():
    robot = NeroRobot(NeroRobotConfig(use_gripper=False), arm=StubArm())
    robot.connect()
    obs = robot.get_observation()

    assert obs[NeroRobot.STATE_FEATURE_NAMES[7]] == 0.0

    robot.disconnect()


def test_observation_gripper_present_when_connected():
    robot = NeroRobot(NeroRobotConfig(use_gripper=True), arm=StubArm())
    robot.connect()
    obs = robot.get_observation()

    assert pytest.approx(obs[NeroRobot.STATE_FEATURE_NAMES[7]], abs=1e-6) == 0.024

    robot.disconnect()


def test_observation_no_camera_key_without_use_camera():
    robot = NeroRobot(NeroRobotConfig(use_camera=False), arm=StubArm())
    robot.connect()
    obs = robot.get_observation()

    assert NeroRobot.CAMERA_KEY not in obs

    robot.disconnect()


def test_observation_camera_key_present_with_stub_capture_fn():
    H, W = 1080, 1920
    stub_frame = np.zeros((H, W, 3), dtype=np.uint8)

    robot = NeroRobot(
        NeroRobotConfig(use_camera=True),
        arm=StubArm(),
        camera_capture_fn=lambda: stub_frame,
    )
    robot.connect()
    obs = robot.get_observation()

    assert NeroRobot.CAMERA_KEY in obs
    img = obs[NeroRobot.CAMERA_KEY]
    assert isinstance(img, np.ndarray), type(img)
    assert img.shape == (H, W, 3), img.shape
    assert img.dtype == np.uint8, img.dtype

    robot.disconnect()


def test_observation_features_shape_consistent_with_get_observation():
    """observation_features keys must match get_observation output keys."""
    robot = NeroRobot(NeroRobotConfig(use_camera=False), arm=StubArm())
    robot.connect()
    feats = robot.observation_features
    obs = robot.get_observation()
    assert set(feats.keys()) == set(obs.keys()), (
        f"feats={set(feats.keys())} obs={set(obs.keys())}"
    )
    robot.disconnect()

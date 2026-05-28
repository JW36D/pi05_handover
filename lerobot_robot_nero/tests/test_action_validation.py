"""Tests for send_action() accepting cartesian (8-dim) actions.

Action layout (pi05 output):
    [0:3]  end_pose position  (x, y, z) metres
    [3:7]  end_pose quaternion (qx, qy, qz, qw)  scalar-last
    [7]    gripper parallel_position (m)

Expected behaviour:
    - Calls arm.move_p([x, y, z, roll, pitch, yaw]) with XYZ extrinsic euler
    - When use_gripper=True, also calls gripper.move_gripper_m(width, force)
    - Rejects shape != (8,)
    - Rejects near-zero quaternion
"""
import math
import sys
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from lerobot_robot_nero import NeroRobot, NeroRobotConfig


class StubGripper:
    def __init__(self, initial_width: float = 0.02):
        self._width = float(initial_width)
        self.last_move: tuple[float, float] | None = None

    def move_gripper_m(self, value: float = 0.0, force: float = 1.0):
        self.last_move = (float(value), float(force))
        self._width = float(value)

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

    def __init__(self, gripper: StubGripper | None = None):
        self._connected = False
        # Nero's DISABLED_HOLD_Q (matches rollio: arm rests near j4=π/2).
        # Using all-zeros confuses IK because the joint-4 elbow is folded.
        self._angles = [0.0, 0.0, 0.0, math.pi / 2, 0.0, 0.0, 0.0]
        self.last_move_p: list[float] | None = None
        self.last_move_j: list[float] | None = None
        self.move_p_count = 0
        self._gripper = gripper
        self.init_effector_arg: str | None = None

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def init_effector(self, effector: str):
        self.init_effector_arg = effector
        if self._gripper is None:
            raise RuntimeError("No gripper attached in stub arm.")
        return self._gripper

    def get_joint_angles(self):
        return list(self._angles)

    def move_j(self, values):
        self.last_move_j = list(values)
        self._angles = list(values)

    def move_p(self, pose):
        self.last_move_p = list(pose)
        self.move_p_count += 1

    def enable(self):
        return None

    def set_speed_percent(self, _percent: int):
        return None


def _identity_action() -> np.ndarray:
    """A reachable in-distribution action (drawn from training data q50).

    Identity quat + arbitrary position is NOT reachable for the gripper-TCP
    Nero model — the IK fails the convergence tolerance. Using a realistic
    AIRBOT-frame pose so the IK pipeline has a solution.
    """
    return np.array(
        [0.5059, -0.0027, 0.2445,        # x, y, z (AIRBOT frame, q50)
         0.0243, 0.0895, 0.0118, 0.9868, # qx, qy, qz, qw (q50, normalised)
         0.0758],                        # gripper (q50)
        dtype=np.float32,
    )


# --------------------------------------------------------------------------


def test_send_action_calls_move_j_via_ik():
    """During inference, send_action runs IK and calls move_j (7 joints)."""
    pytest.importorskip("pinocchio")
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    robot.send_action(_identity_action())

    assert arm.last_move_j is not None, "move_j should have been called"
    assert len(arm.last_move_j) == 7, "Nero is 7-DOF"
    assert arm.last_move_p is None, "move_p must NOT be called (TCP convention mismatch)"

    robot.disconnect()


def test_send_action_ik_recovers_training_joints():
    """Pipeline applied to a recorded training action recovers the recorded
    joint configuration of that frame (modulo redundancy null space).
    This is the key correctness invariant for inference."""
    pytest.importorskip("pinocchio")
    pytest.importorskip("pandas")
    import pandas as pd

    parquet = "/home/guest/Documents/Hand_Over/chunk-000/file-000.parquet"
    if not __import__("pathlib").Path(parquet).is_file():
        pytest.skip("training parquet not available")
    df = pd.read_parquet(parquet)
    ep0 = df[df["episode_index"] == 0]
    action_np = np.asarray(ep0.iloc[0]["action"], dtype=np.float32)
    state_np = np.asarray(ep0.iloc[0]["observation.state"], dtype=np.float32)

    arm = StubArm()
    arm._angles = list(state_np[:7])  # seed measured joints from training
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()
    robot.send_action(action_np)

    # The non-redundant joints (j2, j4, j6, j7 — those that uniquely
    # determine end-effector position+orientation) must be very close.
    assert arm.last_move_j is not None
    diff = np.abs(np.asarray(arm.last_move_j) - state_np[:7])
    for j in (1, 3, 5, 6):
        assert diff[j] < 0.1, f"joint {j+1} drift {diff[j]:.3f} too large"

    robot.disconnect()


def test_send_action_normalizes_quat():
    """Unnormalised input quaternion must not break the pipeline."""
    pytest.importorskip("pinocchio")
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    # Take a reachable action and scale its quat by 3.0 — pipeline must
    # normalise before passing to IK.
    base = _identity_action()
    base[3:7] *= 3.0
    robot.send_action(base)

    assert arm.last_move_j is not None
    assert all(math.isfinite(v) for v in arm.last_move_j)

    robot.disconnect()


def test_send_action_rejects_wrong_shape():
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    with pytest.raises(ValueError, match="shape"):
        robot.send_action(np.zeros(7))

    with pytest.raises(ValueError, match="shape"):
        robot.send_action(np.zeros(9))

    robot.disconnect()


def test_send_action_rejects_zero_quaternion():
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    action = np.concatenate([np.zeros(3), np.zeros(4), [0.0]])
    with pytest.raises(ValueError, match="near-zero"):
        robot.send_action(action)

    robot.disconnect()


def test_send_action_rejects_when_disconnected():
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)

    with pytest.raises(Exception, match="not connected"):
        robot.send_action(_identity_action())


def test_send_action_with_gripper_calls_move_gripper_m():
    gripper = StubGripper(initial_width=0.01)
    arm = StubArm(gripper=gripper)
    cfg = NeroRobotConfig(use_gripper=True, gripper_force=1.5)
    robot = NeroRobot(cfg, arm=arm)
    robot.connect()

    action = _identity_action().copy()
    action[7] = 0.04  # override gripper
    robot.send_action(action)

    assert arm.init_effector_arg == "agx_gripper"
    assert gripper.last_move is not None
    assert pytest.approx(gripper.last_move[0], abs=1e-6) == 0.04
    assert pytest.approx(gripper.last_move[1], abs=1e-6) == 1.5

    robot.disconnect()


def test_send_action_accepts_torch_tensor():
    pytest.importorskip("torch")
    pytest.importorskip("pinocchio")
    import torch

    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    action_np = _identity_action()
    action_t = torch.from_numpy(action_np)
    robot.send_action(action_t)

    assert arm.last_move_j is not None
    assert len(arm.last_move_j) == 7

    robot.disconnect()


def test_send_action_accepts_dict_from_robot_client():
    """RobotClient passes a named dict; send_action must unpack it in key order."""
    pytest.importorskip("pinocchio")
    from lerobot_robot_nero.nero import NeroRobot

    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    expected_np = _identity_action()
    action_dict = {
        key: float(expected_np[i])
        for i, key in enumerate(NeroRobot.ACTION_FEATURE_NAMES)
    }
    robot.send_action(action_dict)

    assert arm.last_move_j is not None
    assert len(arm.last_move_j) == 7

    robot.disconnect()


def test_action_features_keys_match_model_config():
    """action_features must enumerate exactly the 8 pi05 model action keys."""
    from lerobot_robot_nero.nero import NeroRobot

    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    feats = robot.action_features
    assert len(feats) == 8
    keys = list(feats.keys())
    assert keys[0] == "agx_nero__arm.end_pose.0"
    assert keys[6] == "agx_nero__arm.end_pose.6"
    assert keys[7] == "agx_nero__gripper.parallel_mit.0"

    robot.disconnect()


def test_calibrate_uses_move_j_directly():
    """calibrate(execute=...) sends a 7-joint pose via move_j without IK."""
    arm = StubArm()
    cfg = NeroRobotConfig(execute=[0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0])
    robot = NeroRobot(cfg, arm=arm)
    robot.connect(calibrate=False)
    robot.calibrate()

    assert arm.last_move_j is not None, "calibrate must call move_j"
    assert len(arm.last_move_j) == 7
    assert arm.last_move_p is None, "calibrate must not call move_p"

    robot.disconnect()


def test_joint_smoothing_alpha_filters_consecutive_targets():
    robot = NeroRobot(NeroRobotConfig(joint_smoothing_alpha=0.25), arm=StubArm())

    first = robot._smooth_joint_targets([0.0] * 7)
    second = robot._smooth_joint_targets([1.0] * 7)

    np.testing.assert_allclose(first, np.zeros(7), atol=1e-9)
    np.testing.assert_allclose(second, np.full(7, 0.75), atol=1e-9)


def test_joint_smoothing_disabled_passes_targets_through():
    robot = NeroRobot(NeroRobotConfig(joint_smoothing_alpha=0.0), arm=StubArm())
    target = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]

    smoothed = robot._smooth_joint_targets(target)

    np.testing.assert_allclose(smoothed, target, atol=1e-9)


def test_ik_clamp_walks_from_last_sent_command(monkeypatch):
    """A large IK target should keep ramping even if feedback has not moved yet."""
    fake_pkg = types.ModuleType("rollio_device_nero")
    fake_gravity = types.ModuleType("rollio_device_nero.gravity")
    fake_ik = types.ModuleType("rollio_device_nero.ik")

    class FakeNeroModel:
        def __init__(self, with_gripper=True):
            self.with_gripper = with_gripper

    def fake_solve(*_args, **_kwargs):
        return np.ones(7), True, 0.0

    fake_gravity.NeroModel = FakeNeroModel
    fake_ik.solve = fake_solve
    fake_pkg.ik = fake_ik
    monkeypatch.setitem(sys.modules, "rollio_device_nero", fake_pkg)
    monkeypatch.setitem(sys.modules, "rollio_device_nero.gravity", fake_gravity)
    monkeypatch.setitem(sys.modules, "rollio_device_nero.ik", fake_ik)

    arm = StubArm()
    arm._angles = [0.0] * 7
    robot = NeroRobot(NeroRobotConfig(), arm=arm)
    robot.connect()

    last_sent = np.full(7, 0.2, dtype=np.float64)
    robot._last_sent_joint_target = last_sent.copy()
    cmd = np.asarray(robot._solve_ik_to_joints([0.0] * 7), dtype=np.float64)

    expected = last_sent + NeroRobot.MAX_JOINT_DELTA_RAD
    np.testing.assert_allclose(cmd, expected, atol=1e-9)
    np.testing.assert_allclose(robot._last_sent_joint_target, expected, atol=1e-9)

    robot.disconnect()


def test_ik_failure_holds_last_sent_command_not_unclamped_target(monkeypatch):
    """Failure fallback must hold the safe command that actually went to hardware."""
    arm = StubArm()
    robot = NeroRobot(NeroRobotConfig(airbot_aligned_action=False), arm=arm)
    robot.connect()

    safe_hold = np.full(7, 0.123, dtype=np.float64)
    unsafe_unclamped = np.full(7, 9.0, dtype=np.float64)
    robot._last_sent_joint_target = safe_hold.copy()
    robot._latest_ik_target = unsafe_unclamped.copy()

    def fail_ik(*_args, **_kwargs):
        raise RuntimeError("synthetic IK failure")

    monkeypatch.setattr(robot, "_solve_ik_to_joints", fail_ik)
    robot.send_action(_identity_action())

    assert arm.last_move_j is not None
    np.testing.assert_allclose(arm.last_move_j, safe_hold, atol=1e-9)

    robot.disconnect()

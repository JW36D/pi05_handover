from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import deque
from functools import cached_property
from numbers import Real
from typing import Any, Deque, Iterable, Mapping, Sequence, TypeAlias

import numpy as np

from .config_nero import NeroRobotConfig

logger = logging.getLogger(__name__)


def _load_robot_base():
    try:
        from lerobot.robots.robot import Robot  # type: ignore

        return Robot
    except Exception:
        pass

    try:
        from lerobot.robots import Robot  # type: ignore

        return Robot
    except Exception:
        pass

    class _FallbackRobot:
        config_class = object
        name = "robot"

        def __init__(self, config: Any):
            self.config = config
            self.id = getattr(config, "id", "default")
            self.calibration = {}

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(id={self.id})"

    return _FallbackRobot


def _identity_decorator(func):
    return func


def _load_connection_decorators():
    try:
        from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected  # type: ignore

        return check_if_already_connected, check_if_not_connected
    except Exception:
        return _identity_decorator, _identity_decorator


def _load_error_types():
    try:
        from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError  # type: ignore

        return DeviceAlreadyConnectedError, DeviceNotConnectedError
    except Exception:
        pass

    class _DeviceAlreadyConnectedError(RuntimeError):
        pass

    class _DeviceNotConnectedError(RuntimeError):
        pass

    return _DeviceAlreadyConnectedError, _DeviceNotConnectedError


try:
    from lerobot.types import RobotAction as _RobotAction, RobotObservation as _RobotObservation  # type: ignore

    RobotAction = _RobotAction
    RobotObservation = _RobotObservation
except Exception:
    RobotAction: TypeAlias = dict[str, Any]
    RobotObservation: TypeAlias = dict[str, Any]


RobotBase = _load_robot_base()
check_if_already_connected, check_if_not_connected = _load_connection_decorators()
DeviceAlreadyConnectedError, DeviceNotConnectedError = _load_error_types()


class NeroRobot(RobotBase):
    config_class = NeroRobotConfig
    name = "nero"

    def __init__(
        self,
        config: NeroRobotConfig,
        arm: Any | None = None,
        gripper: Any | None = None,
        camera_capture_fn: Any | None = None,
    ):
        super().__init__(config)
        self.config = config
        self._arm = arm
        self._gripper = gripper
        self._connected = False
        self._is_calibrated = True
        # internal keys kept for joint-space reading and calibrate()
        self._joint_keys = tuple(f"joint_{i}.pos" for i in range(1, 8))
        self._gripper_key = "gripper.width_m"
        # camera: either an injectable callable (for tests) or a rs.pipeline
        self._camera_capture_fn = camera_capture_fn
        self._camera_pipeline: Any | None = None
        # Pinocchio model + IK (rollio's exact pipeline) — lazily loaded.
        # Caches the previous IK solution to use as the warm-start seed,
        # mirroring rollio's `_latest_ik_target`.
        self._nero_model: Any | None = None
        self._latest_ik_target: np.ndarray | None = None
        self._last_sent_joint_target: np.ndarray | None = None
        self._last_smoothed_joint_target: np.ndarray | None = None
        self._ik_failure_count: int = 0
        # Diagnostics: ring buffer of recent IK failures + total call counter.
        # Auto-dumped on disconnect when NERO_IK_DIAG_DUMP env var is set or
        # when the buffer is non-empty.
        self._ik_call_index: int = 0
        self._ik_failure_log: Deque[dict[str, Any]] = deque(maxlen=200)
        self._ik_total_failures: int = 0
        self._connect_monotonic: float | None = None

    # ------------------------------------------------------------------
    # LeRobot feature descriptors (used by dataset pipeline if present)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Feature names — must match exactly what was used to convert v2.1 →
    # v3 training data (see convert_rollio_to_lerobot.py STATE_NAMES /
    # ACTION_NAMES) so that the QUANTILES normalizer / unnormalizer applies
    # the right per-dim stats. lerobot internally collapses these into
    # `observation.state` (concat) and `action` (concat) tensors using
    # hw_to_dataset_features().
    # ------------------------------------------------------------------

    STATE_FEATURE_NAMES: tuple[str, ...] = (
        "agx_nero__arm.joint_position.0",
        "agx_nero__arm.joint_position.1",
        "agx_nero__arm.joint_position.2",
        "agx_nero__arm.joint_position.3",
        "agx_nero__arm.joint_position.4",
        "agx_nero__arm.joint_position.5",
        "agx_nero__arm.joint_position.6",
        "agx_nero__gripper.parallel_position.0",
    )

    ACTION_FEATURE_NAMES: tuple[str, ...] = (
        "agx_nero__arm.end_pose.0",
        "agx_nero__arm.end_pose.1",
        "agx_nero__arm.end_pose.2",
        "agx_nero__arm.end_pose.3",
        "agx_nero__arm.end_pose.4",
        "agx_nero__arm.end_pose.5",
        "agx_nero__arm.end_pose.6",
        "agx_nero__gripper.parallel_mit.0",
    )

    CAMERA_KEY: str = "realsense_color"  # → observation.images.realsense_color

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """Hardware-level feature descriptor consumed by lerobot.

        lerobot's `hw_to_dataset_features` interprets:
          • `float`-typed entries as scalar joints → packed into observation.state
          • `tuple`-typed entries as cameras → become observation.images.<key>
        """
        feats: dict[str, Any] = {name: float for name in self.STATE_FEATURE_NAMES}
        if self.config.use_camera:
            feats[self.CAMERA_KEY] = (
                self.config.camera_height,
                self.config.camera_width,
                3,
            )
        return feats

    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {name: float for name in self.ACTION_FEATURE_NAMES}

    # ------------------------------------------------------------------
    # Connection state
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        if not self._connected or self._arm is None:
            return False

        status_attr = getattr(self._arm, "is_connected", None)
        if status_attr is None:
            return self._connected
        if callable(status_attr):
            try:
                return bool(status_attr())
            except Exception:
                return self._connected
        return bool(status_attr)

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @check_if_not_connected
    def move_to_joints(self, joints: Sequence[float] | np.ndarray) -> None:
        """Move the arm to a 7-DOF joint configuration via move_j.

        Used to initialise the arm to a known starting configuration
        (e.g. an episode's first recorded joint state) before kicking
        off cartesian inference. Bypasses IK and the per-tick clamp —
        do NOT call this in a tight 30 Hz loop.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._arm is not None

        target = [float(v) for v in joints]
        if len(target) != 7:
            raise ValueError(f"move_to_joints expects 7 values, got {len(target)}.")
        if not all(math.isfinite(v) for v in target):
            raise ValueError(f"move_to_joints non-finite value in {target}.")

        try:
            self._arm.move_j(target)
        except Exception as exc:
            raise RuntimeError(f"move_to_joints failed: {exc}") from exc
        # Reset IK warm-start so the next send_action() seeds from the
        # measured joints instead of stale state.
        self._latest_ik_target = None
        self._last_sent_joint_target = None
        self._last_smoothed_joint_target = None

    def calibrate(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Nero v1 默认不需要写入零位偏置，校准状态直接标记为已完成。
        self._is_calibrated = True

        # 可选：支持在 calibrate 流程里执行一个 7 关节目标位姿，方便初始化姿态。
        execute_target = self.config.execute
        if execute_target is None:
            return

        target_dict = self._normalize_joint_action(execute_target)
        target = [target_dict[key] for key in self._joint_keys]
        assert self._arm is not None
        try:
            self._arm.move_j(target)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to execute configured calibrate target pose: {exc}"
            ) from exc
        self._latest_ik_target = None
        self._last_sent_joint_target = None
        self._last_smoothed_joint_target = None
        logger.info("%s moved to configured execute pose during calibrate().", self)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected.")

        if self._arm is None:
            self._arm = self._create_default_arm()
        if self.config.use_gripper and self._gripper is None:
            self._gripper = self._init_gripper_driver()

        try:
            self._arm.connect()
        except Exception as exc:
            raise ConnectionError(f"Failed to connect to Nero: {exc}") from exc

        self._connected = True
        self._connect_monotonic = time.monotonic()
        if calibrate and not self.is_calibrated:
            self.calibrate()

        # 连接后先等待首帧关节反馈，避免上层流程第一帧直接失败。
        self._get_joint_angles_with_retry(attempts=50, delay_s=0.02)
        self.configure()

        if self.config.use_camera and self._camera_capture_fn is None:
            self._camera_pipeline = self._start_camera_pipeline()

        logger.info("%s connected.", self)

    @check_if_not_connected
    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Diagnostics: print summary, optionally dump full log.
        if self._ik_call_index > 0:
            rate = self._ik_total_failures / self._ik_call_index
            logger.info(
                "IK summary: %d calls, %d failures (%.2f%%).",
                self._ik_call_index, self._ik_total_failures, 100.0 * rate,
            )
        dump_path = os.environ.get("NERO_IK_DIAG_DUMP")
        if dump_path is None and self._ik_total_failures > 0:
            # Default: drop a timestamped file alongside the user's logs so
            # nothing is lost. Override with NERO_IK_DIAG_DUMP=/path/file.json.
            dump_path = f"/tmp/nero_ik_failures_{int(time.time())}.json"
        if dump_path is not None:
            try:
                self.dump_ik_diagnostics(dump_path)
            except Exception as exc:
                logger.warning("Failed to dump IK diagnostics: %s", exc)

        if self._camera_pipeline is not None:
            try:
                self._camera_pipeline.stop()
            except Exception as exc:
                logger.warning("Camera pipeline stop failed: %s", exc)
            finally:
                self._camera_pipeline = None

        assert self._arm is not None
        try:
            self._arm.disconnect()
        except Exception as exc:
            raise RuntimeError(f"Failed to disconnect Nero cleanly: {exc}") from exc
        finally:
            self._connected = False
            self._last_smoothed_joint_target = None

        logger.info("%s disconnected.", self)

    def configure(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._arm is not None

        speed = int(self.config.speed_percent)
        if speed < 1 or speed > 100:
            raise ValueError("speed_percent must be in [1, 100].")

        if hasattr(self._arm, "enable"):
            enabled = False
            for _ in range(100):
                try:
                    result = self._arm.enable()
                    if result is None or bool(result):
                        enabled = True
                        break
                except Exception as exc:
                    raise RuntimeError(f"Failed to enable Nero before motion: {exc}") from exc
                time.sleep(0.01)

            if not enabled:
                raise RuntimeError(
                    "Failed to enable Nero after retries. "
                    "Please check power state, emergency stop state, and CAN communication."
                )

        if hasattr(self._arm, "set_speed_percent"):
            try:
                self._arm.set_speed_percent(speed)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to set Nero speed_percent={speed}: {exc}"
                ) from exc

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Return a flat hardware observation dict, matching observation_features.

        Keys produced (must align with STATE_FEATURE_NAMES + CAMERA_KEY):
            "agx_nero__arm.joint_position.0..6"      float (rad)
            "agx_nero__gripper.parallel_position.0"  float (m)  [0 if no gripper]
            "<CAMERA_KEY>"                           np.uint8 (H, W, 3) RGB
                                                     [present only with use_camera]

        lerobot's `build_dataset_frame` will pack the 8 floats into
        `observation.state` and the camera into `observation.images.<key>`
        downstream, so the policy server receives the canonical pi05 shape.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_values = self._extract_joint_values(self._get_joint_angles_with_retry())

        if self.config.use_gripper:
            gripper_width = self._extract_gripper_width(self._get_gripper_status_with_retry())
        else:
            gripper_width = 0.0

        flat: dict[str, Any] = {}
        # 7 joints
        for name, value in zip(self.STATE_FEATURE_NAMES[:7], joint_values, strict=True):
            flat[name] = float(value)
        # 1 gripper
        flat[self.STATE_FEATURE_NAMES[7]] = float(gripper_width)

        if self.config.use_camera:
            flat[self.CAMERA_KEY] = self._capture_camera_frame()

        return flat

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    @check_if_not_connected
    def send_action(self, action: Any) -> Any:
        """Send a cartesian action predicted by pi05.

        =====================================================================
        DATA CONTRACT  (verified end-to-end against parquet + checkpoint;
        full proof in lerobot_robot_nero/DATA_CONTRACT.md). DO NOT modify
        without re-running scripts/verify_data_contract.py.
        =====================================================================

        Action layout (8 elements — the names in ACTION_FEATURE_NAMES are
        misleading: the values are AIRBOT-frame, not Nero-frame):
            [0:3]   end-effector POSITION  (x, y, z) metres   AIRBOT-aligned
            [3:7]   end-effector QUATERNION (qx, qy, qz, qw)  AIRBOT-aligned, scalar-last
            [7]     gripper parallel width  (m)               trained range ≤ ~0.087m

        Why "AIRBOT-aligned"?  rollio's runtime publishes every pose on its
        IPC bus pre-transformed by `apply_publish_pose_fix` (Nero base is
        mounted 180° about z relative to AIRBOT). Empirically we verified
        the parquet `action` column is byte-identical to the AIRBOT leader's
        `end_effector_pose`:  max|action[:7] - airbot_play__end_effector_pose| == 0.

        Pipeline (matches rollio runtime, NOT pyAgxArm.move_p — the latter
        uses a different TCP convention than rollio's Pinocchio model and
        leaves orientation mis-tracked):
          1. apply_command_pose_fix → Nero Pinocchio operational pose7
          2. rollio_device_nero.ik.solve   → joint angles (warm-started)
          3. arm.move_j(joint_angles)
          4. gripper.move_gripper_m(width, force)

        Accepts two input shapes:
          (a) dict  {ACTION_FEATURE_NAMES[i]: float}  — from RobotClient
          (b) np.ndarray / torch.Tensor shape (8,)    — from replay/tests
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        assert self._arm is not None

        if isinstance(action, Mapping):
            try:
                values = [float(action[k]) for k in self.ACTION_FEATURE_NAMES]
            except KeyError as exc:
                raise ValueError(
                    f"send_action dict is missing key {exc}. "
                    f"Expected keys: {self.ACTION_FEATURE_NAMES}"
                ) from exc
            a = np.array(values, dtype=np.float32)
        else:
            a = (
                action.detach().cpu().numpy()
                if hasattr(action, "detach")
                else np.asarray(action, dtype=np.float32)
            )
        if a.shape != (8,):
            raise ValueError(f"send_action expects shape (8,), got {a.shape}")

        # Step 1: AIRBOT-aligned → Nero Pinocchio frame (configurable).
        if self.config.airbot_aligned_action:
            try:
                from rollio_device_nero.airbot_aligned_pose import apply_command_pose_fix  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "rollio_device_nero is required. Install it or set "
                    "config.airbot_aligned_action=False to skip the frame fix."
                ) from exc
            native7 = apply_command_pose_fix([float(v) for v in a[0:7]])
        else:
            native7 = [float(v) for v in a[0:7]]

        # Validate quaternion norm before handing pose7 to IK.
        quat_norm = float(np.linalg.norm(native7[3:7]))
        if quat_norm < 1e-6:
            raise ValueError(f"Action quaternion is near-zero (norm={quat_norm}).")

        # Step 2: Pinocchio damped-LS IK with null-space anchoring to
        # current joint configuration (mirrors rollio runtime).
        # During pi05 inference the model occasionally produces a pose that
        # IK can't fit (singularity, near-edge target, noisy quaternion).
        # Crashing the whole process there is unsafe — the policy_server
        # would just keep streaming chunks. Instead, log + hold the last
        # commanded joints (or current measured if nothing cached).
        try:
            joint_targets = self._solve_ik_to_joints(
                native7, airbot_pose7=[float(v) for v in a[0:7]]
            )
        except RuntimeError as exc:
            logger.warning("IK failed, holding last joint target: %s", exc)
            self._ik_failure_count += 1
            if self._ik_failure_count > self.MAX_CONSECUTIVE_IK_FAILURES:
                raise RuntimeError(
                    f"IK failed {self._ik_failure_count} times in a row. "
                    "Aborting send_action to prevent runaway. "
                    "Likely the policy is producing out-of-distribution actions; "
                    "stop the policy_server and inspect."
                ) from exc
            if self._last_sent_joint_target is not None:
                joint_targets = [float(v) for v in self._last_sent_joint_target]
            else:
                # No previous target → hold measured joints
                q_meas = self._extract_joint_values(self._get_joint_angles_with_retry())
                joint_targets = [float(v) for v in q_meas]
        else:
            self._ik_failure_count = 0

        # Step 3: send joint command. We use move_j for compatibility; rollio
        # itself uses MIT mode for higher-rate tracking. At 30 Hz with
        # `chunk_size=50`, move_j on a position controller is sufficient.
        joint_targets = self._smooth_joint_targets(joint_targets)
        try:
            self._arm.move_j(joint_targets)
        except Exception as exc:
            raise RuntimeError(f"Failed to send move_j action to Nero: {exc}") from exc

        # Step 4: gripper.
        if self.config.use_gripper:
            self._send_gripper_width(float(a[7]))

        return a

    # ------------------------------------------------------------------
    # IK (rollio Pinocchio model)
    # ------------------------------------------------------------------

    # SAFETY: maximum per-tick joint motion. Clamps each command step before
    # sending it to the arm. The reference is the previously sent command when
    # available, falling back to measured joints only for the first command.
    # This mirrors rollio's runtime and avoids low-rate feedback aliasing into
    # the command stream.
    MAX_JOINT_DELTA_RAD: float = 0.20  # ≈11.5° per tick — safe at 30Hz

    # SAFETY: how many consecutive IK failures we tolerate before aborting.
    # Single-frame failures are common with pi05 (occasional out-of-distribution
    # action), but a sustained run of failures means the policy has lost its way
    # or the arm is stuck and we should stop instead of holding indefinitely.
    MAX_CONSECUTIVE_IK_FAILURES: int = 30  # at 30Hz ≈ 1 second of held position

    def _smooth_joint_targets(self, joint_targets: Sequence[float]) -> list[float]:
        alpha = float(self.config.joint_smoothing_alpha)
        if alpha <= 0.0:
            self._last_smoothed_joint_target = np.asarray(joint_targets, dtype=np.float64)
            return [float(v) for v in joint_targets]
        if alpha >= 1.0:
            raise ValueError("joint_smoothing_alpha must be in [0.0, 1.0).")

        target = np.asarray(joint_targets, dtype=np.float64)
        if self._last_smoothed_joint_target is None:
            smoothed = target
        else:
            smoothed = alpha * self._last_smoothed_joint_target + (1.0 - alpha) * target

        self._last_smoothed_joint_target = smoothed
        return [float(v) for v in smoothed]

    def _solve_ik_to_joints(
        self,
        target_pose7: list[float] | np.ndarray,
        airbot_pose7: list[float] | np.ndarray | None = None,
    ) -> list[float]:
        """Resolve a target pose7 (Nero Pinocchio frame) to 7 joint angles.

        Uses rollio's `NeroModel` and `ik.solve` — the exact pipeline
        rollio runtime uses to consume cartesian commands. Warm-starts
        from the previous IK solution and anchors the null space to the
        current measured joint configuration to suppress redundancy drift.

        Raises RuntimeError if IK fails to converge — sending the
        non-converged solution is unsafe (the arm can fly to a random
        joint configuration). Failures are also captured into the
        diagnostic ring buffer (`_ik_failure_log`) for offline analysis.

        Args:
            target_pose7: Target in Nero Pinocchio (post-fix) frame.
            airbot_pose7: Original AIRBOT-frame pose, kept for diagnostics
                so we can correlate failures with model output distribution.
        """
        try:
            from rollio_device_nero.gravity import NeroModel  # type: ignore
            from rollio_device_nero import ik as nero_ik  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rollio_device_nero (with pinocchio) is required for IK. "
                "Install pinocchio + rollio_device_nero in the runtime env."
            ) from exc

        if self._nero_model is None:
            # Training data was always recorded with gripper attached; the
            # action's pose7 is therefore in the AGX-gripper TCP frame.
            # Build the IK model with gripper TCP regardless of the runtime
            # `use_gripper` flag — that flag only controls command dispatch,
            # not kinematic frame.
            self._nero_model = NeroModel(with_gripper=True)

        q_meas = np.asarray(
            self._extract_joint_values(self._get_joint_angles_with_retry()),
            dtype=np.float64,
        )
        q_seed = self._latest_ik_target if self._latest_ik_target is not None else q_meas

        # Defaults in rollio's ik.solve are tol=1e-4 and max_iter=50, tuned
        # for high-rate teleop with smooth trajectories. pi05 inference is
        # different — the policy can output edge-of-workspace or near-singular
        # poses where IK only converges to a few millimetres of cartesian
        # residual. tol=5e-3 ≈ 5mm/5mrad is well within Nero's mechanical
        # repeatability and avoids spurious "didn't converge" reports.
        self._ik_call_index += 1
        q_target, converged, err = nero_ik.solve(
            self._nero_model,
            list(target_pose7),
            q0=q_seed,
            q_anchor=q_meas,
            tol=5e-3,
            max_iter=200,
        )
        if not converged:
            # SAFETY: do NOT update _latest_ik_target with a bad solution
            # (would corrupt subsequent warm-starts) and do NOT send.
            self._record_ik_failure(
                err=float(err),
                target_pose7_native=list(target_pose7),
                airbot_pose7=airbot_pose7,
                q_meas=q_meas,
                q_seed=q_seed,
                q_target=q_target,
            )
            raise RuntimeError(
                f"IK did not converge (err={err:.4f}, target_pose7={list(target_pose7)}). "
                "Refusing to send unsafe joint targets. Common causes: target out of "
                "workspace; arm not initialised near the trajectory start (use "
                "robot.move_to_joints() first); orientation not reachable from current "
                "null-space branch."
            )

        # SAFETY: clamp each joint to MAX_JOINT_DELTA_RAD of the previously
        # sent command, not always q_meas. Nero feedback is slower/noisier
        # than the command loop; anchoring every clamp to q_meas can repeatedly
        # send the same first waypoint while the arm is still settling, which
        # looks like "moves once then stops" for long initial chunks.
        q_ref = self._last_sent_joint_target if self._last_sent_joint_target is not None else q_meas
        delta = np.clip(
            q_target - q_ref,
            -self.MAX_JOINT_DELTA_RAD,
            self.MAX_JOINT_DELTA_RAD,
        )
        q_clamped = q_ref + delta
        max_requested = float(np.max(np.abs(q_target - q_ref)))
        if max_requested > self.MAX_JOINT_DELTA_RAD:
            logger.info(
                "Joint delta clamped (max requested %.3f rad > %.3f rad limit).",
                max_requested,
                self.MAX_JOINT_DELTA_RAD,
            )

        # Cache the *unclamped* IK target so the next warm-start uses the true
        # goal, not the slewed waypoint — matches rollio's behaviour.
        self._latest_ik_target = q_target
        # Cache the actual command sent to hardware for the next safety clamp
        # and IK-failure hold path.
        self._last_sent_joint_target = q_clamped
        return [float(v) for v in q_clamped]

    def _record_ik_failure(
        self,
        *,
        err: float,
        target_pose7_native: list[float],
        airbot_pose7: list[float] | np.ndarray | None,
        q_meas: np.ndarray,
        q_seed: np.ndarray,
        q_target: np.ndarray,
    ) -> None:
        """Append a single IK failure record to the diagnostic ring buffer."""
        self._ik_total_failures += 1
        rel_t = (
            time.monotonic() - self._connect_monotonic
            if self._connect_monotonic is not None
            else None
        )
        self._ik_failure_log.append({
            "ik_call_index": self._ik_call_index,
            "t_since_connect_s": None if rel_t is None else round(rel_t, 4),
            "err": err,
            "target_pose7_native": [float(v) for v in target_pose7_native],
            "target_pose7_airbot": (
                [float(v) for v in airbot_pose7] if airbot_pose7 is not None else None
            ),
            "q_meas": [float(v) for v in q_meas],
            "q_seed": [float(v) for v in q_seed],
            "q_target_unconverged": [float(v) for v in q_target],
        })

    def dump_ik_diagnostics(self, path: str | os.PathLike[str]) -> None:
        """Write the IK failure ring buffer (and summary) to a JSON file.

        Useful for offline analysis. Always callable; produces an empty
        summary if no failures have occurred.
        """
        summary = {
            "total_ik_calls": self._ik_call_index,
            "total_ik_failures": self._ik_total_failures,
            "buffered_failures": len(self._ik_failure_log),
            "buffer_capacity": self._ik_failure_log.maxlen,
            "failure_rate": (
                self._ik_total_failures / self._ik_call_index
                if self._ik_call_index > 0
                else 0.0
            ),
            "failures": list(self._ik_failure_log),
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "IK diagnostics dumped to %s (%d failures / %d calls).",
            path, self._ik_total_failures, self._ik_call_index,
        )

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _start_camera_pipeline(self) -> Any:
        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyrealsense2 is required when use_camera=True. "
                "Install it or pass camera_capture_fn= for tests."
            ) from exc

        pipeline = rs.pipeline()
        rs_config = rs.config()
        if self.config.camera_serial:
            rs_config.enable_device(self.config.camera_serial)
        rs_config.enable_stream(
            rs.stream.color,
            self.config.camera_width,
            self.config.camera_height,
            rs.format.rgb8,
            self.config.camera_fps,
        )
        pipeline.start(rs_config)
        logger.info(
            "RealSense pipeline started: %dx%d @ %dfps",
            self.config.camera_width,
            self.config.camera_height,
            self.config.camera_fps,
        )
        return pipeline

    def _capture_camera_frame(self) -> np.ndarray:
        """Return HWC uint8 RGB numpy array from the camera."""
        if self._camera_capture_fn is not None:
            return self._camera_capture_fn()

        if self._camera_pipeline is None:
            raise RuntimeError(
                "Camera pipeline is not started. Connect with use_camera=True."
            )

        frames = self._camera_pipeline.wait_for_frames(timeout_ms=1000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense: no color frame received.")
        img = np.asarray(color_frame.get_data(), dtype=np.uint8)
        assert img.ndim == 3 and img.shape[2] == 3, img.shape
        return img

    # ------------------------------------------------------------------
    # Gripper helpers
    # ------------------------------------------------------------------

    def _init_gripper_driver(self) -> Any:
        assert self._arm is not None
        if not hasattr(self._arm, "init_effector"):
            raise RuntimeError(
                "use_gripper=True but arm driver does not provide init_effector()."
            )

        effector_name = "agx_gripper"
        options = getattr(self._arm, "OPTIONS", None)
        effector_cls = getattr(options, "EFFECTOR", None) if options is not None else None
        if effector_cls is not None and hasattr(effector_cls, "AGX_GRIPPER"):
            effector_name = getattr(effector_cls, "AGX_GRIPPER")

        try:
            return self._arm.init_effector(effector_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize gripper effector ({effector_name}): {exc}"
            ) from exc

    def _send_gripper_width(self, width_m: float) -> None:
        if self._gripper is None:
            raise RuntimeError("Gripper driver is not initialized.")

        force = self._to_finite_float(self.config.gripper_force, "gripper_force")
        width = self._to_finite_float(width_m, "gripper.width_m")

        if force < 0.0:
            raise ValueError("gripper_force must be >= 0.")
        if width < 0.0:
            raise ValueError("gripper.width_m must be >= 0.")

        if hasattr(self._gripper, "move_gripper_m"):
            try:
                self._gripper.move_gripper_m(value=width, force=force)
            except Exception as exc:
                raise RuntimeError(f"Failed to send move_gripper_m: {exc}") from exc
            return

        raise RuntimeError("Gripper driver does not expose move_gripper_m().")

    def _get_gripper_status_with_retry(self, attempts: int = 20, delay_s: float = 0.05) -> Any:
        if self._gripper is None:
            raise RuntimeError("Gripper driver is not initialized.")
        if not hasattr(self._gripper, "get_gripper_status"):
            raise RuntimeError("Gripper driver does not expose get_gripper_status().")

        last = None
        for _ in range(attempts):
            try:
                last = self._gripper.get_gripper_status()
            except Exception as exc:
                raise RuntimeError(f"Failed to read gripper status: {exc}") from exc
            if last is not None:
                return last
            time.sleep(delay_s)

        raise RuntimeError(
            "get_gripper_status() kept returning None. "
            f"attempts={attempts}, delay_s={delay_s}."
        )

    def _extract_gripper_width(self, raw: Any) -> float:
        payload = raw
        if hasattr(payload, "msg"):
            payload = getattr(payload, "msg")
        if isinstance(payload, dict):
            value = payload.get("value", payload.get("width"))
            mode = payload.get("mode")
        else:
            value = getattr(payload, "value", None)
            mode = getattr(payload, "mode", None)

        if value is None:
            raise ValueError(f"Invalid gripper status payload: {raw!r}")

        # The AGX Nero gripper firmware always reports mode="angle" via CAN
        # frame 0x2A8 byte 7, even when physically tracking width-mode
        # commands — this is a known firmware quirk documented in rollio's
        # AgxGripperBackend (runtime/agx_backend.py). Do not warn on it.
        # The reported value is consistent with width mode in v1.11+.

        return self._to_finite_float(value, "gripper.width_m")

    # ------------------------------------------------------------------
    # Joint helpers
    # ------------------------------------------------------------------

    def _extract_joint_values(self, raw: Any) -> list[float]:
        values = self._extract_joint_payload(raw)
        if len(values) != 7:
            raise ValueError(
                f"Expected 7 joint values from get_joint_angles(), got {len(values)}."
            )

        out: list[float] = []
        for idx, value in enumerate(values, start=1):
            if not isinstance(value, Real):
                raise TypeError(
                    f"Observation joint_{idx} is not numeric: {type(value).__name__}."
                )
            f_value = float(value)
            if not math.isfinite(f_value):
                raise ValueError(
                    f"Observation joint_{idx} must be finite, got {value!r}."
                )
            out.append(f_value)
        return out

    def _extract_joint_payload(self, raw: Any) -> list[Any]:
        if raw is None:
            raise ValueError(
                "get_joint_angles() returned None (no valid CAN feedback yet). "
                "Please wait briefly after connect() and retry."
            )

        if isinstance(raw, (list, tuple)):
            return list(raw)

        if hasattr(raw, "msg"):
            return self._extract_joint_payload(getattr(raw, "msg"))

        if hasattr(raw, "data"):
            return self._extract_joint_payload(getattr(raw, "data"))

        if isinstance(raw, dict):
            for key in (
                "joint_angles",
                "joints",
                "angles",
                "value",
                "values",
                "payload",
                "data",
            ):
                if key in raw:
                    return self._extract_joint_payload(raw[key])

            ordered = []
            for idx in range(1, 8):
                for key in (f"joint_{idx}", f"joint{idx}", str(idx)):
                    if key in raw:
                        ordered.append(raw[key])
                        break
            if len(ordered) == 7:
                return ordered

        raise ValueError(
            "Unable to parse get_joint_angles() return value. "
            f"Received type={type(raw).__name__}, value={raw!r}"
        )

    def _get_joint_angles_with_retry(self, attempts: int = 20, delay_s: float = 0.02) -> Any:
        assert self._arm is not None

        last_value = None
        for _ in range(attempts):
            try:
                last_value = self._arm.get_joint_angles()
            except Exception as exc:
                raise RuntimeError(f"Failed to read joint angles from Nero: {exc}") from exc
            if last_value is not None:
                return last_value
            time.sleep(delay_s)

        raise RuntimeError(
            "get_joint_angles() kept returning None after retries. "
            f"attempts={attempts}, delay_s={delay_s}. "
            "Check CAN bus status and whether feedback stream is active."
        )

    # ------------------------------------------------------------------
    # Arm factory helpers
    # ------------------------------------------------------------------

    def _create_default_arm(self) -> Any:
        try:
            from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pyAgxArm is required to instantiate NeroRobot automatically. "
                "Install pyAgxArm or pass a pre-built arm instance via NeroRobot(..., arm=...)."
            ) from exc

        firmware = self._resolve_firmware(NeroFW)
        # Note: 'firmeware_version' is the official (typo'd) parameter name in pyAgxArm API.
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            comm="can",
            firmeware_version=firmware,
            interface=self.config.interface,
            channel=self.config.channel,
            bitrate=self.config.bitrate,
        )
        return AgxArmFactory.create_arm(cfg)

    def _resolve_firmware(self, nero_fw_cls: Any) -> Any:
        """Map config.firmware string to a NeroFW enum value.

        Supported values:
            "default" / ""  → NeroFW.DEFAULT  (firmware ≤ 1.10)
            "v1.11" / "V111"→ NeroFW.V111    (firmware ≥ 1.11)
        """
        firmware = self.config.firmware
        if not isinstance(firmware, str):
            return firmware

        normalized = firmware.strip().upper().replace(".", "").replace("-", "").replace(" ", "").lstrip("V")
        _map = {
            "": "DEFAULT",
            "DEFAULT": "DEFAULT",
            "AUTO": "DEFAULT",
            "111": "V111",
        }
        attr_name = _map.get(normalized, f"V{normalized}")
        if hasattr(nero_fw_cls, attr_name):
            return getattr(nero_fw_cls, attr_name)
        valid = [n for n in dir(nero_fw_cls) if not n.startswith("_")]
        raise ValueError(
            f"Unknown firmware version {firmware!r}. "
            f"Valid NeroFW values: {valid}"
        )

    # ------------------------------------------------------------------
    # Calibrate-only joint action helper (separate from send_action)
    # ------------------------------------------------------------------

    def _normalize_joint_action(
        self, action: Mapping[str, Any] | Iterable[Any]
    ) -> dict[str, float]:
        """Parse a joint-space action for calibrate(). Not used during inference."""
        if isinstance(action, Mapping):
            missing = [k for k in self._joint_keys if k not in action]
            if missing:
                raise ValueError(f"Action missing joint keys: {missing}")
            return {
                key: self._to_finite_float(action[key], key)
                for key in self._joint_keys
            }

        raw_values = list(action)
        if len(raw_values) != 7:
            raise ValueError(
                f"Joint action expected 7 values, got {len(raw_values)}."
            )
        return {
            key: self._to_finite_float(value, f"joint_{idx}")
            for idx, (key, value) in enumerate(
                zip(self._joint_keys, raw_values), start=1
            )
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _to_finite_float(self, value: Any, name: str) -> float:
        if not isinstance(value, Real):
            raise TypeError(f"{name} is not numeric ({type(value).__name__}).")
        f_value = float(value)
        if not math.isfinite(f_value):
            raise ValueError(f"{name} must be finite, got {value!r}.")
        return f_value

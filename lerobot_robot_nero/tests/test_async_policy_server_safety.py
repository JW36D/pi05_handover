"""Safety checks for the local LeRobot async policy server patch."""

import logging
import threading
from queue import Queue
from types import SimpleNamespace

from lerobot.async_inference.policy_server import (
    PolicyServer,
    _disable_torch_compile_for_async,
)
from lerobot.transport import services_pb2


class DummyObservation:
    must_go = False

    def __init__(self, timestep: int):
        self._timestep = timestep

    def get_timestep(self) -> int:
        return self._timestep


def _server_with_observation(observation: DummyObservation) -> PolicyServer:
    server = PolicyServer.__new__(PolicyServer)
    server.observation_queue = Queue(maxsize=1)
    server.observation_queue.put(observation)
    server.config = SimpleNamespace(obs_queue_timeout=0.01, inference_latency=0.0)
    server._predicted_timesteps_lock = threading.Lock()
    server._predicted_timesteps = set()
    server.logger = logging.getLogger("test_async_policy_server_safety")
    return server


def test_failed_inference_does_not_poison_predicted_timesteps():
    server = _server_with_observation(DummyObservation(timestep=49))

    def fail_predict(_obs):
        raise RuntimeError("synthetic inference failure")

    server._predict_action_chunk = fail_predict

    response = server.GetActions(
        request=None,
        context=SimpleNamespace(peer=lambda: "test-client"),
    )

    assert response == services_pb2.Empty()
    assert 49 not in server._predicted_timesteps


def test_successful_inference_marks_predicted_timestep():
    server = _server_with_observation(DummyObservation(timestep=50))
    server._predict_action_chunk = lambda _obs: ["dummy-action"]

    response = server.GetActions(
        request=None,
        context=SimpleNamespace(peer=lambda: "test-client"),
    )

    assert response.data
    assert 50 in server._predicted_timesteps


def test_async_server_disables_torch_compile_by_default(monkeypatch):
    monkeypatch.delenv("LEROBOT_ASYNC_DISABLE_TORCH_COMPILE", raising=False)
    config = SimpleNamespace(compile_model=True, compile_mode="max-autotune")

    changed = _disable_torch_compile_for_async(
        config,
        logging.getLogger("test_async_policy_server_safety"),
    )

    assert changed is True
    assert config.compile_model is False


def test_async_server_can_keep_torch_compile_when_requested(monkeypatch):
    monkeypatch.setenv("LEROBOT_ASYNC_DISABLE_TORCH_COMPILE", "0")
    config = SimpleNamespace(compile_model=True, compile_mode="max-autotune")

    changed = _disable_torch_compile_for_async(
        config,
        logging.getLogger("test_async_policy_server_safety"),
    )

    assert changed is False
    assert config.compile_model is True

"""Regression tests for threshold-triggered async action refresh."""

import logging
import threading
from queue import Queue
from types import SimpleNamespace

from lerobot.async_inference.robot_client import RobotClient


class StubRobot:
    def get_observation(self):
        return {"observation.state": [0.0] * 8}


def _client_with_queue(queue_size: int):
    sent_observations = []

    client = RobotClient.__new__(RobotClient)
    client.robot = StubRobot()
    client.latest_action = 8
    client.latest_action_lock = threading.Lock()
    client.action_queue = Queue()
    client.action_queue_lock = threading.Lock()
    client.must_go = threading.Event()
    client.must_go.set()
    client.logger = logging.getLogger("test_async_robot_client_threshold")
    client.fps_tracker = SimpleNamespace(calculate_fps_metrics=lambda _timestamp: {})

    for _ in range(queue_size):
        client.action_queue.put(object())

    client.send_observation = lambda obs: sent_observations.append(obs) or True
    return client, sent_observations


def test_first_threshold_observation_is_forced_through_server_filters():
    client, sent_observations = _client_with_queue(queue_size=40)

    client.control_loop_observation(task="handover")

    assert len(sent_observations) == 1
    assert sent_observations[0].must_go is True
    assert sent_observations[0].get_timestep() == 8
    assert sent_observations[0].get_observation()["task"] == "handover"
    assert not client.must_go.is_set()


def test_followup_threshold_observation_is_not_forced_until_new_actions_arrive():
    client, sent_observations = _client_with_queue(queue_size=40)

    client.control_loop_observation(task="handover")
    client.control_loop_observation(task="handover")

    assert [obs.must_go for obs in sent_observations] == [True, False]

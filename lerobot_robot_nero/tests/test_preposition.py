import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_preposition_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "preposition.py"
    spec = importlib.util.spec_from_file_location("nero_preposition", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_target_is_training_start_q50():
    preposition = _load_preposition_module()
    args = argparse.Namespace(joints=None, parquet=None, episode=0)

    assert preposition.resolve_target(args) == preposition.TRAINING_START_Q50


def test_resolve_target_reads_v3_observation_state(tmp_path):
    pd = pytest.importorskip("pandas")
    preposition = _load_preposition_module()

    target = np.array([0.1, 0.2, 0.3, 1.4, 0.5, 0.6, 0.7, 0.08], dtype=np.float32)
    parquet = tmp_path / "episode.parquet"
    pd.DataFrame({
        "episode_index": [3],
        "frame_index": [0],
        "observation.state": [target],
    }).to_parquet(parquet)

    args = argparse.Namespace(joints=None, parquet=parquet, episode=3)
    np.testing.assert_allclose(preposition.resolve_target(args), target[:7])


def test_resolve_target_reads_v21_joint_position(tmp_path):
    pd = pytest.importorskip("pandas")
    preposition = _load_preposition_module()

    target = np.array([-0.1, 0.2, -0.3, 1.5, 0.0, 0.1, 0.4], dtype=np.float64)
    parquet = tmp_path / "episode.parquet"
    pd.DataFrame({
        "episode_index": [7],
        "frame_index": [0],
        "observation.state.agx_nero__arm.joint_position": [target],
    }).to_parquet(parquet)

    args = argparse.Namespace(joints=None, parquet=parquet, episode=7)
    np.testing.assert_allclose(preposition.resolve_target(args), target)

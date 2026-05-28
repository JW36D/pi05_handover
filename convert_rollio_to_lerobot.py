#!/usr/bin/env python
"""Convert rollio-exported dataset to LeRobot format (v3.0, for pi05 training).

Source layout (rollio, self-labeled v2.1 but not compatible with lerobot v3):
  output1/data/chunk-000/episode_XXXXXX.parquet
  output1/videos/chunk-000/realsense__color/episode_XXXXXX.mp4

Target: LeRobot v3.0 via LeRobotDataset.create()/add_frame()/save_episode().

Kept features:
  action                                       float32[8]  (agx_nero__arm.end_pose[0..6] + agx_nero__gripper.parallel_mit[0])
  observation.state                            float32[8]  (agx_nero arm.joint_position[7] + gripper.parallel_position[1])
  observation.images.realsense_color           video 1080x1920x3 h264

Dropped: depth, all airbot_play__*, agx_nero velocity/effort/end_effector_pose, gripper velocity/effort.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import av
import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_SOURCES = [
    Path("/home/jiawei/projects/dataset/output1"),
    Path("/home/jiawei/projects/dataset/output"),
]
TASK = "hand over the object"
FPS = 30
VIDEO_H, VIDEO_W = 1080, 1920

ACTION_NAMES = [
    "agx_nero__arm.end_pose.0", "agx_nero__arm.end_pose.1", "agx_nero__arm.end_pose.2",
    "agx_nero__arm.end_pose.3", "agx_nero__arm.end_pose.4", "agx_nero__arm.end_pose.5",
    "agx_nero__arm.end_pose.6", "agx_nero__gripper.parallel_mit.0",
]
STATE_NAMES = [
    "agx_nero__arm.joint_position.0", "agx_nero__arm.joint_position.1",
    "agx_nero__arm.joint_position.2", "agx_nero__arm.joint_position.3",
    "agx_nero__arm.joint_position.4", "agx_nero__arm.joint_position.5",
    "agx_nero__arm.joint_position.6", "agx_nero__gripper.parallel_position.0",
]

FEATURES = {
    "action": {"dtype": "float32", "shape": (8,), "names": ACTION_NAMES},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
    "observation.images.realsense_color": {
        "dtype": "video",
        "shape": (VIDEO_H, VIDEO_W, 3),
        "names": ["height", "width", "channel"],
    },
}


def decode_mp4(path: Path):
    """Yield each frame of the mp4 as an RGB uint8 ndarray of shape (H, W, 3)."""
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def build_episode_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    joint = np.stack(
        [np.asarray(x, dtype=np.float32)
         for x in df["observation.state.agx_nero__arm.joint_position"].values]
    )
    grip = np.stack(
        [np.asarray(x, dtype=np.float32)
         for x in df["observation.state.agx_nero__gripper.parallel_position"].values]
    )
    obs_state = np.concatenate([joint, grip], axis=1)
    action = np.stack(
        [np.asarray(x, dtype=np.float32) for x in df["action"].values]
    )
    if joint.shape[1] != 7 or grip.shape[1] != 1 or action.shape[1] != 8:
        raise ValueError(
            f"Unexpected per-frame shapes: joint={joint.shape}, grip={grip.shape}, action={action.shape}"
        )
    return obs_state, action


def count_video_frames(path: Path) -> int:
    """Read frame count from mp4 container metadata (no full decode)."""
    with av.open(str(path)) as c:
        n = c.streams.video[0].frames
    if n <= 0:
        with av.open(str(path)) as c:
            n = sum(1 for _ in c.decode(video=0))
    return n


def collect_episodes(sources: list[Path]) -> list[tuple[Path, Path, int]]:
    """Return [(parquet_path, mp4_path, source_idx)] in stable order (source order,
    then ascending episode number within each source)."""
    out = []
    for s_idx, src in enumerate(sources):
        pqs = sorted((src / "data/chunk-000").glob("episode_*.parquet"))
        if not pqs:
            raise FileNotFoundError(f"no parquet files under {src/'data/chunk-000'}")
        for pq in pqs:
            ep_num = int(pq.stem.split("_")[1])
            mp4 = src / "videos/chunk-000/realsense__color" / f"episode_{ep_num:06d}.mp4"
            if not mp4.exists():
                raise FileNotFoundError(mp4)
            out.append((pq, mp4, s_idx))
    return out


def convert(out_root: Path, repo_id: str, sources: list[Path], limit: int | None = None) -> None:
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=FEATURES,
        root=str(out_root),
        use_videos=True,
        vcodec="h264",
        streaming_encoding=True,
    )

    episodes = collect_episodes(sources)
    if limit is not None:
        episodes = episodes[:limit]

    total = len(episodes)
    total_frames_written = 0
    total_trimmed = 0
    trimmed_eps = 0
    t0 = time.time()

    for i, (parquet_path, mp4_path, src_idx) in enumerate(episodes):
        ep_num = int(parquet_path.stem.split("_")[1])

        df = pd.read_parquet(parquet_path)
        n_rows = len(df)
        n_vid = count_video_frames(mp4_path)
        n_use = min(n_rows, n_vid)
        if n_rows != n_vid:
            trimmed_eps += 1
            total_trimmed += abs(n_rows - n_vid)
            print(f">>> [align] src{src_idx} ep{ep_num:06d}: parquet={n_rows} "
                  f"video={n_vid} -> using {n_use}")

        obs_state, action = build_episode_arrays(df.iloc[:n_use])

        decoder = decode_mp4(mp4_path)
        for r in range(n_use):
            img = next(decoder)
            if img.shape != (VIDEO_H, VIDEO_W, 3):
                raise RuntimeError(f"src{src_idx} ep{ep_num} frame {r}: unexpected shape {img.shape}")
            ds.add_frame({
                "action": action[r],
                "observation.state": obs_state[r],
                "observation.images.realsense_color": img,
                "task": TASK,
            })
        # drain remaining video frames (when video was longer than parquet)
        for _ in decoder:
            pass

        ds.save_episode()
        total_frames_written += n_use

        done_idx = i + 1
        if done_idx % 10 == 0 or done_idx == total:
            elapsed = time.time() - t0
            eta = elapsed / done_idx * (total - done_idx)
            print(f">>> [{done_idx}/{total}] src{src_idx} ep{ep_num:06d} saved ({n_use} frames) "
                  f"| total={total_frames_written} | "
                  f"elapsed={elapsed/60:.1f}min eta={eta/60:.1f}min", flush=True)

    ds.finalize()
    print(f">>> DONE: {total} episodes, {total_frames_written} frames -> {out_root}", flush=True)
    print(f">>>       alignment trims: {trimmed_eps} episodes, {total_trimmed} frames dropped", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="output dataset root")
    ap.add_argument("--repo-id", default="local/handover_pi05",
                    help="repo_id written into metadata")
    ap.add_argument("--source", type=Path, action="append", default=None,
                    help="rollio source root; repeat to merge multiple. "
                         "Order = append order. Defaults to output1 then output.")
    ap.add_argument("--n", type=int, default=None,
                    help="limit total episode count across sources (for smoke test)")
    args = ap.parse_args()

    out = args.out.expanduser().resolve()
    if out.exists():
        raise SystemExit(
            f"refusing to write to existing path: {out}\n"
            f"LeRobotDataset.create() requires a non-existent root — delete it first."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    sources = args.source if args.source else DEFAULT_SOURCES
    sources = [s.expanduser().resolve() for s in sources]

    print(f"sources  = {[str(s) for s in sources]}")
    print(f"out      = {out}")
    print(f"repo_id  = {args.repo_id}")
    print(f"task     = {TASK!r}")
    print(f"limit    = {args.n}")
    convert(out, args.repo_id, sources, args.n)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Profile each phase of pi05 policy loading.

Tells you exactly where the 2-3 min are going so we can target the right
optimisation.

Usage:
    python scripts/profile_policy_load.py \\
        --checkpoint ../checkpoints/050000/pretrained_model \\
        --device cuda
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def stage(name: str):
    print(f"\n[ START ] {name}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        print(f"[ DONE  ] {name}  ({dt:.2f}s)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to pretrained_model directory")
    p.add_argument("--device", default="cuda",
                   help="cuda | cpu | cuda:0")
    p.add_argument("--policy_type", default="pi05")
    args = p.parse_args()

    # ----------------------------------------------------------------
    # Phase 0: heavy library imports
    # ----------------------------------------------------------------
    with stage("import torch"):
        import torch  # noqa: F401
    with stage("import transformers"):
        import transformers  # noqa: F401
    with stage("import lerobot.policies.factory"):
        from lerobot.policies.factory import (  # type: ignore
            get_policy_class,
            make_pre_post_processors,
        )

    # ----------------------------------------------------------------
    # Phase 1: model class construction
    # ----------------------------------------------------------------
    with stage(f"get_policy_class({args.policy_type!r})"):
        policy_class = get_policy_class(args.policy_type)

    # ----------------------------------------------------------------
    # Phase 2: from_pretrained — disk → CPU
    # ----------------------------------------------------------------
    with stage(f"policy_class.from_pretrained({args.checkpoint!s})"):
        policy = policy_class.from_pretrained(str(args.checkpoint))

    # Check param count + memory
    import torch
    n_params = sum(p.numel() for p in policy.parameters())
    bytes_per_param = next(policy.parameters()).element_size()
    mem_mb = n_params * bytes_per_param / 1e6
    print(f"[ INFO  ] params: {n_params/1e9:.2f}B  ({mem_mb:.1f} MB on {next(policy.parameters()).dtype})", flush=True)

    # ----------------------------------------------------------------
    # Phase 3: CPU → GPU transfer
    # ----------------------------------------------------------------
    if args.device != "cpu":
        with stage(f"policy.to({args.device!r})"):
            policy.to(args.device)
        with stage("torch.cuda.synchronize()"):
            if "cuda" in args.device:
                torch.cuda.synchronize()

    # ----------------------------------------------------------------
    # Phase 4: pre/post processors
    # ----------------------------------------------------------------
    with stage("make_pre_post_processors"):
        pre, post = make_pre_post_processors(
            policy.config,
            pretrained_path=str(args.checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": args.device},
            },
            postprocessor_overrides={
                "device_processor": {"device": args.device},
            },
        )

    # ----------------------------------------------------------------
    # Phase 5: warm-up inference (first call triggers cuDNN benchmark etc.)
    # ----------------------------------------------------------------
    print("\n[ INFO  ] All loading done. Now timing first inference (warm-up)…", flush=True)
    H, W = 1080, 1920
    dummy_obs = {
        "observation.state": torch.zeros(1, 8, device=args.device),
        "observation.images.realsense_color": torch.zeros(
            1, 3, H, W, dtype=torch.float32, device=args.device
        ),
        "task": ["test task"],
    }
    try:
        dummy_obs = pre(dummy_obs)
        with stage("first predict_action_chunk (cuDNN warmup)"):
            with torch.no_grad():
                _ = policy.predict_action_chunk(dummy_obs)
            if "cuda" in args.device:
                torch.cuda.synchronize()
        with stage("second predict_action_chunk (steady-state)"):
            with torch.no_grad():
                _ = policy.predict_action_chunk(dummy_obs)
            if "cuda" in args.device:
                torch.cuda.synchronize()
    except Exception as exc:
        print(f"[ SKIP  ] inference warmup failed: {exc}")

    print("\nProfile complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Drop-in replacement for `lerobot.async_inference.policy_server` that wires
Real-Time Chunking (RTC) into the existing async inference loop.

Why a separate file?
    Upstream `policy_server.py` calls `policy.predict_action_chunk(observation)`
    without the kwargs RTC needs (`prev_chunk_left_over`, `inference_delay`).
    Even if you set `rtc_config.enabled = True` in the checkpoint config.json,
    the RTCProcessor inside pi05 receives `prev_chunk_left_over=None` and
    silently no-ops (returns plain v_t). This script subclasses PolicyServer
    to (a) inject the RTC config into the loaded policy and (b) track the
    previous action chunk so each call passes the correct kwargs.

Approximations (worth understanding before tuning):

  • `prev_chunk_left_over` — the server has no direct view into the client's
    action queue. We estimate the consumed portion as
        consumed = (now - chunk_returned_at) * fps
    and treat the tail as the "left-over". Network jitter and the client's
    `chunk_size_threshold`-driven send timing add ±2-3 frame noise. RTC's
    soft attention schedule (EXP) tolerates this in practice.

  • `inference_delay` — set to the previous inference's wall-clock duration
    times fps. pi05 inference time is fairly stable, so this estimate
    converges within a few chunks.

Usage (drop-in replacement for upstream command — flags identical):

    python lerobot_robot_nero/scripts/policy_server_rtc.py \\
        --host=127.0.0.1 --port=8080 --fps=30

Robot client side does NOT change. Same `--policy_type=pi05`,
`--pretrained_name_or_path=...`, etc.

Disable RTC at runtime (without restarting):
    NERO_RTC_DISABLED=1  → falls back to upstream behaviour (kwargs not passed,
                          rtc_config.enabled stays at whatever the checkpoint
                          says, which is `null` for our pi05 → no RTC).
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Make sure the local lerobot/ source (not any pip-installed copy) is found.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lerobot.async_inference import policy_server as upstream  # noqa: E402
from lerobot.configs import RTCAttentionSchedule  # noqa: E402
from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: E402

logger = logging.getLogger("policy_server_rtc")


# --- RTC defaults — match the official docs' recommended values ---
RTC_EXECUTION_HORIZON = 10
RTC_MAX_GUIDANCE_WEIGHT = 10.0
RTC_PREFIX_ATTENTION_SCHEDULE = RTCAttentionSchedule.EXP


class RTCPolicyServer(upstream.PolicyServer):
    """PolicyServer that injects RTC config and passes the RTC kwargs each step.

    Drop-in: every upstream method except `SendPolicyInstructions` and
    `_get_action_chunk` is inherited unchanged.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._rtc_disabled: bool = bool(int(os.environ.get("NERO_RTC_DISABLED", "0")))
        # Tracking state for prev_chunk_left_over estimation.
        self._rtc_prev_chunk: Any = None  # torch.Tensor (T, A) or None
        self._rtc_chunk_returned_at: float | None = None
        self._rtc_inference_duration: float | None = None

        if self._rtc_disabled:
            self.logger.warning(
                "NERO_RTC_DISABLED=1 set — RTC integration is OFF; behaving like upstream."
            )

    # ---------------- Hook 1: inject RTCConfig after model is loaded ----------------

    def SendPolicyInstructions(self, request, context):  # noqa: N802 (gRPC method name)
        # Let upstream do all the heavy lifting (load checkpoint, .to(device),
        # build pre/post processors). Only after that do we mutate the policy.
        result = super().SendPolicyInstructions(request, context)
        if not self._rtc_disabled:
            self._enable_rtc_on_loaded_policy()
        return result

    def _enable_rtc_on_loaded_policy(self) -> None:
        if not hasattr(self, "policy") or self.policy is None:
            self.logger.warning("RTC inject skipped: policy not loaded.")
            return
        if not hasattr(self.policy, "init_rtc_processor"):
            self.logger.info(
                "Policy %s has no init_rtc_processor — skipping RTC.",
                type(self.policy).__name__,
            )
            return

        cfg = getattr(self.policy, "config", None)
        if cfg is None:
            self.logger.warning("RTC inject skipped: policy has no .config.")
            return

        # If checkpoint already enabled RTC, respect that and just log it.
        existing = getattr(cfg, "rtc_config", None)
        if existing is not None and getattr(existing, "enabled", False):
            self.logger.info(
                "RTC already enabled in checkpoint config (no override). "
                "execution_horizon=%s max_guidance_weight=%s schedule=%s",
                existing.execution_horizon,
                existing.max_guidance_weight,
                existing.prefix_attention_schedule,
            )
            return

        cfg.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=RTC_EXECUTION_HORIZON,
            max_guidance_weight=RTC_MAX_GUIDANCE_WEIGHT,
            prefix_attention_schedule=RTC_PREFIX_ATTENTION_SCHEDULE,
        )
        self.policy.init_rtc_processor()
        self.logger.info(
            "RTC enabled on loaded %s policy (horizon=%d, guidance=%.1f, schedule=%s).",
            type(self.policy).__name__,
            RTC_EXECUTION_HORIZON,
            RTC_MAX_GUIDANCE_WEIGHT,
            RTC_PREFIX_ATTENTION_SCHEDULE.value,
        )

    # ---------------- Hook 2: feed RTC kwargs into each chunk inference -------------

    def _get_action_chunk(self, observation):
        """Run policy inference; pass prev_chunk + inference_delay if available."""
        if self._rtc_disabled:
            return super()._get_action_chunk(observation)

        kwargs: dict[str, Any] = {}

        fps = float(getattr(self.config, "fps", 30.0)) or 30.0

        # Estimate prev_chunk_left_over from wall-clock since the last chunk
        # was returned. The client has been consuming at `fps`, so:
        #   consumed_steps ≈ elapsed * fps
        if self._rtc_prev_chunk is not None and self._rtc_chunk_returned_at is not None:
            elapsed = time.monotonic() - self._rtc_chunk_returned_at
            chunk_len = self._rtc_prev_chunk.shape[0]
            consumed = max(0, min(int(round(elapsed * fps)), chunk_len))
            tail = self._rtc_prev_chunk[consumed:]
            if tail.shape[0] > 0:
                kwargs["prev_chunk_left_over"] = tail

        # Estimate inference_delay from previous inference duration. First
        # call has no estimate → don't pass; pi05 will fall back to its
        # internal default.
        if self._rtc_inference_duration is not None:
            kwargs["inference_delay"] = max(1, int(round(self._rtc_inference_duration * fps)))

        # Run the underlying inference, time it.
        t0 = time.monotonic()
        chunk = self.policy.predict_action_chunk(observation, **kwargs)
        self._rtc_inference_duration = time.monotonic() - t0

        # Mirror upstream's shape handling.
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)
        chunk = chunk[:, : self.actions_per_chunk, :]

        # Cache for next call. Detach so the saved tensor doesn't hold the
        # autograd graph (we're in eval mode anyway, but be explicit).
        self._rtc_prev_chunk = chunk.squeeze(0).detach()
        self._rtc_chunk_returned_at = time.monotonic()
        return chunk


def main() -> None:
    # Monkey-patch the PolicyServer class in upstream so `serve()` instantiates
    # our subclass without us having to re-implement the gRPC plumbing.
    upstream.PolicyServer = RTCPolicyServer
    # Reuse upstream's @draccus.wrap()'d serve() — this gives us the same
    # CLI surface (--host, --port, --fps, ...).
    upstream.serve()


if __name__ == "__main__":
    main()

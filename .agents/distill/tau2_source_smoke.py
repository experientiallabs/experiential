"""End-to-end smoke of the tau2 rollout source: one real episode, real Tinker sampling.

Runs `collect_tau2_rollouts` on a single pinned airline task with the Qwen3.5-9B
BASE model as the sampled policy (student-before shape: `model == model_type`, no
adapter) and the pinned azure/gpt-5.4-mini user simulator. Verifies the whole
chain end to end: tau2 subprocess -> loopback proxy -> TinkerChatProvider (tool
rendering + parsing) -> TokenRecorder span sink -> results.json reward join.

Cost: one episode (a few cents of Tinker sampling + user-sim tokens).

Run from the worktree root with TINKER_API_KEY, AZURE_API_KEY, AZURE_API_BASE,
AZURE_API_VERSION set:

    uv run python .agents/distill/tau2_source_smoke.py \
        --tau2-root ~/Desktop/Projects/world-model-harness/packages/environment-capture/tau-bench
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from wmo.distill.config import DistillConfig
from wmo.distill.tau2 import collect_tau2_rollouts
from wmo.harness.doc import HarnessDoc
from wmo.providers.base import ProviderConfig, ProviderKind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("tau2_source_smoke")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau2-root", required=True, help="packages/environment-capture/tau-bench")
    parser.add_argument("--task-id", default="airline/0")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--run-dir", default=".wmo/distill-runs/tau2-source-smoke")
    args = parser.parse_args()

    tau2_root = Path(args.tau2_root).expanduser()
    cfg = DistillConfig.model_validate(
        {
            "student": {"base_model": args.model},
            "teacher": {"model": "Qwen/Qwen3.6-27B"},
            "tau2": {
                "tau2_bin": str(tau2_root / ".venv" / "bin" / "tau2"),
                "data_dir": str(tau2_root / "tau2-bench" / "data"),
                "user_llm": "azure/gpt-5.4-mini",
            },
            "rollout": {"max_turns": 40, "episode_timeout_s": 600.0},
            "sampling": {"temperature": 1.0, "max_tokens": 4096},
            "train": {"group_size": 1, "trial_concurrency": 1},
        }
    )
    provider = ProviderConfig(kind=ProviderKind.TINKER, model=args.model, model_type=args.model)
    run_dir = Path(args.run_dir)

    records, stats = collect_tau2_rollouts(
        0, [args.task_id], cfg, HarnessDoc.baseline(), provider, run_dir
    )

    [record] = records
    logger.info("record: %s", json.dumps(record.model_dump(mode="json", exclude={"spans"})))
    logger.info(
        "spans: %d (turns), first span: prompt %d ids -> sampled %d ids, logprobs %d",
        len(record.spans),
        len(record.spans[0].prompt_token_ids) if record.spans else 0,
        len(record.spans[0].sampled_token_ids) if record.spans else 0,
        len(record.spans[0].sampled_logprobs) if record.spans else 0,
    )
    # The prefix property: each later span's prompt must extend the previous
    # span's prompt + sampled ids exactly (this is what makes the episode one
    # datum and proves nothing was re-encoded).
    prefix_ok = True
    for earlier, later in zip(record.spans, record.spans[1:], strict=False):
        expected = earlier.prompt_token_ids + earlier.sampled_token_ids
        if later.prompt_token_ids[: len(expected)] != expected:
            prefix_ok = False
            logger.warning(
                "prefix break between call %d and %d (fallback re-render)",
                earlier.call_index,
                later.call_index,
            )
    logger.info(
        "stats: %s",
        json.dumps(stats.model_dump(mode="json")),
    )
    ok = bool(record.spans) and not record.infra_failed
    logger.info(
        "VERDICT: %s (spans=%d, infra_failed=%s, reward=%.2f, stop=%s, prefix_chain=%s)",
        "PASS" if ok else "FAIL",
        len(record.spans),
        record.infra_failed,
        record.reward,
        record.stop_reason,
        "intact" if prefix_ok else "broken (see warnings)",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Smoke the text bridge end to end: real teacher transcripts -> one live CE step.

Proves the tier-1 path with real money and real weights, cheaply: read recorded
tau2 teacher transcripts (any provider's; cycle 1's Qwen3.6-27B episodes are
the handy corpus), re-encode them under the STUDENT's renderer through
`wmo.distill.text_episodes`, then run ONE `forward_backward` +  `optim_step`
against a live Tinker LoRA using the same cross_entropy wire format the
supervised phase uses.

What it proves that unit tests cannot: the re-encoded datums are accepted by
the live cross_entropy loss (the keyset and dtypes are right), a real LoRA
takes a gradient step on them, and the whole path costs what the forecast says.

Cost: one minibatch of student training tokens (cents; capped by --max-datums).

    uv run python .agents/distill/text_bridge_smoke.py \
        --episodes-root .wmo/distill-runs/tau2-cycle1/warmup-rollouts/tau2/step-0000 \
        --teacher-model Qwen/Qwen3.6-27B --max-datums 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from wmo.distill.config import DistillConfig
from wmo.distill.data import build_datums, to_tinker_sft_datums
from wmo.distill.rendering import build_offline_rendering
from wmo.distill.tau2 import episodes_from_tau2_results
from wmo.distill.text_episodes import text_warmup_manifest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
logger = logging.getLogger("text_bridge_smoke")

CROSS_ENTROPY_LOSS = "cross_entropy"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", required=True, help="dir of <episode>/results.json")
    parser.add_argument("--teacher-model", required=True, help="who wrote the transcripts")
    parser.add_argument("--student", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-datums", type=int, default=8, help="cost cap: datums per step")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--dry-run", action="store_true", help="build datums, spend nothing")
    args = parser.parse_args()

    root = Path(args.episodes_root)
    episode_dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "results.json").exists())
    if not episode_dirs:
        logger.error("no <episode>/results.json under %s", root)
        return 2

    episodes = []
    for episode_dir in episode_dirs:
        task_id = episode_dir.name.rsplit("-a", 1)[0].replace("-", "/", 1)
        episodes.extend(
            episodes_from_tau2_results(
                episode_dir / "results.json",
                teacher_model=args.teacher_model,
                task_id=task_id,
            )
        )
    if not episodes:
        logger.error("no graded transcripts with resolvable system prompts under %s", root)
        return 2
    kept = [episode for episode in episodes if episode.passed]
    logger.info(
        "read %d graded episode(s) from %d dir(s); %d passed the keep filter",
        len(episodes),
        len(episode_dirs),
        len(kept),
    )
    if not kept:
        logger.error("no passing episodes to train on")
        return 2

    rendering = build_offline_rendering(args.student)
    manifest = text_warmup_manifest(kept, rendering, teacher_model=args.teacher_model)
    records = manifest.records
    cfg = DistillConfig.model_validate(
        {
            "student": {"base_model": args.student, "lora_rank": args.lora_rank},
            "teacher": {"model": args.teacher_model},
            "tau2": {"tau2_bin": "/unused", "data_dir": "/unused"},
            "train": {"steps": 0, "learning_rate": args.learning_rate},
            "warmup": {"steps": 1},
        }
    )
    datums, stats = build_datums(records, cfg)
    logger.info(
        "datums: %d from %d episode(s) (drops: overflow %d, overlong %d)",
        len(datums),
        len(records),
        stats.overflow_drops,
        stats.overlong_drops,
    )
    batch = datums[: args.max_datums]
    loss_tokens = sum(int(sum(d.loss_mask)) for d in batch)
    total_tokens = sum(len(d.model_input_tokens) for d in batch)
    logger.info(
        "training batch: %d datum(s), %d total tokens, %d loss tokens; all hard-target: %s",
        len(batch),
        total_tokens,
        loss_tokens,
        all(d.hard_targets_only for d in batch),
    )
    if args.dry_run:
        logger.info("VERDICT: DRY RUN ok (nothing spent)")
        return 0

    wire = to_tinker_sft_datums(batch)
    logger.info("converted %d datum(s) to the live cross_entropy wire format", len(wire))

    from wmo.providers.tinker import shared_service_client

    service = shared_service_client()
    training = service.create_lora_training_client(
        base_model=args.student, rank=args.lora_rank
    )
    logger.info("training client up for %s (rank %d)", args.student, args.lora_rank)
    forward = training.forward_backward(wire, loss_fn=CROSS_ENTROPY_LOSS)
    optim = training.optim_step(dict(learning_rate=args.learning_rate))
    forward_result = forward.result()
    optim.result()
    metrics = getattr(forward_result, "metrics", None) or {}
    logger.info("forward_backward metrics: %s", metrics)
    logger.info(
        "VERDICT: PASS - the live cross_entropy loss accepted %d text-derived datum(s) "
        "(%d loss tokens) and the LoRA took a step",
        len(wire),
        loss_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

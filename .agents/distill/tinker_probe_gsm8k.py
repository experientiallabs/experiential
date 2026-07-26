"""Standalone Tinker-codepath probe: on-policy distillation on inline GSM8K-style prompts.

Purpose: isolate the Tinker training codepath (sample -> teacher
compute_logprobs -> reverse-KL advantages -> forward_backward with the
importance_sampling loss -> optim_step -> save_weights_for_sampler refresh)
from the harbor/E2B/pi rollout stack. Rollouts are single-turn math
completions sampled directly from the current student weights; on-policy
distillation needs no rewards, so there are no graders. The pass signal is
train/reverse_kl_per_token trending down across steps.

Secondary purpose: per-SDK-call wall-clock timings (count/mean/p95/max for
every sample, compute_logprobs, forward_backward, optim_step, save_state, and
save_weights_for_sampler call) as the "normal" baseline for the planned
wedge-hardening around the SDK's internal retry loop.

Probe-only seams, documented deviations from the product path:
- `wmh.distill.loop.collect_rollouts` is monkeypatched (plain attribute
  assignment) with `ProbeRollouts`, mirroring `loop_test._FakeRollouts`. The
  probe never imports harbor; `harbor.job_template` points at the checked-in
  placeholder YAML next to this script purely to satisfy the config shape.
- Pricing uses placeholder USD/Mtok rates (see `_PLACEHOLDER_PRICING`) so the
  budget cap enforces and spend is reported; re-verify against the live
  Tinker models page before treating the USD numbers as real.
- The probe owns its training client (`ProbeTrainingClient`) instead of the
  loop's `SdkTrainingClient`: the live Tinker service (SDK 0.23.3) rejects
  the `mask` loss_fn_inputs key for the importance_sampling loss
  (`ImportanceSamplingLossExtraArgs.__init__() got an unexpected keyword
  argument 'mask'`), which `wmh.distill.data.to_tinker_datums` always sends.
  `attach_advantages` zeroes advantages at every non-loss position, so the
  mask is redundant with the advantages and `_compat_tinker_datums` drops it,
  loss-equivalently; everything else mirrors the shifted next-token layout.

Run from the repo root with only TINKER_API_KEY set:

    uv run python .agents/distill/tinker_probe_gsm8k.py --steps 4 \
        --tasks-per-batch 2 --group-size 2 --max-tokens 128 \
        --budget-usd 2 --run-dir .wmh/distill-runs/probe-gsm8k-dev
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4

import tinker
from llm_waterfall.types import ChatMessage

import wmh.distill.loop as loop_module
from wmh.agents.default import default_agent
from wmh.distill.config import (
    BudgetConfig,
    DistillConfig,
    EvalConfig,
    GateConfig,
    HarborConfig,
    PricingConfig,
    RolloutConfig,
    SamplingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
    WandbConfig,
)
from wmh.distill.data import TrainDatum
from wmh.distill.loop import (
    DistillBudgetError,
    DistillProgress,
    DistillResult,
    OptimStepOutput,
    SdkSamplingClient,
    TrainStepOutput,
    run_distillation,
)
from wmh.distill.rendering import ChatRendering, RendererTokenizer, build_renderer
from wmh.distill.rollouts import RolloutStats
from wmh.distill.store import AdapterStore, DistillRunStore
from wmh.distill.teacher import EncodingTokenizer
from wmh.distill.tokens import TrialRecord
from wmh.harness.doc import HarnessDoc
from wmh.providers.base import ProviderConfig
from wmh.providers.tinker import TINKER_API_KEY_ENV, SampledSequenceLike, TokenSpan

logger = logging.getLogger("tinker_probe_gsm8k")

RUN_NAME = "probe-gsm8k"

_JOB_TEMPLATE = Path(__file__).resolve().parent / "tb2-harbor-job-template.yaml"
"""Placeholder harbor JobConfig; the probe monkeypatches rollouts and never reads it."""

_PLACEHOLDER_PRICING = PricingConfig(
    student_prefill=0.25,
    student_sample=1.0,
    student_train=1.0,
    teacher_prefill=0.5,
)
"""Conservative placeholder USD/Mtok rates for small Qwen models.

Set so the budget cap enforces and the run reports a spend number; re-verify
against the live Tinker models page before reading the USD as real spend.
"""

SYSTEM_PROMPT = (
    "You are a concise math tutor. Solve the word problem step by step, "
    "then state the final numeric answer on its own last line."
)

USER_SUFFIX = "Think step by step, then give the final answer."

PROBLEMS: tuple[str, ...] = (
    "Maya has 24 stickers. She gives 7 to her brother and buys 12 more. "
    "How many stickers does she have now?",
    "A bakery sells muffins for $3 each. Leo buys 5 muffins and pays with a "
    "$20 bill. How much change does he get?",
    "A train travels 60 miles per hour for 3 hours, then 40 miles per hour "
    "for 2 hours. How many miles does it travel in total?",
    "Sara reads 15 pages of her book every night. The book has 210 pages. "
    "How many nights will it take her to finish?",
    "A farmer has 4 fields. Each field grows 35 pumpkins. He sells 48 "
    "pumpkins at the market. How many pumpkins are left?",
    "Tom saves $8 every week. After 9 weeks he spends $25 on a game. "
    "How much money does he have left?",
    "A classroom has 6 rows of desks with 7 desks in each row. If 5 desks "
    "are empty, how many desks are occupied?",
    "Nina bakes 3 dozen cookies. She keeps 8 for herself and splits the rest "
    "equally among 4 friends. How many cookies does each friend get?",
    "A movie ticket costs $12 for adults and $8 for children. What is the "
    "total cost for 2 adults and 3 children?",
    "Jake runs 4 miles on Monday, twice as far on Tuesday, and 3 miles on "
    "Wednesday. How many miles does he run in total?",
    "A shop had 120 apples. It sold 45 in the morning and 38 in the "
    "afternoon. How many apples remain?",
    "Each bus holds 40 students. How many buses are needed to carry 130 students?",
    "Lily buys 3 notebooks for $4 each and 2 pens for $1.50 each. "
    "How much does she spend in total?",
    "A rectangle is 9 cm long and 5 cm wide. What is its perimeter?",
    "Ben has twice as many marbles as Ava. Ava has 17 marbles. "
    "How many marbles do they have together?",
    "A water tank holds 500 liters. A pump fills it at 25 liters per minute. "
    "How many minutes does it take to fill the tank from empty?",
    "Emma earns $15 per hour and works 6 hours on Saturday. She spends $32 "
    "of her earnings. How much does she keep?",
    "There are 96 chairs arranged equally in 8 rows. How many chairs are in each row?",
    "Carlos had 50 baseball cards. He traded 12 cards away for 5 rare ones. "
    "How many cards does he have now?",
    "A pizza is cut into 8 slices. Three pizzas are ordered for a party and "
    "19 slices are eaten. How many slices are left?",
    "A library lends 74 books on Monday and 58 on Tuesday. If 39 books are "
    "returned on Wednesday, how many books are still out?",
    "Sophie plants 5 rows of tulips with 12 tulips per row. Rabbits eat 14 "
    "of them. How many tulips survive?",
    "A phone costs $240. Dan pays $60 up front and the rest in 6 equal "
    "monthly payments. How much is each payment?",
    "Mia swims 20 laps per day, 5 days a week. How many laps does she swim in 3 weeks?",
    "A jar contains 34 red, 27 blue, and 19 green candies. How many candies are in the jar?",
    "Oliver is 3 times as old as his sister, who is 6. In 4 years, how old will Oliver be?",
    "A car uses 8 liters of fuel per 100 km. How many liters does it use on a 250 km trip?",
    "Zoe scores 78, 85, and 92 on three tests. What is her average score?",
    "A store discounts an $80 jacket by 25 percent. What is the sale price?",
    "Sam packs 144 eggs into cartons of 12. How many cartons does he fill?",
    "A garden hose fills a 60 liter barrel in 15 minutes. How many liters per minute is that?",
    "Priya walks 1.5 km to school and 1.5 km back home every school day. "
    "How many km does she walk in a 5 day school week?",
)

TASK_IDS: tuple[str, ...] = tuple(f"gsm8k-{index:03d}" for index in range(len(PROBLEMS)))
PROBLEM_BY_ID: dict[str, str] = dict(zip(TASK_IDS, PROBLEMS, strict=True))

N_TRAIN = 28
TRAIN_TASK_IDS: tuple[str, ...] = TASK_IDS[:N_TRAIN]
HOLDOUT_TASK_IDS: tuple[str, ...] = TASK_IDS[N_TRAIN:]


# -- SDK-call timing -------------------------------------------------------------------------


@dataclass(frozen=True)
class CallStats:
    """Wall-clock summary of one SDK call kind."""

    count: int
    mean_s: float
    p95_s: float
    max_s: float


class SdkCallTimer:
    """Accumulates wall-clock durations per SDK call kind (single-threaded loop)."""

    def __init__(self) -> None:
        self._durations: dict[str, list[float]] = {}

    def record(self, name: str, seconds: float) -> None:
        """Record one call's duration and log it live (the hang-watch trail)."""
        self._durations.setdefault(name, []).append(seconds)
        logger.info("[sdk] %s took %.2fs", name, seconds)

    def summary(self) -> dict[str, CallStats]:
        """Per-call-kind stats, keyed by call name."""
        stats: dict[str, CallStats] = {}
        for name, values in sorted(self._durations.items()):
            ordered = sorted(values)
            p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
            stats[name] = CallStats(
                count=len(ordered),
                mean_s=statistics.fmean(ordered),
                p95_s=ordered[p95_index],
                max_s=ordered[-1],
            )
        return stats

    def log_summary(self) -> None:
        """Log the timing table (the wedge-hardening baseline)."""
        logger.info("SDK call timings (wall clock):")
        logger.info("  %-36s %5s %8s %8s %8s", "call", "count", "mean_s", "p95_s", "max_s")
        for name, stats in self.summary().items():
            logger.info(
                "  %-36s %5d %8.2f %8.2f %8.2f",
                name,
                stats.count,
                stats.mean_s,
                stats.p95_s,
                stats.max_s,
            )


class TimedSamplingClient:
    """Times every sample/compute_logprobs call on one real sampling client."""

    def __init__(self, inner: SdkSamplingClient, timer: SdkCallTimer, role: str) -> None:
        self._inner = inner
        self._timer = timer
        self._role = role

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
    ) -> SampledSequenceLike:
        """One timed synchronous sample."""
        started = time.monotonic()
        try:
            return self._inner.sample(
                prompt_token_ids, max_tokens=max_tokens, temperature=temperature
            )
        finally:
            self._timer.record(f"sample.{self._role}", time.monotonic() - started)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        """One timed synchronous compute_logprobs call."""
        started = time.monotonic()
        try:
            return self._inner.compute_logprobs(token_ids)
        finally:
            self._timer.record(f"compute_logprobs.{self._role}", time.monotonic() - started)

    def get_tokenizer(self) -> EncodingTokenizer:
        """The base model's tokenizer (untimed; local after the first fetch)."""
        return self._inner.get_tokenizer()


def _compat_tinker_datums(train_datums: Sequence[TrainDatum]) -> list[tinker.Datum]:
    """`to_tinker_datums` minus the `mask` loss_fn_inputs key (probe-only compat).

    The live service's importance_sampling loss args (SDK 0.23.3) accept only
    target_tokens, logprobs, and advantages; sending `mask` fails the request
    server-side. `attach_advantages` zeroes advantages at every non-loss
    position, so dropping the mask is loss-equivalent (a zero advantage
    contributes zero gradient). The shifted next-token layout is unchanged.

    Args:
        train_datums: Datums with advantages attached (`attach_advantages`).

    Returns:
        One `tinker.Datum` per input datum, in order.

    Raises:
        ValueError: If a datum has no advantages attached or fewer than 2
            tokens (too short for the input/target shift).
    """
    out: list[tinker.Datum] = []
    for datum in train_datums:
        if not datum.advantages:
            raise ValueError(
                f"datum (trial {datum.trial_name}) has no advantages attached; "
                "attach_advantages must run before the forward/backward"
            )
        tokens = datum.model_input_tokens
        if len(tokens) < 2:
            raise ValueError(
                f"datum (trial {datum.trial_name}) has {len(tokens)} token(s); at "
                "least 2 are needed for the next-token input/target shift"
            )
        length = len(tokens) - 1
        out.append(
            tinker.Datum(
                model_input=tinker.ModelInput.from_ints(tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64", shape=[length]
                    ),
                    "logprobs": tinker.TensorData(
                        data=datum.sampled_logprobs[1:], dtype="float32", shape=[length]
                    ),
                    "advantages": tinker.TensorData(
                        data=datum.advantages[1:], dtype="float32", shape=[length]
                    ),
                },
            )
        )
    return out


class ProbeTrainingClient:
    """`DistillTrainingClient` over the raw SDK, timed, with the compat datums.

    Mirrors `wmh.distill.loop.SdkTrainingClient` (blocking calls, per-session
    nonce on save names) except that `forward_backward` converts through
    `_compat_tinker_datums` so the batch survives the live service's
    importance_sampling loss-args validation (no `mask` key).
    """

    def __init__(self, client: tinker.TrainingClient, timer: SdkCallTimer) -> None:
        self._client = client
        self._timer = timer
        self._session = uuid4().hex[:8]
        self._save_counter = 0

    def get_tokenizer(self) -> EncodingTokenizer:
        """The student base model's HF tokenizer."""
        return cast("EncodingTokenizer", self._client.get_tokenizer())

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        """One timed blocking forward/backward batch (compat datum conversion).

        Returns a real `TrainStepOutput`. It used to return None, which violated the
        `TrainingClient` protocol it stands in for -- the loop reads `.loss` off this and
        crashed with `'NoneType' object has no attribute 'loss'`, so the probe could not run
        at all. The loss is read from the response's metrics rather than fabricated: absent
        or non-numeric becomes None, which the loop already treats as "backend reported no
        loss metric".
        """
        converted = _compat_tinker_datums(datums)
        started = time.monotonic()
        try:
            response = self._client.forward_backward(
                converted, cast("tinker.types.LossFnType", loss_fn)
            ).result()
        finally:
            self._timer.record("train.forward_backward", time.monotonic() - started)
        metrics = getattr(response, "metrics", None) or {}
        loss = metrics.get("total_loss:sum", metrics.get("total_loss"))
        return TrainStepOutput(loss=float(loss) if isinstance(loss, (int, float)) else None)

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        """One timed blocking Adam step.

        Returns a real `OptimStepOutput` for the same reason as `forward_backward`: the loop
        reads `.grad_norm` off it, and returning None broke the protocol. Grad norm is
        reported only if the backend supplies one; never fabricated.
        """
        started = time.monotonic()
        try:
            response = self._client.optim_step(
                tinker.types.AdamParams(learning_rate=learning_rate)
            ).result()
        finally:
            self._timer.record("train.optim_step", time.monotonic() - started)
        norm = getattr(response, "grad_norm", None)
        return OptimStepOutput(grad_norm=float(norm) if isinstance(norm, (int, float)) else None)

    def save_state(self) -> str:
        """One timed blocking save_state under a session-unique name."""
        name = f"wmh-probe-{self._session}-state-{self._save_counter:04d}"
        self._save_counter += 1
        started = time.monotonic()
        try:
            return self._client.save_state(name).result().path
        finally:
            self._timer.record("train.save_state", time.monotonic() - started)

    def load_state(self, path: str) -> None:
        """One timed blocking load_state."""
        started = time.monotonic()
        try:
            self._client.load_state(path).result()
        finally:
            self._timer.record("train.load_state", time.monotonic() - started)

    def save_weights_for_sampler(self, name: str) -> str:
        """One timed blocking sampler-weights save."""
        started = time.monotonic()
        try:
            return self._client.save_weights_for_sampler(f"{name}-{self._session}").result().path
        finally:
            self._timer.record("train.save_weights_for_sampler", time.monotonic() - started)


class TimedServiceClient:
    """`DistillServiceClient` over the real SDK, timing every call it hands out.

    Sampling clients are cached by model path so the probe's rollout sampling
    reuses the exact client the loop created for the same weights.
    """

    def __init__(self, inner: tinker.ServiceClient, timer: SdkCallTimer) -> None:
        self._inner = inner
        self._timer = timer
        self._sampling_clients: dict[str, TimedSamplingClient] = {}

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> ProbeTrainingClient:
        """Create (timed) the real LoRA training client."""
        started = time.monotonic()
        try:
            client = self._inner.create_lora_training_client(base_model=base_model, rank=rank)
        finally:
            self._timer.record("service.create_lora_training_client", time.monotonic() - started)
        return ProbeTrainingClient(client, self._timer)

    def create_sampling_client(self, model_path: str) -> TimedSamplingClient:
        """Create or reuse a timed sampling client for a sampler path or base model."""
        cached = self._sampling_clients.get(model_path)
        if cached is not None:
            return cached
        role = "student" if model_path.startswith("tinker://") else "teacher"
        started = time.monotonic()
        try:
            if model_path.startswith("tinker://"):
                client = self._inner.create_sampling_client(model_path=model_path)
            else:
                client = self._inner.create_sampling_client(base_model=model_path)
        finally:
            self._timer.record("service.create_sampling_client", time.monotonic() - started)
        timed = TimedSamplingClient(SdkSamplingClient(client), self._timer, role)
        self._sampling_clients[model_path] = timed
        return timed


# -- the probe rollout collector -------------------------------------------------------------


class ProbeRollouts:
    """Probe-only `collect_rollouts` replacement (mirrors `loop_test._FakeRollouts`).

    For each (task_id, attempt) it renders a single-turn chat with the base
    model's cookbook renderer, samples one completion from the provider's
    model through the SAME service client the loop uses, and wraps the result
    as one `TrialRecord` with a single `TokenSpan`. Works for both provider
    models: the student's tinker:// sampler paths and the teacher base-model
    name (baseline evals run the teacher through this same seam). Rewards are
    always 0.0 / not passed: on-policy distillation needs no graders.
    """

    def __init__(self, service: TimedServiceClient) -> None:
        self._service = service
        self._renderers: dict[str, ChatRendering] = {}

    def _rendering_for(self, base_model: str, client: TimedSamplingClient) -> ChatRendering:
        """Build (once per base model) the cookbook renderer the provider would use."""
        rendering = self._renderers.get(base_model)
        if rendering is None:
            tokenizer = cast("RendererTokenizer", client.get_tokenizer())
            rendering = build_renderer(base_model, tokenizer)
            self._renderers[base_model] = rendering
        return rendering

    def __call__(
        self,
        step_index: int,
        task_ids: Sequence[str],
        cfg: DistillConfig,
        harness: HarnessDoc,
        provider_config: ProviderConfig,
        run_dir: Path,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[list[TrialRecord], RolloutStats]:
        """Sample one single-turn completion per (task, attempt) as a TrialRecord."""
        del harness, should_cancel  # the probe has no harbor jobs to pin or cancel
        client = self._service.create_sampling_client(provider_config.model)
        base_model = provider_config.model_type or provider_config.model
        rendering = self._rendering_for(base_model, client)
        records: list[TrialRecord] = []
        for task_id in task_ids:
            problem = PROBLEM_BY_ID[task_id]
            prompt_ids = rendering.build_generation_prompt(
                [
                    ChatMessage(role="system", content=SYSTEM_PROMPT),
                    ChatMessage(role="user", content=f"{problem}\n\n{USER_SUFFIX}"),
                ]
            )
            for attempt in range(1, cfg.train.group_size + 1):
                sequence = client.sample(
                    prompt_ids,
                    max_tokens=cfg.sampling.max_tokens,
                    temperature=cfg.sampling.temperature,
                )
                sampled = list(sequence.tokens)
                logprobs = sequence.logprobs
                if logprobs is None or len(logprobs) != len(sampled):
                    got = "no logprobs" if logprobs is None else f"{len(logprobs)} logprobs"
                    raise RuntimeError(
                        f"sampling returned {got} for {len(sampled)} sampled tokens on "
                        f"{provider_config.model}; per-token logprobs are required for "
                        "tokens-in-tokens-out training and are never fabricated"
                    )
                trial_name = f"{task_id}__s{attempt}"
                records.append(
                    TrialRecord(
                        task_id=task_id,
                        attempt=attempt,
                        trial_name=trial_name,
                        reward=0.0,
                        passed=False,
                        spans=[
                            TokenSpan(
                                call_index=0,
                                prompt_token_ids=list(prompt_ids),
                                sampled_token_ids=sampled,
                                sampled_logprobs=list(logprobs),
                            )
                        ],
                        stop_reason="submitted",
                        artifact_dir=str(
                            run_dir / "probe-trials" / f"step-{step_index:04d}" / trial_name
                        ),
                    )
                )
        # `RolloutStats` grew four required fields (raw_solve_rate, executed_trials,
        # infra_failed_trials, scaffold_loss_rate) after this probe was written, and the probe
        # was not re-run in between, so it bit-rotted into a hard ValidationError -- the
        # fast-iteration tool the goal file points at for cheap pre-checks was unusable. Every
        # probe trial completes by construction (there is no harness to lose an episode to), so
        # the honest values are: nothing infra-failed, everything executed, no scaffold loss.
        # solve_rate stays 0.0 because these are ungraded math completions with no verifier;
        # on-policy distillation needs no reward, and the probe's pass signal is the reverse-KL
        # trend, not a solve rate.
        stats = RolloutStats(
            trials=len(records),
            trials_with_spans=len(records),
            solve_rate=0.0,
            raw_solve_rate=0.0,
            executed_trials=len(records),
            infra_failed_trials=0,
            empty_span_trials=0,
            scaffold_loss_rate=0.0,
        )
        logger.info(
            "probe rollouts step %d: %d trial(s) from %s (%d task(s) x %d attempt(s))",
            step_index,
            len(records),
            provider_config.model,
            len(list(task_ids)),
            cfg.train.group_size,
        )
        return records, stats


# -- run wiring ------------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> DistillConfig:
    """The probe's DistillConfig from the parsed CLI arguments."""
    return DistillConfig(
        student=StudentConfig(base_model=args.student, lora_rank=16),
        teacher=TeacherConfig(model=args.teacher),
        harbor=HarborConfig(job_template=str(_JOB_TEMPLATE), backend="local"),
        rollout=RolloutConfig(max_turns=1),
        train=TrainConfig(
            steps=args.steps,
            tasks_per_batch=args.tasks_per_batch,
            group_size=args.group_size,
            sampler_refresh_every=1,
            save_state_every=4,
            trial_concurrency=1,
            learning_rate=args.learning_rate,
            num_substeps=args.num_substeps,
            advantage_clip=args.advantage_clip,
        ),
        sampling=SamplingConfig(temperature=1.0, max_tokens=args.max_tokens),
        eval=EvalConfig(every=0, tasks=len(HOLDOUT_TASK_IDS), k=1),
        gate=GateConfig(k=1),
        pricing=_PLACEHOLDER_PRICING,
        budget=BudgetConfig(max_usd=args.budget_usd),
        wandb=WandbConfig(enabled=args.wandb, project="wmh-distill", run_name=RUN_NAME),
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The probe's CLI surface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B", help="student base model")
    parser.add_argument("--teacher", default="Qwen/Qwen3.5-9B", help="teacher base model")
    parser.add_argument("--steps", type=int, default=8, help="training steps")
    parser.add_argument("--tasks-per-batch", type=int, default=4, help="tasks per train batch")
    parser.add_argument("--group-size", type=int, default=2, help="attempts per task")
    parser.add_argument("--max-tokens", type=int, default=256, help="per-completion output cap")
    parser.add_argument(
        "--budget-usd", type=float, default=3.0, help="hard budget cap (placeholder pricing)"
    )
    parser.add_argument(
        "--run-dir",
        default=".wmh/distill-runs/probe-gsm8k",
        help="fresh run directory for this probe's artifacts",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="enable [wandb] tracking (project wmh-distill)"
    )
    # The three knobs that decide whether a run trains or collapses, exposed here so the
    # question can be answered for pennies instead of on a $14/step agentic run. Both real
    # model pairs collapsed at their first honest update count -- Qwen upward (length 5.33x),
    # Nano downward (entropy 0.35x) -- and the suspected cause is the UNBOUNDED advantage
    # magnitude, not the sign (audited: advantage = teacher_lp - sampled_lp = -reverse_kl,
    # which is the cookbook's construction and correct). This probe is the cheap way to test
    # that: hold everything else fixed and vary the bound.
    parser.add_argument(
        "--num-substeps",
        type=int,
        default=1,
        help="minibatches (and optimizer updates) per collected batch; 1 = historical behaviour",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=None,
        help="bound |advantage| per token; unset (default) trains the raw teacher-minus-student gap",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="optimizer lr")
    return parser.parse_args(argv)


def _log_progress(event: DistillProgress) -> None:
    """Log spend as steps land (the loop's own logger carries the message)."""
    if event.phase in ("training", "gate"):
        logger.info(
            "[probe] %s step=%s total spend $%.4f",
            event.phase,
            "-" if event.step is None else event.step,
            event.spent_usd,
        )


def _log_metrics_rows(run_dir: Path) -> list[tuple[int, float]]:
    """Log every persisted metrics row and return the (step, reverse KL) series."""
    rows = DistillRunStore(run_dir).read_metrics()
    series: list[tuple[int, float]] = []
    logger.info("persisted step metrics (%d row(s)):", len(rows))
    for row in rows:
        step = row.get("step")
        kl = row.get("reverse_kl_per_token")
        logger.info(
            "  step %s: reverse_kl_per_token=%s datums=%s loss_tokens=%s "
            "sample_tokens=%s teacher_prefill_tokens=%s usd=%s",
            step,
            kl,
            row.get("datums"),
            row.get("loss_tokens"),
            row.get("student_sample_tokens"),
            row.get("teacher_prefill_tokens"),
            row.get("usd"),
        )
        if isinstance(step, int) and isinstance(kl, int | float) and not isinstance(kl, bool):
            series.append((step, float(kl)))
    return series


def _log_verdict(series: list[tuple[int, float]], total_usd: float | None) -> None:
    """The probe's pass signal: reverse KL first step vs last step, plus spend."""
    spend = "unknown (run aborted before a result)" if total_usd is None else f"${total_usd:.4f}"
    if len(series) < 2:
        logger.warning(
            "VERDICT: not enough scored steps for a KL trend (%d row(s)); total spend %s "
            "(placeholder pricing)",
            len(series),
            spend,
        )
        return
    first_step, first_kl = series[0]
    last_step, last_kl = series[-1]
    trend = "DOWN (pass)" if last_kl < first_kl else "NOT DOWN (investigate)"
    logger.info(
        "VERDICT: reverse KL/token step %d: %.4f -> step %d: %.4f, trend %s; "
        "total spend %s (placeholder pricing, see _PLACEHOLDER_PRICING)",
        first_step,
        first_kl,
        last_step,
        last_kl,
        trend,
        spend,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe end to end and log the KL trajectory, spend, and timings."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not os.environ.get(TINKER_API_KEY_ENV):
        logger.error(
            "%s is not set; export your Tinker API key before running the probe",
            TINKER_API_KEY_ENV,
        )
        return 2
    if not _JOB_TEMPLATE.is_file():
        logger.error("placeholder harbor job template missing at %s", _JOB_TEMPLATE)
        return 2
    run_dir = Path(args.run_dir)
    if (run_dir / "config.toml").exists():
        logger.error(
            "run dir %s already holds a distillation run; the probe has no resume "
            "surface, so choose a fresh --run-dir",
            run_dir,
        )
        return 2

    cfg = build_config(args)
    timer = SdkCallTimer()
    service = TimedServiceClient(tinker.ServiceClient(), timer)
    rollouts = ProbeRollouts(service)
    # Probe-only monkeypatch: run_distillation resolves collect_rollouts through
    # its module global, exactly what loop_test patches with _FakeRollouts.
    loop_module.collect_rollouts = rollouts
    harness = default_agent("pi")

    logger.info(
        "probe run: student %s <- teacher %s, %d step(s) x %d task(s) x %d attempt(s), "
        "%d train / %d holdout task(s), max_tokens %d, budget $%.2f -> %s",
        args.student,
        args.teacher,
        args.steps,
        args.tasks_per_batch,
        args.group_size,
        len(TRAIN_TASK_IDS),
        len(HOLDOUT_TASK_IDS),
        args.max_tokens,
        args.budget_usd,
        run_dir,
    )
    started = time.monotonic()
    result: DistillResult | None = None
    exit_code = 0
    try:
        result = run_distillation(
            RUN_NAME,
            cfg,
            harness,
            list(TRAIN_TASK_IDS),
            list(HOLDOUT_TASK_IDS),
            run_dir,
            on_progress=_log_progress,
            service_client=service,
            adapter_store=AdapterStore(run_dir),
        )
    except DistillBudgetError as exc:
        logger.error("budget abort: %s", exc)
        exit_code = 1
    elapsed = time.monotonic() - started

    logger.info("probe wall clock: %.1fs", elapsed)
    series = _log_metrics_rows(run_dir)
    timer.log_summary()
    total_usd = result.spend.total_usd if result is not None else None
    if result is not None:
        logger.info("gate: %s; final sampler %s", result.gate.reason, result.final_sampler_path)
    _log_verdict(series, total_usd)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

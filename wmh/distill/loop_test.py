"""Offline tests for the distillation orchestrator against the tinker fakes.

The whole loop runs without any SDK or network: the injected service client
wraps `wmh.distill.fake_tinker` (converting the loop's `TrainDatum`s into
`FakeDatum`s so the fake training client's TITO assertion sees every batch),
`collect_rollouts` is monkeypatched with a collector that samples real spans
from the CURRENT fake sampler weights, and `build_renderer` is monkeypatched
with a stub since no cookbook renderer exists for the fake base model.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, cast

import pytest
from llm_waterfall.types import ChatMessage, ChatTool

import wmh.distill.loop as loop_module
import wmh.providers.tinker as providers_tinker

if TYPE_CHECKING:
    import tinker
from wmh.core.types import JsonObject
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
    WarmupConfig,
)
from wmh.distill.data import TrainDatum
from wmh.distill.deadlines import TinkerDeadlineError
from wmh.distill.fake_tinker import (
    FakeDatum,
    FakeSamplingClient,
    FakeServiceClient,
    FakeTokenizer,
    FakeTrainingClient,
)
from wmh.distill.loop import (
    EVAL_ROLLOUTS_DIR,
    MAX_CONSECUTIVE_EMPTY_STEPS,
    SDK_GRAD_NORM_METRIC_NAMES,
    SDK_LOSS_METRIC_NAMES,
    STUDENT_AFTER_EVAL,
    STUDENT_BEFORE_EVAL,
    WARMUP_ROLLOUTS_DIR,
    DistillBudgetError,
    DistillEmptyBatchError,
    DistillProgress,
    DistillResult,
    DistillSamplingClient,
    OptimStepOutput,
    SdkSamplingClient,
    SdkServiceClient,
    SdkTrainingClient,
    StepMetrics,
    TaskSampler,
    TrainStepOutput,
    WarmupMetrics,
    pin_rollout_params,
    resume_command,
    run_distillation,
    sdk_metric_value,
)
from wmh.distill.rollouts import RolloutStats
from wmh.distill.store import AdapterStore, DistillRunStore
from wmh.distill.teacher import EncodingTokenizer
from wmh.distill.tokens import TrialRecord
from wmh.distill.tracking import DistillTracker
from wmh.harness.doc import HarnessDoc
from wmh.providers.base import ProviderConfig
from wmh.providers.tinker import SampledSequenceLike, TokenSpan

_NAME = "distill-loop-test"
_TRAIN_IDS = ("task-a", "task-b", "task-c", "task-d")
_HOLDOUT_IDS = ("hold-a", "hold-b")
_STUDENT = "fake/student-4b"
_TEACHER = "fake/teacher-70b"


# -- fakes and shims -------------------------------------------------------------------------


class _StubRendering:
    """Just enough `ChatRendering` for the loop's preflight prompt."""

    @property
    def stop_sequences(self) -> list[str]:
        return []

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        del tools
        text = messages[-1].content
        return [ord(ch) for ch in (text if isinstance(text, str) else "ping")]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token) for token in token_ids)

    def parse_response(self, sampled_ids: list[int]) -> None:
        raise NotImplementedError("the loop never parses responses during preflight")


def _fake_build_renderer(base_model: str, tokenizer: EncodingTokenizer) -> _StubRendering:
    del base_model, tokenizer
    return _StubRendering()


def _number(row: JsonObject, key: str) -> float:
    """Read one numeric metrics-row value with the narrowing ty needs."""
    value = row[key]
    assert isinstance(value, int | float) and not isinstance(value, bool), (key, value)
    return float(value)


def _fake_datum(datum: TrainDatum) -> FakeDatum:
    """Convert a loop datum to the fake client's shifted next-token layout."""
    tokens = datum.model_input_tokens
    return FakeDatum(
        model_input_tokens=tokens[:-1],
        target_tokens=tokens[1:],
        weights=datum.loss_mask[1:],
        advantages=datum.advantages[1:],
        logprobs=datum.sampled_logprobs[1:],
    )


class _Training:
    """`DistillTrainingClient` shim converting loop datums to `FakeDatum`s."""

    def __init__(self, inner: FakeTrainingClient) -> None:
        self.inner = inner
        self.load_state_calls: list[str] = []

    def get_tokenizer(self) -> FakeTokenizer:
        return self.inner.get_tokenizer()

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> TrainStepOutput:
        # Mirror SdkTrainingClient: extract the loss from the SDK-shaped
        # output's metrics dict through the same suffix-tolerant helper.
        output = self.inner.forward_backward([_fake_datum(datum) for datum in datums], loss_fn)
        return TrainStepOutput(loss=sdk_metric_value(output.metrics, SDK_LOSS_METRIC_NAMES))

    def optim_step(self, learning_rate: float) -> OptimStepOutput:
        response = self.inner.optim_step(learning_rate)
        return OptimStepOutput(
            grad_norm=sdk_metric_value(response.metrics, SDK_GRAD_NORM_METRIC_NAMES)
        )

    def save_state(self) -> str:
        return self.inner.save_state()

    def load_state(self, path: str) -> None:
        self.load_state_calls.append(path)
        self.inner.load_state(path)

    def save_weights_for_sampler(self, name: str) -> str:
        return self.inner.save_weights_for_sampler(name)


class _Service:
    """`DistillServiceClient` over the fakes.

    The training client is memoized so its saved states and registered
    sampler paths survive across a budget abort and the resumed session, the
    way real tinker:// artifacts do.
    """

    def __init__(self) -> None:
        self.inner = FakeServiceClient()
        self.training: _Training | None = None

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> _Training:
        if self.training is None:
            self.training = _Training(self.inner.create_lora_training_client(base_model, rank))
        return self.training

    def create_sampling_client(self, model_path: str) -> DistillSamplingClient:
        return self.inner.create_sampling_client(model_path)


@dataclass(frozen=True)
class _RolloutCall:
    """One recorded `collect_rollouts` invocation."""

    step_index: int
    task_ids: tuple[str, ...]
    attempts: int
    provider_model: str
    run_dir: Path
    doc_hash: str


class _FakeRollouts:
    """Offline `collect_rollouts`: samples real spans from the current weights.

    Every trial makes two prefix-extending sampling calls through the service
    client for the provider's model path, so student trials carry spans the
    fake training client's ledger knows about (the TITO ground truth). Tasks
    at even positions pass, so every batch's solve rate is 0.5.
    """

    def __init__(self, service: _Service) -> None:
        self.service = service
        self.calls: list[_RolloutCall] = []
        self.fabricate_spans = False
        self.teacher_fail_all = False
        """When True, every trial of the teacher provider fails (warmup skip path)."""

        self.fail_on_train_step: int | None = None
        """Raise on the TRAIN batch of this step (a crash between phases)."""

        self.empty_span_train_steps: set[int] = set()
        """TRAIN steps whose every trial records zero spans (a dead provider)."""

        self.zero_trial_train_steps: set[int] = set()
        """TRAIN steps whose batch returns no trials at all (harbor scored nothing)."""

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
        del should_cancel
        is_train_batch = (
            run_dir.parent.name != "eval-rollouts" and run_dir.name != WARMUP_ROLLOUTS_DIR
        )
        if is_train_batch and step_index == self.fail_on_train_step:
            raise RuntimeError("injected rollout crash")
        # A dead student provider: every trial ends span-less ("submitted"
        # with zero turns), exactly what swallowed worker failures produce.
        empty_batch = is_train_batch and step_index in self.empty_span_train_steps
        if is_train_batch and step_index in self.zero_trial_train_steps:
            self.calls.append(
                _RolloutCall(
                    step_index=step_index,
                    task_ids=tuple(task_ids),
                    attempts=cfg.train.group_size,
                    provider_model=provider_config.model,
                    run_dir=run_dir,
                    doc_hash=harness.doc_hash,
                )
            )
            stats = RolloutStats(trials=0, trials_with_spans=0, solve_rate=0.0, empty_span_trials=0)
            return [], stats
        client = self.service.create_sampling_client(provider_config.model)
        records: list[TrialRecord] = []
        for task_index, task_id in enumerate(task_ids):
            for attempt in range(1, cfg.train.group_size + 1):
                passed = task_index % 2 == 0 and not empty_batch
                if self.teacher_fail_all and provider_config.model == _TEACHER:
                    passed = False
                trial_name = f"{task_id}__s{attempt}"
                spans = [] if empty_batch else self._spans(client, task_id, step_index, attempt)
                records.append(
                    TrialRecord(
                        task_id=task_id,
                        attempt=attempt,
                        trial_name=trial_name,
                        reward=1.0 if passed else 0.0,
                        passed=passed,
                        spans=spans,
                        stop_reason="submitted",
                        artifact_dir=str(
                            run_dir / "harbor" / f"step-{step_index:04d}" / trial_name
                        ),
                    )
                )
        with_spans = sum(1 for record in records if record.spans)
        stats = RolloutStats(
            trials=len(records),
            trials_with_spans=with_spans,
            solve_rate=sum(1 for record in records if record.passed) / len(records),
            empty_span_trials=len(records) - with_spans,
        )
        self.calls.append(
            _RolloutCall(
                step_index=step_index,
                task_ids=tuple(task_ids),
                attempts=cfg.train.group_size,
                provider_model=provider_config.model,
                run_dir=run_dir,
                doc_hash=harness.doc_hash,
            )
        )
        return records, stats

    def _spans(
        self,
        client: DistillSamplingClient,
        task_id: str,
        step_index: int,
        attempt: int,
    ) -> list[TokenSpan]:
        if self.fabricate_spans:
            # Token ids no sampler ever issued: a TITO violation by construction.
            return [
                TokenSpan(
                    call_index=0,
                    prompt_token_ids=[65, 66, 67],
                    sampled_token_ids=[1, 2, 3],
                    sampled_logprobs=[-0.1, -0.2, -0.3],
                )
            ]
        prompt = [ord(ch) for ch in f"{task_id}:{step_index}:{attempt}:"]
        first = client.sample(prompt, max_tokens=5, temperature=0.7)
        assert first.logprobs is not None
        follow = [*prompt, *first.tokens, *(ord(ch) for ch in "|obs|")]
        second = client.sample(follow, max_tokens=5, temperature=0.7)
        assert second.logprobs is not None
        return [
            TokenSpan(
                call_index=0,
                prompt_token_ids=prompt,
                sampled_token_ids=list(first.tokens),
                sampled_logprobs=list(first.logprobs),
            ),
            TokenSpan(
                call_index=1,
                prompt_token_ids=follow,
                sampled_token_ids=list(second.tokens),
                sampled_logprobs=list(second.logprobs),
            ),
        ]


@dataclass
class _Env:
    """One test's wired-up offline environment."""

    service: _Service
    rollouts: _FakeRollouts
    run_dir: Path
    adapters: AdapterStore
    progress: list[DistillProgress] = field(default_factory=list)


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Env:
    service = _Service()
    rollouts = _FakeRollouts(service)
    monkeypatch.setattr(loop_module, "collect_rollouts", rollouts)
    monkeypatch.setattr(loop_module, "build_renderer", _fake_build_renderer)
    return _Env(
        service=service,
        rollouts=rollouts,
        run_dir=tmp_path / "run",
        adapters=AdapterStore(tmp_path / ".wmh"),
    )


def _cfg(*, budget_max: float | None = None, pricing: PricingConfig | None = None) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model=_STUDENT, lora_rank=8),
        teacher=TeacherConfig(model=_TEACHER),
        harbor=HarborConfig(job_template="unused-by-the-stubbed-collector.yaml"),
        train=TrainConfig(
            steps=3,
            tasks_per_batch=2,
            group_size=2,
            learning_rate=1e-4,
            sampler_refresh_every=1,
            save_state_every=2,
            trial_concurrency=2,
        ),
        eval=EvalConfig(every=2, tasks=2, k=1),
        gate=GateConfig(k=1),
        pricing=pricing
        if pricing is not None
        else PricingConfig(
            student_prefill=1.0, student_sample=1.0, student_train=1.0, teacher_prefill=1.0
        ),
        budget=BudgetConfig(max_usd=budget_max),
    )


def _train_priced_cfg(budget_max: float) -> DistillConfig:
    """A config whose spend lands ONLY on the student_train meter.

    Baselines never charge student_train, so the run deterministically
    survives the baselines and hits the cap at the first training step's
    budget check.
    """
    return _cfg(
        budget_max=budget_max,
        pricing=PricingConfig(
            student_prefill=0.0, student_sample=0.0, student_train=1e9, teacher_prefill=0.0
        ),
    )


def _warmup_cfg(
    *,
    warmup_steps: int = 2,
    rollouts_per_task: int = 2,
    keep: Literal["passed", "all"] = "passed",
    warmup_lr: float | None = 5e-5,
    budget_max: float | None = None,
    pricing: PricingConfig | None = None,
) -> DistillConfig:
    """The 3-step config with the supervised warmup phase enabled."""
    return _cfg(budget_max=budget_max, pricing=pricing).model_copy(
        update={
            "warmup": WarmupConfig(
                steps=warmup_steps,
                rollouts_per_task=rollouts_per_task,
                keep=keep,
                learning_rate=warmup_lr,
            )
        }
    )


def _warmup_calls(env: _Env) -> list[_RolloutCall]:
    warmup_dir = env.run_dir / WARMUP_ROLLOUTS_DIR
    return [call for call in env.rollouts.calls if call.run_dir == warmup_dir]


def _eval_run_counts(env: _Env) -> dict[str, int]:
    """How many collector batches each eval key actually ran (reuse skips runs)."""
    eval_root = env.run_dir / EVAL_ROLLOUTS_DIR
    counts: dict[str, int] = {}
    for call in env.rollouts.calls:
        if call.run_dir.parent == eval_root:
            counts[call.run_dir.name] = counts.get(call.run_dir.name, 0) + 1
    return counts


def _loss_fns(env: _Env) -> list[str]:
    training = env.service.training
    assert training is not None
    return [loss for _, loss in training.inner.forward_backward_calls]


def _run(
    env: _Env,
    cfg: DistillConfig,
    *,
    resume: bool = False,
    tracker: DistillTracker | None = None,
    cli_agent: str | None = None,
) -> DistillResult:
    return run_distillation(
        _NAME,
        cfg,
        HarnessDoc.baseline(),
        _TRAIN_IDS,
        _HOLDOUT_IDS,
        env.run_dir,
        resume=resume,
        on_progress=env.progress.append,
        service_client=env.service,
        adapter_store=env.adapters,
        tracker=tracker,
        cli_agent=cli_agent,
    )


@dataclass(frozen=True)
class _TrackedSummary:
    """One recorded `log_summary` call."""

    gate_accepted: bool
    gate_reason: str
    teacher_solve_rate: float
    student_before_solve_rate: float
    student_after_solve_rate: float
    total_usd: float
    steps_completed: int


class _RecordingTracker:
    """A `DistillTracker` that records every call for assertions."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, StepMetrics]] = []
        self.warmup_steps: list[tuple[int, WarmupMetrics]] = []
        self.evals: list[tuple[str, float, int | None]] = []
        self.summaries: list[_TrackedSummary] = []
        self.finish_calls = 0

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        self.steps.append((step, metrics))

    def log_warmup_step(self, warmup_step: int, metrics: WarmupMetrics) -> None:
        self.warmup_steps.append((warmup_step, metrics))

    def log_eval(self, name: str, solve_rate: float, step: int | None) -> None:
        self.evals.append((name, solve_rate, step))

    def log_summary(
        self,
        *,
        gate_accepted: bool,
        gate_reason: str,
        teacher_solve_rate: float,
        student_before_solve_rate: float,
        student_after_solve_rate: float,
        total_usd: float,
        steps_completed: int,
    ) -> None:
        self.summaries.append(
            _TrackedSummary(
                gate_accepted=gate_accepted,
                gate_reason=gate_reason,
                teacher_solve_rate=teacher_solve_rate,
                student_before_solve_rate=student_before_solve_rate,
                student_after_solve_rate=student_after_solve_rate,
                total_usd=total_usd,
                steps_completed=steps_completed,
            )
        )

    def finish(self) -> None:
        self.finish_calls += 1


# -- the 3-step end-to-end run ---------------------------------------------------------------


def test_three_step_run_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _setup(tmp_path, monkeypatch)

    result = _run(env, _cfg())

    assert result.steps_completed == 3
    assert result.gate.accepted
    assert result.adapter_version == 1
    assert result.final_sampler_path.startswith("tinker://fake/sampler/")
    assert result.run_dir == str(env.run_dir)
    assert result.spend.total_usd > 0.0

    # Metrics rows: one per step, carrying the loss/health/meter numbers.
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["solve_rate"] == 0.5
        assert row["fragmentation_rate"] == 0.0
        assert row["datums"] == 4  # 2 tasks x 2 attempts, each merged to one datum
        assert isinstance(row["reverse_kl_per_token"], float)
        # RL metrics: rewards are binary here, so the mean equals solve_rate;
        # the fake backend reports a deterministic loss and grad norm.
        assert row["reward_mean"] == 0.5
        assert isinstance(row["advantage_mean"], float)
        assert _number(row, "advantage_std") >= 0.0
        assert 0.0 <= _number(row, "clip_fraction") <= 1.0
        assert isinstance(row["pg_loss"], float) and row["pg_loss"] > 0.0
        assert isinstance(row["grad_norm"], float) and row["grad_norm"] > 0.0
        for key in (
            "usd",
            "student_prefill_tokens",
            "student_cached_prefill_tokens",
            "student_sample_tokens",
            "student_train_tokens",
            "teacher_prefill_tokens",
        ):
            assert _number(row, key) > 0
    # Teacher-in-harness billing (teacher_sample plus cached teacher prefill)
    # happens only in the pre-step teacher baseline, which folds into row 0;
    # training steps score the teacher in one full-price request per datum.
    assert _number(rows[0], "teacher_sample_tokens") > 0
    assert _number(rows[0], "teacher_cached_prefill_tokens") > 0
    for row in rows[1:]:
        assert _number(row, "teacher_sample_tokens") == 0
        assert _number(row, "teacher_cached_prefill_tokens") == 0
    # Cumulative spend: the first row folds in the pre-step baseline spend
    # exactly, every later row advances by exactly its own delta, and the
    # finalize eval charges after the last row (so the ledger total is higher).
    assert _number(rows[0], "cumulative_usd") == pytest.approx(_number(rows[0], "usd"))
    for previous, row in zip(rows, rows[1:], strict=False):
        assert _number(row, "cumulative_usd") == pytest.approx(
            _number(previous, "cumulative_usd") + _number(row, "usd")
        )
    assert _number(rows[-1], "cumulative_usd") < result.spend.total_usd

    # TITO held through every forward_backward: the fake training client
    # asserts spans against its ledger BEFORE recording a call, so three
    # recorded importance_sampling batches mean three passing TITO checks.
    training = env.service.training
    assert training is not None
    batches = training.inner.forward_backward_calls
    assert len(batches) == 3
    assert all(loss_fn == "importance_sampling" for _, loss_fn in batches)
    assert all(len(batch) == 4 for batch, _ in batches)
    assert training.inner.optim_step_lrs == [1e-4] * 3

    # Cadences: refresh_every=1 gives every training step its own sampler path.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert [call.step_index for call in train_calls] == [0, 1, 2]
    assert all(call.attempts == 2 for call in train_calls)
    assert len({call.provider_model for call in train_calls}) == 3
    # save_state_every=2 checkpoints after step 1; finalize checkpoints step 2.
    assert [checkpoint.step for checkpoint in store.checkpoints()] == [1, 2]
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.sampler_path == result.final_sampler_path

    # Evals: two holdout baselines, one interim train-subsample eval after
    # step 1 (eval.every=2), and the final holdout student-after eval.
    eval_calls = [call for call in env.rollouts.calls if call.run_dir != env.run_dir]
    assert [call.run_dir.name for call in eval_calls] == [
        "baseline-teacher",
        "baseline-student-before",
        "step-0001",
        "student-after",
    ]
    teacher_call, before_call, interim_call, after_call = eval_calls
    assert teacher_call.provider_model == _TEACHER
    assert teacher_call.task_ids == _HOLDOUT_IDS
    assert teacher_call.attempts == 1  # gate.k
    assert before_call.provider_model == train_calls[0].provider_model
    assert len(interim_call.task_ids) == 2
    assert set(interim_call.task_ids) <= set(_TRAIN_IDS)
    assert after_call.provider_model == result.final_sampler_path
    assert after_call.task_ids == _HOLDOUT_IDS
    for key in ("baseline-teacher", "baseline-student-before", "step-0001", "student-after"):
        assert (store.evals_dir / f"{key}.json").is_file()

    # Terminal artifacts: config snapshot, gate, model card, handoff, adapter.
    assert store.config_path.is_file()
    gate = json.loads(store.gate_path.read_text(encoding="utf-8"))
    assert gate["accepted"] is True
    assert gate["teacher_solve_rate"] == 0.5
    assert result.final_sampler_path in store.handoff_path.read_text(encoding="utf-8")
    assert env.adapters.versions(_NAME) == [1]
    assert env.adapters.aliases(_NAME) == {"champion": 1}
    card = env.adapters.resolve(_NAME)
    assert card.base_model == _STUDENT
    assert card.teacher_model == _TEACHER
    assert card.sampler_path == result.final_sampler_path
    assert card.state_path == result.final_state_path
    assert card.gate is not None and card.gate.accepted

    phases = {event.phase for event in env.progress}
    assert {"preflight", "baseline", "rollouts", "training", "eval", "finalize", "gate"} <= phases

    # Every rollout batch (train and eval) ran the SAME pinned document: the
    # seed doc with [sampling]/[rollout] written into its param surfaces.
    pinned_hash = pin_rollout_params(HarnessDoc.baseline(), _cfg()).doc_hash
    assert {call.doc_hash for call in env.rollouts.calls} == {pinned_hash}

    # The spend ledger tracked every charge, including the finalize eval that
    # no metrics row ever carries.
    assert store.read_spend() == pytest.approx(result.spend.total_usd)


def test_tracker_sees_every_step_eval_summary_and_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    result = _run(env, _cfg(), tracker=tracker)

    # One log_step per training step, with the SAME rows the store persisted.
    assert [step for step, _ in tracker.steps] == [0, 1, 2]
    persisted = DistillRunStore(env.run_dir).read_metrics()
    for (step, metrics), row in zip(tracker.steps, persisted, strict=True):
        assert row == {"step": step, **metrics.model_dump(mode="json")}

    # Every recorded eval report was tracked under its store key: the two
    # pre-training baselines (no step), the interim eval after step 1
    # (eval.every=2), and the finalize student-after eval.
    assert tracker.evals == [
        ("baseline-teacher", 0.5, None),
        ("baseline-student-before", 0.5, None),
        ("step-0001", 0.5, 1),
        ("student-after", 0.5, 2),
    ]

    (summary,) = tracker.summaries
    assert summary.gate_accepted is True
    assert summary.gate_reason == result.gate.reason
    assert summary.teacher_solve_rate == 0.5
    assert summary.student_before_solve_rate == 0.5
    assert summary.student_after_solve_rate == 0.5
    assert summary.total_usd == pytest.approx(result.spend.total_usd)
    assert summary.steps_completed == 3

    assert tracker.finish_calls == 1


def test_tracker_finish_fires_on_the_budget_abort_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0), tracker=tracker)

    # Step 0 completed (its row was tracked before the abort), the gate was
    # never reached, and the tracker was still finished.
    assert [step for step, _ in tracker.steps] == [0]
    assert [name for name, _, _ in tracker.evals] == [
        "baseline-teacher",
        "baseline-student-before",
    ]
    assert tracker.summaries == []
    assert tracker.finish_calls == 1


def test_rollout_params_are_pinned_from_the_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    doc = HarnessDoc.baseline()
    cfg = _cfg().model_copy(
        update={
            "rollout": RolloutConfig(max_turns=7),
            "sampling": SamplingConfig(temperature=0.5, max_tokens=256),
        }
    )

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        pinned = pin_rollout_params(doc, cfg)

    assert pinned.max_turns() == 7
    assert pinned.temperature() == pytest.approx(0.5)
    assert pinned.max_output_tokens() == 256
    # Off-1.0 temperatures bias the reverse-KL advantages; the pinning warns.
    assert "temperature" in caplog.text and "1.0" in caplog.text
    # Pure function of (doc, cfg): repeated pinning yields the identical identity.
    assert pin_rollout_params(doc, cfg).doc_hash == pinned.doc_hash
    assert pinned.doc_hash != doc.doc_hash

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        neutral = pin_rollout_params(doc, _cfg())  # default temperature 1.0
    assert neutral.temperature() == pytest.approx(1.0)
    assert "temperature" not in caplog.text


def test_a_tito_violation_fails_the_forward_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fabricated spans (ids no sampler issued) must die in the TITO assertion."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fabricate_spans = True
    with pytest.raises(AssertionError, match="TITO violation"):
        _run(env, _cfg())


def test_teacher_in_harness_episodes_charge_the_teacher_sample_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warmup collection and the teacher baseline bill sampling, not prefill.

    Every other price is zero (the cached rates derive 20% of zero), so the
    whole run's spend is exactly the teacher-in-harness sampled volume at the
    teacher_sample rate.
    """
    env = _setup(tmp_path, monkeypatch)
    pricing = PricingConfig(
        student_prefill=0.0,
        student_sample=0.0,
        student_train=0.0,
        teacher_prefill=0.0,
        teacher_sample=1e6,  # $1 per token, so USD equals the token count
    )

    result = _run(env, _warmup_cfg(pricing=pricing))

    lines = {line.meter: line for line in result.spend.lines}
    # Teacher-in-harness trials: the gate baseline (2 holdout tasks x gate.k=1)
    # plus the warmup collection (4 train tasks x 2 attempts); the stub
    # collector samples 2 calls x 5 tokens per trial.
    expected_sampled = (2 * 1 + 4 * 2) * 2 * 5
    assert lines["teacher_sample"].tokens == expected_sampled
    assert result.spend.total_usd == pytest.approx(float(expected_sampled))


# -- budget abort and resume -----------------------------------------------------------------


def test_budget_abort_persists_state_and_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)

    with pytest.raises(DistillBudgetError) as excinfo:
        _run(env, _train_priced_cfg(budget_max=1.0))

    error = excinfo.value
    expected_command = resume_command(_NAME, env.run_dir)
    assert error.resume_command == expected_command
    assert expected_command in str(error)
    assert "budget.max_usd" in str(error)
    assert error.max_usd == 1.0
    assert error.spent_usd > 1.0

    # Step 0 completed (its metrics row exists) and its state was checkpointed.
    store = DistillRunStore(env.run_dir)
    assert [row["step"] for row in store.read_metrics()] == [0]
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 0
    # The checkpointed state is real: the fake training client can restore it.
    training = env.service.training
    assert training is not None
    training.inner.load_state(latest.state_path)
    # The run never reached the gate.
    assert not store.gate_path.exists()


def test_resume_command_prints_the_agent_string_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run seeded from a stored version is invoked as 'name@ref'; the printed
    resume command must carry that exact string or the CLI's resume conflict
    check rejects the one command the abort message tells the user to run."""
    env = _setup(tmp_path, monkeypatch)

    with pytest.raises(DistillBudgetError) as excinfo:
        _run(env, _train_priced_cfg(budget_max=1.0), cli_agent=f"{_NAME}@v3")

    expected = resume_command(f"{_NAME}@v3", env.run_dir)
    assert excinfo.value.resume_command == expected
    assert expected in str(excinfo.value)


def test_resume_rejects_a_changed_gate_k(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Baselines recorded at one k must not gate a student-after measured at
    another; the resume refuses instead of comparing different estimators."""
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))

    changed = _cfg().model_copy(update={"gate": GateConfig(k=2)})
    with pytest.raises(RuntimeError, match="measured at k=1"):
        _run(env, changed, resume=True)


def test_resume_rejects_a_changed_teacher_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The teacher baseline is reusable precisely because the teacher identity
    is stable; swapping the teacher mid-run would gate against a stale rate."""
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))

    changed = _cfg().model_copy(update={"teacher": TeacherConfig(model="other/teacher")})
    with pytest.raises(RuntimeError, match="mid-run model swap"):
        _run(env, changed, resume=True)


def test_zero_trial_steps_count_toward_the_empty_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch that scores no trials at all is at least as dead as an all-empty
    one; it must extend the dead-provider streak, not reset it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.zero_trial_train_steps = {0}
    env.rollouts.empty_span_train_steps = {1}

    with pytest.raises(DistillEmptyBatchError) as excinfo:
        _run(env, _cfg())

    assert excinfo.value.consecutive_steps == MAX_CONSECUTIVE_EMPTY_STEPS


def test_two_consecutive_all_empty_steps_abort_with_the_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dead-provider guard: all-empty steps must not burn the whole budget."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.empty_span_train_steps = {0, 1}

    with pytest.raises(DistillEmptyBatchError) as excinfo:
        _run(env, _cfg())

    error = excinfo.value
    assert error.consecutive_steps == MAX_CONSECUTIVE_EMPTY_STEPS
    expected_command = resume_command(_NAME, env.run_dir)
    assert error.resume_command == expected_command
    assert expected_command in str(error)
    assert "no completions" in str(error)
    assert "runner logs" in str(error)
    # Exactly two empty train batches ran: the abort fired at the second,
    # before a third batch could spend more on span-less rollouts.
    train_steps = [call.step_index for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert train_steps == [0, 1]
    # Artifacts persisted: both steps' metrics rows and a resumable checkpoint.
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1]
    assert all(_number(row, "datums") == 0 for row in rows)
    assert all(_number(row, "empty_span_trials") == _number(row, "trials") for row in rows)
    latest = store.latest_checkpoint()
    assert latest is not None
    assert latest.step == 1


def test_a_non_empty_step_resets_the_empty_batch_streak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.empty_span_train_steps = {0, 2}  # the healthy step 1 sits between

    result = _run(env, _cfg())

    # Two non-consecutive empty steps never abort: the run finishes all steps.
    assert result.steps_completed == 3
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert _number(rows[0], "datums") == 0
    assert _number(rows[1], "datums") > 0
    assert _number(rows[2], "datums") == 0
    # A step that trained nothing surfaces no backend or advantage metrics
    # (absent, never fabricated); the healthy step carries them all.
    for row in (rows[0], rows[2]):
        assert row["pg_loss"] is None
        assert row["grad_norm"] is None
        assert row["advantage_mean"] is None
        assert row["advantage_std"] is None
        assert _number(row, "clip_fraction") == 0.0
        assert _number(row, "reward_mean") == 0.0  # trials ran; every one failed
    assert isinstance(rows[1]["pg_loss"], float)
    assert isinstance(rows[1]["grad_norm"], float)
    assert isinstance(rows[1]["advantage_mean"], float)


def test_resume_continues_the_step_count_and_reuses_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(DistillBudgetError):
        _run(env, _train_priced_cfg(budget_max=1.0))
    baseline_calls = [
        call for call in env.rollouts.calls if call.run_dir.name.startswith("baseline")
    ]
    assert len(baseline_calls) == 2

    # Resume with the budget lifted (the documented recovery path).
    result = _run(env, _cfg(), resume=True)

    assert result.steps_completed == 3
    assert result.gate.accepted
    store = DistillRunStore(env.run_dir)
    assert [row["step"] for row in store.read_metrics()] == [0, 1, 2]
    training = env.service.training
    assert training is not None
    latest_before_resume = store.checkpoints()[0]
    assert training.load_state_calls == [latest_before_resume.state_path]
    # Baselines were reused from the recorded eval payloads, not re-run.
    baseline_calls = [
        call for call in env.rollouts.calls if call.run_dir.name.startswith("baseline")
    ]
    assert len(baseline_calls) == 2
    # Training resumed at step 1: across both sessions each step ran exactly once.
    train_steps = [call.step_index for call in env.rollouts.calls if call.run_dir == env.run_dir]
    assert train_steps == [0, 1, 2]
    # Prior-session spend was restored from the spend ledger, and the resumed
    # session's rows carry cumulative totals that INCLUDE it.
    assert result.spend.prior_usd > 0.0
    for row in store.read_metrics()[1:]:
        assert _number(row, "cumulative_usd") > result.spend.prior_usd


def test_resume_restores_spend_charged_between_metrics_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 'cannot spend the budget twice' regression (spend ledger).

    A budget abort during the holdout baselines leaves NO metrics row (rows
    land only when a training step completes), so a resume that derived prior
    spend from metrics.jsonl would restore $0 and happily spend budget.max_usd
    all over again. The ledger is written on every charge, so the resumed
    meter must equal the pre-abort meter exactly.
    """
    env = _setup(tmp_path, monkeypatch)
    cfg = _cfg(budget_max=1e-6)  # the teacher baseline alone exceeds the cap

    with pytest.raises(DistillBudgetError) as first:
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    assert store.read_metrics() == []  # nothing for a metrics-derived resume to see
    assert first.value.spent_usd > 0.0
    assert store.read_spend() == pytest.approx(first.value.spent_usd)
    calls_after_abort = len(env.rollouts.calls)

    # Resuming with the SAME cap must abort immediately at the restored meter
    # (prior spend + nothing new), not run a fresh budget.max_usd worth.
    with pytest.raises(DistillBudgetError) as second:
        _run(env, cfg, resume=True)
    assert second.value.spent_usd == pytest.approx(first.value.spent_usd)
    assert len(env.rollouts.calls) == calls_after_abort  # no new spend before the abort

    # The documented recovery (raise the cap, resume) carries the prior spend
    # forward into the final accounting.
    result = _run(env, _cfg(), resume=True)
    assert result.spend.prior_usd == pytest.approx(first.value.spent_usd)
    assert result.spend.total_usd == pytest.approx(first.value.spent_usd + result.spend.session_usd)
    assert store.read_spend() == pytest.approx(result.spend.total_usd)


def test_fresh_run_into_a_used_run_dir_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    _run(env, _cfg())
    with pytest.raises(ValueError, match="resume=True"):
        _run(env, _cfg())


def test_resume_with_an_empty_run_dir_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="nothing to resume"):
        _run(env, _cfg(), resume=True)


# -- the supervised warmup phase ---------------------------------------------------------------


def test_warmup_trains_on_passing_teacher_trials_then_opd_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    tracker = _RecordingTracker()

    result = _run(env, _warmup_cfg(), tracker=tracker)

    assert result.steps_completed == 3
    assert result.gate.accepted

    # The warmup collection ran the TEACHER on the full train split under the
    # isolated warmup-rollouts root, rollouts_per_task attempts per task.
    (warmup_call,) = _warmup_calls(env)
    assert warmup_call.provider_model == _TEACHER
    assert warmup_call.task_ids == _TRAIN_IDS
    assert warmup_call.attempts == 2
    assert warmup_call.step_index == 0

    # Two cross_entropy passes over the passed-filter datums (tasks at even
    # positions pass: 2 tasks x 2 attempts = 4 kept trials, one datum each),
    # then the three importance_sampling OPD steps; every cross_entropy batch
    # cleared the fake's TITO check on TEACHER-issued spans. The warmup LR
    # applies to the warmup passes only.
    training = env.service.training
    assert training is not None
    assert _loss_fns(env) == ["cross_entropy"] * 2 + ["importance_sampling"] * 3
    ce_batches = [
        batch for batch, loss in training.inner.forward_backward_calls if loss == "cross_entropy"
    ]
    assert all(len(batch) == 4 for batch in ce_batches)
    assert training.inner.optim_step_lrs == [5e-5, 5e-5, 1e-4, 1e-4, 1e-4]

    # OPD step 0 sampled the WARMED student: the post-warmup forced refresh
    # produced a fresh sampler path, distinct from the pre-warmup weights the
    # student-before baseline sampled.
    train_calls = [call for call in env.rollouts.calls if call.run_dir == env.run_dir]
    before_call = next(
        call for call in env.rollouts.calls if call.run_dir.name == STUDENT_BEFORE_EVAL
    )
    assert "warmup" in train_calls[0].provider_model
    assert train_calls[0].provider_model != before_call.provider_model

    # Metrics: one phase-tagged warmup row per warmup step, then the OPD rows
    # (their step indices restart at 0, so the phase key is the discriminator).
    store = DistillRunStore(env.run_dir)
    rows = store.read_metrics()
    warmup_rows = [row for row in rows if row.get("phase") == "warmup"]
    step_rows = [row for row in rows if "phase" not in row]
    assert [row["step"] for row in warmup_rows] == [0, 1]
    assert [row["step"] for row in step_rows] == [0, 1, 2]
    for row in warmup_rows:
        assert row["trials"] == 8
        assert row["kept_trials"] == 4
        assert row["solve_rate"] == 0.5
        assert row["datums"] == 4
        assert _number(row, "learning_rate") == pytest.approx(5e-5)
        assert _number(row, "student_train_tokens") > 0
        assert _number(row, "usd") > 0
    # The teacher collection's charge folds into warmup row 0 only, billed as
    # a teacher-in-harness batch: sampled tokens at teacher_sample plus
    # per-request prefill (unique full-rate, repeats cached).
    assert _number(warmup_rows[0], "teacher_prefill_tokens") > 0
    assert _number(warmup_rows[0], "teacher_cached_prefill_tokens") > 0
    assert _number(warmup_rows[0], "teacher_sample_tokens") > 0
    assert _number(warmup_rows[1], "teacher_prefill_tokens") == 0
    assert _number(warmup_rows[1], "teacher_cached_prefill_tokens") == 0
    assert _number(warmup_rows[1], "teacher_sample_tokens") == 0

    # The tracker saw the same warmup rows the store persisted.
    assert [step for step, _ in tracker.warmup_steps] == [0, 1]
    for (step, metrics), row in zip(tracker.warmup_steps, warmup_rows, strict=True):
        assert row == {"step": step, **metrics.model_dump(mode="json")}

    # The completion marker records the phase, and warmup never lands in the
    # checkpoint manifest (checkpoint steps drive the resume step count).
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 2
    assert record.trials == 8
    assert record.kept_trials == 4
    assert record.datums == 4
    assert record.skipped_reason is None
    assert record.state_path is not None
    assert record.sampler_path == train_calls[0].provider_model
    assert [checkpoint.step for checkpoint in store.checkpoints()] == [1, 2]

    assert "warmup" in {event.phase for event in env.progress}


def test_warmup_zero_passing_trials_skips_to_pure_opd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Zero passing teacher trials degrade the run to pure OPD, never abort it."""
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True

    with caplog.at_level(logging.WARNING, logger="wmh.distill.loop"):
        result = _run(env, _warmup_cfg())

    assert "pure on-policy distillation" in caplog.text
    assert result.steps_completed == 3
    assert _loss_fns(env) == ["importance_sampling"] * 3  # nothing to warm up on
    assert len(_warmup_calls(env)) == 1  # the collection itself did run

    store = DistillRunStore(env.run_dir)
    warmup_rows = [row for row in store.read_metrics() if row.get("phase") == "warmup"]
    assert len(warmup_rows) == 1
    assert warmup_rows[0]["trials"] == 8
    assert warmup_rows[0]["kept_trials"] == 0
    assert warmup_rows[0]["datums"] == 0
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 0
    assert record.skipped_reason is not None
    assert "keep='passed'" in record.skipped_reason
    assert record.state_path is None and record.sampler_path is None


def test_warmup_keep_all_trains_on_failing_trials_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.teacher_fail_all = True  # nothing passes; keep="all" still trains

    result = _run(env, _warmup_cfg(warmup_steps=1, keep="all"))

    assert result.steps_completed == 3
    assert _loss_fns(env) == ["cross_entropy"] + ["importance_sampling"] * 3
    training = env.service.training
    assert training is not None
    (ce_batch, _) = training.inner.forward_backward_calls[0]
    assert len(ce_batch) == 8  # every trial kept: 4 tasks x 2 attempts
    record = DistillRunStore(env.run_dir).read_warmup()
    assert record is not None
    assert record.kept_trials == 8
    assert record.datums == 8


def test_resume_never_reruns_a_completed_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finalize interrupted after the gate eval resumes without re-running
    warmup or the student-after eval. (Resuming a fully COMPLETED run is
    refused outright: see test_resume_of_a_completed_run_is_refused.)"""
    env = _setup(tmp_path, monkeypatch)
    cfg = _warmup_cfg()
    _run(env, cfg)
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 2

    store = DistillRunStore(env.run_dir)
    after_evals_before = _eval_run_counts(env)[STUDENT_AFTER_EVAL]
    # Simulate a crash between the student-after eval and the gate verdict.
    store.gate_path.unlink()

    result = _run(env, cfg, resume=True)

    assert result.steps_completed == 3
    # Neither the teacher collection nor the SFT passes ran again.
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 2
    # The recorded student-after eval was reused, not re-spent.
    assert _eval_run_counts(env)[STUDENT_AFTER_EVAL] == after_evals_before


def test_resume_of_a_completed_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resuming a run whose gate verdict is recorded would re-spend the holdout
    eval and could promote a duplicate adapter version; it must refuse."""
    env = _setup(tmp_path, monkeypatch)
    cfg = _warmup_cfg()
    _run(env, cfg)

    with pytest.raises(RuntimeError, match="already completed"):
        _run(env, cfg, resume=True)


def test_budget_abort_mid_warmup_reruns_warmup_whole_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted warmup holds no completion marker, so it re-runs whole."""
    env = _setup(tmp_path, monkeypatch)
    capped = _warmup_cfg(
        budget_max=1.0,
        # Only student_train is priced, so the run survives the baselines and
        # the teacher collection, then the first warmup pass blows the cap.
        pricing=PricingConfig(
            student_prefill=0.0, student_sample=0.0, student_train=1e9, teacher_prefill=0.0
        ),
    )

    with pytest.raises(DistillBudgetError):
        _run(env, capped)

    store = DistillRunStore(env.run_dir)
    assert store.read_warmup() is None  # no marker: the phase never finished
    assert len(_warmup_calls(env)) == 1
    assert _loss_fns(env).count("cross_entropy") == 1  # the aborted first pass

    result = _run(env, _warmup_cfg(), resume=True)  # cap lifted: the documented recovery

    assert result.steps_completed == 3
    # The resumed session re-collected and re-trained the whole warmup phase.
    assert len(_warmup_calls(env)) == 2
    assert _loss_fns(env).count("cross_entropy") == 1 + 2
    record = store.read_warmup()
    assert record is not None
    assert record.steps == 2


def test_resume_restores_post_warmup_state_when_no_step_checkpoint_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between warmup and the first step checkpoint keeps the warmup.

    Without the warmup record's state_path restore, this resume would start
    OPD from the COLD student and silently lose the warmup it already paid
    for (no step checkpoint exists yet for load_state to use).
    """
    env = _setup(tmp_path, monkeypatch)
    env.rollouts.fail_on_train_step = 0
    cfg = _warmup_cfg()

    with pytest.raises(RuntimeError, match="injected rollout crash"):
        _run(env, cfg)

    store = DistillRunStore(env.run_dir)
    record = store.read_warmup()
    assert record is not None
    assert record.state_path is not None
    assert store.latest_checkpoint() is None  # nothing step-level to restore

    env.rollouts.fail_on_train_step = None
    result = _run(env, cfg, resume=True)

    assert result.steps_completed == 3
    training = env.service.training
    assert training is not None
    assert training.load_state_calls == [record.state_path]
    assert len(_warmup_calls(env)) == 1  # warmup itself was not re-run


# -- preflight failure paths -----------------------------------------------------------------


def test_missing_api_key_without_an_injected_client_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """service_client=None requires the key (or the extra) before any client is built."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises((RuntimeError, ImportError), match="TINKER_API_KEY|--extra distill"):
        run_distillation(
            _NAME,
            _cfg(),
            HarnessDoc.baseline(),
            _TRAIN_IDS,
            _HOLDOUT_IDS,
            tmp_path / "run",
        )


class _ExplodingScorer:
    """A sampling client whose every call fails (a retired teacher model)."""

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        raise RuntimeError("model not found: the teacher was retired")

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        raise RuntimeError("model not found: the teacher was retired")


def test_teacher_ping_failure_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        if model_path == _TEACHER:
            return _ExplodingScorer()
        return original(model_path)

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(RuntimeError, match="teacher preflight ping failed") as excinfo:
        _run(env, _cfg())
    assert "the teacher was retired" in str(excinfo.value)


class _SkewedClient:
    """Delegates sampling but perturbs recomputed logprobs (scoring-path drift)."""

    def __init__(self, inner: DistillSamplingClient) -> None:
        self._inner = inner

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return [
            None if logprob is None else logprob + 1.0
            for logprob in self._inner.compute_logprobs(token_ids)
        ]


def test_tito_recompute_disagreement_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        client = original(model_path)
        if model_path.startswith("tinker://"):
            return _SkewedClient(client)
        return client

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(RuntimeError, match="TITO recompute disagreement"):
        _run(env, _cfg())


class _PerturbedClient:
    """Delegates sampling but shifts recomputed logprobs by per-index offsets.

    Offsets are applied cyclically over the recomputed sequence, letting tests
    model zero-mean kernel noise (small alternating offsets) or one
    catastrophic position (a single large offset).
    """

    def __init__(self, inner: DistillSamplingClient, offsets: Sequence[float]) -> None:
        self._inner = inner
        self._offsets = list(offsets)

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        recomputed = self._inner.compute_logprobs(token_ids)
        return [
            None if lp is None else lp + self._offsets[index % len(self._offsets)]
            for index, lp in enumerate(recomputed)
        ]


def test_tito_recompute_tolerates_zero_mean_kernel_noise() -> None:
    """Small alternating sampler/scorer drift (the observed live regime) passes."""
    inner = FakeSamplingClient("tito-noise-probe")
    noisy = _PerturbedClient(inner, [0.08, -0.08])
    loop_module.tito_recompute_check(noisy, [1, 2, 3, 4])


def test_tito_recompute_single_catastrophic_position_fails() -> None:
    """One multi-nat outlier position trips the per-position bound."""
    inner = FakeSamplingClient("tito-spike-probe")
    prompt = [1, 2, 3, 4]
    probe = inner.sample(prompt, max_tokens=16, temperature=1.0)
    spike_index = len(prompt) + len(list(probe.tokens)) - 1
    offsets = [0.0] * spike_index + [2.0]
    spiky = _PerturbedClient(FakeSamplingClient("tito-spike-probe"), offsets)
    with pytest.raises(RuntimeError, match="per-position bound"):
        loop_module.tito_recompute_check(spiky, prompt)


class _OffsetTokenizer:
    """Encodes every character one id higher than the student's tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [ord(ch) + 1 for ch in text]


class _WrongTokenizerTeacher:
    """A teacher client that CAN supply a tokenizer, and it disagrees."""

    def __init__(self, inner: DistillSamplingClient) -> None:
        self._inner = inner

    def sample(
        self, prompt_token_ids: list[int], *, max_tokens: int, temperature: float
    ) -> SampledSequenceLike:
        return self._inner.sample(prompt_token_ids, max_tokens=max_tokens, temperature=temperature)

    def compute_logprobs(self, token_ids: list[int]) -> list[float | None]:
        return self._inner.compute_logprobs(token_ids)

    def get_tokenizer(self) -> _OffsetTokenizer:
        return _OffsetTokenizer()


def test_tokenizer_fingerprint_mismatch_is_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _setup(tmp_path, monkeypatch)
    original = env.service.create_sampling_client

    def patched(model_path: str) -> DistillSamplingClient:
        client = original(model_path)
        if model_path == _TEACHER:
            return _WrongTokenizerTeacher(client)
        return client

    monkeypatch.setattr(env.service, "create_sampling_client", patched)
    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        _run(env, _cfg())


# -- input validation and the task sampler ---------------------------------------------------


def test_overlapping_splits_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BOTH splits"):
        run_distillation(
            _NAME,
            _cfg(),
            HarnessDoc.baseline(),
            ("task-a", "task-b"),
            ("task-b", "hold-a"),
            tmp_path / "run",
            service_client=_Service(),
        )


def test_task_sampler_is_seeded_unique_and_covering() -> None:
    ids = ["a", "b", "c", "d", "e"]
    first = TaskSampler(ids, seed=7)
    second = TaskSampler(ids, seed=7)
    batches = [first.next_batch(2) for _ in range(6)]
    assert batches == [second.next_batch(2) for _ in range(6)]  # deterministic
    for batch in batches:
        assert len(batch) == 2
        assert len(set(batch)) == 2  # unique within a batch
    # The cycle visits every task before repeating any: the first three
    # batches (six slots over a five-task split) cover the whole split.
    assert {task for batch in batches[:3] for task in batch} == set(ids)
    # Oversized requests clamp to the split size.
    assert TaskSampler(["only"], seed=0).next_batch(5) == ["only"]
    with pytest.raises(ValueError, match="duplicates"):
        TaskSampler(["a", "a"], seed=0)
    with pytest.raises(ValueError, match="empty"):
        TaskSampler([], seed=0)


# -- SdkTrainingClient deadlines: retry-once vs abort per call idempotency -----------------------


class _NeverResolvingFuture:
    """Mimics the SDK future of a wedged session: result(timeout) honors the timeout."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


@dataclass(frozen=True)
class _SavedArtifact:
    path: str


class _ReadyFuture:
    """A fake SDK future whose result is immediately available."""

    def __init__(self, value: object = None) -> None:
        self._value = value

    def result(self, timeout: float | None = None) -> object:
        del timeout
        return self._value


class _WedgedOnceTrainingClient:
    """Fake tinker.TrainingClient: the FIRST call of each method wedges, retries succeed."""

    def __init__(self) -> None:
        self.forward_backward_calls = 0
        self.optim_step_calls = 0
        self.save_state_names: list[str] = []
        self.load_state_paths: list[str] = []
        self.save_weights_names: list[str] = []

    def forward_backward(
        self, datums: object, loss_fn: object
    ) -> _NeverResolvingFuture | _ReadyFuture:
        del datums, loss_fn
        self.forward_backward_calls += 1
        return _NeverResolvingFuture()

    def optim_step(self, params: object) -> _NeverResolvingFuture | _ReadyFuture:
        del params
        self.optim_step_calls += 1
        return _NeverResolvingFuture()

    def save_state(self, name: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.save_state_names.append(name)
        if len(self.save_state_names) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/state/{name}"))

    def load_state(self, path: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.load_state_paths.append(path)
        if len(self.load_state_paths) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture()

    def save_weights_for_sampler(self, name: str) -> _NeverResolvingFuture | _ReadyFuture:
        self.save_weights_names.append(name)
        if len(self.save_weights_names) == 1:
            return _NeverResolvingFuture()
        return _ReadyFuture(_SavedArtifact(path=f"tinker://fake/sampler/{name}"))


def _sdk_training_client(fake: _WedgedOnceTrainingClient) -> SdkTrainingClient:
    return SdkTrainingClient(cast("tinker.TrainingClient", fake))


def _short_deadlines(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in (
        "WMH_TINKER_DEADLINE_FORWARD_BACKWARD",
        "WMH_TINKER_DEADLINE_OPTIM_STEP",
        "WMH_TINKER_DEADLINE_SAVE_STATE",
        "WMH_TINKER_DEADLINE_LOAD_STATE",
        "WMH_TINKER_DEADLINE_SAVE_WEIGHTS_FOR_SAMPLER",
        "WMH_TINKER_DEADLINE_CONNECT",
    ):
        monkeypatch.setenv(env_var, "0.05")


def test_save_weights_for_sampler_retries_once_under_a_fresh_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    path = client.save_weights_for_sampler("run-step-0001")

    # Retried exactly once, and the retry saved under a distinct "-r1" name so
    # a first attempt that completed server-side after abandonment cannot collide.
    assert len(fake.save_weights_names) == 2
    first, second = fake.save_weights_names
    assert first.startswith("run-step-0001-")
    assert second == f"{first}-r1"
    assert path == f"tinker://fake/sampler/{second}"


def test_save_state_retries_once_with_a_fresh_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    path = client.save_state()

    assert len(fake.save_state_names) == 2
    first, second = fake.save_state_names
    assert first != second  # the retry advanced the per-session counter
    assert first.endswith("-state-0000")
    assert second.endswith("-state-0001")
    assert path == f"tinker://fake/state/{second}"


def test_load_state_retries_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    client.load_state("tinker://fake/state/x")

    assert fake.load_state_paths == ["tinker://fake/state/x"] * 2


def _attached_datum() -> TrainDatum:
    return TrainDatum(
        trial_name="task-a__x1",
        fragment_index=0,
        model_input_tokens=[1, 2, 3],
        loss_mask=[0.0, 1.0, 1.0],
        sampled_logprobs=[0.0, -0.5, -0.7],
        advantages=[0.0, 0.1, 0.2],
    )


def test_forward_backward_deadline_aborts_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # NOT idempotent: gradients may have accumulated server-side before the
    # deadline fired, so a retry could count the batch twice. Exactly one call.
    pytest.importorskip("tinker")
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    with pytest.raises(TinkerDeadlineError, match="tinker forward_backward timed out"):
        client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert fake.forward_backward_calls == 1


def test_optim_step_deadline_aborts_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    # NOT idempotent: the step may have been applied server-side; a retry
    # would double-step the optimizer. Exactly one call.
    pytest.importorskip("tinker")
    _short_deadlines(monkeypatch)
    fake = _WedgedOnceTrainingClient()
    client = _sdk_training_client(fake)

    with pytest.raises(TinkerDeadlineError, match="tinker optim_step timed out"):
        client.optim_step(1e-5)
    assert fake.optim_step_calls == 1


# -- SdkTrainingClient metric extraction (what the SDK actually exposes) --------------------------


def test_sdk_metric_value_matches_bare_and_suffixed_names() -> None:
    """Server metric keys carry a ':reduction' suffix; both spellings match."""
    assert sdk_metric_value({"total_loss:sum": 2.0}, SDK_LOSS_METRIC_NAMES) == 2.0
    assert sdk_metric_value({"total_loss": 1.5}, SDK_LOSS_METRIC_NAMES) == 1.5
    assert sdk_metric_value({"loss:mean": 0.25}, SDK_LOSS_METRIC_NAMES) == 0.25
    assert sdk_metric_value({"grad_norm:mean": 0.5}, SDK_GRAD_NORM_METRIC_NAMES) == 0.5
    # Unrelated keys (e.g. the documented MoE diagnostics), an empty dict, and
    # the OptimStepResponse's metrics=None all surface nothing.
    assert sdk_metric_value({"e_frac_with_tokens:mean": 1.0}, SDK_LOSS_METRIC_NAMES) is None
    assert sdk_metric_value({}, SDK_GRAD_NORM_METRIC_NAMES) is None
    assert sdk_metric_value(None, SDK_LOSS_METRIC_NAMES) is None


@dataclass(frozen=True)
class _FwdBwdOutput:
    """The ForwardBackwardOutput slice the adapter reads (no typed loss exists)."""

    metrics: dict[str, float]


@dataclass(frozen=True)
class _OptimResponse:
    """The OptimStepResponse slice the adapter reads."""

    metrics: dict[str, float] | None


class _ReadyTrainingClient:
    """Fake tinker.TrainingClient whose futures resolve immediately."""

    def __init__(
        self, fwdbwd_metrics: dict[str, float], optim_metrics: dict[str, float] | None
    ) -> None:
        self._fwdbwd_metrics = fwdbwd_metrics
        self._optim_metrics = optim_metrics

    def forward_backward(self, datums: object, loss_fn: object) -> _ReadyFuture:
        del datums, loss_fn
        return _ReadyFuture(_FwdBwdOutput(metrics=self._fwdbwd_metrics))

    def optim_step(self, params: object) -> _ReadyFuture:
        del params
        return _ReadyFuture(_OptimResponse(metrics=self._optim_metrics))


def test_sdk_training_client_extracts_reported_metrics() -> None:
    pytest.importorskip("tinker")
    fake = _ReadyTrainingClient(
        fwdbwd_metrics={"total_loss:sum": 1.25}, optim_metrics={"grad_norm:mean": 3.5}
    )
    client = SdkTrainingClient(cast("tinker.TrainingClient", fake))
    output = client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert output.loss == 1.25
    assert client.optim_step(1e-5).grad_norm == 3.5


def test_sdk_training_client_surfaces_none_when_the_service_reports_nothing() -> None:
    """No fabricated values: absent metrics stay None end to end."""
    pytest.importorskip("tinker")
    fake = _ReadyTrainingClient(fwdbwd_metrics={}, optim_metrics=None)
    client = SdkTrainingClient(cast("tinker.TrainingClient", fake))
    output = client.forward_backward([_attached_datum()], loss_fn="importance_sampling")
    assert output.loss is None
    assert client.optim_step(1e-5).grad_norm is None


def test_client_construction_is_deadline_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Sampling-client construction now goes through the process-wide shared
    # cache (bounded there; see wmh/providers/tinker_test.py), so the loop's
    # remaining direct construction is the training client.
    _short_deadlines(monkeypatch)
    never = threading.Event()

    class _WedgedService:
        def create_lora_training_client(self, base_model: str, rank: int) -> NoReturn:
            del base_model, rank
            never.wait()
            raise AssertionError("unreachable: the event is never set")

    service = SdkServiceClient(cast("tinker.ServiceClient", _WedgedService()))
    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        service.create_lora_training_client(_STUDENT, rank=8)


def test_loop_sampling_adapter_deadline_evicts_the_shared_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop's per-refresh sampling clients come from the process-wide
    # shared cache; a deadline expiry must evict the entry so every future
    # user (the harbor trial providers included) rebuilds a fresh session.
    monkeypatch.setenv("WMH_TINKER_DEADLINE_CONNECT", "0.05")
    never = threading.Event()

    class _WedgedSamplingClient:
        def get_tokenizer(self) -> NoReturn:
            never.wait()
            raise AssertionError("unreachable: the event is never set")

    wedged = cast("tinker.SamplingClient", _WedgedSamplingClient())
    path = "tinker://fake/sampler/x"
    monkeypatch.setattr(providers_tinker, "_shared_samplers", {path: wedged})
    adapter = SdkSamplingClient(wedged, model=path)

    with pytest.raises(TinkerDeadlineError, match="tinker connect timed out"):
        adapter.get_tokenizer()

    assert path not in providers_tinker._shared_samplers

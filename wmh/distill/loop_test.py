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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from llm_waterfall.types import ChatMessage, ChatTool

import wmh.distill.loop as loop_module
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
)
from wmh.distill.data import TrainDatum
from wmh.distill.fake_tinker import (
    FakeDatum,
    FakeSamplingClient,
    FakeServiceClient,
    FakeTokenizer,
    FakeTrainingClient,
)
from wmh.distill.loop import (
    DistillBudgetError,
    DistillProgress,
    DistillResult,
    DistillSamplingClient,
    StepMetrics,
    TaskSampler,
    pin_rollout_params,
    resume_command,
    run_distillation,
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

    def forward_backward(self, datums: Sequence[TrainDatum], loss_fn: str) -> None:
        self.inner.forward_backward([_fake_datum(datum) for datum in datums], loss_fn)

    def optim_step(self, learning_rate: float) -> None:
        self.inner.optim_step(learning_rate)

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
        client = self.service.create_sampling_client(provider_config.model)
        records: list[TrialRecord] = []
        for task_index, task_id in enumerate(task_ids):
            for attempt in range(1, cfg.train.group_size + 1):
                passed = task_index % 2 == 0
                trial_name = f"{task_id}__s{attempt}"
                records.append(
                    TrialRecord(
                        task_id=task_id,
                        attempt=attempt,
                        trial_name=trial_name,
                        reward=1.0 if passed else 0.0,
                        passed=passed,
                        spans=self._spans(client, task_id, step_index, attempt),
                        stop_reason="submitted",
                        artifact_dir=str(
                            run_dir / "harbor" / f"step-{step_index:04d}" / trial_name
                        ),
                    )
                )
        stats = RolloutStats(
            trials=len(records),
            trials_with_spans=len(records),
            solve_rate=sum(1 for record in records if record.passed) / len(records),
            empty_span_trials=0,
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


def _run(
    env: _Env,
    cfg: DistillConfig,
    *,
    resume: bool = False,
    tracker: DistillTracker | None = None,
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
        self.evals: list[tuple[str, float, int | None]] = []
        self.summaries: list[_TrackedSummary] = []
        self.finish_calls = 0

    def log_step(self, step: int, metrics: StepMetrics) -> None:
        self.steps.append((step, metrics))

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
        for key in (
            "usd",
            "student_prefill_tokens",
            "student_sample_tokens",
            "student_train_tokens",
            "teacher_prefill_tokens",
        ):
            assert _number(row, key) > 0

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
    # Prior-session spend was restored from the spend ledger.
    assert result.spend.prior_usd > 0.0


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

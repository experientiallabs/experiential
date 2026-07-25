"""Offline validation for `probe_scaffold.py`: no Tinker, no E2B, no docker, no spend.

The probe's seams are monkeypatched the way the distill tests do it: the harbor
scorer is stubbed exactly as `wmh/distill/rollouts_test.py` stubs it, so the
REAL `collect_rollouts` runs (real per-step job dir, real token sinks, real
`TrialRecord` assembly), and the sampling client is `wmh.distill.fake_tinker`'s
deterministic fake. The fake service raises if anyone asks it for a training
client, which is the invariant that keeps this probe from ever training.

`.agents/` sits outside the root pytest `testpaths`, so run this file explicitly:

    uv run pytest -q .agents/distill/probe_scaffold_test.py
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import probe_scaffold
import pytest
from harbor.models.job.config import JobConfig

from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    PricingConfig,
    RolloutConfig,
    SamplingConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
    snapshot_toml,
)
from wmh.distill.fake_tinker import FakeSamplingClient, FakeTokenizer
from wmh.evals.harbor.scorer import HarborScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_reap import CapacityCheck
from wmh.harness.runtime import StopReason
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.providers.tinker import TokenRecorder, TokenSpan

_STUDENT = "Qwen/Qwen3-4B"
_TEACHER = "Qwen/Qwen3-8B"
_TASK_IDS = ("task-a", "task-b")
_GROUP_SIZE = 2
_SERVED_WINDOW = 262_144


# -- fakes -----------------------------------------------------------------------------------


class _PingSampler(FakeSamplingClient):
    """The deterministic fake sampler plus the tokenizer `SdkSamplingClient` exposes."""

    def get_tokenizer(self) -> FakeTokenizer:
        """The student tokenizer the preflight ping encodes with."""
        return FakeTokenizer()


class _FakeService:
    """A sampler-only service: asking it for a training client is a test failure."""

    def __init__(self) -> None:
        self.sampled: list[str] = []

    def create_lora_training_client(self, base_model: str, rank: int = 32) -> object:
        """Never legal here: the probe must not be able to train."""
        raise AssertionError(
            f"the scaffold probe opened a training client for {base_model!r} (rank {rank})"
        )

    def create_sampling_client(self, model_path: str) -> _PingSampler:
        """A fake sampler seeded by the model string."""
        self.sampled.append(model_path)
        return _PingSampler(seed=model_path)


@dataclass(frozen=True)
class _Trial:
    """One canned trial outcome, as a real harbor trial would leave it behind."""

    task_id: str
    attempt: int
    passed: bool = False
    stop_reason: str | None = StopReason.SUBMITTED.value
    turns: int = 3
    infra_failed: bool = False


class _StubScorer:
    """Stands in for a created HarborScorer: writes trial evidence, returns cells."""

    def __init__(self, trials: Sequence[_Trial], jobs_dir: Path, token_sink_dir: Path) -> None:
        self.trials = list(trials)
        self.jobs_dir = jobs_dir
        self.token_sink_dir = token_sink_dir
        self.score_calls = 0

    def candidate_job_dir(self, doc: HarnessDoc) -> Path:
        """Mirror HarborScorer's deterministic per-candidate job dir."""
        return self.jobs_dir / f"wmh-{doc.doc_hash[:12]}"

    def score(
        self, doc: HarnessDoc, *, should_cancel: Callable[[], bool] | None = None
    ) -> ScoreReport:
        """Write each trial's spans and run trace, then report its reward cell."""
        del should_cancel
        self.score_calls += 1
        candidate = self.candidate_job_dir(doc)
        cells: list[ScoreCell] = []
        for trial in self.trials:
            trial_name = f"{trial.task_id}__s{trial.attempt}"
            trial_dir = candidate / trial_name
            agent_dir = trial_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)
            if trial.stop_reason is not None:
                (agent_dir / "wmh-run.json").write_text(
                    json.dumps({"stop_reason": trial.stop_reason}), encoding="utf-8"
                )
            if trial.turns:
                recorder = TokenRecorder(jsonl_path=self.token_sink_dir / f"{trial_name}.jsonl")
                for call_index in range(trial.turns):
                    recorder.record(
                        TokenSpan(
                            call_index=call_index,
                            prompt_token_ids=list(range(10 + call_index * 5)),
                            sampled_token_ids=[70, 71, 72],
                            sampled_logprobs=[-0.5, -0.25, -1.0],
                        )
                    )
            cells.append(
                ScoreCell(
                    task_id=trial.task_id,
                    attempt=trial.attempt,
                    reward=1.0 if trial.passed else 0.0,
                    passed=trial.passed,
                    artifact_dir=str(trial_dir),
                    infra_failed=trial.infra_failed,
                )
            )
        return ScoreReport(
            doc_hash=doc.doc_hash,
            request=ScoreRequest(task_ids=_TASK_IDS, attempts=_GROUP_SIZE),
            reward_mode="raw",
            cells=tuple(cells),
        )


@dataclass
class _Capture:
    """What the stubbed `HarborScorer.create` was asked for."""

    kwargs: list[dict[str, object]]
    service: _FakeService


# -- harness -----------------------------------------------------------------------------------


def _config(tmp_path: Path, *, backend: str = "local", trial_concurrency: int = 3) -> Path:
    """Write a real distill TOML the probe loads through `load_distill_config`."""
    template = tmp_path / "job-template.yaml"
    template.write_text(
        "job_name: template\n"
        "jobs_dir: /tmp/overridden-by-the-collector\n"
        "n_concurrent_trials: 1\n"
        "datasets:\n"
        f"- path: {tmp_path / 'tasks'}\n"
        "agents:\n"
        "- {}\n",
        encoding="utf-8",
    )
    cfg = DistillConfig(
        student=StudentConfig(base_model=_STUDENT),
        teacher=TeacherConfig(model=_TEACHER),
        harbor=HarborConfig(
            job_template=str(template),
            backend="e2b" if backend == "e2b" else "local",
        ),
        rollout=RolloutConfig(
            max_turns=100, episode_timeout_s=1800.0, context_budget_tokens=240_000
        ),
        train=TrainConfig(group_size=_GROUP_SIZE, trial_concurrency=trial_concurrency),
        sampling=SamplingConfig(temperature=1.0, max_tokens=16_384),
        pricing=PricingConfig(student_prefill=0.57, student_sample=1.44),
    )
    path = tmp_path / "probe.toml"
    path.write_text(snapshot_toml(cfg), encoding="utf-8")
    return path


def _task_ids_file(tmp_path: Path, task_ids: Sequence[str] = _TASK_IDS) -> Path:
    path = tmp_path / "task-ids.json"
    path.write_text(json.dumps(list(task_ids)), encoding="utf-8")
    return path


def _install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, trials: Sequence[_Trial]) -> _Capture:
    """Stub every live seam: the scorer, the sampling service, and the capability probe."""
    capture = _Capture(kwargs=[], service=_FakeService())

    async def fake_create(
        _cls: type[HarborScorer],
        job_template: JobConfig,
        task_ids: Sequence[str],
        **kwargs: object,
    ) -> _StubScorer:
        capture.kwargs.append({**kwargs, "task_ids": list(task_ids)})
        sink = kwargs["extra_agent_kwargs"]
        assert isinstance(sink, dict)
        return _StubScorer(trials, Path(job_template.jobs_dir), Path(str(sink["token_sink_dir"])))

    monkeypatch.setattr(HarborScorer, "create", classmethod(fake_create))
    monkeypatch.setattr(probe_scaffold, "shared_service_client", lambda: object())
    monkeypatch.setattr(probe_scaffold, "SdkServiceClient", lambda _service: capture.service)
    monkeypatch.setattr(probe_scaffold, "served_context_window", lambda _model: _SERVED_WINDOW)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    return capture


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    trials: Sequence[_Trial],
    *,
    extra: Sequence[str] = (),
    backend: str = "local",
    trial_concurrency: int = 3,
) -> tuple[int, str, dict[str, object]]:
    """Run the probe end to end; return (exit code, stdout report, report JSON)."""
    capture = _install(monkeypatch, tmp_path, trials)
    run_dir = tmp_path / "run"
    code = probe_scaffold.main(
        [
            "--config",
            str(_config(tmp_path, backend=backend, trial_concurrency=trial_concurrency)),
            "--task-ids",
            str(_task_ids_file(tmp_path)),
            "--run-dir",
            str(run_dir),
            *extra,
        ]
    )
    out = capsys.readouterr().out
    report_path = run_dir / probe_scaffold.REPORT_FILENAME
    payload: dict[str, object] = {}
    if report_path.exists():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        payload = loaded
    # The probe pinged the student and NEVER asked for a training client (the fake
    # service raises on that), which is what keeps this measurement free of training.
    assert capture.service.sampled == [_STUDENT]
    return code, out, payload


# -- tests -------------------------------------------------------------------------------------


def test_a_fully_submitting_wave_passes_and_writes_per_episode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trials = [
        _Trial("task-a", 1, passed=True, turns=4),
        _Trial("task-a", 2, turns=2),
        _Trial("task-b", 1, turns=9),
        _Trial("task-b", 2, passed=True, turns=6),
    ]

    code, out, payload = _run(monkeypatch, tmp_path, capsys, trials)

    assert code == 0
    assert "PASS" in out
    assert "SCAFFOLD LOSS RATE" in out and "0.0%" in out
    assert "submitted" in out and "a real completion" in out
    # Turn and token distributions come from the joined token sinks.
    assert "turns per episode" in out and "sampled tokens per episode" in out
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["trials"] == 4
    assert stats["executed_trials"] == 4
    assert stats["scaffold_loss_rate"] == 0.0
    assert stats["solve_rate"] == 0.5
    assert stats["stop_reason_counts"] == {"submitted": 4}
    assert payload["passed"] is True
    # Every episode is auditable: its harbor dir, its sink file, and per-call lengths
    # instead of a gigabyte of duplicated token ids.
    episodes = payload["episodes"]
    assert isinstance(episodes, list) and len(episodes) == 4
    first = episodes[0]
    assert isinstance(first, dict)
    assert first["turns"] == 4
    assert first["sampled_tokens"] == 12
    assert first["stop_reason"] == "submitted"
    assert Path(str(first["token_sink"])).is_file()
    assert Path(str(first["artifact_dir"])).is_dir()
    assert first["spans"] == [
        {"call_index": 0, "prompt_tokens": 10, "sampled_tokens": 3},
        {"call_index": 1, "prompt_tokens": 15, "sampled_tokens": 3},
        {"call_index": 2, "prompt_tokens": 20, "sampled_tokens": 3},
        {"call_index": 3, "prompt_tokens": 25, "sampled_tokens": 3},
    ]
    # Priced from the config's own [pricing] student rates.
    spend = payload["student_spend_usd"]
    assert isinstance(spend, float) and spend > 0.0


def test_scaffold_loss_over_the_threshold_fails_with_the_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trials = [
        _Trial("task-a", 1, passed=True, turns=4),
        _Trial("task-a", 2, stop_reason=StopReason.MAX_TURNS.value, turns=100),
        _Trial("task-b", 1, stop_reason=StopReason.NO_TOOL_CALL.value, turns=5),
        _Trial("task-b", 2, stop_reason=StopReason.OUTPUT_TRUNCATED.value, turns=7),
    ]

    code, out, payload = _run(monkeypatch, tmp_path, capsys, trials)

    assert code == 1
    assert "FAIL" in out
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["scaffold_loss_rate"] == 0.75
    assert stats["stop_reason_counts"] == {
        "max_turns": 1,
        "no_tool_call": 1,
        "output_truncated": 1,
        "submitted": 1,
    }
    # Each bucket is spelled out in plain language, with its share of the wave.
    assert "cut off at the turn cap" in out
    assert "prose-only turns exhausted the nudge budget" in out
    assert "cut at the output-token cap" in out
    assert "25.0%" in out  # each of the four buckets is one episode of four


def test_the_markers_harbor_writes_into_partial_traces_are_named_not_shrugged_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real wave also shows two non-`StopReason` markers; both are scaffold losses."""
    trials = [
        _Trial("task-a", 1, passed=True, turns=4),
        _Trial("task-a", 2, stop_reason=StopReason.BUDGET.value, turns=60),
        _Trial("task-b", 1, stop_reason="cancelled-by-harbor-timeout", turns=12),
        _Trial("task-b", 2, stop_reason="agent-exception:TimeoutError", turns=3),
    ]

    code, out, payload = _run(monkeypatch, tmp_path, capsys, trials)

    assert code == 1
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["scaffold_loss_rate"] == 0.75
    assert "ran past the episode wall budget" in out
    assert "harbor cancelled the trial on its own timeout" in out
    assert "the agent process raised TimeoutError" in out


def test_a_threshold_override_decides_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One cut-off episode of four is 25% loss: over 5%, under a 30% threshold."""
    trials = [
        _Trial("task-a", 1, turns=3),
        _Trial("task-a", 2, turns=3),
        _Trial("task-b", 1, turns=3),
        _Trial("task-b", 2, stop_reason=StopReason.PROVIDER_ERROR.value, turns=1),
    ]

    code, out, _ = _run(
        monkeypatch, tmp_path, capsys, trials, extra=["--max-scaffold-loss", "0.30"]
    )

    assert code == 0
    assert "PASS" in out
    assert "threshold (--max-scaffold-loss)" in out and "30.0%" in out


def test_infra_failures_are_excluded_and_an_all_dead_wave_is_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dead = [
        _Trial(task_id, attempt, stop_reason=None, turns=0, infra_failed=True)
        for task_id in _TASK_IDS
        for attempt in (1, 2)
    ]

    code, out, payload = _run(monkeypatch, tmp_path, capsys, dead)

    assert code == 1
    assert "NULL MEASUREMENT" in out
    stats = payload["stats"]
    assert isinstance(stats, dict)
    assert stats["executed_trials"] == 0
    assert stats["infra_failed_trials"] == 4
    # A wave with no verifier evidence reports no rate at all, never 0.0% as a pass.
    assert stats["scaffold_loss_rate"] == 0.0
    assert payload["passed"] is False


def test_a_zero_reward_submit_with_no_turns_is_flagged_as_a_possible_false_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stale runner's reason-less `done` frame degrades to `submitted`; say so."""
    trials = [
        _Trial("task-a", 1, passed=True, turns=5),
        _Trial("task-a", 2, turns=0),
        _Trial("task-b", 1, turns=1),
        _Trial("task-b", 2, turns=6),
    ]

    code, out, _ = _run(monkeypatch, tmp_path, capsys, trials)

    assert code == 0  # the recorded stop reasons all say `submitted`
    assert "possible FALSE PASS" in out
    assert "2 episode(s) reported `submitted` with reward 0" in out


def test_the_e2b_wave_reserves_two_sandboxes_per_concurrent_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Capacity is checked against the trials this wave can really run at once."""
    asked: list[int] = []

    def record(*, required: int) -> CapacityCheck:
        asked.append(required)
        return CapacityCheck(cap=100, alive_before=0, alive=0, required=required)

    monkeypatch.setattr(probe_scaffold, "check_capacity", record)

    trials = [_Trial(task_id, attempt, turns=2) for task_id in _TASK_IDS for attempt in (1, 2)]
    code, _, _ = _run(monkeypatch, tmp_path, capsys, trials, backend="e2b", trial_concurrency=64)

    assert code == 0
    # 2 tasks x 2 attempts = 4 trials, so concurrency clamps to 4, not the configured 64.
    assert asked == [4 * probe_scaffold.E2B_SANDBOXES_PER_TRIAL]


def test_a_full_e2b_account_refuses_to_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        probe_scaffold,
        "check_capacity",
        lambda *, required: CapacityCheck(cap=100, alive_before=99, alive=99, required=required),
    )
    trials = [_Trial(task_id, attempt) for task_id in _TASK_IDS for attempt in (1, 2)]

    code, out, _ = _run(monkeypatch, tmp_path, capsys, trials, backend="e2b")

    assert code == 2  # a setup error, not a measured failure
    assert out == ""  # no report is printed for a wave that never ran


def test_a_second_probe_into_the_same_run_dir_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    trials = [_Trial(task_id, attempt, turns=2) for task_id in _TASK_IDS for attempt in (1, 2)]
    first, _, _ = _run(monkeypatch, tmp_path, capsys, trials)
    assert first == 0

    second = probe_scaffold.main(
        [
            "--config",
            str(_config(tmp_path)),
            "--task-ids",
            str(_task_ids_file(tmp_path)),
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )

    assert second == 2


def test_task_ids_are_validated_like_the_cli(tmp_path: Path) -> None:
    good = _task_ids_file(tmp_path)
    assert probe_scaffold.load_task_ids(good) == _TASK_IDS

    not_a_list = tmp_path / "bad.json"
    not_a_list.write_text('{"task": "a"}', encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON array"):
        probe_scaffold.load_task_ids(not_a_list)

    duplicated = tmp_path / "dupes.json"
    duplicated.write_text(json.dumps(["task-a", "task-a"]), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid task ids"):
        probe_scaffold.load_task_ids(duplicated)

    with pytest.raises(ValueError, match="cannot load task ids"):
        probe_scaffold.load_task_ids(tmp_path / "nowhere.json")


def test_the_probe_pins_the_configs_rollout_params_onto_the_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wave must run the same document a training step would, or it measures nothing."""
    trials = [_Trial(task_id, attempt, turns=2) for task_id in _TASK_IDS for attempt in (1, 2)]
    capture = _install(monkeypatch, tmp_path, trials)
    config_path = _config(tmp_path)
    run_dir = tmp_path / "run"

    code = probe_scaffold.main(
        [
            "--config",
            str(config_path),
            "--task-ids",
            str(_task_ids_file(tmp_path)),
            "--run-dir",
            str(run_dir),
        ]
    )
    capsys.readouterr()

    assert code == 0
    cfg = probe_scaffold.load_distill_config(config_path)
    pinned = probe_scaffold.pin_rollout_params(probe_scaffold.probe_harness(), cfg)
    assert pinned.max_turns() == 100
    assert pinned.max_output_tokens() == 16_384
    payload = json.loads((run_dir / probe_scaffold.REPORT_FILENAME).read_text(encoding="utf-8"))
    assert payload["harness_doc_hash"] == pinned.doc_hash
    # The collector got the config's budgets and the base-model provider, no adapter.
    [kwargs] = capture.kwargs
    assert kwargs["episode_timeout_s"] == 1800.0
    assert kwargs["context_window"] == 240_000
    assert kwargs["attempts"] == _GROUP_SIZE
    assert kwargs["agent_concurrency"] == 1  # local pi shares one runner dir
    provider = payload["provider_config"]
    assert isinstance(provider, dict)
    assert provider["kind"] == "tinker"
    assert provider["model"] == _STUDENT and provider["model_type"] == _STUDENT

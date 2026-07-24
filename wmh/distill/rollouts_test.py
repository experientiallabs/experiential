"""Tests for the rollout collector against a stubbed harbor scorer."""

from __future__ import annotations

import ast
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from harbor.models.job.config import JobConfig

import wmh.distill.rollouts as rollouts_module
from wmh.distill.agents import WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH
from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from wmh.distill.rollouts import collect_rollouts
from wmh.evals.harbor.scorer import HarborScorer
from wmh.harness.doc import HarnessDoc
from wmh.harness.scoring import ScoreCell, ScoreReport, ScoreRequest
from wmh.providers.base import ProviderConfig, ProviderKind
from wmh.providers.tinker import TokenRecorder, TokenSpan

_TASK_IDS = ("task-a", "task-b")
_GROUP_SIZE = 2


def _provider_config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="Qwen/Qwen3-8B",
        model="tinker://run/weights/4",
    )


def _write_template(tmp_path: Path) -> Path:
    template_path = tmp_path / "job-template.yaml"
    template_path.write_text(
        "job_name: template\n"
        "jobs_dir: /tmp/overridden-by-the-collector\n"
        "n_concurrent_trials: 1\n"
        "datasets:\n"
        f"- path: {tmp_path / 'tasks'}\n"
        "agents:\n"
        "- {}\n",
        encoding="utf-8",
    )
    return template_path


def _cfg(
    tmp_path: Path,
    *,
    backend: str = "local",
    trial_concurrency: int = 3,
) -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-4B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-8B"),
        harbor=HarborConfig(
            job_template=str(_write_template(tmp_path)),
            backend="e2b" if backend == "e2b" else "local",
            reward_key="reward",
        ),
        train=TrainConfig(group_size=_GROUP_SIZE, trial_concurrency=trial_concurrency),
    )


def _span(call_index: int) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2, call_index],
        sampled_token_ids=[70, 71],
        sampled_logprobs=[-0.5, -1.0],
    )


def _report(harness: HarnessDoc, trials_dir: Path) -> ScoreReport:
    """The canned (task x attempt) matrix: task-a passes both, task-b fails both."""
    cells = []
    for task_id in _TASK_IDS:
        for attempt in (1, 2):
            reward = 1.0 if task_id == "task-a" else 0.0
            cells.append(
                ScoreCell(
                    task_id=task_id,
                    attempt=attempt,
                    reward=reward,
                    passed=reward == 1.0,
                    artifact_dir=str(trials_dir / f"{task_id}__s{attempt}"),
                )
            )
    return ScoreReport(
        doc_hash=harness.doc_hash,
        request=ScoreRequest(task_ids=_TASK_IDS, attempts=_GROUP_SIZE),
        reward_mode="raw",
        cells=tuple(cells),
    )


class _StubScorer:
    """Stands in for a created HarborScorer; returns the canned report."""

    def __init__(self, report: ScoreReport) -> None:
        self.report = report
        self.score_calls: list[tuple[str, Callable[[], bool] | None]] = []
        self.jobs_dir: Path | None = None

    def candidate_job_dir(self, doc: HarnessDoc) -> Path:
        """Mirror HarborScorer's deterministic per-candidate job dir."""
        assert self.jobs_dir is not None
        return self.jobs_dir / f"wmh-{doc.doc_hash[:12]}"

    def score(
        self,
        doc: HarnessDoc,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ScoreReport:
        self.score_calls.append((doc.doc_hash, should_cancel))
        return self.report


class _CreateCapture:
    """Monkeypatch target for HarborScorer.create; records every construction."""

    def __init__(self, stub: _StubScorer) -> None:
        self.stub = stub
        self.templates: list[JobConfig] = []
        self.task_ids: list[list[str]] = []
        self.kwargs: list[dict[str, object]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        capture = self

        async def fake_create(
            _cls: type[HarborScorer],
            job_template: JobConfig,
            task_ids: Sequence[str],
            **kwargs: object,
        ) -> _StubScorer:
            capture.templates.append(job_template)
            capture.task_ids.append(list(task_ids))
            capture.kwargs.append(dict(kwargs))
            capture.stub.jobs_dir = job_template.jobs_dir
            return capture.stub

        monkeypatch.setattr(HarborScorer, "create", classmethod(fake_create))


def test_collect_rollouts_wires_the_distill_agent_and_joins_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    trials_dir = tmp_path / "trials"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, trials_dir))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    # Prewrite sinks for 3 of the 4 trials plus one run trace, as the trials would.
    sink_dir = run_dir / "tokens" / "step-0004"
    sink_dir.mkdir(parents=True)
    for trial_name in ("task-a__s1", "task-a__s2", "task-b__s1"):
        recorder = TokenRecorder(jsonl_path=sink_dir / f"{trial_name}.jsonl")
        recorder.record(_span(0))
        recorder.record(_span(1))
    agent_dir = trials_dir / "task-a__s1" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "wmh-run.json").write_text(
        json.dumps({"stop_reason": "submitted"}), encoding="utf-8"
    )

    def cancel_poll() -> bool:
        return False

    records, stats = collect_rollouts(
        4,
        _TASK_IDS,
        cfg,
        harness,
        _provider_config(),
        run_dir,
        should_cancel=cancel_poll,
    )

    # The scorer was created on a FRESH per-step jobs dir with the distill agent.
    [template] = capture.templates
    assert template.jobs_dir == run_dir / "harbor" / "step-0004"
    assert capture.task_ids == [list(_TASK_IDS)]
    [kwargs] = capture.kwargs
    assert kwargs["agent_import_path"] == WMH_DISTILL_HARBOR_AGENT_IMPORT_PATH
    assert kwargs["extra_agent_kwargs"] == {"token_sink_dir": str(sink_dir)}
    assert kwargs["attempts"] == _GROUP_SIZE
    assert kwargs["reward_key"] == "reward"
    assert kwargs["harness_backend"] == "local"
    assert kwargs["task_environment"] == "docker"
    assert kwargs["agent_concurrency"] == 1  # local pi shares one runner dir
    assert kwargs["provider_config"] == _provider_config()

    # Cancellation flows through to the blocking score call.
    assert stub.score_calls == [(harness.doc_hash, cancel_poll)]

    # Spans correlate by trial name; the span-less trial is kept, not dropped.
    assert [record.trial_name for record in records] == [
        "task-a__s1",
        "task-a__s2",
        "task-b__s1",
        "task-b__s2",
    ]
    assert all(len(record.spans) == 2 for record in records[:3])
    assert records[0].stop_reason == "submitted"
    assert records[3].spans == []
    assert records[3].stop_reason is None
    assert stats.trials == 4
    assert stats.trials_with_spans == 3
    assert stats.empty_span_trials == 1
    assert stats.solve_rate == 0.5


def test_each_step_gets_a_fresh_jobs_dir_and_sink_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Job dirs are keyed by doc hash only; reusing one jobs dir across steps would
    resume step N's trials as step N+1's results with stale-weights tokens."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)
    collect_rollouts(5, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert [template.jobs_dir for template in capture.templates] == [
        run_dir / "harbor" / "step-0004",
        run_dir / "harbor" / "step-0005",
    ]
    assert [kwargs["extra_agent_kwargs"] for kwargs in capture.kwargs] == [
        {"token_sink_dir": str(run_dir / "tokens" / "step-0004")},
        {"token_sink_dir": str(run_dir / "tokens" / "step-0005")},
    ]
    # The sink dirs were created ahead of the trials that write into them.
    assert (run_dir / "tokens" / "step-0004").is_dir()
    assert (run_dir / "tokens" / "step-0005").is_dir()


def _write_recorded_job_config(candidate_dir: Path, provider_config: ProviderConfig) -> None:
    """Persist the slice of a harbor config.json the stale-policy check reads."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    payload = {"agents": [{"kwargs": {"provider_config": provider_config.model_dump(mode="json")}}]}
    (candidate_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_stale_policy_job_dir_is_wiped_before_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job dir left by a previous session under DIFFERENT sampler weights can
    never pass the scorer's strict resume check, and its trials sampled another
    policy: the collector wipes it (and the step's token sinks) so the step
    re-runs whole from the current weights instead of dying on resume."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    old_provider = _provider_config().model_copy(update={"model": "tinker://run/weights/OLD"})
    _write_recorded_job_config(candidate, old_provider)
    (candidate / "task-a__s1").mkdir(parents=True)  # a stale completed trial
    stale_sink = run_dir / "tokens" / "step-0004" / "task-a__s1.jsonl"
    stale_sink.parent.mkdir(parents=True)
    stale_sink.write_text("{}\n", encoding="utf-8")

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert not candidate.exists()  # the stale-policy job dir was wiped whole
    assert not stale_sink.exists()  # and its token sinks with it
    assert (run_dir / "tokens" / "step-0004").is_dir()  # recreated for the re-run
    assert len(stub.score_calls) == 1


def test_matching_policy_job_dir_is_kept_for_harbor_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same provider identity (e.g. the teacher's stable ref, which carries no
    session nonce): the dir is left alone so harbor's native trial-level
    resume re-runs only what is missing."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    _write_recorded_job_config(candidate, _provider_config())
    completed_trial = candidate / "task-a__s1"
    completed_trial.mkdir(parents=True)
    kept_sink = run_dir / "tokens" / "step-0004" / "task-a__s1.jsonl"
    kept_sink.parent.mkdir(parents=True)
    TokenRecorder(jsonl_path=kept_sink).record(_span(0))

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert completed_trial.is_dir()
    assert kept_sink.exists()


def test_unreadable_job_config_is_left_for_the_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable config.json is not evidence of another policy; the dir is
    left untouched so the scorer raises its own actionable error rather than
    the collector destroying evidence silently."""
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path)

    candidate = run_dir / "harbor" / "step-0004" / f"wmh-{harness.doc_hash[:12]}"
    candidate.mkdir(parents=True)
    (candidate / "config.json").write_text("{not json", encoding="utf-8")

    collect_rollouts(4, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    assert (candidate / "config.json").exists()


def test_e2b_backend_routes_environment_and_concurrency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    harness = HarnessDoc.baseline()
    stub = _StubScorer(_report(harness, tmp_path / "trials"))
    capture = _CreateCapture(stub)
    capture.install(monkeypatch)
    cfg = _cfg(tmp_path, backend="e2b", trial_concurrency=6)

    collect_rollouts(0, _TASK_IDS, cfg, harness, _provider_config(), run_dir)

    [kwargs] = capture.kwargs
    assert kwargs["harness_backend"] == "e2b"
    assert kwargs["task_environment"] == "e2b"
    assert kwargs["agent_concurrency"] == 6


def test_negative_step_index_is_rejected(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="step_index must be >= 0"):
        collect_rollouts(
            -1, _TASK_IDS, cfg, HarnessDoc.baseline(), _provider_config(), tmp_path / "run"
        )


def test_template_load_failures_are_actionable(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    harness = HarnessDoc.baseline()

    missing = cfg.model_copy(deep=True)
    missing.harbor.job_template = str(tmp_path / "nowhere.yaml")
    with pytest.raises(ValueError, match="cannot load the harbor job template"):
        collect_rollouts(0, _TASK_IDS, missing, harness, _provider_config(), tmp_path / "run")

    not_mapping = tmp_path / "list.yaml"
    not_mapping.write_text("- 1\n- 2\n", encoding="utf-8")
    listy = cfg.model_copy(deep=True)
    listy.harbor.job_template = str(not_mapping)
    with pytest.raises(ValueError, match="must be a mapping"):
        collect_rollouts(0, _TASK_IDS, listy, harness, _provider_config(), tmp_path / "run")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("agents: nope\n", encoding="utf-8")
    broken = cfg.model_copy(deep=True)
    broken.harbor.job_template = str(invalid)
    with pytest.raises(ValueError, match="invalid harbor job template"):
        collect_rollouts(0, _TASK_IDS, broken, harness, _provider_config(), tmp_path / "run")


def test_missing_harbor_extra_names_the_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "harbor.models.job.config", None)
    cfg = _cfg(tmp_path)
    with pytest.raises(ImportError, match="uv sync --extra harbor"):
        collect_rollouts(
            0, _TASK_IDS, cfg, HarnessDoc.baseline(), _provider_config(), tmp_path / "run"
        )


def test_module_scope_never_imports_the_harbor_extra() -> None:
    """The collector module must stay importable without the harbor extra."""
    assert rollouts_module.__file__ is not None
    tree = ast.parse(Path(rollouts_module.__file__).read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    assert not roots & {"harbor", "yaml"}
    # wmh.distill.agents and wmh.evals.harbor import harbor at module scope; the
    # collector may only pull them inside the guarded lazy block.
    banned_wmh = {"wmh.distill.agents", "wmh.evals"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not any(
                node.module == name or node.module.startswith(name + ".") for name in banned_wmh
            )

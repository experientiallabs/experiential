"""CLI and resource-lifecycle tests for scorer-driven harness optimization."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

import pytest
import typer
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from pydantic import ValidationError
from typer.testing import CliRunner

from wmh.cli import app
from wmh.cli.harness_optimize import (
    HarnessOptimizeOutcome,
    HarnessOptimizeProgress,
    _execute_optimization,
)
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.evals.harbor.canonical import canonical_harbor_job_config
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxUsage, e2b_create_rate_policy_payload
from wmh.harness.live_session import SessionEvent
from wmh.harness.population import (
    HarnessPopulationOptimizer,
    PopulationIteration,
    PopulationOptimizationResult,
)
from wmh.harness.population_checkpoint import (
    PopulationCheckpointError,
    PopulationCheckpointIdentity,
    PopulationCheckpointStore,
)
from wmh.harness.project_proposer import (
    CandidateProposal,
    CandidateProposalError,
    EvaluatedCandidate,
    ProjectCandidateProposer,
)
from wmh.harness.runtime import TokenUsage
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore
from wmh.providers.base import ProviderConfig, ProviderKind

module = importlib.import_module("wmh.cli.harness_optimize")
runner = CliRunner()
_DIGEST = "sha256:" + "a" * 64


class _OptimizeCommonArguments(TypedDict):
    name: str
    root: str
    run_dir: Path
    job_config: JobConfig
    task_ids: tuple[str, ...]
    reward_key: str
    iterations: int
    attempts: int
    max_score_cells: int
    seed_reference: str | None
    meta_config: ProviderConfig
    agent_config: ProviderConfig
    harness_backend: Literal["local", "e2b"]
    e2b_template: str | None
    environment_command_timeout_sec: int
    episode_timeout_sec: float
    project_timeout_sec: float
    max_history_candidates: int
    max_history_bytes: int
    max_new_boundaries: int | None


class _OptimizeArguments(_OptimizeCommonArguments):
    seed: HarnessSourceTree | None
    resume: bool


class _Reader:
    def read_bytes(self, path: str) -> bytes:
        assert path == "raw/result.json"
        return b"{}"


class _ToolProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def complete_chat(self, request: object) -> object:
        raise NotImplementedError


class _Project:
    def __init__(self) -> None:
        self.workspace = "/workspace/project"
        self.closed = False

    def __enter__(self) -> _Project:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.closed = True

    def usage(self) -> SandboxUsage:
        return SandboxUsage(count=1, seconds=12.5)


class _NoPreparationScorer:
    """Test scorer contract for local task environments with no preparation."""

    @property
    def preparation_required(self) -> bool:
        return False

    def preparation_summary(self) -> None:
        return None

    @property
    def preparation_artifact_hash(self) -> None:
        return None

    def preparation_artifact_bytes(self) -> None:
        return None

    async def verify_preparation(self) -> None:
        return None

    def evaluator_policy(self) -> dict[str, object]:
        return {}


def _source(prompt: str) -> HarnessSourceTree:
    return HarnessSourceTree(
        files=(
            HarnessSourceFile(path="SYSTEM.md", content=prompt),
            HarnessSourceFile(
                path="config.toml",
                content=('[harness]\ntools = ["bash", "submit"]\nruntime_kind = "pi-node"\n'),
            ),
        )
    )


def _request() -> ScoreRequest:
    return ScoreRequest(
        context=ScoreContext(
            task_set_digest=_DIGEST,
            evaluator_digest=_DIGEST,
            execution_config_digest=_DIGEST,
        ),
        task_ids=("task-a",),
        attempts=1,
    )


def _result(source: HarnessSourceTree | None = None) -> PopulationOptimizationResult:
    source = source or _source("selected")
    evaluated = _evaluated("candidate-0000", source, value=1.0)
    return PopulationOptimizationResult(population=(evaluated,), iterations=(), best=evaluated)


def _evaluated(
    candidate_id: str,
    source: HarnessSourceTree,
    *,
    value: float,
) -> EvaluatedCandidate:
    artifact = EvaluationArtifact.from_bytes(path="raw/result.json", content=b"{}")
    report = HarnessScoreReport(
        source_run_id=f"run-{candidate_id}",
        candidate_doc_hash=source.to_doc(candidate_id).doc_hash,
        request=_request(),
        cells=(
            ScoreCell(
                task_id="task-a",
                attempt=1,
                score=value,
                passed=value >= 0.5,
                summary="official result",
                artifact_paths=("raw/result.json",),
            ),
        ),
        artifacts=(artifact,),
    )
    return EvaluatedCandidate(
        candidate_id=candidate_id,
        source=source,
        score=HarnessScore(report=report, artifacts=_Reader()),
    )


def _valid_proposal(candidate_id: str, source: HarnessSourceTree) -> CandidateProposal:
    candidate = source.to_doc(candidate_id)
    return CandidateProposal(
        candidate_id=candidate_id,
        source=source,
        candidate=candidate,
        events=(SessionEvent(kind="submit", payload={"answer": "done"}),),
        worker_usage=TokenUsage(input_tokens=10, output_tokens=5, calls=1),
        request=f"produce {candidate_id}",
        status_json=json.dumps(
            {
                "agent_error": None,
                "candidate_doc_hash": candidate.doc_hash,
                "candidate_id": candidate_id,
                "source_tree_hash": source.tree_hash,
                "valid": True,
                "validation_error": None,
            },
            indent=2,
            sort_keys=True,
        ),
    )


def _result_with_invalid_iterations(
    count: int,
    *,
    seed_source: HarnessSourceTree | None = None,
) -> PopulationOptimizationResult:
    seed = _result(seed_source).population[0]
    iterations = tuple(
        PopulationIteration(
            index=index,
            error=CandidateProposalError(
                f"candidate-{index:04d}",
                "invalid candidate",
                request=f"produce candidate-{index:04d}",
                status_json=json.dumps(
                    {
                        "agent_error": None,
                        "candidate_doc_hash": None,
                        "candidate_id": f"candidate-{index:04d}",
                        "source_tree_hash": None,
                        "valid": False,
                        "validation_error": "invalid candidate",
                    },
                    indent=2,
                    sort_keys=True,
                ),
            ),
        )
        for index in range(1, count + 1)
    )
    return PopulationOptimizationResult(population=(seed,), iterations=iterations, best=seed)


def _job_config(tmp_path: Path) -> JobConfig:
    return JobConfig(
        job_name="template",
        jobs_dir=tmp_path / "unused",
        n_attempts=1,
        n_concurrent_trials=2,
        retry=RetryConfig(max_retries=0),
        environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
        agents=[AgentConfig(n_concurrent=1)],
        datasets=[DatasetConfig(path=tmp_path / "tasks")],
    )


def _raw_job_config_for_hash_seed(hash_seed: int) -> dict[str, object]:
    """Harbor's JSON-mode set ordering from a genuinely distinct interpreter seed."""
    script = """
from harbor.models.job.config import JobConfig, RetryConfig

config = JobConfig(
    job_name="template",
    jobs_dir="run/harbor",
    retry=RetryConfig(
        max_retries=0,
        include_exceptions={"ZetaError", "AlphaError"},
    ),
    artifacts=["second", "first"],
)
print(config.model_dump_json())
"""
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(hash_seed)
    output = subprocess.check_output(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
    )
    value = json.loads(output)
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _configs() -> tuple[ProviderConfig, ProviderConfig]:
    return (
        ProviderConfig(
            kind=ProviderKind.OPENAI_RESPONSES,
            model="meta-model",
            reasoning_effort="high",
        ),
        ProviderConfig(kind=ProviderKind.ANTHROPIC, model="agent-model"),
    )


def _checkpoint_identity(
    *,
    run_dir: Path,
    root: Path,
    job_config: JobConfig,
    seed: HarnessSourceTree,
    request: ScoreRequest,
    iterations: int,
    max_score_cells: int,
    meta_config: ProviderConfig,
    agent_config: ProviderConfig,
) -> PopulationCheckpointIdentity:
    effective_job_config = JobConfig.model_validate(
        job_config.model_copy(
            update={"jobs_dir": run_dir / "harbor"},
            deep=True,
        ).model_dump(mode="python")
    )
    return PopulationCheckpointIdentity(
        output_name="selected",
        artifact_root=str(root.resolve()),
        seed_reference=None,
        seed_source_tree_hash=seed.tree_hash,
        score_request=request,
        iterations=iterations,
        planned_score_cells=(iterations + 1) * len(request.task_ids) * request.attempts,
        max_score_cells=max_score_cells,
        harbor_job_template=effective_job_config.model_dump(mode="json"),
        meta_provider=meta_config,
        agent_provider=agent_config,
        optimizer_document_hash=module.optimizer_agent().doc_hash,
        harness_backend="local",
        e2b_template=None,
        project_e2b_create_rate_policy=e2b_create_rate_policy_payload(),
        environment_command_timeout_sec=300,
        episode_timeout_sec=300,
        project_timeout_sec=900,
        max_source_files=module.DEFAULT_SOURCE_TREE_MAX_FILES,
        max_source_bytes=module.DEFAULT_SOURCE_TREE_MAX_BYTES,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
    )


def _commit_ready_boundary(
    checkpoint: PopulationCheckpointStore,
    result: PopulationOptimizationResult,
    *,
    usage: SandboxUsage,
) -> None:
    checkpoint.begin_setup()
    checkpoint.before_step(len(result.iterations))
    checkpoint.commit_boundary(result)
    checkpoint.finish_project_segment(usage)


def test_harbor_job_identity_and_resume_are_stable_across_process_hash_seeds(
    tmp_path: Path,
) -> None:
    """Harbor set order cannot fork an otherwise identical durable optimization."""
    raw_templates = [_raw_job_config_for_hash_seed(seed) for seed in (0, 1, 42)]
    raw_retry_orders = {
        json.dumps(cast("dict[str, object]", template["retry"]), separators=(",", ":"))
        for template in raw_templates
    }
    assert len(raw_retry_orders) > 1  # prove the subprocesses exercised the original defect

    canonical_templates = [canonical_harbor_job_config(template) for template in raw_templates]
    assert canonical_templates[0] == canonical_templates[1] == canonical_templates[2]
    retry = cast("dict[str, object]", canonical_templates[0]["retry"])
    assert retry["include_exceptions"] == ["AlphaError", "ZetaError"]
    assert retry["exclude_exceptions"] == sorted(cast("list[str]", retry["exclude_exceptions"]))
    assert canonical_templates[0]["artifacts"] == ["second", "first"]
    ordered = JobConfig(
        job_name="ordered",
        agents=[
            AgentConfig(import_path="agents.First"),
            AgentConfig(import_path="agents.Second"),
        ],
        datasets=[DatasetConfig(path=Path("second")), DatasetConfig(path=Path("first"))],
        artifacts=["second", "first"],
    )
    ordered_payload = canonical_harbor_job_config(ordered)
    ordered_agents = cast("list[dict[str, object]]", ordered_payload["agents"])
    ordered_datasets = cast("list[dict[str, object]]", ordered_payload["datasets"])
    assert [agent["import_path"] for agent in ordered_agents] == [
        "agents.First",
        "agents.Second",
    ]
    assert [dataset["path"] for dataset in ordered_datasets] == ["second", "first"]
    assert ordered_payload["artifacts"] == ["second", "first"]
    reordered = ordered.model_copy(
        update={
            "agents": list(reversed(ordered.agents)),
            "datasets": list(reversed(ordered.datasets)),
            "artifacts": list(reversed(ordered.artifacts)),
        },
        deep=True,
    )
    assert canonical_harbor_job_config(reordered) != ordered_payload

    unknown_top_level = {**raw_templates[0], "unknown_identity_input": True}
    unknown_nested = json.loads(json.dumps(raw_templates[0]))
    cast("dict[str, object]", unknown_nested["retry"])["unknown_retry_input"] = True
    with pytest.raises(ValidationError):
        canonical_harbor_job_config(unknown_top_level)
    with pytest.raises(ValidationError):
        canonical_harbor_job_config(unknown_nested)

    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    meta_config, agent_config = _configs()
    base_identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=_job_config(tmp_path),
        seed=seed,
        request=_request(),
        iterations=1,
        max_score_cells=2,
        meta_config=meta_config,
        agent_config=agent_config,
    )
    # model_copy intentionally bypasses validation to reproduce a checkpoint written by the
    # pre-fix process, whose Harbor JSON list retained that process's arbitrary set order.
    legacy_identity = base_identity.model_copy(update={"harbor_job_template": raw_templates[0]})
    expected_identity = PopulationCheckpointIdentity.model_validate(
        {
            **base_identity.model_dump(mode="json"),
            "harbor_job_template": raw_templates[1],
        }
    )
    serialized_identities = [
        json.dumps(
            PopulationCheckpointIdentity.model_validate(
                {
                    **base_identity.model_dump(mode="json"),
                    "harbor_job_template": template,
                }
            ).model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        for template in raw_templates
    ]
    assert len(set(serialized_identities)) == 1
    assert len({hashlib.sha256(value).hexdigest() for value in serialized_identities}) == 1

    with PopulationCheckpointStore.create(
        run_dir,
        identity=legacy_identity,
        seed=seed,
    ) as checkpoint:
        _commit_ready_boundary(
            checkpoint,
            _result(seed),
            usage=SandboxUsage(count=1, seconds=1.0),
        )
    identity_path = run_dir / "checkpoint/identity.json"
    raw_identity_before_resume = identity_path.read_bytes()
    with PopulationCheckpointStore.open(run_dir) as resumed:
        resumed.assert_identity(expected_identity)
        assert resumed.control.committed_step == 0
    assert identity_path.read_bytes() == raw_identity_before_resume


def test_execute_composes_exact_scorer_project_optimizer_archive_and_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    scorer_calls: list[dict[str, object]] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **kwargs: object) -> FakeScorer:
            scorer_calls.append(kwargs)
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

    projects: list[_Project] = []
    project_calls: list[dict[str, object]] = []

    class FakeProjectFactory:
        @classmethod
        def create(cls, **kwargs: object) -> _Project:
            project_calls.append(kwargs)
            control = json.loads((tmp_path / "local-run/checkpoint/control.json").read_text())
            assert control["state"] == "in_progress"
            assert control["active_kind"] == "setup"
            assert control["project_sandbox_usage"] is None
            project = _Project()
            projects.append(project)
            return project

    optimizer_calls: list[dict[str, object]] = []
    result = _result_with_invalid_iterations(3, seed_source=_source("seed"))

    class FakeOptimizer:
        def __init__(self, proposer: object, scorer: object) -> None:
            optimizer_calls.append({"proposer": proposer, "scorer": scorer})

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            optimizer_calls[-1].update(kwargs)
            before_step = cast(Callable[[int], None], kwargs["before_step"])
            on_boundary = cast(
                Callable[[PopulationOptimizationResult], None], kwargs["on_boundary"]
            )
            resumed = cast(PopulationOptimizationResult | None, kwargs["resume"])
            if resumed is None:
                boundary = _result(_source("seed"))
                before_step(0)
                on_boundary(boundary)
                return boundary
            index = len(resumed.iterations) + 1
            boundary = PopulationOptimizationResult(
                population=result.population,
                iterations=result.iterations[:index],
                best=result.best,
            )
            before_step(index)
            on_boundary(boundary)
            return boundary

    archived: list[tuple[Path, PopulationOptimizationResult]] = []

    def fake_archive(path: Path, value: PopulationOptimizationResult) -> Path:
        archived.append((path, value))
        path.mkdir(parents=True)
        manifest = path / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    monkeypatch.setattr(module, "write_population_archive", fake_archive)
    run_dir = tmp_path / "local-run"
    root = tmp_path / ".wmh"
    seed = _source("seed")

    outcome = _execute_optimization(
        name="selected",
        root=str(root),
        run_dir=run_dir,
        job_config=_job_config(tmp_path),
        task_ids=("task-a",),
        reward_key="reward",
        iterations=3,
        attempts=1,
        max_score_cells=4,
        seed=seed,
        seed_reference=None,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="e2b",
        e2b_template="template-x",
        environment_command_timeout_sec=300,
        episode_timeout_sec=12_000,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
        resume=False,
    )

    [scorer_call] = scorer_calls
    assert scorer_call["task_ids"] == ("task-a",)
    assert scorer_call["provider_config"] == agent_config
    assert scorer_call["harness_backend"] == "e2b"
    assert scorer_call["e2b_template"] == "template-x"
    assert scorer_call["episode_timeout_sec"] == 12_000
    assert scorer_call["reward_mode"] == "raw"
    assert cast(JobConfig, scorer_call["job_config"]).jobs_dir == run_dir / "harbor"
    assert len(project_calls) == 3
    assert len({id(project) for project in projects}) == 3
    assert all(call["template"] == "template-x" for call in project_calls)
    assert all(call["timeout"] == 900 for call in project_calls)
    assert all(project.closed for project in projects)
    assert len(optimizer_calls) == 4
    assert all(
        cast(FakeScorer, call["scorer"]).request(attempts=1) == request for call in optimizer_calls
    )
    assert all(call["seed"] == seed for call in optimizer_calls)
    assert all(call["request"] == request for call in optimizer_calls)
    assert all(call["iterations"] == 3 for call in optimizer_calls)
    assert all(call["max_new_boundaries"] == 1 for call in optimizer_calls)
    assert optimizer_calls[0]["resume"] is None
    proposer = cast(ProjectCandidateProposer, optimizer_calls[-1]["proposer"])
    assert proposer._max_history_candidates == 20
    assert proposer._max_history_bytes == 1_000_000
    assert len(archived) == 1
    assert archived[0][0].parent == run_dir
    assert archived[0][0].name.startswith(".population.tmp-")
    assert archived[0][1] == result
    assert isinstance(outcome, HarnessOptimizeOutcome)
    assert outcome.saved.name == "selected"
    assert outcome.saved.version == 1
    assert outcome.saved.doc_hash == result.best.candidate.doc_hash
    assert (root / "harnesses/selected/aliases.toml").exists()
    inputs = json.loads((run_dir / "inputs.json").read_text())
    assert inputs["iterations"] == 3
    assert inputs["episode_timeout_sec"] == 12_000
    assert inputs["evaluator_policy"] == {}
    retry = inputs["harbor_job_template"]["retry"]
    assert retry["exclude_exceptions"] == sorted(retry["exclude_exceptions"])
    written = json.loads((run_dir / "outcome.json").read_text())
    assert written["best_candidate_id"] == "candidate-0000"
    assert written["project_sandbox_usage"] == {"count": 3, "seconds": 37.5}
    assert written["known_score_cells"] == 1
    assert written["max_score_cells"] == 4


def test_execute_checkpoints_preparation_then_resumes_inspect_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()
    seed = _source("seed")
    receipt = b'{"complete":true}\n'
    receipt_hash = "sha256:" + hashlib.sha256(receipt).hexdigest()
    summary = {"schema_version": 1, "unique_template_count": 1}
    run_dir = tmp_path / "run"
    events: list[str] = []

    class PreparedScorer:
        mismatch_hash = False

        def __init__(self) -> None:
            self.artifact_path: Path | None = None

        @classmethod
        async def create(cls, **_kwargs: object) -> PreparedScorer:
            events.append("create")
            return cls()

        @property
        def preparation_required(self) -> bool:
            return True

        def preparation_summary(self) -> dict[str, int]:
            events.append("summary")
            return summary

        async def prepare(self, artifact_path: Path) -> None:
            events.append("prepare")
            assert artifact_path == tmp_path / ".run.scorer-preparation.bin"
            assert not run_dir.exists()
            artifact_path.write_bytes(receipt)
            self.artifact_path = artifact_path

        def load_preparation(self, artifact_path: Path) -> None:
            events.append("load")
            assert artifact_path.read_bytes() == receipt
            self.artifact_path = artifact_path

        def request(self, *, attempts: int) -> ScoreRequest:
            events.append("request")
            assert attempts == 1
            assert self.artifact_path is not None
            return request

        @property
        def preparation_artifact_hash(self) -> str:
            events.append("hash")
            if self.mismatch_hash:
                return "sha256:" + "b" * 64
            return receipt_hash

        def preparation_artifact_bytes(self) -> bytes:
            events.append("bytes")
            assert self.artifact_path is not None
            return self.artifact_path.read_bytes()

        def bind_preparation(self, artifact_path: Path) -> None:
            events.append("bind")
            assert artifact_path.read_bytes() == receipt
            self.artifact_path = artifact_path

        async def verify_preparation(self) -> None:
            events.append("verify")
            assert self.artifact_path == run_dir / "checkpoint/scorer-preparation.bin"

        def evaluator_policy(self) -> dict[str, object]:
            return {
                "reward_key": "reward",
                "reward_interpretation": {
                    "mode": "positive_binary",
                    "threshold": 0.0,
                    "comparator": "gt",
                },
            }

    class FakeOptimizer:
        def __init__(self, *_args: object) -> None:
            pass

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            resumed = cast(PopulationOptimizationResult | None, kwargs["resume"])
            before_step = cast(Callable[[int], None], kwargs["before_step"])
            on_boundary = cast(
                Callable[[PopulationOptimizationResult], None],
                kwargs["on_boundary"],
            )
            if resumed is None:
                step = 0
                boundary = _result(seed)
            else:
                step = 1
                boundary = _result_with_invalid_iterations(1, seed_source=seed)
            events.append("boundary")
            before_step(step)
            on_boundary(boundary)
            return boundary

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            events.append("project")
            return _Project()

    def confirm(observed: object) -> None:
        events.append("confirm")
        assert observed == summary

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", PreparedScorer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    arguments: _OptimizeCommonArguments = {
        "name": "selected",
        "root": str(tmp_path / ".wmh"),
        "run_dir": run_dir,
        "job_config": _job_config(tmp_path),
        "task_ids": ("task-a",),
        "reward_key": "reward",
        "iterations": 1,
        "attempts": 1,
        "max_score_cells": 2,
        "seed_reference": None,
        "meta_config": meta_config,
        "agent_config": agent_config,
        "harness_backend": "e2b",
        "e2b_template": "template-x",
        "environment_command_timeout_sec": 300,
        "episode_timeout_sec": 12_000,
        "project_timeout_sec": 900,
        "max_history_candidates": 20,
        "max_history_bytes": 1_000_000,
        "max_new_boundaries": 1,
    }

    progress = _execute_optimization(
        **arguments,
        seed=seed,
        resume=False,
        before_preparation=confirm,
    )

    assert isinstance(progress, HarnessOptimizeProgress)
    assert events == [
        "create",
        "summary",
        "confirm",
        "prepare",
        "request",
        "hash",
        "bytes",
        "bind",
        "verify",
        "boundary",
    ]
    artifact_path = run_dir / "checkpoint/scorer-preparation.bin"
    assert artifact_path.read_bytes() == receipt
    assert not (tmp_path / ".run.scorer-preparation.bin").exists()
    identity = json.loads((run_dir / "checkpoint/identity.json").read_text())
    assert identity["scorer_preparation_artifact_hash"] == receipt_hash

    events.clear()
    PreparedScorer.mismatch_hash = True
    with pytest.raises(
        PopulationCheckpointError,
        match="scorer_preparation_artifact_hash",
    ):
        _execute_optimization(
            **arguments,
            seed=None,
            resume=True,
            before_preparation=confirm,
        )
    assert events == ["create", "summary", "confirm", "load", "request", "hash"]

    events.clear()
    PreparedScorer.mismatch_hash = False
    outcome = _execute_optimization(
        **arguments,
        seed=None,
        resume=True,
        before_preparation=confirm,
    )

    assert isinstance(outcome, HarnessOptimizeOutcome)
    assert events == [
        "create",
        "summary",
        "confirm",
        "load",
        "request",
        "hash",
        "verify",
        "project",
        "boundary",
    ]


def test_execute_preparation_failure_leaves_only_audit_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    staging_path = tmp_path / ".run.scorer-preparation.bin"
    events: list[str] = []

    class BrokenPreparationScorer:
        @classmethod
        async def create(cls, **_kwargs: object) -> BrokenPreparationScorer:
            events.append("create")
            return cls()

        @property
        def preparation_required(self) -> bool:
            return True

        def preparation_summary(self) -> dict[str, int]:
            events.append("summary")
            return {"schema_version": 1}

        async def prepare(self, artifact_path: Path) -> None:
            events.append("prepare")
            assert artifact_path == staging_path
            artifact_path.write_bytes(b'{"complete":false}\n')
            raise RuntimeError("provider preparation failed")

    class UnreachedOptimizer:
        def __init__(self, *_args: object) -> None:
            raise AssertionError("preparation failure must precede the seed boundary")

    class UnreachedProject:
        @classmethod
        def create(cls, **_kwargs: object) -> None:
            raise AssertionError("preparation failure must precede project creation")

    def confirm(_summary: object) -> None:
        events.append("confirm")

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", BrokenPreparationScorer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", UnreachedOptimizer)
    monkeypatch.setattr(module, "AgentProject", UnreachedProject)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    with pytest.raises(RuntimeError, match="provider preparation failed"):
        _execute_optimization(
            name="selected",
            root=str(tmp_path / ".wmh"),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="e2b",
            e2b_template="template-x",
            environment_command_timeout_sec=300,
            episode_timeout_sec=12_000,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
            before_preparation=confirm,
        )

    assert events == ["create", "summary", "confirm", "prepare"]
    assert not run_dir.exists()
    assert staging_path.read_bytes() == b'{"complete":false}\n'


def test_execute_verification_failure_stops_before_seed_or_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    receipt = b'{"complete":true}\n'
    receipt_hash = "sha256:" + hashlib.sha256(receipt).hexdigest()
    events: list[str] = []

    class UnverifiedScorer:
        def __init__(self) -> None:
            self.artifact_path: Path | None = None

        @classmethod
        async def create(cls, **_kwargs: object) -> UnverifiedScorer:
            return cls()

        @property
        def preparation_required(self) -> bool:
            return True

        def preparation_summary(self) -> dict[str, int]:
            return {"schema_version": 1}

        async def prepare(self, artifact_path: Path) -> None:
            artifact_path.write_bytes(receipt)
            self.artifact_path = artifact_path

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return _request()

        @property
        def preparation_artifact_hash(self) -> str:
            return receipt_hash

        def preparation_artifact_bytes(self) -> bytes:
            assert self.artifact_path is not None
            return self.artifact_path.read_bytes()

        def bind_preparation(self, artifact_path: Path) -> None:
            self.artifact_path = artifact_path

        async def verify_preparation(self) -> None:
            events.append("verify")
            raise RuntimeError("provider mapping changed")

        def evaluator_policy(self) -> dict[str, object]:
            return {}

    class UnreachedOptimizer:
        def __init__(self, *_args: object) -> None:
            events.append("seed")
            raise AssertionError("verification failure must precede the seed boundary")

    class UnreachedProject:
        @classmethod
        def create(cls, **_kwargs: object) -> None:
            events.append("project")
            raise AssertionError("verification failure must precede project creation")

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", UnverifiedScorer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", UnreachedOptimizer)
    monkeypatch.setattr(module, "AgentProject", UnreachedProject)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    with pytest.raises(RuntimeError, match="provider mapping changed"):
        _execute_optimization(
            name="selected",
            root=str(tmp_path / ".wmh"),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="e2b",
            e2b_template="template-x",
            environment_command_timeout_sec=300,
            episode_timeout_sec=12_000,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    assert events == ["verify"]
    assert (run_dir / "checkpoint/scorer-preparation.bin").read_bytes() == receipt
    assert not (tmp_path / ".run.scorer-preparation.bin").exists()
    control = json.loads((run_dir / "checkpoint/control.json").read_text())
    assert control["committed_step"] == -1
    assert control["state"] == "ready"


def test_execute_stops_ready_then_resumes_without_publishing_partial_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    seed = _source("seed")
    first_proposal = _result_with_invalid_iterations(1, seed_source=seed)
    completed = _result_with_invalid_iterations(2, seed_source=seed)
    optimizer_resumes: list[PopulationOptimizationResult | None] = []
    projects: list[_Project] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            project = _Project()
            projects.append(project)
            return project

    class FakeOptimizer:
        def __init__(self, *_args: object) -> None:
            pass

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            before_step = cast(Callable[[int], None], kwargs["before_step"])
            on_boundary = cast(
                Callable[[PopulationOptimizationResult], None], kwargs["on_boundary"]
            )
            resumed = cast(PopulationOptimizationResult | None, kwargs["resume"])
            optimizer_resumes.append(resumed)
            if resumed is None:
                boundary = _result(seed)
                before_step(0)
                on_boundary(boundary)
                return boundary
            index = len(resumed.iterations) + 1
            boundary = first_proposal if index == 1 else completed
            before_step(index)
            on_boundary(boundary)
            return boundary

    def fake_archive(path: Path, _result: PopulationOptimizationResult) -> Path:
        path.mkdir(parents=True)
        manifest = path / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    monkeypatch.setattr(module, "write_population_archive", fake_archive)
    root = tmp_path / ".wmh"
    run_dir = tmp_path / "run"
    arguments: _OptimizeCommonArguments = {
        "name": "selected",
        "root": str(root),
        "run_dir": run_dir,
        "job_config": _job_config(tmp_path),
        "task_ids": ("task-a",),
        "reward_key": "reward",
        "iterations": 2,
        "attempts": 1,
        "max_score_cells": 3,
        "seed_reference": None,
        "meta_config": meta_config,
        "agent_config": agent_config,
        "harness_backend": "local",
        "e2b_template": None,
        "environment_command_timeout_sec": 300,
        "episode_timeout_sec": 300.0,
        "project_timeout_sec": 900,
        "max_history_candidates": 20,
        "max_history_bytes": 1_000_000,
        "max_new_boundaries": 2,
    }

    progress = _execute_optimization(**arguments, seed=seed, resume=False)

    assert isinstance(progress, HarnessOptimizeProgress)
    assert len(progress.result.iterations) == 1
    assert len(projects) == 1
    assert projects[0].closed
    assert json.loads((run_dir / "checkpoint/control.json").read_text())["state"] == "ready"
    assert "max_new_boundaries" not in json.loads((run_dir / "inputs.json").read_text())
    assert not (run_dir / "population").exists()
    assert not (run_dir / "outcome.json").exists()
    assert not (root / "harnesses/selected").exists()

    arguments["max_new_boundaries"] = 1
    outcome = _execute_optimization(**arguments, seed=None, resume=True)

    assert isinstance(outcome, HarnessOptimizeOutcome)
    assert len(projects) == 2
    assert all(project.closed for project in projects)
    assert optimizer_resumes[0] is None
    assert optimizer_resumes[1] is not None
    assert optimizer_resumes[1].iterations == ()
    assert optimizer_resumes[2] is not None
    assert len(optimizer_resumes[2].iterations) == 1
    assert len(outcome.result.iterations) == 2
    assert json.loads((run_dir / "checkpoint/control.json").read_text())["state"] == "complete"
    assert (run_dir / "population/manifest.json").is_file()
    assert (run_dir / "outcome.json").is_file()
    assert (root / "harnesses/selected/aliases.toml").is_file()


def test_execute_rejects_score_cell_plan_before_scorer_or_run_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scorer_calls: list[dict[str, object]] = []

    class UnreachedScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **kwargs: object) -> UnreachedScorer:
            scorer_calls.append(kwargs)
            return cls()

    monkeypatch.setattr(module, "HarborScorer", UnreachedScorer)
    meta_config, agent_config = _configs()
    run_dir = tmp_path / "run"

    with pytest.raises(typer.BadParameter, match="exceed"):
        _execute_optimization(
            name="selected",
            root=str(tmp_path / ".wmh"),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=1,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=300,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    assert scorer_calls == []
    assert not run_dir.exists()


def test_execute_resumes_ready_seed_at_next_slot_without_rescoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    meta_config, agent_config = _configs()
    job_config = _job_config(tmp_path)
    effective_job_config = JobConfig.model_validate(
        job_config.model_copy(
            update={"jobs_dir": run_dir / "harbor"},
            deep=True,
        ).model_dump(mode="python")
    )
    identity = PopulationCheckpointIdentity(
        output_name="selected",
        artifact_root=str(root.resolve()),
        seed_reference=None,
        seed_source_tree_hash=seed.tree_hash,
        score_request=request,
        iterations=1,
        planned_score_cells=2,
        max_score_cells=2,
        harbor_job_template=effective_job_config.model_dump(mode="json"),
        meta_provider=meta_config,
        agent_provider=agent_config,
        optimizer_document_hash=module.optimizer_agent().doc_hash,
        harness_backend="local",
        e2b_template=None,
        project_e2b_create_rate_policy=e2b_create_rate_policy_payload(),
        environment_command_timeout_sec=300,
        episode_timeout_sec=300,
        project_timeout_sec=900,
        max_source_files=module.DEFAULT_SOURCE_TREE_MAX_FILES,
        max_source_bytes=module.DEFAULT_SOURCE_TREE_MAX_BYTES,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
    )
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as checkpoint:
        checkpoint.begin_setup()
        checkpoint.before_step(0)
        checkpoint.commit_boundary(_result(seed))
        checkpoint.finish_project_segment(SandboxUsage(count=1, seconds=1.5))

    score_calls: list[str] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def score(self, candidate: object, *, request: ScoreRequest) -> object:
            del request
            score_calls.append(str(candidate))
            raise AssertionError("committed seed must not be rescored")

    restore_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    propose_calls: list[tuple[str, ...]] = []

    class FakeResumableProposer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def restore(
            self,
            history: Sequence[EvaluatedCandidate],
            turns: Sequence[CandidateProposal | CandidateProposalError],
        ) -> None:
            history_items = tuple(history)
            turn_items = tuple(turns)
            restore_calls.append(
                (
                    tuple(item.candidate_id for item in history_items),
                    tuple(item.candidate_id for item in turn_items),
                )
            )

        def propose(
            self,
            history: Sequence[EvaluatedCandidate],
            **_kwargs: object,
        ) -> CandidateProposal:
            history_items = tuple(history)
            propose_calls.append(tuple(item.candidate_id for item in history_items))
            raise CandidateProposalError(
                "candidate-0001",
                "invalid candidate",
                request="produce candidate-0001",
                status_json=json.dumps(
                    {
                        "agent_error": None,
                        "candidate_doc_hash": None,
                        "candidate_id": "candidate-0001",
                        "source_tree_hash": None,
                        "valid": False,
                        "validation_error": "invalid candidate",
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )

    project = _Project()

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            return project

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "ProjectCandidateProposer", FakeResumableProposer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", HarnessPopulationOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    outcome = _execute_optimization(
        name="selected",
        root=str(root),
        run_dir=run_dir,
        job_config=job_config,
        task_ids=("task-a",),
        reward_key="reward",
        iterations=1,
        attempts=1,
        max_score_cells=2,
        seed=None,
        seed_reference=None,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="local",
        e2b_template=None,
        environment_command_timeout_sec=300,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
        resume=True,
    )

    assert score_calls == []
    assert restore_calls == [(("candidate-0000",), ())]
    assert propose_calls == [("candidate-0000",)]
    assert len(outcome.result.iterations) == 1
    assert outcome.result.iterations[0].error is not None


def test_resume_rejects_episode_timeout_drift_before_any_score_or_project_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    job_config = _job_config(tmp_path)
    meta_config, agent_config = _configs()
    identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=job_config,
        seed=seed,
        request=request,
        iterations=1,
        max_score_cells=2,
        meta_config=meta_config,
        agent_config=agent_config,
    ).model_copy(
        update={
            "harness_backend": "e2b",
            "e2b_template": "template",
            "episode_timeout_sec": 300,
        }
    )
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed):
        pass

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def score(self, *_args: object, **_kwargs: object) -> HarnessScore:
            raise AssertionError("identity drift must fail before scoring")

    class NoProject:
        @classmethod
        def create(cls, **_kwargs: object) -> None:
            raise AssertionError("identity drift must fail before project creation")

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", NoProject)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    with pytest.raises(PopulationCheckpointError, match="episode_timeout_sec"):
        _execute_optimization(
            name="selected",
            root=str(root),
            run_dir=run_dir,
            job_config=job_config,
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=None,
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="e2b",
            e2b_template="template",
            environment_command_timeout_sec=300,
            episode_timeout_sec=12_000,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=True,
        )


def test_resume_rejects_evaluator_policy_drift_before_verification_or_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    job_config = _job_config(tmp_path)
    meta_config, agent_config = _configs()
    identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=job_config,
        seed=seed,
        request=request,
        iterations=1,
        max_score_cells=2,
        meta_config=meta_config,
        agent_config=agent_config,
    )
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed):
        pass
    events: list[str] = []

    class PositiveModeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **kwargs: object) -> PositiveModeScorer:
            assert kwargs["reward_mode"] == "positive_binary"
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def evaluator_policy(self) -> dict[str, object]:
            return {
                "reward_interpretation": {
                    "mode": "positive_binary",
                    "threshold": 0.0,
                    "comparator": "gt",
                }
            }

        async def verify_preparation(self) -> None:
            events.append("verify")

    class NoProject:
        @classmethod
        def create(cls, **_kwargs: object) -> None:
            events.append("project")
            raise AssertionError("evaluator drift must fail before project creation")

    monkeypatch.setattr(module, "HarborScorer", PositiveModeScorer)
    monkeypatch.setattr(module, "AgentProject", NoProject)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    with pytest.raises(PopulationCheckpointError, match="evaluator_policy"):
        _execute_optimization(
            name="selected",
            root=str(root),
            run_dir=run_dir,
            job_config=job_config,
            task_ids=("task-a",),
            reward_key="reward",
            reward_mode="positive_binary",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=None,
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=300,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=True,
        )

    assert events == []


def test_execute_resumes_mixed_prefix_at_next_id_without_replaying_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    job_config = _job_config(tmp_path)
    meta_config, agent_config = _configs()
    identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=job_config,
        seed=seed,
        request=request,
        iterations=3,
        max_score_cells=4,
        meta_config=meta_config,
        agent_config=agent_config,
    )

    seed_result = _result(seed)
    first_source = _source("first")
    first = _evaluated("candidate-0001", first_source, value=0.8)
    first_iteration = PopulationIteration(
        index=1,
        proposal=_valid_proposal("candidate-0001", first_source),
        evaluation=first,
    )
    first_result = PopulationOptimizationResult(
        population=(seed_result.population[0], first),
        iterations=(first_iteration,),
        best=seed_result.best,
    )
    invalid_source = HarnessSourceTree(
        files=(HarnessSourceFile(path="notes.txt", content="unfinished"),)
    )
    invalid = CandidateProposalError(
        "candidate-0002",
        "invalid candidate",
        source=invalid_source,
        worker_usage=TokenUsage(input_tokens=4, output_tokens=2, calls=1),
        request="produce candidate-0002",
        status_json=json.dumps(
            {
                "agent_error": None,
                "candidate_doc_hash": None,
                "candidate_id": "candidate-0002",
                "source_tree_hash": invalid_source.tree_hash,
                "valid": False,
                "validation_error": "invalid candidate",
            },
            indent=2,
            sort_keys=True,
        ),
    )
    mixed_result = PopulationOptimizationResult(
        population=first_result.population,
        iterations=(first_iteration, PopulationIteration(index=2, error=invalid)),
        best=seed_result.best,
    )
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as checkpoint:
        _commit_ready_boundary(checkpoint, seed_result, usage=SandboxUsage())
        _commit_ready_boundary(
            checkpoint,
            first_result,
            usage=SandboxUsage(count=1, seconds=1.0),
        )
        _commit_ready_boundary(
            checkpoint,
            mixed_result,
            usage=SandboxUsage(count=1, seconds=2.0),
        )

    proposed_source = _source("third")
    proposed = _valid_proposal("candidate-0003", proposed_source)
    score_calls: list[str] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            assert request == _request()
            assert candidate.doc_hash == proposed.candidate.doc_hash
            score_calls.append(candidate.name)
            return _evaluated(candidate.name, proposed_source, value=0.9).score

    restore_calls: list[tuple[tuple[str, ...], tuple[str, ...]]]
    restore_calls = []

    class FakeResumableProposer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def restore(
            self,
            history: Sequence[EvaluatedCandidate],
            turns: Sequence[CandidateProposal | CandidateProposalError],
        ) -> None:
            history_items = tuple(history)
            turn_items = tuple(turns)
            restore_calls.append(
                (
                    tuple(item.candidate_id for item in history_items),
                    tuple(item.candidate_id for item in turn_items),
                )
            )
            assert history_items[1].source == first_source
            assert turn_items[0].source == first_source
            assert turn_items[1].source == invalid_source

        def propose(
            self,
            history: Sequence[EvaluatedCandidate],
            **_kwargs: object,
        ) -> CandidateProposal:
            assert tuple(item.candidate_id for item in history) == (
                "candidate-0000",
                "candidate-0001",
            )
            return proposed

    projects: list[_Project] = []

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            project = _Project()
            projects.append(project)
            return project

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "ProjectCandidateProposer", FakeResumableProposer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", HarnessPopulationOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    outcome = _execute_optimization(
        name="selected",
        root=str(root),
        run_dir=run_dir,
        job_config=job_config,
        task_ids=("task-a",),
        reward_key="reward",
        iterations=3,
        attempts=1,
        max_score_cells=4,
        seed=None,
        seed_reference=None,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="local",
        e2b_template=None,
        environment_command_timeout_sec=300,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
        resume=True,
    )

    assert restore_calls == [
        (
            ("candidate-0000", "candidate-0001"),
            ("candidate-0001", "candidate-0002"),
        )
    ]
    assert score_calls == ["candidate-0003"]
    assert [item.candidate_id for item in outcome.result.population] == [
        "candidate-0000",
        "candidate-0001",
        "candidate-0003",
    ]
    assert [item.candidate_id for item in outcome.result.iterations] == [
        "candidate-0001",
        "candidate-0002",
        "candidate-0003",
    ]
    assert len(projects) == 1
    assert projects[0].closed


def test_execute_uses_fresh_project_and_restores_committed_prefix_for_each_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    seed = _source("seed")
    score_calls: list[str] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            assert request == _request()
            score_calls.append(candidate.name)
            assert candidate.name == "candidate-0000"
            return _evaluated(candidate.name, seed, value=1.0).score

    restore_calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    class FakeResumableProposer:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.turns: tuple[CandidateProposal | CandidateProposalError, ...] = ()

        def restore(
            self,
            history: Sequence[EvaluatedCandidate],
            turns: Sequence[CandidateProposal | CandidateProposalError],
        ) -> None:
            self.turns = tuple(turns)
            restore_calls.append(
                (
                    tuple(item.candidate_id for item in history),
                    tuple(item.candidate_id for item in self.turns),
                )
            )

        def propose(
            self,
            history: Sequence[EvaluatedCandidate],
            **_kwargs: object,
        ) -> CandidateProposal:
            assert tuple(item.candidate_id for item in history) == ("candidate-0000",)
            index = len(self.turns) + 1
            candidate_id = f"candidate-{index:04d}"
            raise CandidateProposalError(
                candidate_id,
                "invalid candidate",
                request=f"produce {candidate_id}",
                status_json=json.dumps(
                    {
                        "agent_error": None,
                        "candidate_doc_hash": None,
                        "candidate_id": candidate_id,
                        "source_tree_hash": None,
                        "valid": False,
                        "validation_error": "invalid candidate",
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )

    projects: list[_Project] = []

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            project = _Project()
            projects.append(project)
            return project

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "ProjectCandidateProposer", FakeResumableProposer)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", HarnessPopulationOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    outcome = _execute_optimization(
        name="selected",
        root=str(tmp_path / ".wmh"),
        run_dir=tmp_path / "run",
        job_config=_job_config(tmp_path),
        task_ids=("task-a",),
        reward_key="reward",
        iterations=2,
        attempts=1,
        max_score_cells=3,
        seed=seed,
        seed_reference=None,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="local",
        e2b_template=None,
        environment_command_timeout_sec=300,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
        resume=False,
    )

    assert score_calls == ["candidate-0000"]
    assert restore_calls == [
        (("candidate-0000",), ()),
        (("candidate-0000",), ("candidate-0001",)),
    ]
    assert len(projects) == 2
    assert len({id(project) for project in projects}) == 2
    assert all(project.closed for project in projects)
    assert [item.candidate_id for item in outcome.result.iterations] == [
        "candidate-0001",
        "candidate-0002",
    ]


def test_execute_finalizes_fully_committed_ready_prefix_without_project_or_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    job_config = _job_config(tmp_path)
    meta_config, agent_config = _configs()
    identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=job_config,
        seed=seed,
        request=request,
        iterations=2,
        max_score_cells=3,
        meta_config=meta_config,
        agent_config=agent_config,
    )
    seed_result = _result(seed)
    first_result = _result_with_invalid_iterations(1, seed_source=seed)
    final_result = _result_with_invalid_iterations(2, seed_source=seed)
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as checkpoint:
        _commit_ready_boundary(checkpoint, seed_result, usage=SandboxUsage())
        _commit_ready_boundary(
            checkpoint,
            first_result,
            usage=SandboxUsage(count=1, seconds=1.0),
        )
        _commit_ready_boundary(
            checkpoint,
            final_result,
            usage=SandboxUsage(count=1, seconds=2.0),
        )

    score_calls: list[str] = []
    project_calls: list[dict[str, object]] = []

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScore:
            del request
            score_calls.append(candidate.name)
            raise AssertionError("fully committed checkpoint must not score")

    class UnreachedProjectFactory:
        @classmethod
        def create(cls, **kwargs: object) -> _Project:
            project_calls.append(kwargs)
            raise AssertionError("fully committed checkpoint must not create a project")

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", UnreachedProjectFactory)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))

    outcome = _execute_optimization(
        name="selected",
        root=str(root),
        run_dir=run_dir,
        job_config=job_config,
        task_ids=("task-a",),
        reward_key="reward",
        iterations=2,
        attempts=1,
        max_score_cells=3,
        seed=None,
        seed_reference=None,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="local",
        e2b_template=None,
        environment_command_timeout_sec=300,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
        resume=True,
    )

    assert score_calls == []
    assert project_calls == []
    assert [item.candidate_id for item in outcome.result.population] == ["candidate-0000"]
    assert [item.candidate_id for item in outcome.result.iterations] == [
        "candidate-0001",
        "candidate-0002",
    ]
    assert outcome.result.best.candidate.doc_hash == final_result.best.candidate.doc_hash
    control = json.loads((run_dir / "checkpoint/control.json").read_text())
    assert control["state"] == "complete"
    assert (run_dir / "population/manifest.json").is_file()
    assert (run_dir / "outcome.json").is_file()


def test_execute_resumes_exact_publication_after_alias_before_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    seed = _source("seed")
    request = _request()
    job_config = _job_config(tmp_path)
    meta_config, agent_config = _configs()
    identity = _checkpoint_identity(
        run_dir=run_dir,
        root=root,
        job_config=job_config,
        seed=seed,
        request=request,
        iterations=2,
        max_score_cells=3,
        meta_config=meta_config,
        agent_config=agent_config,
    )
    with PopulationCheckpointStore.create(run_dir, identity=identity, seed=seed) as checkpoint:
        _commit_ready_boundary(checkpoint, _result(seed), usage=SandboxUsage())
        _commit_ready_boundary(
            checkpoint,
            _result_with_invalid_iterations(1, seed_source=seed),
            usage=SandboxUsage(count=1, seconds=1.0),
        )
        _commit_ready_boundary(
            checkpoint,
            _result_with_invalid_iterations(2, seed_source=seed),
            usage=SandboxUsage(count=1, seconds=2.0),
        )

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

    class UnreachedProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            raise AssertionError("fully committed checkpoint must not create a project")

    original_mark_complete = PopulationCheckpointStore.mark_complete
    completion_calls = 0

    def fail_once(
        self: PopulationCheckpointStore,
        *,
        saved: HarnessDoc,
        archive_manifest: Path,
        outcome_path: Path,
    ) -> None:
        nonlocal completion_calls
        completion_calls += 1
        if completion_calls == 1:
            raise RuntimeError("completion interrupted")
        original_mark_complete(
            self,
            saved=saved,
            archive_manifest=archive_manifest,
            outcome_path=outcome_path,
        )

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", UnreachedProjectFactory)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    monkeypatch.setattr(PopulationCheckpointStore, "mark_complete", fail_once)
    arguments: _OptimizeArguments = {
        "name": "selected",
        "root": str(root),
        "run_dir": run_dir,
        "job_config": job_config,
        "task_ids": ("task-a",),
        "reward_key": "reward",
        "iterations": 2,
        "attempts": 1,
        "max_score_cells": 3,
        "seed": None,
        "seed_reference": None,
        "meta_config": meta_config,
        "agent_config": agent_config,
        "harness_backend": "local",
        "e2b_template": None,
        "environment_command_timeout_sec": 300,
        "episode_timeout_sec": 300.0,
        "project_timeout_sec": 900,
        "max_history_candidates": 20,
        "max_history_bytes": 1_000_000,
        "resume": True,
        "max_new_boundaries": None,
    }

    with pytest.raises(RuntimeError, match="completion interrupted"):
        _execute_optimization(**arguments)

    store = HarnessStore(root)
    assert store.versions("selected") == [1]
    assert store.aliases("selected")[CHAMPION_ALIAS] == 1
    assert (run_dir / "population/manifest.json").is_file()
    assert (run_dir / "outcome.json").is_file()
    interrupted = json.loads((run_dir / "checkpoint/control.json").read_text())
    assert interrupted["state"] == "in_progress"
    assert interrupted["active_kind"] == "finalize"
    assert interrupted["publication_intent"]["harness_version"] == 1

    manifest_path = run_dir / "population/manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PopulationCheckpointError, match="archive differs"):
        _execute_optimization(**arguments)
    manifest_path.write_bytes(manifest_bytes)

    outcome_path = run_dir / "outcome.json"
    outcome_bytes = outcome_path.read_bytes()
    outcome_path.write_text(
        json.dumps(json.loads(outcome_bytes)),
        encoding="utf-8",
    )
    with pytest.raises(PopulationCheckpointError, match="JSON bytes differ"):
        _execute_optimization(**arguments)
    outcome_path.write_bytes(outcome_bytes)

    outcome = _execute_optimization(**arguments)

    assert isinstance(outcome, HarnessOptimizeOutcome)
    assert outcome.saved.version == 1
    assert store.versions("selected") == [1]
    assert store.aliases("selected")[CHAMPION_ALIAS] == 1
    assert json.loads((run_dir / "checkpoint/control.json").read_text())["state"] == "complete"


def test_project_teardown_failure_keeps_final_boundary_nonresumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()

    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

    class TeardownFailsProject(_Project):
        def __exit__(self, *exc_info: object) -> None:
            super().__exit__(*exc_info)
            raise RuntimeError("project teardown failed")

    project = TeardownFailsProject()

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            return project

    result = _result_with_invalid_iterations(1, seed_source=_source("seed"))

    class FakeOptimizer:
        def __init__(self, *_args: object) -> None:
            pass

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            before_step = cast(Callable[[int], None], kwargs["before_step"])
            on_boundary = cast(
                Callable[[PopulationOptimizationResult], None], kwargs["on_boundary"]
            )
            resumed = cast(PopulationOptimizationResult | None, kwargs["resume"])
            if resumed is None:
                boundary = _result(_source("seed"))
                before_step(0)
                on_boundary(boundary)
                return boundary
            before_step(1)
            on_boundary(result)
            return result

    archive_calls: list[Path] = []
    meta_config, agent_config = _configs()
    run_dir = tmp_path / "run"
    root = tmp_path / ".wmh"
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", FakeOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    monkeypatch.setattr(
        module,
        "write_population_archive",
        lambda path, _result: archive_calls.append(path),
    )

    with pytest.raises(RuntimeError, match="teardown failed"):
        _execute_optimization(
            name="selected",
            root=str(root),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=300,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    control = json.loads((run_dir / "checkpoint/control.json").read_text())
    assert control["state"] == "in_progress"
    assert control["active_kind"] == "cleanup"
    assert control["project_sandbox_usage"] is None
    assert archive_calls == []
    assert not (root / "harnesses/selected").exists()
    assert not (run_dir / "outcome.json").exists()


def test_execute_closes_project_and_never_publishes_winner_after_optimizer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return _request()

    project = _Project()

    class FakeProjectFactory:
        @classmethod
        def create(cls, **_kwargs: object) -> _Project:
            return project

    class BrokenOptimizer:
        def __init__(self, proposer: object, scorer: object) -> None:
            del proposer, scorer

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            resumed = cast(PopulationOptimizationResult | None, kwargs["resume"])
            if resumed is None:
                before_step = cast(Callable[[int], None], kwargs["before_step"])
                on_boundary = cast(
                    Callable[[PopulationOptimizationResult], None], kwargs["on_boundary"]
                )
                boundary = _result(_source("seed"))
                before_step(0)
                on_boundary(boundary)
                return boundary
            raise RuntimeError("scorer unavailable")

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "AgentProject", FakeProjectFactory)
    monkeypatch.setattr(module, "HarnessPopulationOptimizer", BrokenOptimizer)
    monkeypatch.setattr(module, "get_provider", lambda config: _ToolProvider(config))
    root = tmp_path / ".wmh"

    with pytest.raises(RuntimeError, match="scorer unavailable"):
        _execute_optimization(
            name="unpublished",
            root=str(root),
            run_dir=tmp_path / "local-run",
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=300,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    assert project.closed is True
    assert not (root / "harnesses/unpublished").exists()
    assert not (tmp_path / "local-run/outcome.json").exists()


def test_execute_does_not_claim_result_directory_when_scorer_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> BrokenScorer:
            raise ValueError("invalid scorer configuration")

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", BrokenScorer)
    run_dir = tmp_path / "local-run"

    with pytest.raises(ValueError, match="invalid scorer configuration"):
        _execute_optimization(
            name="unpublished",
            root=str(tmp_path / ".wmh"),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=240,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    assert not run_dir.exists()


def test_execute_does_not_claim_result_directory_for_non_tool_meta_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeScorer(_NoPreparationScorer):
        @classmethod
        async def create(cls, **_kwargs: object) -> FakeScorer:
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return _request()

    meta_config, agent_config = _configs()
    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "get_provider", lambda config: object())
    run_dir = tmp_path / "local-run"

    with pytest.raises(typer.BadParameter, match="structured tool calling"):
        _execute_optimization(
            name="unpublished",
            root=str(tmp_path / ".wmh"),
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            iterations=1,
            attempts=1,
            max_score_cells=2,
            seed=_source("seed"),
            seed_reference=None,
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=240,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
            resume=False,
        )

    assert not run_dir.exists()


def test_cli_uses_required_roles_complete_default_pi_seed_and_local_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                meta=ModelRole(
                    provider="openai_responses",
                    model="meta-model",
                    reasoning_effort="high",
                ),
                agent=ModelRole(provider="anthropic", model="agent-model"),
            )
        ),
        root,
    )
    config_path = tmp_path / "harbor.yaml"
    config_path.write_text(_job_config(tmp_path).model_dump_json(), encoding="utf-8")
    task_ids_path = tmp_path / "task-ids.json"
    task_ids_path.write_text('["task-a"]\n', encoding="utf-8")
    calls: list[dict[str, object]] = []
    fake_result = _result()

    def fake_execute(**kwargs: object) -> HarnessOptimizeOutcome | HarnessOptimizeProgress:
        calls.append(kwargs)
        run_dir = cast(Path, kwargs["run_dir"])
        if kwargs["max_new_boundaries"] is not None:
            return HarnessOptimizeProgress(
                result=fake_result,
                run_dir=run_dir,
                project_usage=None,
            )
        saved = fake_result.best.candidate.model_copy(update={"name": "selected", "version": 1})
        return HarnessOptimizeOutcome(
            result=fake_result,
            saved=saved,
            run_dir=run_dir,
            archive_manifest=run_dir / "population/manifest.json",
            project_usage=_Project().usage(),
        )

    monkeypatch.setattr(module, "_execute_optimization", fake_execute)
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "template-from-env")
    result_out = tmp_path / "local-run"

    invoked = runner.invoke(
        app,
        [
            "harness",
            "optimize",
            "selected",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
            "--reward-mode",
            "positive-binary",
            "--iterations",
            "2",
            "--attempts",
            "1",
            "--max-score-cells",
            "3",
            "--result-out",
            str(result_out),
            "--root",
            str(root),
            "--yes",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    assert "optimized" in invoked.output
    [call] = calls
    assert call["harness_backend"] == "local"
    assert call["reward_mode"] == "positive_binary"
    assert call["iterations"] == 2
    assert call["task_ids"] == ("task-a",)
    assert call["e2b_template"] == "template-from-env"
    assert cast(ProviderConfig, call["meta_config"]).reasoning_effort == "high"
    assert cast(ProviderConfig, call["agent_config"]).kind is ProviderKind.ANTHROPIC
    seed = call["seed"]
    assert isinstance(seed, HarnessSourceTree)
    assert seed.to_doc("seed").runtime_kind() == "pi-node"
    assert len(seed.to_doc("seed").code_files()) > 1

    partial_out = tmp_path / "partial-run"
    partial = runner.invoke(
        app,
        [
            "harness",
            "optimize",
            "selected",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
            "--iterations",
            "2",
            "--attempts",
            "1",
            "--max-score-cells",
            "3",
            "--harness-backend",
            "e2b",
            "--episode-timeout",
            "12000",
            "--max-new-boundaries",
            "1",
            "--result-out",
            str(partial_out),
            "--root",
            str(root),
            "--yes",
        ],
    )

    assert partial.exit_code == 0, partial.output
    assert calls[-1]["max_new_boundaries"] == 1
    assert calls[-1]["harness_backend"] == "e2b"
    assert calls[-1]["episode_timeout_sec"] == 12_000
    assert calls[-1]["reward_mode"] == "raw"
    assert "episode timeout=12000s" in " ".join(partial.output.split())
    assert "checkpointed" in partial.output
    assert "no winner published" in partial.output
    assert "--resume --result-out" in partial.output
    assert "partial-run" in partial.output


def test_cli_requires_explicit_confirmation_in_noninteractive_mode(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                meta=ModelRole(provider="openai_responses", model="meta-model"),
                agent=ModelRole(provider="anthropic", model="agent-model"),
            )
        ),
        root,
    )
    config_path = tmp_path / "harbor.json"
    config_path.write_text(_job_config(tmp_path).model_dump_json(), encoding="utf-8")
    task_ids_path = tmp_path / "task-ids.json"
    task_ids_path.write_text('["task-a"]\n', encoding="utf-8")

    invoked = runner.invoke(
        app,
        [
            "harness",
            "optimize",
            "selected",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
            "--iterations",
            "1",
            "--attempts",
            "1",
            "--max-score-cells",
            "2",
            "--root",
            str(root),
        ],
    )

    assert invoked.exit_code == 2
    assert "pass --yes" in invoked.output


@pytest.mark.parametrize("payload", ['["same", "same"]', '{"task": "a"}', "not-json"])
def test_cli_rejects_nonexact_task_id_manifests(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "task-ids.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(typer.BadParameter):
        module.load_task_ids(path)


def test_cli_help_preserves_model_roles_seed_syntax_and_local_output_wording() -> None:
    invoked = runner.invoke(app, ["harness", "optimize", "--help"])

    assert invoked.exit_code == 0, invoked.output
    assert "models.meta" in invoked.output
    assert "models.agent" in invoked.output
    assert "name@ref" in invoked.output
    assert "Local run directory" in invoked.output
    assert "--episode-timeout" in invoked.output
    assert "--reward-mode" in invoked.output
    assert "seconds" in invoked.output

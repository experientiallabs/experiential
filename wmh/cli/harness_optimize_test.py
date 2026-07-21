"""CLI and resource-lifecycle tests for scorer-driven harness optimization."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import cast

import pytest
import typer
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from typer.testing import CliRunner

from wmh.cli import app
from wmh.cli.harness_optimize import HarnessOptimizeOutcome, _execute_optimization
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.harness.e2b_sandbox import SandboxUsage
from wmh.harness.population import PopulationOptimizationResult
from wmh.harness.project_proposer import EvaluatedCandidate, ProjectCandidateProposer
from wmh.harness.scoring import (
    EvaluationArtifact,
    HarnessScore,
    HarnessScoreReport,
    ScoreCell,
    ScoreContext,
    ScoreRequest,
)
from wmh.harness.source_tree import HarnessSourceFile, HarnessSourceTree
from wmh.providers.base import ProviderConfig, ProviderKind

module = importlib.import_module("wmh.cli.harness_optimize")
runner = CliRunner()
_DIGEST = "sha256:" + "a" * 64


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


def _result() -> PopulationOptimizationResult:
    source = _source("selected")
    candidate_id = "candidate-0000"
    artifact = EvaluationArtifact.from_bytes(path="raw/result.json", content=b"{}")
    report = HarnessScoreReport(
        source_run_id="run-1",
        candidate_doc_hash=source.to_doc(candidate_id).doc_hash,
        request=_request(),
        cells=(
            ScoreCell(
                task_id="task-a",
                attempt=1,
                score=1.0,
                passed=True,
                summary="official result",
                artifact_paths=("raw/result.json",),
            ),
        ),
        artifacts=(artifact,),
    )
    evaluated = EvaluatedCandidate(
        candidate_id=candidate_id,
        source=source,
        score=HarnessScore(report=report, artifacts=_Reader()),
    )
    return PopulationOptimizationResult(population=(evaluated,), iterations=(), best=evaluated)


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


def _configs() -> tuple[ProviderConfig, ProviderConfig]:
    return (
        ProviderConfig(
            kind=ProviderKind.OPENAI_RESPONSES,
            model="meta-model",
            reasoning_effort="high",
        ),
        ProviderConfig(kind=ProviderKind.ANTHROPIC, model="agent-model"),
    )


def test_execute_composes_exact_scorer_project_optimizer_archive_and_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    scorer_calls: list[dict[str, object]] = []

    class FakeScorer:
        @classmethod
        async def create(cls, **kwargs: object) -> FakeScorer:
            scorer_calls.append(kwargs)
            return cls()

        def request(self, *, attempts: int) -> ScoreRequest:
            assert attempts == 1
            return request

    project = _Project()
    project_calls: list[dict[str, object]] = []

    class FakeProjectFactory:
        @classmethod
        def create(cls, **kwargs: object) -> _Project:
            project_calls.append(kwargs)
            return project

    optimizer_calls: list[dict[str, object]] = []
    result = _result()

    class FakeOptimizer:
        def __init__(self, proposer: object, scorer: object) -> None:
            optimizer_calls.append({"proposer": proposer, "scorer": scorer})

        def optimize(self, **kwargs: object) -> PopulationOptimizationResult:
            optimizer_calls[-1].update(kwargs)
            return result

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
        seed=seed,
        meta_config=meta_config,
        agent_config=agent_config,
        harness_backend="e2b",
        e2b_template="template-x",
        environment_command_timeout_sec=300,
        project_timeout_sec=900,
        max_history_candidates=20,
        max_history_bytes=1_000_000,
    )

    [scorer_call] = scorer_calls
    assert scorer_call["task_ids"] == ("task-a",)
    assert scorer_call["provider_config"] == agent_config
    assert scorer_call["harness_backend"] == "e2b"
    assert scorer_call["e2b_template"] == "template-x"
    assert cast(JobConfig, scorer_call["job_config"]).jobs_dir == run_dir / "harbor"
    [project_call] = project_calls
    assert project_call["template"] == "template-x"
    assert project_call["timeout"] == 900
    assert project.closed is True
    [optimizer_call] = optimizer_calls
    assert cast(FakeScorer, optimizer_call["scorer"]).request(attempts=1) == request
    assert optimizer_call["seed"] == seed
    assert optimizer_call["request"] == request
    assert optimizer_call["iterations"] == 3
    proposer = cast(ProjectCandidateProposer, optimizer_call["proposer"])
    assert proposer._max_history_candidates == 20
    assert proposer._max_history_bytes == 1_000_000
    assert archived == [(run_dir / "population", result)]
    assert outcome.saved.name == "selected"
    assert outcome.saved.version == 1
    assert outcome.saved.doc_hash == result.best.candidate.doc_hash
    assert (root / "harnesses/selected/aliases.toml").exists()
    assert json.loads((run_dir / "inputs.json").read_text())["iterations"] == 3
    written = json.loads((run_dir / "outcome.json").read_text())
    assert written["best_candidate_id"] == "candidate-0000"
    assert written["project_sandbox_usage"] == {"count": 1, "seconds": 12.5}


def test_execute_closes_project_and_never_publishes_winner_after_optimizer_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeScorer:
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

        def optimize(self, **_kwargs: object) -> PopulationOptimizationResult:
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
            seed=_source("seed"),
            meta_config=meta_config,
            agent_config=agent_config,
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=300,
            project_timeout_sec=900,
            max_history_candidates=20,
            max_history_bytes=1_000_000,
        )

    assert project.closed is True
    assert not (root / "harnesses/unpublished").exists()
    assert not (tmp_path / "local-run/outcome.json").exists()


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

    def fake_execute(**kwargs: object) -> HarnessOptimizeOutcome:
        calls.append(kwargs)
        saved = fake_result.best.candidate.model_copy(update={"name": "selected", "version": 1})
        run_dir = cast(Path, kwargs["run_dir"])
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
            "--iterations",
            "2",
            "--attempts",
            "1",
            "--result-out",
            str(result_out),
            "--root",
            str(root),
            "--yes",
        ],
    )

    assert invoked.exit_code == 0, invoked.output
    [call] = calls
    assert call["harness_backend"] == "local"
    assert call["iterations"] == 2
    assert call["task_ids"] == ("task-a",)
    assert call["e2b_template"] == "template-from-env"
    assert cast(ProviderConfig, call["meta_config"]).reasoning_effort == "high"
    assert cast(ProviderConfig, call["agent_config"]).kind is ProviderKind.ANTHROPIC
    seed = call["seed"]
    assert isinstance(seed, HarnessSourceTree)
    assert seed.to_doc("seed").runtime_kind() == "pi-node"
    assert len(seed.to_doc("seed").code_files()) > 1


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
        module._load_task_ids(path)


def test_cli_help_preserves_model_roles_seed_syntax_and_local_output_wording() -> None:
    invoked = runner.invoke(app, ["harness", "optimize", "--help"])

    assert invoked.exit_code == 0, invoked.output
    assert "models.meta" in invoked.output
    assert "models.agent" in invoked.output
    assert "name@ref" in invoked.output
    assert "Local run directory" in invoked.output

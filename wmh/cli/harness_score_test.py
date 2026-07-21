"""CLI composition tests for proposer-free multi-harness scoring."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import cast

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from typer.testing import CliRunner

from wmh.cli import app
from wmh.cli.harness_score import HarnessScoreOutcome, _execute_scoring
from wmh.config.settings import ModelRole, ModelsSettings, ProjectSettings, save_settings
from wmh.harness.score_batch import HarnessScoreBatch, HarnessScoreTarget, ScoredHarness
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

module = importlib.import_module("wmh.cli.harness_score")
runner = CliRunner()
_DIGEST = "sha256:" + "a" * 64


class _Reader:
    def read_bytes(self, path: str) -> bytes:
        assert path == "raw/result.json"
        return b"{}"


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


def _score(target: HarnessScoreTarget, request: ScoreRequest) -> HarnessScore:
    artifact = EvaluationArtifact.from_bytes(path="raw/result.json", content=b"{}")
    return HarnessScore(
        report=HarnessScoreReport(
            source_run_id=f"run-{target.label}",
            candidate_doc_hash=target.harness.doc_hash,
            request=request,
            cells=(
                ScoreCell(
                    task_id="task-a",
                    attempt=1,
                    score=1.0,
                    passed=True,
                    artifact_paths=("raw/result.json",),
                ),
            ),
            artifacts=(artifact,),
        ),
        artifacts=_Reader(),
    )


def _batch(targets: tuple[HarnessScoreTarget, ...]) -> HarnessScoreBatch:
    request = _request()
    return HarnessScoreBatch(
        request=request,
        entries=tuple(
            ScoredHarness(target=target, score=_score(target, request)) for target in targets
        ),
    )


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


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "harbor.yaml"
    config_path.write_text(_job_config(tmp_path).model_dump_json(), encoding="utf-8")
    task_ids_path = tmp_path / "task-ids.json"
    task_ids_path.write_text('["task-a"]\n', encoding="utf-8")
    return config_path, task_ids_path


def _agent_settings(root: Path) -> None:
    save_settings(
        ProjectSettings(
            models=ModelsSettings(
                agent=ModelRole(provider="anthropic", model="agent-model"),
            )
        ),
        root,
    )


def test_execute_resolves_one_scorer_request_and_neutral_archive(
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

    target = HarnessScoreTarget(
        label="candidate@v1",
        harness=_source("p").to_doc("candidate"),
    )
    scored_calls: list[tuple[object, tuple[HarnessScoreTarget, ...], ScoreRequest]] = []

    def fake_score_harnesses(
        scorer: object,
        targets: tuple[HarnessScoreTarget, ...],
        *,
        request: ScoreRequest,
    ) -> HarnessScoreBatch:
        scored_calls.append((scorer, targets, request))
        return _batch(targets)

    archived: list[tuple[Path, HarnessScoreBatch]] = []

    def fake_archive(path: Path, batch: HarnessScoreBatch) -> Path:
        archived.append((path, batch))
        path.mkdir(parents=True)
        manifest = path / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        return manifest

    monkeypatch.setattr(module, "HarborScorer", FakeScorer)
    monkeypatch.setattr(module, "score_harnesses", fake_score_harnesses)
    monkeypatch.setattr(module, "write_score_archive", fake_archive)
    run_dir = tmp_path / "run"
    agent_config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="agent-model")

    outcome = _execute_scoring(
        run_dir=run_dir,
        job_config=_job_config(tmp_path),
        task_ids=("task-a",),
        reward_key="reward",
        attempts=1,
        targets=(target,),
        agent_config=agent_config,
        harness_backend="e2b",
        e2b_template="template-x",
        environment_command_timeout_sec=240,
    )

    [scorer_call] = scorer_calls
    assert cast(JobConfig, scorer_call["job_config"]).jobs_dir == run_dir / "harbor"
    assert scorer_call["provider_config"] == agent_config
    assert scorer_call["harness_backend"] == "e2b"
    assert scorer_call["e2b_template"] == "template-x"
    [scored_call] = scored_calls
    assert scored_call[1] == (target,)
    assert scored_call[2] is request
    assert archived == [(run_dir / "scores", outcome.result)]
    assert outcome.archive_manifest == run_dir / "scores/manifest.json"
    inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
    assert inputs["score_request"] == request.model_dump(mode="json")
    assert inputs["targets"] == [
        {
            "document_hash": target.harness.doc_hash,
            "label": "candidate@v1",
            "name": "candidate",
            "source_tree_hash": target.source.tree_hash,
            "version": 0,
        }
    ]
    assert "meta_provider" not in inputs


def test_execute_does_not_claim_result_directory_when_scorer_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenScorer:
        @classmethod
        async def create(cls, **_kwargs: object) -> BrokenScorer:
            raise RuntimeError("task resolution failed")

    monkeypatch.setattr(module, "HarborScorer", BrokenScorer)
    run_dir = tmp_path / "run"
    target = HarnessScoreTarget(
        label="candidate@v1",
        harness=_source("p").to_doc("candidate"),
    )

    with pytest.raises(RuntimeError, match="task resolution failed"):
        _execute_scoring(
            run_dir=run_dir,
            job_config=_job_config(tmp_path),
            task_ids=("task-a",),
            reward_key="reward",
            attempts=1,
            targets=(target,),
            agent_config=ProviderConfig(
                kind=ProviderKind.ANTHROPIC,
                model="agent-model",
            ),
            harness_backend="local",
            e2b_template=None,
            environment_command_timeout_sec=240,
        )

    assert not run_dir.exists()


def test_cli_resolves_default_then_immutable_stored_version_with_agent_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / ".wmh"
    _agent_settings(root)
    store = HarnessStore(root)
    store.save_version(_source("v1").to_doc("stored"))
    selected = store.save_version(
        _source("v2").to_doc("stored"),
        alias=CHAMPION_ALIAS,
    )
    config_path, task_ids_path = _write_inputs(tmp_path)
    calls: list[dict[str, object]] = []

    def fake_execute(**kwargs: object) -> HarnessScoreOutcome:
        calls.append(kwargs)
        targets = cast(tuple[HarnessScoreTarget, ...], kwargs["targets"])
        run_dir = cast(Path, kwargs["run_dir"])
        return HarnessScoreOutcome(
            result=_batch(targets),
            run_dir=run_dir,
            archive_manifest=run_dir / "scores/manifest.json",
        )

    monkeypatch.setattr(module, "_execute_scoring", fake_execute)
    monkeypatch.setenv("WMH_E2B_TEMPLATE", "template-from-env")
    result_out = tmp_path / "run"

    invoked = runner.invoke(
        app,
        [
            "harness",
            "score",
            "--include-default",
            "--harness",
            "stored@champion",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
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
    targets = cast(tuple[HarnessScoreTarget, ...], call["targets"])
    assert [target.label for target in targets] == ["default", "stored@v2"]
    assert targets[1].harness.version == 2
    assert targets[1].harness.doc_hash == selected.doc_hash
    assert call["harness_backend"] == "local"
    assert call["e2b_template"] is None
    agent_config = cast(ProviderConfig, call["agent_config"])
    assert agent_config.kind is ProviderKind.ANTHROPIC
    assert "models.meta" not in invoked.output


def test_cli_requires_a_target_before_model_or_scorer_resolution(tmp_path: Path) -> None:
    config_path, task_ids_path = _write_inputs(tmp_path)

    invoked = runner.invoke(
        app,
        [
            "harness",
            "score",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
            "--attempts",
            "1",
            "--root",
            str(tmp_path / ".wmh"),
            "--yes",
        ],
    )

    assert invoked.exit_code == 2
    assert "--include-default or at least one --harness" in invoked.output


def test_cli_requires_explicit_confirmation_in_noninteractive_mode(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    _agent_settings(root)
    config_path, task_ids_path = _write_inputs(tmp_path)

    invoked = runner.invoke(
        app,
        [
            "harness",
            "score",
            "--include-default",
            "--harbor-config",
            str(config_path),
            "--task-ids",
            str(task_ids_path),
            "--reward-key",
            "reward",
            "--attempts",
            "1",
            "--root",
            str(root),
        ],
    )

    assert invoked.exit_code == 2
    assert "pass --yes" in invoked.output


def test_cli_help_is_proposer_free_and_requires_only_agent_role() -> None:
    invoked = runner.invoke(app, ["harness", "score", "--help"])

    assert invoked.exit_code == 0, invoked.output
    assert "models.agent" in invoked.output
    assert "models.meta" not in invoked.output
    assert "proposer" not in invoked.output.lower()
    assert "world model" not in invoked.output.lower()

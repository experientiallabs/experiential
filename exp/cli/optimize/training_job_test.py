"""CLI initial-scoring and TailSFT lifecycle using the production artifact owners."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.optimize import model as model_command
from exp.cli.optimize.training_job import TrainingJob
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.model.sft.composition_test import _prepared
from exp.optimize.model.sft.initial_scoring import saved_initial_scoring
from exp.optimize.model.sft.tail_sft import TailSFTPolicy
from exp.optimize.model.sft.tail_sft_test import _ScoringBackend
from exp.optimize.model.sft.training_test import _spec


def test_cli_scores_trains_and_replays_without_automatic_runtime_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing admitted dataset can reach a candidate without routed production logs."""
    spec = _spec(batch_size=2, maximum_cost_usd=1, training_usd_per_million_tokens=1000)
    prepared = _prepared(tmp_path, training=spec)
    store = prepared.store
    job = TrainingJob(
        action="score-initial",
        dataset=artifact_input(store.artifacts.read(prepared.dataset_id).manifest),
        connection="tinker",
        base_model_alias="base",
        training=spec,
    )
    path = tmp_path / "job.json"
    path.write_text(job.model_dump_json())
    backend = _ScoringBackend(initial=True)
    compositions = []

    def compose(project: ProjectStore, name: str, digest: str) -> _ScoringBackend:
        """Record credential-bound composition without creating any real credentials."""
        compositions.append((project.paths.project_id, name, digest))
        return backend

    monkeypatch.setattr(model_command, "_compose_tinker_backend", compose)
    monkeypatch.setattr(model_command, "installed_execution_revision", lambda: "w14m-test")
    args = [
        "optimize",
        "model",
        store.paths.project_id,
        "--root",
        str(store.paths.root),
        "--training-job",
        str(path),
        "--yes",
        "--non-interactive",
    ]
    runner = CliRunner()
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert compositions == []
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    assert backend.score_calls == 1 and backend.train_calls == 0
    reference = saved_initial_scoring(store, job.dataset, spec, code_revision="w14m-test")
    assert reference is not None
    assert runner.invoke(app, args).exit_code == 0
    assert len(compositions) == 1
    backend.initial = False
    training = spec.model_copy(
        update={"tail_sft": TailSFTPolicy(reference=reference, drop_fraction=0.5)}
    )
    path.write_text(
        job.model_copy(update={"action": "train", "training": training}).model_dump_json()
    )
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "Candidate saved" in result.output
    assert backend.train_calls == 1
    replay = runner.invoke(app, args)
    assert replay.exit_code == 0, replay.output
    assert len(compositions) == 2
    assert backend.train_calls == 1

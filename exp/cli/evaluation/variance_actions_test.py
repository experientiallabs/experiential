"""The ``rules-stage`` and ``actor-critic`` actions end to end through ``exp eval --continuations``.

Provider wires are scripted; spec, exposure and trace dataset are real saved inputs, so dry runs,
spend consent, execution and replay all run through the public command.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.variance_actions import (
    ActorCriticAction,
    RulesStageAction,
    rules_stage_identity,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.release_revision import installed_execution_revision
from exp.common.traces.ingest.dataset import persist_trace_dataset
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.rules.extraction_test import (
    RoutedTransport,
    _answer,
    _setup,
    _tool_reply,
    _wire,
)
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageSpec
from exp.optimize.evaluation.rules.stage_test import _plan, _spec, _traces
from exp.optimize.evaluation.variance import actor_critic_test as critic_fixture
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec

_TRACES = ("alpha", "beta")


def _write(path: Path, model: BaseModel) -> Path:
    """Write one contract model as a JSON file and return its path."""
    path.write_text(model.model_dump_json())
    return path


def _command(root: Path, request: Path, *extra: str) -> list[str]:
    """The public ``exp eval`` command for one continuation spec."""
    return ["eval", "case", "--root", str(root), "--continuations", str(request), *extra]


def _forbid(*args: object, **kwargs: object) -> object:
    """Fail the test: the stubbed callable must not run."""
    raise AssertionError("must not be called")


def _rules_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, RulesStageSpec, Path]:
    """Save a two-trace dataset and write the spec, exposure and command files.

    Args:
        tmp_path: Root of the project.
        monkeypatch: Fixture configuring the author credential.

    Returns:
        The project, the registered spec and the command file.
    """
    project, settings = _setup(tmp_path, monkeypatch)
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=_traces(*_TRACES), issues=()),
        project.artifacts,
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    spec = _spec(settings)
    action = RulesStageAction(
        spec=_write(tmp_path / "rules-spec.json", spec),
        exposure=_write(tmp_path / "exposure.json", _plan(_TRACES)),
        traces=_TRACES,
        dataset=artifact_input(persisted.manifest),
    )
    request = _write(
        tmp_path / "rules.json",
        ContinuationCommand(action="rules-stage", rules_stage=action, spend=UNPRICED_SPEND),
    )
    return project, spec, request


def test_rules_stage_dry_run_prints_the_summary_and_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run shows traces, draws, author, concurrency and cost, then stops."""
    project, _, request = _rules_fixture(tmp_path, monkeypatch)
    transport = RoutedTransport({})
    _wire(monkeypatch, transport)

    result = CliRunner().invoke(app, _command(tmp_path, request, "--dry-run"))

    assert result.exit_code == 0, result.output
    assert "Rules stage: 2 traces, 2 extraction draws with judge, 3 at once" in result.output
    assert "physical attempts" in result.output
    assert not transport.requests
    assert not project.artifacts.list_ids(artifact_type=REPORT_TYPE)


def test_rules_stage_refused_consent_stops_before_any_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the estimate constructs no transport and runs no stage."""
    project, _, request = _rules_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuation_context.require_spend_consent", lambda *a, **k: False
    )
    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", _forbid)
    monkeypatch.setattr("exp.cli.evaluation.variance_actions.run_rules_stage", _forbid)

    result = CliRunner().invoke(app, _command(tmp_path, request, "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert not project.artifacts.list_ids(artifact_type=REPORT_TYPE)


def test_rules_stage_runs_then_replays_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consented run saves the rule book; the saved report replays with no new spend."""
    project, spec, request = _rules_fixture(tmp_path, monkeypatch)
    transport = RoutedTransport({name: _tool_reply(_answer(name)) for name in _TRACES})
    _wire(monkeypatch, transport)
    runner = CliRunner()

    result = runner.invoke(app, _command(tmp_path, request, "--yes", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert len(transport.requests) == 2
    assert '"rule_counts"' in result.output and '"requires_witness"' in result.output
    (report_id,) = project.artifacts.list_ids(artifact_type=REPORT_TYPE)
    identity = rules_stage_identity(spec, _traces(*_TRACES), installed_execution_revision())
    assert report_id == identity + "-result"

    monkeypatch.setattr("exp.cli.evaluation.continuation_context.require_spend_consent", _forbid)
    replay = runner.invoke(app, _command(tmp_path, request, "--non-interactive"))

    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == 2
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]


def _critic_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ActorCriticSpec, Path]:
    """Write a registered actor-critic spec, a mining-probe exposure plan and the command.

    Args:
        tmp_path: Root of the project.
        monkeypatch: Fixture configuring the actor and role credential.

    Returns:
        The project, the spec and the command file.
    """
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = critic_fixture._spec(project, job)
    action = ActorCriticAction(
        spec=_write(tmp_path / "critic-spec.json", spec),
        exposure=_write(
            tmp_path / "exposure.json", critic_fixture._exposure(project, job, "mining_probes")
        ),
    )
    request = _write(
        tmp_path / "critic.json",
        ContinuationCommand(action="actor-critic", actor_critic=action, spend=UNPRICED_SPEND),
    )
    return project, spec, request


def test_actor_critic_dry_run_prints_the_summary_and_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run shows G, J, C, K and M with the cost inventory, then stops."""
    project, _, request = _critic_fixture(tmp_path, monkeypatch)
    actor, roles = critic_fixture._wire(monkeypatch, [], [])

    result = CliRunner().invoke(app, _command(tmp_path, request, "--dry-run"))

    assert result.exit_code == 0, result.output
    assert "G=3 attempts, J=1 judges, C=1 critics, up to K=2 points" in result.output
    assert "M=2 continuations" in result.output and "physical attempts" in result.output
    assert not actor.requests and not roles.requests
    assert not project.artifacts.list_ids(artifact_type="actor-critic-report")


def test_actor_critic_refused_consent_stops_before_any_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the estimate constructs no transport and runs nothing."""
    project, _, request = _critic_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuation_context.require_spend_consent", lambda *a, **k: False
    )
    monkeypatch.setattr("exp.optimize.evaluation.continuations.job.HttpxJsonTransport", _forbid)
    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", _forbid)
    monkeypatch.setattr("exp.cli.evaluation.variance_actions.run_actor_critic", _forbid)

    result = CliRunner().invoke(app, _command(tmp_path, request, "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert not project.artifacts.list_ids(artifact_type="actor-critic-report")


def test_actor_critic_runs_then_replays_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consented run prints ranking and eligibility; the saved report replays free."""
    _, _, request = _critic_fixture(tmp_path, monkeypatch)
    group = [
        reply for row in (2, 3, 4) for reply in (critic_fixture._edit(row), critic_fixture._done())
    ]
    suffixes = [
        critic_fixture._edit(2),
        critic_fixture._done(),
        critic_fixture._edit(3),
        critic_fixture._done(),
        critic_fixture._done(),
        critic_fixture._done(),
    ]
    limits = ["Simulated grades only."]
    estimates = [
        {"step": step, "probability": 0.5, "rationale": "Base rate.", "evidence_refs": ["group"]}
        for step in (0, 1)
    ]
    roles = [
        critic_fixture._tool_reply(
            "choose_points",
            {
                "points": [critic_fixture._point(0), critic_fixture._point(1)],
                "audit": [],
                "limitations": limits,
            },
        ),
        critic_fixture._tool_reply(
            "estimate_points", {"estimates": estimates, "limitations": limits}
        ),
    ]
    actor, role = critic_fixture._wire(monkeypatch, group + suffixes, roles)
    runner = CliRunner()

    result = runner.invoke(app, _command(tmp_path, request, "--yes", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert '"ranking"' in result.output and '"training_eligible": false' in result.output
    assert '"ineligible_reasons"' in result.output
    calls = (len(actor.requests), len(role.requests))

    monkeypatch.setattr("exp.cli.evaluation.continuation_context.require_spend_consent", _forbid)
    replay = runner.invoke(app, _command(tmp_path, request, "--non-interactive"))

    assert replay.exit_code == 0, replay.output
    assert (len(actor.requests), len(role.requests)) == calls
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]

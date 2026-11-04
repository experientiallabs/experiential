"""The variance-build, rules-stage and actor-critic actions end to end through ``exp eval``.

Provider wires are scripted; spec, exposure and trace dataset are real saved inputs, so dry runs,
spend consent, execution and replay all run through the public command.
"""

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from rich.console import Console
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.continuation_context_test import UNPRICED_SPEND
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.evaluation.variance_actions import (
    ActorCriticAction,
    RulesStageAction,
    VarianceBuildAction,
    VarianceScheduleAction,
    actor_critic_lines,
    handle_variance_schedule,
    point_option_costs,
    rules_stage_identity,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.release_revision import installed_execution_revision
from exp.common.traces.ingest.dataset import persist_trace_dataset
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.costs import job_costs
from exp.optimize.evaluation.rules.extraction_test import (
    RoutedTransport,
    _setup,
    _wire,
)
from exp.optimize.evaluation.rules.research_test import AgentScript
from exp.optimize.evaluation.rules.stage import REPORT_TYPE, RulesStageSpec
from exp.optimize.evaluation.rules.stage_test import _answers, _plan, _spec, _traces
from exp.optimize.evaluation.variance import actor_critic_test as critic_fixture
from exp.optimize.evaluation.variance import build_test as build_fixture
from exp.optimize.evaluation.variance import schedule_test as schedule_fixture
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.actor_critic_costs import actor_critic_costs
from exp.optimize.evaluation.variance.build_records import REPORT_TYPE as VARIANCE_REPORT_TYPE
from exp.optimize.evaluation.variance.build_records import VarianceBuildSpec
from exp.optimize.evaluation.variance.decision_measure_test import FORCED
from exp.optimize.evaluation.variance.evidence_tools import EvidenceBudget, EvidenceToolProfile
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judge_prompt import (
    DEFAULT_AGENTIC_JUDGE,
    judge_model_calls,
    judge_profile,
)
from exp.optimize.evaluation.variance.point_options import PointOptionsSpec
from exp.optimize.evaluation.variance.point_options_test import _proposal
from exp.optimize.evaluation.variance.rewards import SelectorObjectiveV2
from exp.optimize.evaluation.variance.schedule import (
    RevisionRegistration,
    TrainingScheduleSpec,
    TransitionRule,
    next_rollout_plan,
    schedule_state,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.spend_ceiling import current_spend_ceiling, operation_spend_scope

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
    output = " ".join(result.output.split())
    assert "Rules stage: 2 traces; 2 research draws (up to 12 calls)" in output
    assert "one weighing draw per merged rule (up to 6 calls" in output
    assert "one condensation draw (up to 6 calls) with judge; 3 at once" in output
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
    transport = AgentScript(_answers)
    _wire(monkeypatch, transport)
    runner = CliRunner()

    result = runner.invoke(app, _command(tmp_path, request, "--yes", "--non-interactive"))

    assert result.exit_code == 0, result.output
    roles = transport.roles()
    assert {"research", "condensation"} <= set(roles)
    assert "extraction" not in roles
    bought = len(transport.requests)
    assert '"rule_counts"' in result.output and '"requires_witness"' in result.output
    (report_id,) = project.artifacts.list_ids(artifact_type=REPORT_TYPE)
    identity = rules_stage_identity(spec, _traces(*_TRACES), installed_execution_revision())
    assert report_id == identity + "-result"

    monkeypatch.setattr("exp.cli.evaluation.continuation_context.require_spend_consent", _forbid)
    replay = runner.invoke(app, _command(tmp_path, request, "--non-interactive"))

    assert replay.exit_code == 0, replay.output
    assert len(transport.requests) == bought
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
    assert "Upper bound" in result.output
    assert not actor.requests and not roles.requests
    assert not project.artifacts.list_ids(artifact_type="actor-critic-report")


def test_actor_critic_cost_scales_role_draws_by_their_model_call_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each selector and critic row prices every model call its evidence-tool budget allows."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    base = critic_fixture._spec(project, job, judges=2)
    tools = EvidenceToolProfile(
        mode="read_only_tools", budget=EvidenceBudget(model_calls=3, tool_calls=4)
    )
    spec = base.model_copy(update={"selector_profile": tools, "critic_profile": tools})
    forced, scaled = actor_critic_costs(project, base), actor_critic_costs(project, spec)
    group = len(job_costs(project, base.initial))
    roles = len(base.judges) + len(base.critics)
    role_attempts = base.judges[0].maximum_physical_attempts

    assert len(forced) == len(scaled)
    assert [row.attempts for row in forced[group : group + roles]] == [role_attempts] * roles
    assert [row.attempts for row in scaled[group : group + roles]] == [3 * role_attempts] * roles
    assert forced[group + roles :] == scaled[group + roles :]
    suffix = forced[group + roles]
    assert suffix.attempts == base.continuation_physical_attempts * len(base.judges) * base.points
    summary = " ".join(actor_critic_lines(spec))
    assert (
        "Upper bound (error-mining-v1): 2 selector draws x 3 model calls "
        "(read_only_tools, 4 tool calls)" in summary
    )
    assert "1 critic draws x 3 model calls" in summary and "4 suffix jobs of 2" in summary
    judged = base.initial.study.repeats + len(base.judges) * base.points * base.continuations
    judge = base.episode_judge.settings.maximum_physical_attempts
    assert forced[-1].attempts == judge * judged
    assert f"Episode judge: up to {judged} episodes x 1 judge requests" in summary
    # An agentic judge adds its plan, criterion-check and self-check draws to every read.
    profile = base.episode_judge.definition.profile
    agentic = judge_profile(
        model_alias=profile.model_alias,
        model=profile.model,
        maximum_output_tokens=profile.maximum_output_tokens,
        evidence_tool_budget=2,
        prompt_set=DEFAULT_AGENTIC_JUDGE,
    )
    definition = base.episode_judge.definition.model_copy(update={"profile": agentic})
    episode_judge = base.episode_judge.model_copy(update={"definition": definition})
    deep = base.model_copy(update={"episode_judge": episode_judge})
    assert judge_model_calls(agentic) == 2 + 4
    assert actor_critic_costs(project, deep)[-1].attempts == 6 * judge * judged
    assert f"{judged} episodes x 6 judge requests" in " ".join(actor_critic_lines(deep))


def test_version_2_schedule_rollouts_price_every_point_option_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Options for J * K points, every stance, a reconcile and each continuation, per role."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    base = critic_fixture._spec(project, job, judges=2)
    plain = TrainingScheduleSpec(name="plain", base=base)
    assert point_option_costs(project, plain, base) == ()
    options = PointOptionsSpec(proposer=base.judges[0], classifier=base.critics[0])
    scored = plain.model_copy(update={"point_options": options})
    rows = point_option_costs(project, scored, base)
    draws = len(base.judges) * base.points * options.profile.budget.model_calls
    draws *= len(options.stances) + 1 + base.continuations
    attempts = base.judges[0].maximum_physical_attempts
    assert [row.attempts for row in rows] == [draws * attempts, draws * attempts]


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


def _variance_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, Path, Path, build_fixture.DecisionWire]:
    """Complete a build, write a variance-build command and route decisions to a fake wire.

    Args:
        tmp_path: Directory holding the build root and the command.
        monkeypatch: Fixture keeping the build local and the decision wire scripted.

    Returns:
        The project, its root, the command file and the decision wire.
    """
    project = build_fixture.variance_project(tmp_path, monkeypatch)
    wire = build_fixture.DecisionWire()
    monkeypatch.setattr("exp.optimize.evaluation.variance.build.HttpxJsonTransport", lambda: wire)
    action = VarianceBuildAction(
        spec=_write(
            tmp_path / "variance-spec.json", VarianceBuildSpec(task_bank=build_fixture.TASK_BANK)
        )
    )
    request = _write(
        tmp_path / "variance.json",
        ContinuationCommand(action="variance-build", variance_build=action, spend=UNPRICED_SPEND),
    )
    return project, tmp_path / ".exp", request, wire


def _variance_command(root: Path, request: Path, *extra: str) -> list[str]:
    """The public ``exp eval`` command for the variance build of the fixture project."""
    return [
        "eval",
        build_fixture.PROJECT,
        "--root",
        str(root),
        "--continuations",
        str(request),
        *extra,
    ]


def test_variance_build_dry_run_prints_calls_and_reuse_and_makes_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run shows traces per role, decision calls per view and embedding reuse."""
    project, root, request, wire = _variance_fixture(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, _variance_command(root, request, "--dry-run"))

    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())
    assert "final_evaluation is never tagged" in output
    assert "x ceil(3/32) =" in output and "decision calls" in output
    assert "Embedding reuse:" in output and "no embedding call" in output
    assert not wire.states
    assert not project.artifacts.list_ids(artifact_type=VARIANCE_REPORT_TYPE)


def test_variance_build_refused_consent_stops_before_any_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Declining the estimate runs no stage and sends nothing."""
    project, root, request, wire = _variance_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuation_context.require_spend_consent", lambda *a, **k: False
    )
    monkeypatch.setattr("exp.cli.evaluation.variance_actions.run_variance_build", _forbid)

    result = CliRunner().invoke(app, _variance_command(root, request, "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert not wire.states
    assert not project.artifacts.list_ids(artifact_type=VARIANCE_REPORT_TYPE)


def test_variance_build_runs_over_the_catalog_alias_then_replays_without_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consented run tags through the decision alias, writes the exposure file and replays."""
    project, root, request, wire = _variance_fixture(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, _variance_command(root, request, "--yes", "--non-interactive"))

    assert result.exit_code == 0, result.output
    assert wire.states
    assert '"tag_retrieval"' in result.output and '"world_model"' in result.output
    (report_id,) = project.artifacts.list_ids(artifact_type=VARIANCE_REPORT_TYPE)
    identity = report_id.removesuffix("-result")
    exposure = project.paths.runtime_directory / "variance-build" / identity / "exposure.json"
    plan = ExposurePlan.model_validate_json(exposure.read_bytes())
    assert "final_evaluation" in plan.roles.values()
    calls = len(wire.states)

    monkeypatch.setattr("exp.cli.evaluation.continuation_context.require_spend_consent", _forbid)
    replay = runner.invoke(app, _variance_command(root, request, "--non-interactive"))

    assert replay.exit_code == 0, replay.output
    assert len(wire.states) == calls
    assert project.artifacts.list_ids(artifact_type=VARIANCE_REPORT_TYPE) == (report_id,)
    assert replay.output[replay.output.index("{") :] == result.output[result.output.index("{") :]


def _schedule_context(
    root: Path, *, dry_run: bool = False, yes: bool = True
) -> tuple[ContinuationContext, StringIO]:
    """A direct handler context over the project at ``root``."""
    output = StringIO()
    context = ContinuationContext(
        store=ProjectStore(root, "case"),
        console=Console(file=output, width=200),
        revision="test",
        root=root,
        project="case",
        spec_path=root / "operation.json",
        yes=yes,
        non_interactive=True,
        dry_run=dry_run,
        operation="operation-fixture",
        spend=UNPRICED_SPEND,
    )
    return context, output


def _schedule_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, TrainingScheduleSpec, Path, Path]:
    """Write a one-warm-up schedule spec and the mining-probe exposure file."""
    project, spec = schedule_fixture._schedule(
        tmp_path, monkeypatch, steady_draws=1, transition=TransitionRule(warmup_rollouts=1)
    )
    exposure = critic_fixture._exposure(project, spec.base.initial, "mining_probes")
    return (
        project,
        spec,
        _write(tmp_path / "schedule.json", spec),
        _write(tmp_path / "exposure.json", exposure),
    )


def test_variance_schedule_status_and_dry_run_plan_save_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status folds saved events and a dry-run plan previews without saving or calling."""
    project, spec, path, exposure = _schedule_files(tmp_path, monkeypatch)
    monkeypatch.setattr("exp.cli.evaluation.variance_actions.run_actor_critic", _forbid)
    context, output = _schedule_context(tmp_path, dry_run=True)

    handle_variance_schedule(context, VarianceScheduleAction(operation="status", spec=path))
    handle_variance_schedule(
        context,
        VarianceScheduleAction(operation="plan", spec=path, execute=True, exposure=exposure),
    )

    text = output.getvalue()
    assert '"phase": "warmup"' in text and '"kind": "warmup"' in text
    assert '"selector_draws": 2' in text and "no provider calls" in text
    assert not project.artifacts.exists(spec.identity)
    with pytest.raises(ValidationError, match="exposure"):
        VarianceScheduleAction(operation="status", spec=path, execute=True)
    with pytest.raises(ValidationError, match="revision file"):
        VarianceScheduleAction(operation="register-revision", spec=path)


def test_variance_schedule_declined_consent_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decision is saved, but a declined estimate buys nothing and completes nothing."""
    project, spec, path, exposure = _schedule_files(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "exp.cli.evaluation.continuation_context.require_spend_consent", lambda *a, **k: False
    )
    monkeypatch.setattr("exp.cli.evaluation.variance_actions.run_actor_critic", _forbid)
    context, _ = _schedule_context(tmp_path)

    handle_variance_schedule(
        context,
        VarianceScheduleAction(operation="plan", spec=path, execute=True, exposure=exposure),
    )

    state = schedule_state(project, spec)
    assert state.pending is not None and state.completed == ()
    assert not project.artifacts.list_ids(artifact_type="actor-critic-report")


def test_variance_schedule_executes_role_and_actor_rollouts_and_registers_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consented role and actor-only rollouts run and are recorded; a revision opens epoch 1."""
    project, spec, path, exposure = _schedule_files(tmp_path, monkeypatch)
    context, output = _schedule_context(tmp_path)
    plan = VarianceScheduleAction(operation="plan", spec=path, execute=True, exposure=exposure)
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
    choose = critic_fixture._tool_reply(
        "choose_points",
        {"points": [critic_fixture._point(0), critic_fixture._point(1)], "limitations": ["x"]},
    )
    estimate = critic_fixture._tool_reply("estimate_points", critic_fixture._estimates([0, 1]))
    for actor, roles in (
        (group + suffixes, [choose, choose, estimate]),
        (group + suffixes, [choose, estimate]),
        (group, []),
    ):
        critic_fixture._wire(monkeypatch, actor, roles)
        with operation_spend_scope():
            handle_variance_schedule(context, plan)

    state = schedule_state(project, spec)
    assert [row.plan.kind for row in state.completed] == ["warmup", "steady_roles", "actor_only"]
    assert state.completed[2].metrics.actor_scored == 3
    assert '"training": "records only' in output.getvalue()

    selector = spec.base.judges[0]
    trained = selector.model_copy(
        update={"model": selector.model.model_copy(update={"model_id": "selector-r1"})}
    )
    revision = _write(
        tmp_path / "revision.json",
        RevisionRegistration(role="selector", settings=trained, source="external-trainer"),
    )
    register = VarianceScheduleAction(operation="register-revision", spec=path, revision=revision)
    handle_variance_schedule(_schedule_context(tmp_path, dry_run=True)[0], register)
    assert schedule_state(project, spec).epoch == 0
    handle_variance_schedule(context, register)
    assert schedule_state(project, spec).epoch == 1


@pytest.mark.parametrize("approved", [False, True])
def test_saved_schedule_report_requires_consent_for_pending_point_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, approved: bool
) -> None:
    """Resuming after role evidence saved still governs unsaved option completion calls."""
    project, base = schedule_fixture._schedule(tmp_path, monkeypatch)
    spec = base.model_copy(
        update={
            "objective": SelectorObjectiveV2(),
            "point_options": PointOptionsSpec(
                proposer=base.base.judges[0],
                classifier=base.base.critics[0],
                profile=FORCED,
                stances=("action_clustering",),
            ),
        }
    )
    pending = next_rollout_plan(project, spec, code_revision="test")
    schedule_fixture._run_roles(project, monkeypatch, pending)
    path = _write(tmp_path / "schedule.json", spec)
    exposure = _write(
        tmp_path / "exposure.json",
        critic_fixture._exposure(project, pending.actor_job, "mining_probes"),
    )
    wire = ScriptedJsonTransport(
        tuple(_proposal({"decision": "Choose a row", "options": []}) for _ in range(4))
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    consent: list[bool] = []

    def confirm(*args: object, **kwargs: object) -> bool:
        """Return the explicit invocation decision and count consent requests."""
        consent.append(approved)
        return approved

    monkeypatch.setattr("exp.cli.evaluation.continuation_context.require_spend_consent", confirm)
    context, _ = _schedule_context(tmp_path, yes=False)
    with operation_spend_scope():
        assert current_spend_ceiling() is None
        handle_variance_schedule(
            context,
            VarianceScheduleAction(
                operation="plan",
                spec=path,
                execute=True,
                exposure=exposure,
            ),
        )
        assert consent == [approved]
        assert (current_spend_ceiling() is not None) == approved
    assert bool(wire.requests) == approved
    assert len(schedule_state(project, spec).completed) == int(approved)

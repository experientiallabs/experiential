"""Explicit repair retains original scores and recovers a publication without redrawing."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner, Result

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import ArtifactEnvelope, JsonValue
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.audit.panel import plan_held_out_panel
from exp.optimize.evaluation.audit.panel_test import REVISION, build_project, episode
from exp.optimize.evaluation.fidelity.memory import (
    refuse_ineligible_memory,
    saved_memory_receipt,
    trajectory_record,
)
from exp.optimize.evaluation.fidelity.runner import FidelitySpec, PreparedFidelity
from exp.optimize.evaluation.learning import repair as learning_repair
from exp.optimize.evaluation.learning import retirement, traces
from exp.optimize.evaluation.learning.repair import LearningRepairSpec, repair_learning
from exp.optimize.evaluation.learning.retirement import retirement_of
from exp.optimize.evaluation.learning.retirement_test import OUTAGE, learn, refusing_project
from exp.optimize.evaluation.learning.traces import (
    _saved_attempts,
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
)
from exp.optimize.evaluation.tests.replay_retirement_test import RedrawnPrediction, now
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    memory_catalog,
    memory_head,
    memory_version_id,
    publish_trajectory,
)


def failed_run(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, PreparedFidelity, RedrawnPrediction, LearningRepairSpec]:
    """Leave a saved malformed investigation while unrelated learning advances memory."""
    episodes = [episode(i) for i in range(4)]
    provider = RedrawnPrediction(episodes)
    provider.failures = 1
    _, project, _ = build_project(root, monkeypatch, provider, episodes=episodes)
    outcome = learn_from_traces(
        project, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
    )
    assert outcome.incomplete == 1 and outcome.head_version == 3
    case = learning_cases(project, "all")[0]
    prepared = _saved_attempts(project, learning_plan(project), case)[0][0]
    head = memory_head(project.artifacts, outcome.plan.lineage)
    assert head is not None
    spec = LearningRepairSpec(
        run=artifact_input(project.artifacts.read(prepared.id).manifest),
        report=artifact_input(project.artifacts.read(prepared.id + "-report").manifest),
        parent=artifact_input(
            project.artifacts.read(memory_version_id(head.lineage, head.version)).manifest
        ),
        reason="Retain disclosed observations; abstain from the failed hypothesis.",
    )
    return project, prepared, provider, spec


def test_repair_advances_named_head_without_changing_any_prediction_or_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original miss remains scored while the unconsumed trace joins memory v4."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    store = project.artifacts
    original = store.read_bytes(prepared.id + "-report", "report.json")
    first_receipt = store.read_bytes(prepared.id + "-memory", "memory.json")
    before_calls = len(provider.requests)
    before_curve = learning_curve(project)
    before_ids = set(store.list_ids())
    assert (
        repair_learning(project, spec, code_revision=REVISION, dry_run=True)["provider_calls"] == 0
    )
    assert set(store.list_ids()) == before_ids
    result = repair_learning(project, spec, code_revision=REVISION)
    assert result["published"] is True and result["version"] == 4
    assert result["original_parent_version"] == 0 and result["parent_version"] == 3
    assert repair_learning(project, spec, code_revision=REVISION) == result
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["repair"] == result["repair"]
    assert len(provider.requests) == before_calls and provider.draws == 1
    assert store.read_bytes(prepared.id + "-report", "report.json") == original
    assert store.read_bytes(prepared.id + "-memory", "memory.json") == first_receipt
    curve = learning_curve(project)
    assert curve is not None and before_curve is not None
    assert [(p.exact, p.steps) for p in curve.points] == [
        (p.exact, p.steps) for p in before_curve.points
    ]
    assert curve.incomplete == 0
    with pytest.raises(ValueError, match="different disposition"):
        repair_learning(
            project, spec.model_copy(update={"reason": "Different reason"}), code_revision=REVISION
        )


@pytest.mark.parametrize(
    ("source", "when"),
    [("report", "before"), ("run", "before"), ("run", "dry_run"), ("report", "after")],
)
def test_repair_refuses_excluded_evidence_and_cites_what_it_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, when: str
) -> None:
    """Excluded evidence never enters repaired memory, and later ancestry checks reach it."""
    project, _, _, spec = failed_run(tmp_path, monkeypatch)
    store = project.artifacts
    audit = store.write_json(
        artifact_id="qualification-audit",
        artifact_type="qualification-audit",
        envelope=ArtifactEnvelope(schema_version=1, created_at=now(), code_revision="test"),
        files={"audit.json": {"reason": "The run's evidence was audited unsafe"}},
    )

    def exclude() -> None:
        """Exclude the run or its exact report from qualification."""
        exclude_source_from_qualification(
            store,
            spec.run if source == "run" else spec.report,
            audit=artifact_input(audit),
            rationale="Audited unsafe evidence",
            code_revision="test",
        )

    if when != "after":
        exclude()
        before = set(store.list_ids())
        with pytest.raises(ValueError, match="excluded from qualification"):
            repair_learning(project, spec, code_revision=REVISION, dry_run=when == "dry_run")
        assert set(store.list_ids()) == before
        return
    lineage = str(repair_learning(project, spec, code_revision=REVISION)["lineage"])
    head = memory_head(store, lineage)
    assert head is not None and {spec.run, spec.report} <= set(head.inputs)
    exclude()
    with pytest.raises(ValueError, match="excluded from qualification"):
        refuse_ineligible_memory(project, lineage)


def test_a_version_published_without_its_evidence_is_never_recovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repair interrupted after an uncited publication fails closed instead of recovering."""
    project, _, _, spec = failed_run(tmp_path, monkeypatch)
    original_write = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Simulate process loss precisely after the memory owner published."""
        if artifact_type == "fidelity-learning-repair-result":
            raise KeyboardInterrupt("loss before receipt")
        return original_write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    uncited = Mock(
        side_effect=lambda *args, evidence, **kwargs: publish_trajectory(*args, **kwargs)
    )
    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", crash)
        patch.setattr(learning_repair, "publish_trajectory", uncited)
        with pytest.raises(KeyboardInterrupt):
            repair_learning(project, spec, code_revision=REVISION)
    assert uncited.call_count == 1
    with pytest.raises(ValueError, match="published without citing"):
        repair_learning(project, spec, code_revision=REVISION)


def test_stale_parent_refuses_before_writing_and_explicit_repair_keeps_unknowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old empty parent cannot overwrite the advanced bank or silently rebase."""
    project, _, provider, spec = failed_run(tmp_path, monkeypatch)
    before = set(project.artifacts.list_ids())
    calls = len(provider.requests)
    with pytest.raises(ValueError, match="memory advanced"):
        repair_learning(project, spec.model_copy(update={"parent": None}), code_revision=REVISION)
    assert set(project.artifacts.list_ids()) == before and len(provider.requests) == calls


def test_crash_after_publication_recovers_same_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing repair receipt does not publish a second version after restart."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    original_write = project.artifacts.write_json

    def crash(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Simulate process loss precisely after the memory owner published."""
        if artifact_type == "fidelity-learning-repair-result":
            raise KeyboardInterrupt("loss before receipt")
        return original_write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", crash)
        with pytest.raises(KeyboardInterrupt):
            repair_learning(project, spec, code_revision=REVISION)
    calls = len(provider.requests)
    result = repair_learning(project, spec, code_revision=REVISION)
    assert result["version"] == 4 and result["published"] is True
    assert len(provider.requests) == calls and provider.draws == 1
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["version"] == 4


def test_public_cli_repairs_then_replays_without_catalog_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The explicit operation publishes no replacement score and needs no provider setup."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    command = tmp_path / "repair.json"
    command.write_text(
        ContinuationCommand(action="repair-learning", learning_repair=spec).model_dump_json()
    )
    args = [
        "eval",
        project.paths.project_id,
        "--root",
        str(project.paths.root),
        "--continuations",
        str(command),
    ]
    calls = len(provider.requests)
    for suffix in (["--dry-run"], ["--yes", "--non-interactive"], []):
        result = CliRunner().invoke(app, args + suffix)
        assert result.exit_code == 0, (result.output, result.exception)
    assert len(provider.requests) == calls
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["version"] == 4


def test_an_advanced_head_after_interruption_needs_a_new_explicit_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpublished attempt can be superseded without overwriting its registered intent."""
    project, prepared, provider, spec = failed_run(tmp_path, monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(
            learning_repair, "publish_trajectory", Mock(side_effect=ValueError("head raced"))
        )
        with pytest.raises(ValueError, match="head raced"):
            repair_learning(project, spec, code_revision=REVISION)
    assert prepared.spec.memory is not None
    lineage = prepared.spec.memory.lineage
    parent = memory_head(project.artifacts, lineage)
    different = {
        **trajectory_record(prepared.spec),
        "task_lineage": "unrelated-source",
        "task": "Another observed task",
    }
    head, consumed = publish_trajectory(
        project.artifacts,
        lineage=lineage,
        parent=parent,
        world_model=prepared.spec.world_model,
        selection_policy_sha256=str(prepared.spec.memory_selection_identity),
        trajectory=different,
        research=(),
        created_at=now(),
        code_revision=REVISION,
        publisher="other-run",
    )
    assert consumed and head.version == 4
    with pytest.raises(ValueError, match="memory advanced"):
        repair_learning(project, spec, code_revision=REVISION)
    current = artifact_input(project.artifacts.read(memory_version_id(lineage, 4)).manifest)
    calls = len(provider.requests)
    result = repair_learning(
        project, spec.model_copy(update={"parent": current}), code_revision=REVISION
    )
    assert result["version"] == 5 and len(provider.requests) == calls
    assert len(project.artifacts.list_ids(artifact_type="fidelity-learning-repair")) == 2


def test_observation_only_repair_never_admits_candidate_text_as_original_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate-only token cannot bypass a hidden source through fabricated research."""
    project, prepared, _, spec = failed_run(tmp_path, monkeypatch)
    report = project.artifacts.read_bytes(prepared.id + "-report", "report.json")
    assert b"first draw missed" in project.artifacts.read_bytes(
        prepared.id + "-report-intent", "evidence.json"
    )
    repair_learning(project, spec, code_revision=REVISION)
    assert prepared.spec.memory is not None
    head = memory_head(project.artifacts, prepared.spec.memory.lineage)
    records, _ = memory_catalog(
        project.artifacts,
        head,
        eligibility=Eligibility(hidden_task_lineage="hidden-prediction-source", strict=True),
    )
    assert "first draw missed" not in json.dumps(records)
    observations = [row for row in records.values() if row.get("kind") == "original-observation"]
    assert len(observations) == 12
    assert project.artifacts.read_bytes(prepared.id + "-report", "report.json") == report


def test_an_operator_retires_a_stuck_trace_and_learning_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace an outage keeps blocking is retired by name, recorded, and skipped thereafter.

    The retirement makes no call and edits no memory. It refuses a learned trace and a held-out
    trace, and the audit checkpoint and learning curve both name and count it as not learned.
    """
    store, provider, trace_id = refusing_project(tmp_path, monkeypatch, OUTAGE)
    with pytest.raises(ProviderTransportError):
        learn(store)
    lineage = learning_plan(store, scope="fit").lineage
    head = memory_head(store.artifacts, lineage)
    learned_id = learning_cases(store, "fit")[0].trace_id
    held_id = next(
        case.trace_id
        for case in learning_cases(store, "all")
        if case.trace_id not in {row.trace_id for row in learning_cases(store, "fit")}
    )

    def retire(trace: str, *suffix: str) -> Result:
        """Run the public repair-learning retirement for one trace."""
        command = tmp_path / "retire.json"
        command.write_text(
            json.dumps(
                {
                    "action": "repair-learning",
                    "learning_repair": {
                        "resolution": "retire-trace",
                        "trace_id": trace,
                        "reason": "gateway outage on this trace only; investigated separately",
                    },
                }
            )
        )
        args = ["eval", store.paths.project_id, "--root", str(store.paths.root)]
        return CliRunner().invoke(app, [*args, "--continuations", str(command), *suffix])

    calls = len(provider.requests)
    for refused, why in ((learned_id, "already learned"), (held_id, "held-out traces")):
        result = retire(refused, "--non-interactive")
        assert result.exit_code != 0 and why in " ".join(result.output.split()), result.output
    dry = retire(trace_id, "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert retirement_of(store, lineage, trace_id) is None, "a dry run writes nothing"
    for _ in range(2):  # the second identical retirement replays
        result = retire(trace_id, "--non-interactive")
        assert result.exit_code == 0, (result.output, result.exception)
    assert len(provider.requests) == calls, "retirement makes no model call"
    assert memory_head(store.artifacts, lineage) == head, "retirement never edits memory"
    retirement = retirement_of(store, lineage, trace_id)
    assert retirement is not None and retirement.cause == "operator"
    assert retirement.run is not None and retirement.refusals == ()
    provider.task_id = None  # the outage would end, but the trace is never sent again
    outcome = learn(store, retry_incomplete=True)
    cases = learning_cases(store, "fit")
    assert (outcome.learned, outcome.retired, outcome.stopped) == (len(cases) - 1, 1, None)
    assert learning_plan(store, scope="fit").lineage == lineage
    curve = learning_curve(store, "fit")
    assert curve is not None and curve.retired == 1 and len(curve.points) == len(cases) - 1
    planned = plan_held_out_panel(
        store, 3, profile="memory", code_revision=REVISION, clock=lambda: datetime.now(UTC)
    )
    assert planned.checkpoint is not None
    (row,) = planned.checkpoint.retired
    assert (row.trace_id, row.cause) == (trace_id, "operator")
    assert planned.checkpoint.fit_traces_learned == len(cases) - 1


def test_an_operator_cannot_retire_a_trace_a_learning_pass_has_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retirement arriving after the pass checked retirement, before it ran, is refused.

    Otherwise the retirement is recorded and the pass then learns the trace anyway: the trace
    ends up both learned and retired, and every learning report drops its score.
    """
    store, provider, trace_id = refusing_project(tmp_path, monkeypatch, OUTAGE)
    with pytest.raises(ProviderTransportError):
        learn(store)
    provider.task_id = None  # the outage ends: the next pass learns the trace
    refusals: list[str] = []
    checked = traces.require_unlearned

    def operator_retires(
        project: ProjectStore, lineage: str, spec: FidelitySpec, earlier: set[str]
    ) -> None:
        """The pass's unlearned check of the resumed trace, then a racing operator retirement."""
        checked(project, lineage, spec, earlier)
        if refusals:
            return  # later traces of the pass
        request = learning_repair.TraceRetirementRequest(
            resolution="retire-trace", trace_id=trace_id, reason="racing operator"
        )
        with pytest.raises(
            ValueError, match="being learned or retired by another process"
        ) as refused:
            learning_repair.retire_trace(project, request, code_revision=REVISION)
        refusals.append(str(refused.value))

    monkeypatch.setattr(traces, "require_unlearned", operator_retires)
    outcome = learn(store, retry_incomplete=True)

    assert len(refusals) == 1 and outcome.retired == 0
    assert outcome.learned == len(learning_cases(store, "fit"))
    lineage = learning_plan(store, scope="fit").lineage
    assert retirement_of(store, lineage, trace_id) is None


def test_a_pass_meeting_a_held_trace_stops_with_an_operator_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contention is a refusal naming the trace, not a raw lock timeout taken as an outage."""
    store, provider, trace_id = refusing_project(tmp_path, monkeypatch, OUTAGE)
    with pytest.raises(ProviderTransportError):
        learn(store)
    provider.task_id = None
    lineage = learning_plan(store, scope="fit").lineage

    with (
        retirement.trace_held(store, lineage, trace_id),
        pytest.raises(ValueError, match=f"trace {trace_id} is being learned or retired"),
    ):
        learn(store, retry_incomplete=True)

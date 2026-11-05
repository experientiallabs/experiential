"""Admission order and real committed-owner completion at the native pipeline boundary."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.council_prelude_test import (
    _prelude_fixture,
    _register_fixture,
)
from exp.optimize.evaluation.variance.pipeline import _execute
from exp.optimize.evaluation.variance.pipeline_downstream import downstream_inventory
from exp.optimize.evaluation.variance.pipeline_spec import (
    CostLine,
    DecisionVarianceSpec,
    FamiliesOutputs,
    StageContext,
    StageDependencies,
    StagePlan,
    StageRecord,
    StageRun,
)


class _Admission:
    """Observe the real owner boundary without granting any provider or runtime authority."""

    def __init__(self, events: list[str], *, refuse: bool = False) -> None:
        """Keep one test's ordered events and optional pre-dispatch refusal."""
        self.events, self.refuse = events, refuse

    @contextmanager
    def __call__(self, context: StageContext) -> Iterator[None]:
        """Refuse before the stage or hold its scope through committed-owner completion."""
        if self.refuse:
            raise ValueError("native admission refused")
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")

    def completed(self, context: StageContext, owner: ArtifactInput) -> None:
        """Require the pipeline owner's real persisted record, including during replay."""
        stored = context.project.artifacts.read(owner.artifact_id)
        assert artifact_input(stored.manifest) == owner
        assert stored.manifest.code_revision == context.code_revision
        record = StageRecord.model_validate_json(stored.payloads["record.json"])
        assert context.plan is not None
        assert record.identity == context.identity and record.stage == context.plan.stage
        self.events.append("completed")


class _Stage:
    """Supply an explicit empty fixture result to exercise only the commit boundary."""

    def __init__(self, events: list[str]) -> None:
        """Record native executor invocation order."""
        self.events = events

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """No external fixture inputs contribute to this boundary check."""
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """The boundary fixture dispatches nothing."""
        return (), ()

    def run(self, context: StageContext) -> StageRun:
        """Return fixture output; this makes no task-quality or qualification claim."""
        self.events.append("run")
        return StageRun(FamiliesOutputs(families=()), (), {})


def test_stage_admission_precedes_execution_and_completion_uses_real_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusal prevents dispatch; success commits before completion; replay never runs again."""
    project, prepared, resume, contract, _ = _prelude_fixture(tmp_path, monkeypatch)
    registration = _register_fixture(project, prepared, resume, contract)
    spec = resolve_bootstrap(project, contract.pipeline)
    events: list[str] = []
    stage = _Stage(events)
    row = StagePlan(stage="families", identity="native-stage-hook-boundary", saved=False)
    with pytest.raises(ValueError, match="native admission refused"):
        _execute(
            project,
            spec,
            row,
            stage,
            {},
            [],
            "council-executor",
            StageDependencies(stage_admission=_Admission(events, refuse=True)),
        )
    assert events == [] and not project.artifacts.exists(row.identity + "-stage")
    dependencies = StageDependencies(stage_admission=_Admission(events))
    result = _execute(project, spec, row, stage, {}, [], "council-executor", dependencies)
    assert events == ["enter", "run", "completed", "exit"]
    events.clear()
    assert _execute(project, spec, row, stage, {}, [], "council-executor", dependencies) == result
    assert events == ["completed"]
    context = StageContext(project, spec, row.identity, "council-executor", {}, dependencies, row)
    with pytest.raises(ValueError, match="every exact native upstream owner"):
        downstream_inventory(context, registration.qualification)

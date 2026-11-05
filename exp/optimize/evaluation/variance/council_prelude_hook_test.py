"""Actual configured pipeline owners execute inside their native prelude admission seam."""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.pipeline_spec import StageContext
from exp.optimize.evaluation.variance.pipeline_test import TIER_B, make_spec, wired


def test_real_pipeline_composes_trace_groups_labels_and_authored_rubric_under_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record real native calls at scope entry/exit, not only a mocked collaborator setter."""
    run = wired(tmp_path, monkeypatch, agentic=True)
    active: ContextVar[str | None] = ContextVar("offline-prelude-phase", default=None)
    events: list[tuple[str, str | None, int]] = []
    observed: list[CouncilPreludeCheckpoint] = []

    @contextmanager
    def scope(checkpoint: CouncilPreludeCheckpoint) -> Iterator[None]:
        """One admission covers the actual owner and restores scope for the following phase."""
        assert active.get() is None
        token = active.set(checkpoint.phase)
        before = sum(run.calls())
        observed.append(checkpoint)
        try:
            yield
        finally:
            events.append((checkpoint.phase, checkpoint.actor_alias, sum(run.calls()) - before))
            active.reset(token)

    def admission(
        context: StageContext, checkpoint: CouncilPreludeCheckpoint
    ) -> AbstractContextManager[object]:
        """Use the actual pipeline composition inputs to validate native owner identity."""
        assert context.project == run.project and checkpoint.job.actor_alias in (
            "actor",
            TIER_B,
        )
        if checkpoint.phase == "traceml_labels":
            assert checkpoint.group is not None
            assert run.project.artifacts.exists(checkpoint.group.artifact_id)
        if checkpoint.phase == "behavior_reference":
            assert checkpoint.judge is not None
        return scope(checkpoint)

    run.dependencies = replace(run.dependencies, council_prelude_admission=admission)
    spec = make_spec(run.fixture, run.rubric, run.bank, agentic=True)
    report = run.run(spec)
    assert report.complete, report.stages
    assert active.get() is None
    phases = [phase for phase, _, _ in events]
    assert phases.count("traceml_group") == phases.count("traceml_labels") == 2
    assert phases.count("mining_rubric") == 1
    assert phases.count("behavior_reference") == 2
    assert all(calls > 0 for phase, _, calls in events if phase != "behavior_reference")
    before = (run.calls(), len(events))
    replay = run.run(spec, approve=False)
    assert replay.complete and (run.calls(), len(events)) == before

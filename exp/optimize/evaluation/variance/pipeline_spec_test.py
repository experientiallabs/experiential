"""Offline tests of the decision-variance stage graph contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from exp.optimize.evaluation.variance.pipeline_spec import (
    COUNCIL_ACTORS,
    STAGES,
    CostLine,
    DecisionVarianceSpec,
    MiningSettings,
    PipelinePlan,
    StagePlan,
    TraceMLSettings,
    stage_identity,
)
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired


def test_the_council_is_five_open_weight_tiers_in_stage_order() -> None:
    """The default actors are the five council actors, and the graph has six ordered stages."""
    assert COUNCIL_ACTORS == (
        "gemma-4-26b-a4b-it",
        "glm-5.3-flash",
        "glm-5.3",
        "kimi-k3",
        "mimo-v2.6-pro",
    )
    assert STAGES[0] == "variance-build" and STAGES[-1] == "records" and len(STAGES) == 6


def test_identities_chain_so_a_changed_slice_rekeys_only_later_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the records slice keeps every earlier identity; changing mining re-keys three."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)

    def chain(spec: DecisionVarianceSpec) -> list[str]:
        """Every stage identity of a spec, chained in order."""
        identities: list[str] = []
        for stage in STAGES:
            identities.append(stage_identity(stage, spec, tuple(identities), "test"))
        return identities

    original = chain(base)
    records = chain(
        base.model_copy(update={"records": base.records.model_copy(update={"harbor": False})})
    )
    mining = chain(base.model_copy(update={"mining": base.mining.model_copy(update={"seed": 1})}))
    assert records[:5] == original[:5] and records[5] != original[5]
    assert mining[:3] == original[:3]
    assert all(a != b for a, b in zip(mining[3:], original[3:], strict=True))


def test_duplicate_or_invalid_tiers_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each actor is its own tier, named by a lowercase tier label."""
    run = wired(tmp_path, monkeypatch)
    base = make_spec(run.fixture, run.rubric, run.bank)
    data = base.mining.model_dump(mode="json")
    with pytest.raises(ValueError, match="distinct"):
        MiningSettings.model_validate({**data, "actors": ["actor", "actor"]})
    with pytest.raises(ValueError, match="tier labels"):
        MiningSettings.model_validate({**data, "actors": ["Actor"]})


def test_a_plan_totals_only_unsaved_stages_and_keeps_unpriced_lines_visible() -> None:
    """Saved stages cost nothing; unpriced components stay listed beside the priced bound."""
    plan = PipelinePlan(
        spec_sha256="a" * 64,
        code_revision="test",
        stages=(
            StagePlan(
                stage="rules",
                identity="r",
                saved=True,
                costs=(CostLine(name="author", attempts=3, usd=9.0),),
            ),
            StagePlan(
                stage="records",
                identity="s",
                saved=False,
                costs=(
                    CostLine(name="judge", attempts=2, usd=1.5),
                    CostLine(name="actor", attempts=4, usd=None),
                ),
            ),
        ),
    )
    assert plan.known_usd == 1.5 and not plan.complete
    assert [row.name for row in plan.unpriced] == ["actor"]


def test_an_enabled_traceml_stage_names_its_move_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No rule stands in for the pivot and return judgment, so the judge is required."""
    run = wired(tmp_path, monkeypatch)
    settings = run.fixture.settings
    with pytest.raises(ValueError, match="move_judge"):
        TraceMLSettings()
    with pytest.raises(ValueError, match="unknown traceml move variants"):
        TraceMLSettings(move_judge=settings, move_variants=("tool-change-rules",))
    assert TraceMLSettings(enabled=False).move_judge is None
    narrowed = TraceMLSettings(move_judge=settings, move_variants=("evidence-first",))
    spec = make_spec(run.fixture, run.rubric, run.bank)
    changed = spec.model_copy(update={"traceml": narrowed})
    assert spec.slice("traceml") != changed.slice("traceml")

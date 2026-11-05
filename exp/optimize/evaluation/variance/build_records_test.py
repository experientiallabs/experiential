"""Saved variance-build artifacts are content-addressed, adopted when identical and pinned."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectConfig, ProjectStore
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.variance import build as build_module
from exp.optimize.evaluation.variance import build_records as records
from exp.optimize.evaluation.variance.build_records import (
    EXPOSURE_TYPE,
    MATRIX_TYPE,
    ScopeRule,
    VarianceBuildPlan,
    corpus_scope,
    load_exposure_plan,
    load_tag_matrix,
    probe_selection,
    save_artifact,
    trace_admissions,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.runtime.gateway.decisions_contracts import DecisionRequest, NoulQuestion
from exp.simulation.mining.decisions_test import _SNAPSHOT, _bank, _unit
from exp.simulation.mining.questions import Question
from exp.simulation.mining.tag_clusters import (
    AdmittedTrace,
    ClusterAdmission,
    ClusterAdmissionSpec,
)
from exp.simulation.mining.tag_selection import TagRule
from exp.simulation.mining.tagging import (
    DecisionCache,
    TraceTaggingError,
    decision_key,
    question_chunks,
)
from exp.simulation.mining.views import Unit
from exp.simulation.retrieval.tags import TagView


def _project(root: Path) -> ProjectStore:
    """Initialize an empty project."""
    project = ProjectStore(root, "records")
    project.initialize(ProjectConfig(project_id="records"))
    return project


def _plan(seed: int) -> ExposurePlan:
    """A two-lineage exposure plan."""
    return ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=seed,
        probe_fraction=0.5,
        roles={"a": "world_learning", "b": "mining_probes"},
        traces={"t-a": "a", "t-b": "b"},
    )


def test_an_identical_save_adopts_the_saved_artifact(tmp_path: Path) -> None:
    """Saving the same content twice returns one reference and loads it back."""
    project = _project(tmp_path)
    files = {"exposure.json": _plan(0)}

    first = save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test")

    assert save_artifact(project.artifacts, EXPOSURE_TYPE, files, (), "test") == first
    assert load_exposure_plan(project, first) == _plan(0)
    other = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(1)}, (), "t")
    assert other != first
    assert len(project.artifacts.list_ids(artifact_type=EXPOSURE_TYPE)) == 2


def test_a_reader_refuses_another_artifact_type(tmp_path: Path) -> None:
    """A reference is read only as the artifact type it was saved as."""
    project = _project(tmp_path)
    saved = save_artifact(project.artifacts, EXPOSURE_TYPE, {"exposure.json": _plan(0)}, (), "t")

    with pytest.raises(ValueError, match=f"pinned {MATRIX_TYPE}"):
        load_tag_matrix(project, saved)


def _admission(traces: tuple[str, ...]) -> ClusterAdmission:
    """An admission of the given traces from one cluster."""
    return ClusterAdmission(
        spec=ClusterAdmissionSpec(),
        clustering="tag-clusters-x",
        admitted=tuple(
            AdmittedTrace(
                trace_id=trace, cluster_id="c00", confidence=0.9, rank=index + 1, reason="c00 why"
            )
            for index, trace in enumerate(traces)
        ),
        quotas=(),
    )


def test_corpora_come_from_a_rule_else_admission_else_the_role() -> None:
    """Admission narrows a corpus; an admitted trace outside world learning is refused."""
    plan = _plan(0).model_copy(
        update={
            "roles": {"a": "world_learning", "b": "mining_probes", "c": "world_learning"},
            "traces": {"t-a": "a", "t-b": "b", "t-c": "c"},
        }
    )
    assert corpus_scope(plan, "construction", None, None).ids == ("t-a", "t-c")
    admitted = corpus_scope(plan, "construction", None, _admission(("t-c",)))
    assert admitted.ids == ("t-c",) and admitted.name == "world-learning-construction"
    with pytest.raises(ValueError, match="not reserved for world_learning"):
        corpus_scope(plan, "retrieval", None, _admission(("t-b",)))
    rule = ScopeRule(rule=TagRule(name="keep", require_any=("q",)))
    rows = trace_admissions(
        (admitted, None), (corpus_scope(plan, "retrieval", None, None), rule), _admission(("t-c",))
    )
    assert [(row.trace_id, row.corpora, row.cluster_id) for row in rows] == [
        ("t-a", ("retrieval",), None),
        ("t-c", ("construction", "retrieval"), "c00"),
    ]
    assert (
        rows[1].reason
        == "construction: c00 why; retrieval: tag rule 'keep' over the task view selected it"
    )


def test_probe_selection_is_empty_without_error_tags() -> None:
    """No error bank means no ranking and no selection."""
    assert probe_selection(_plan(0), (), None, 3) == ((), (), ())


def test_remaining_costs_discount_committed_calls_without_changing_dispatch_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached request removes its future retry allowance while full dispatch bounds stay fixed."""
    plan = VarianceBuildPlan(
        identity="fixture",
        saved=False,
        decision_model=_SNAPSHOT,
        roles={},
        views=(),
        decision_calls=2,
        decision_attempts=6,
        embedder=_SNAPSHOT,
        embedding_inputs=0,
        embedding_inputs_saved=0,
        construction=False,
    ).model_copy(update={"cached_decision_calls": 1})

    def price(project: ProjectStore, alias: str, attempts: int) -> CostComponent:
        """Return unpriced attempt inventory without consulting a provider."""
        return CostComponent(alias, attempts, None)

    monkeypatch.setattr(build_module, "decision_cost", price)
    costs = build_module.variance_build_costs(_project(tmp_path), plan, "fixture")

    assert sum(row.attempts for row in costs) == 3
    assert plan.decision_attempts == 6


def _seed_decision(cache: DecisionCache, unit: Unit, questions: Sequence[Question]) -> None:
    """Commit one exact decision batch through the real cache without a provider."""
    request = DecisionRequest(
        state=unit.text,
        questions={q.question_id: NoulQuestion(instructions=q.text) for q in questions},
    )
    key = decision_key(unit, questions, _SNAPSHOT.model_dump(mode="json"))

    def answer() -> JsonObject:
        """Return valid probabilities for this selected question batch."""
        return {"answers": {q.question_id: {"type": "noul", "noul": 0.5} for q in questions}}

    cache.purchase(key, request, answer)


def test_cache_inventory_matches_model_questions_views_and_chunks(tmp_path: Path) -> None:
    """Only exact committed request keys reduce inventory, even for duplicate source text."""
    project = _project(tmp_path)
    cache = DecisionCache(project.paths)
    bank = _bank(33)
    first, _ = question_chunks(bank.questions, 32)
    unit = _unit("one", "same source")
    _seed_decision(cache, unit, first)
    _seed_decision(cache, _unit("unrelated", "other source"), first)
    before = cache.records.list_ids()
    units: dict[TagView, tuple[Unit, ...]] = {
        "task": (unit, _unit("duplicate", unit.text), _unit("missing", "new source"))
    }

    assert records.cached_decision_calls(project, _SNAPSHOT, {"task": bank}, units, None) == 1
    changed = _SNAPSHOT.model_copy(update={"model_id": "other-model"})
    assert records.cached_decision_calls(project, changed, {"task": bank}, units, None) == 0
    reworded = bank.model_copy(
        update={
            "questions": (
                bank.questions[0].model_copy(update={"text": "A different proposition?"}),
                *bank.questions[1:],
            )
        }
    )
    assert records.cached_decision_calls(project, _SNAPSHOT, {"task": reworded}, units, None) == 0
    assert records.cached_decision_calls(project, _SNAPSHOT, {}, {}, (bank, (unit,))) == 1
    assert cache.records.list_ids() == before


def test_cache_inventory_does_not_discount_or_remove_unresolved_claims(tmp_path: Path) -> None:
    """An abandoned or unknown request retains its entire future allowance and claim."""
    project = _project(tmp_path)
    cache = DecisionCache(project.paths)
    bank = _bank(1)
    unit = _unit("unresolved")
    key = decision_key(unit, bank.questions, _SNAPSHOT.model_dump(mode="json"))
    request = DecisionRequest(
        state=unit.text,
        questions={q.question_id: NoulQuestion(instructions=q.text) for q in bank.questions},
    )

    def unresolved() -> JsonObject:
        """Simulate a request whose provider outcome cannot be proved."""
        raise RuntimeError("unknown outcome")

    with pytest.raises(RuntimeError, match="unknown outcome"):
        cache.purchase(key, request, unresolved)
    before = cache.records.list_ids()

    assert (
        records.cached_decision_calls(project, _SNAPSHOT, {"task": bank}, {"task": (unit,)}, None)
        == 0
    )
    assert before == (f"intent/{key}",)
    assert cache.records.list_ids() == before


def test_cache_inventory_refuses_corrupt_committed_metadata(tmp_path: Path) -> None:
    """A malformed committed cache entry is never silently treated as completed work."""
    project = _project(tmp_path)
    bank = _bank(1)
    unit = _unit("corrupt")
    cache = DecisionCache(project.paths)
    key = decision_key(unit, bank.questions, _SNAPSHOT.model_dump(mode="json"))
    cache.records.write(f"decision/{key}", b"not-json", exclusive=True)

    with pytest.raises(TraceTaggingError, match="invalid"):
        records.cached_decision_calls(project, _SNAPSHOT, {"task": bank}, {"task": (unit,)}, None)

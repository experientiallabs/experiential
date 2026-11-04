"""The variance build end to end over a real completed ``exp build``, offline.

``exp build`` runs on a twenty-lineage OTLP export with deterministic local clients. Decisions are
answered by a fake decision boundary; embeddings are never requested, so a completed stage proves
the scoped index reused the build's saved vectors.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.build.app_test import _catalog, _otlp_export, _RuntimeCatalog
from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelRecord,
    ModelRequest,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectConfig, ProjectStore
from exp.common.project.errors import ArtifactStoreError
from exp.common.project.paths import ProjectPaths
from exp.common.tasks import load_task_set
from exp.common.traces import load_trace_dataset
from exp.optimize.evaluation.variance.build import (
    _tag_units,
    plan_variance_build,
    run_variance_build,
)
from exp.optimize.evaluation.variance.build_records import (
    REPORT_TYPE,
    ConstructionSettings,
    ScopeRule,
    VarianceBuildSpec,
    load_exposure_plan,
    load_tag_matrix,
    load_tag_retrieval_binding,
)
from exp.optimize.evaluation.variance.coalition import BehaviorPattern
from exp.optimize.evaluation.variance.exposure import plan_exposure
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport
from exp.runtime.models.spend_ceiling import SpendCeilingReached
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.error_tags import ERROR_BANK_ID, error_patterns, load_error_tags
from exp.simulation.mining.questions import (
    LOCAL_BANK_ROOT_VARIABLE,
    Question,
    QuestionBank,
    resolve_question_bank,
)
from exp.simulation.mining.tag_cluster_names import (
    ClusterNamer,
    ClusterNamingSpec,
    load_cluster_names,
)
from exp.simulation.mining.tag_cluster_names_test import SNAPSHOT as NAMER
from exp.simulation.mining.tag_cluster_names_test import ScriptedNamer, name_reply
from exp.simulation.mining.tag_clusters import (
    ClusterAdmissionSpec,
    ClusteringSpec,
    load_tag_clustering,
)
from exp.simulation.mining.tag_selection import TagRule
from exp.simulation.mining.tagging import DecisionCache, TagMatrix, TraceTaggingError
from exp.simulation.mining.views import Unit, observations_view, task_view
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

PROJECT = "support"
TASK_BANK = "shape-task-v1"
OBSERVATIONS_BANK = "shape-observations-v1"
_QUESTIONS = ("shape.q0", "shape.q1", "shape.q2")
ERROR_QUESTIONS = (
    ("error.check.unverified_success_claim", "error.check"),
    ("error.feedback.ignored_error_output", "error.feedback"),
    ("risk.destructive_request", "risk"),
)


def _bank_toml(bank_id: str, prefix: str) -> str:
    """Render one three-question bank file."""
    questions = "".join(
        f'\n[[questions]]\nquestion_id = "{prefix}.q{index}"\nfamily = "shape"\n'
        f'text = "Does proposition {index} hold?"\n'
        for index in range(3)
    )
    return f'bank_id = "{bank_id}"\nversion = 1\n{questions}'


def _error_bank_toml() -> str:
    """Render a small error bank with two error propositions and one task-view risk."""
    questions = "".join(
        f'\n[[questions]]\nquestion_id = "{question}"\nfamily = "{family}"\n'
        f'text = "Does {question} hold?"\n'
        for question, family in ERROR_QUESTIONS
    )
    return f'bank_id = "{ERROR_BANK_ID}"\nversion = 1\n{questions}'


def _probability(state: str, question: str) -> float:
    """Answer one proposition deterministically from the text and question."""
    digest = hashlib.sha256(f"{state}\0{question}".encode()).digest()
    return 0.9 if digest[0] % 2 else 0.1


def _answers(state: str, questions: Mapping[str, object], *, drop: str | None) -> JsonObject:
    """Answer every asked question, omitting all of them for the ``drop`` text."""
    answers: JsonObject = {
        question: {"type": "noul", "noul": _probability(state, question)}
        for question in questions
        if state != drop
    }
    return {"answers": answers, "usage": {"input_tokens": 10, "cost": 0.0}}


@dataclass
class FakeDecisions:
    """A decision boundary recording every rendered state it was asked about.

    Attributes:
        drop: Exact state whose answers are withheld, making that trace fail.
        states: Every state sent, in order.
        asked: Every state with the questions asked about it, in order.
    """

    drop: str | None = None
    states: list[str] = field(default_factory=list)
    asked: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer one typed decision request."""
        state = request.state
        assert isinstance(state, str)
        self.states.append(state)
        self.asked.append((state, tuple(request.questions)))
        return _answers(state, request.questions, drop=self.drop)


class DecisionWire(JsonHttpTransport):
    """An HTTP wire answering the native decision route; anything else fails the test."""

    def __init__(self) -> None:
        """Start with no recorded state."""
        self.states: list[str] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one decision POST."""
        assert url.endswith("/systemone"), url
        assert payload is not None
        state, questions = payload["state"], payload["questions"]
        assert isinstance(state, str) and isinstance(questions, dict)
        self.states.append(state)
        return JsonHttpResponse(200, _answers(state, questions, drop=None))


def variance_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProjectStore:
    """Complete a real ``exp build``, then add a decision alias and two local banks.

    Args:
        tmp_path: Directory holding the export, the root and the banks.
        monkeypatch: Fixture keeping build clients local and pointing at the banks.

    Returns:
        The built project.
    """
    monkeypatch.setattr("exp.cli.build.app.RuntimeModelCatalog", _RuntimeCatalog)
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    source = _otlp_export(tmp_path, count=20)
    root = tmp_path / ".exp"
    root.mkdir()
    _catalog(root)
    result = CliRunner().invoke(
        app, ["build", PROJECT, "--traces", str(source), "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    project = ProjectStore(root, PROJECT)
    catalog = load_model_catalog(project.model_catalog_path)
    write_model_catalog(
        project.model_catalog_path,
        catalog.model_copy(
            update={
                "connections": {
                    **catalog.connections,
                    "decisions": ConnectionConfig(
                        provider="typesafe",
                        base_url="https://fixture.invalid/v1",
                        api_key_env="DECISION_TEST_KEY",
                        trusted_custom_origin=True,
                    ),
                },
                "models": {
                    **catalog.models,
                    "jev": ModelRecord(
                        connection="decisions",
                        model="jev-fixture",
                        billing_source=BillingSource.CUSTOMER_MANAGED,
                        capabilities=ModelCapabilities(supports_completions=False),
                    ),
                },
            }
        ),
    )
    banks = tmp_path / "banks"
    banks.mkdir()
    (banks / f"{TASK_BANK}.toml").write_text(_bank_toml(TASK_BANK, "shape"))
    (banks / f"{OBSERVATIONS_BANK}.toml").write_text(_bank_toml(OBSERVATIONS_BANK, "obs"))
    (banks / f"{ERROR_BANK_ID}.toml").write_text(_error_bank_toml())
    monkeypatch.setenv(LOCAL_BANK_ROOT_VARIABLE, str(banks))
    monkeypatch.setenv("DECISION_TEST_KEY", "fixture-only")
    return project


def _tagger(
    project: ProjectStore, decide: Callable[[DecisionRequest], JsonObject]
) -> DecisionsTraceTagger:
    """A cached tagger over the fake boundary, identified as the catalog's decision alias."""
    catalog = load_model_catalog(project.model_catalog_path)
    snapshot, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("jev")
    return DecisionsTraceTagger(decide, snapshot, cache=DecisionCache(project.paths))


def _rendered(project: ProjectStore) -> tuple[dict[str, str], dict[str, str]]:
    """Return every build trace's task and observations views, by trace."""
    build = project.load_project().build
    assert build is not None
    traces = load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    return (
        {trace.trace_id: task_view(trace).text for trace in traces},
        {trace.trace_id: observations_view(trace).text for trace in traces},
    )


def _refuse(request: DecisionRequest) -> JsonObject:
    """Fail the test: a replay must not decide anything."""
    raise AssertionError(f"replay made a decision call about {str(request.state)[:40]!r}")


def test_variance_build_tags_scopes_indexes_binds_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole stage runs offline, keeps every role apart and replays with no call."""
    project = variance_project(tmp_path, monkeypatch)
    spec = VarianceBuildSpec(task_bank=TASK_BANK, observations_bank=OBSERVATIONS_BANK)
    planned = plan_variance_build(project, spec, code_revision="test")
    assert not planned.saved
    assert planned.embedding_inputs > 0
    assert planned.embedding_inputs_saved == planned.embedding_inputs
    decide = FakeDecisions()

    report = run_variance_build(
        project, spec, code_revision="test", tagger=_tagger(project, decide)
    )

    plan = load_exposure_plan(project, report.exposure)
    roles = {trace: plan.role_of(trace) for trace in plan.traces}
    final = {trace for trace, role in roles.items() if role == "final_evaluation"}
    probes = {trace for trace, role in roles.items() if role == "mining_probes"}
    learning = {trace for trace, role in roles.items() if role == "world_learning"}
    assert final and probes and learning
    tasks, observations = _rendered(project)
    sent = set(decide.states)
    assert not sent & ({tasks[trace] for trace in final} | {observations[t] for t in final})
    probe_views = {observations[trace] for trace in probes}
    assert probe_views <= sent
    assert all(
        all(q.startswith("error.") for q in questions)
        for state, questions in decide.asked
        if state in probe_views
    )
    assert {tasks[trace] for trace in learning | probes} <= sent
    assert {observations[trace] for trace in learning} <= sent
    assert len(decide.states) == report.plan.decision_calls

    obs, task = report.tagging
    assert task.plan.view == "task" and obs.plan.view == "observations"
    assert task.tagged == {"world_learning": len(learning), "mining_probes": len(probes)}
    assert obs.tagged == {"world_learning": len(learning)}
    assert not task.failures and not obs.failures
    matrix = load_tag_matrix(project, task.matrix)
    assert {row.trace_id for row in matrix.rows} == learning | probes

    assert set(report.retrieval_scope.ids) == learning == set(report.construction_scope.ids)
    assert report.rules_stage_traces == tuple(sorted(learning))
    assert report.plan.errors is not None and report.errors is not None
    assert report.clustering is not None and report.admission is not None
    assert {row.trace_id for row in report.admissions} == learning
    assert all(row.corpora == ("construction", "retrieval") for row in report.admissions)
    assert all(
        row.cluster_id is not None and "cluster c" in row.reason for row in report.admissions
    )
    index = load_rag_index(project.artifacts, report.index.artifact_id)
    indexed = {row.trace_id for row in index.transitions}
    assert indexed == learning
    assert index.index.embedder.model_id == "embed-id"
    assert index.index.corpus_scope == report.retrieval_scope
    recorded = index.index.trace_tags
    assert recorded is not None
    for view in ("observations", "task"):
        record = recorded.view(view)
        assert record is not None
        assert {row.trace_id for row in record.traces} == indexed
    world = load_grounded_world_model_artifact(project.artifacts, report.world_model)
    assert world.serving_rag == report.index

    build = project.load_project().build
    assert build is not None
    probe_tasks = {
        case.task_id
        for case in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
        if plan.roles[case.lineage_group_id] == "mining_probes"
    }
    treatment = load_tag_retrieval_binding(project, report.tag_retrieval)
    control = load_tag_retrieval_binding(project, report.control_retrieval)
    assert set(treatment.tasks) == probe_tasks == {row.task_id for row in report.probes}
    assert treatment.policy.weight == 0.25 and control.policy.weight == 0.0
    assert control.tasks == treatment.tasks

    replay = run_variance_build(
        project, spec, code_revision="test", tagger=_tagger(project, FakeDecisions())
    )
    assert replay == report
    assert plan_variance_build(project, spec, code_revision="test").saved
    (saved,) = project.artifacts.list_ids(artifact_type=REPORT_TYPE)
    assert saved == report.identity + "-result"


def test_tag_rules_narrow_retrieval_inside_world_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrieval rule selects only world-learning traces whose tags fired."""
    project = variance_project(tmp_path, monkeypatch)
    rule = ScopeRule(rule=TagRule(name="retrieval", require_any=("shape.q0",)))
    spec = VarianceBuildSpec(task_bank=TASK_BANK, retrieval_rule=rule)
    decide = FakeDecisions()

    report = run_variance_build(
        project, spec, code_revision="test", tagger=_tagger(project, decide)
    )

    plan = load_exposure_plan(project, report.exposure)
    tasks, _ = _rendered(project)
    fired = {
        trace
        for trace in plan.traces
        if plan.role_of(trace) == "world_learning" and _probability(tasks[trace], "shape.q0") > 0.5
    }
    assert set(report.retrieval_scope.ids) == fired
    assert len(report.construction_scope.ids) > len(fired)
    index = load_rag_index(project.artifacts, report.index.artifact_id)
    assert {row.trace_id for row in index.transitions} == fired


def test_clusters_choose_world_learning_and_error_tags_choose_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cluster admission narrows construction, a rule still overrides retrieval, error tags rank
    the probes, names are bought once, and the whole run replays without any call."""
    project = variance_project(tmp_path, monkeypatch)
    rule = ScopeRule(rule=TagRule(name="retrieval", require_any=("shape.q0",)))
    spec = VarianceBuildSpec(
        task_bank=TASK_BANK,
        retrieval_rule=rule,
        clustering=ClusteringSpec(max_clusters=4, seeds=4),
        cluster_admission=ClusterAdmissionSpec(total=4),
        cluster_naming=ClusterNamingSpec(),
        probe_limit=1,
    )
    with pytest.raises(ValueError, match="admitted namer"):
        run_variance_build(project, spec, code_revision="test", tagger=_tagger(project, _refuse))
    planned = plan_variance_build(project, spec, code_revision="test")
    assert planned.errors is not None and planned.naming_calls == 16
    decide = FakeDecisions()
    namer = ScriptedNamer(name_reply)

    report = run_variance_build(
        project,
        spec,
        code_revision="test",
        tagger=_tagger(project, decide),
        namer=ClusterNamer(namer, NAMER),
    )

    assert len(decide.states) == report.plan.decision_calls
    plan = load_exposure_plan(project, report.exposure)
    learning = {t for t in plan.traces if plan.role_of(t) == "world_learning"}
    held_out = {t for t in plan.traces if plan.role_of(t) == "final_evaluation"}
    assert report.clustering is not None and report.admission is not None
    clustering = load_tag_clustering(project.artifacts, report.clustering)
    assert set(clustering.trace_ids) == learning and not set(clustering.trace_ids) & held_out
    assert {s.split(":", 1)[0] for s in clustering.columns} == {"task", "errors"}
    assert len(report.construction_scope.ids) == 4
    assert set(report.construction_scope.ids) == set(report.admission.trace_ids())
    assert report.rules_stage_traces == report.construction_scope.ids
    tasks, _ = _rendered(project)
    fired = {t for t in learning if _probability(tasks[t], "shape.q0") > 0.5}
    assert set(report.retrieval_scope.ids) == fired
    by_trace = {row.trace_id: row for row in report.admissions}
    assert set(by_trace) == set(report.construction_scope.ids) | fired
    for trace in report.construction_scope.ids:
        assert "construction: cluster c" in by_trace[trace].reason
    for trace in fired:
        assert "retrieval: tag rule 'retrieval'" in by_trace[trace].reason
    assert report.cluster_names is not None
    names = load_cluster_names(project.artifacts, report.cluster_names)
    assert all(row.status == "named" for row in names.names)
    assert len(namer.requests) == 4 * len(clustering.clusters)
    assert all(row.cluster_name is not None for row in report.admission.admitted)

    assert report.errors is not None
    errors = load_error_tags(project.artifacts, report.errors)
    assert {row.trace_id for row in errors.matrix.rows} == learning | {
        t for t in plan.traces if plan.role_of(t) == "mining_probes"
    }
    assert {row.task_id for row in report.probe_ranking} == {row.task_id for row in report.probes}
    scores = [row.score for row in report.probe_ranking]
    assert scores == sorted(scores, reverse=True)
    assert report.selected_probes == (report.probe_ranking[0].task_id,)
    patterns = error_patterns(errors, resolve_question_bank(ERROR_BANK_ID), include_uncertain=True)
    assert patterns
    assert BehaviorPattern.model_validate(patterns[0].model_dump()).kind == "mistake"

    def refuse_naming(request: ModelRequest) -> str:
        """Fail the test: a replay must not name anything."""
        raise AssertionError("replay made a naming call")

    replay = run_variance_build(
        project,
        spec,
        code_revision="test",
        tagger=_tagger(project, _refuse),
        namer=ClusterNamer(ScriptedNamer(refuse_naming), NAMER),
    )
    assert replay == report


def test_a_failed_trace_is_recorded_and_kept_out_of_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace whose answers are missing is a recorded failure, never a fabricated row.

    Without clustering construction keeps the trace; retrieval, which records tags, does not.
    """
    project = variance_project(tmp_path, monkeypatch)
    spec = VarianceBuildSpec(task_bank=TASK_BANK, clustering=None)
    build = project.load_project().build
    assert build is not None
    rows = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id).bindings
    plan = plan_exposure(build.task_set, rows, seed=spec.seed, probe_fraction=spec.probe_fraction)
    victim = min(trace for trace in plan.traces if plan.role_of(trace) == "world_learning")
    tasks, _ = _rendered(project)

    report = run_variance_build(
        project, spec, code_revision="test", tagger=_tagger(project, FakeDecisions(tasks[victim]))
    )

    (task,) = report.tagging
    assert [(row.trace_id, row.role) for row in task.failures] == [(victim, "world_learning")]
    assert "no answer" in task.failures[0].error
    assert victim not in report.retrieval_scope.ids
    assert victim in report.construction_scope.ids
    index = load_rag_index(project.artifacts, report.index.artifact_id)
    assert victim not in {row.trace_id for row in index.transitions}
    replay = run_variance_build(
        project, spec, code_revision="test", tagger=_tagger(project, _refuse)
    )
    assert replay == report


def test_construction_needs_a_constructor_and_records_its_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paid reconstruction is opt-in; when on, it reads exactly the construction scope."""
    project = variance_project(tmp_path, monkeypatch)
    spec = VarianceBuildSpec(task_bank=TASK_BANK, construction=ConstructionSettings())
    with pytest.raises(ValueError, match="admitted constructor"):
        run_variance_build(project, spec, code_revision="test", tagger=_tagger(project, _refuse))
    requests: list[str] = []

    def complete(body: JsonObject, operation: str) -> JsonObject:
        """Return a fixed constructor proposal and record what it read."""
        requests.append(json.dumps(body))
        return {
            "interface": {
                "tools": [{"name": "reply", "description": "Observed", "parameters": {}}],
                "state_schema": {"type": "object"},
            },
            "claims": [
                {
                    "statement": "Unobserved behavior is unknown",
                    "status": "unknown",
                    "origin": "model_prior",
                    "scope": "unseen calls",
                }
            ],
            "text_rules": "Retain uncertainty.",
        }

    report = run_variance_build(
        project,
        spec,
        code_revision="test",
        tagger=_tagger(project, FakeDecisions()),
        constructor=complete,
    )

    assert report.constructed is not None and requests
    world = load_grounded_world_model_artifact(project.artifacts, report.world_model)
    assert world.construction_scope == report.construction_scope
    assert world.knowledge is not None
    assert set(world.knowledge.source_trace_ids) == set(report.construction_scope.ids)
    assert world.serving_rag == report.index


def test_a_project_without_a_build_is_refused(tmp_path: Path) -> None:
    """The stage needs a completed build and says so."""
    project = ProjectStore(tmp_path, "empty")
    project.initialize(ProjectConfig(project_id="empty"))
    with pytest.raises(ValueError, match="completed `exp build`"):
        plan_variance_build(project, VarianceBuildSpec(), code_revision="test")


def _recovery_bank() -> QuestionBank:
    """Return a single proposition for focused recovery checks."""
    return QuestionBank(
        bank_id="recovery-v1",
        version=1,
        questions=(Question(question_id="shape.valid", family="shape", text="Is this valid?"),),
    )


def test_unsafe_input_is_quarantined_without_losing_valid_rows(tmp_path: Path) -> None:
    """One unsafe observation leaves valid source rows and an explicit trace failure."""
    bank = _recovery_bank()
    units = (
        Unit("unsafe", "observations", "password: synthetic-placeholder"),
        Unit("safe", "observations", "Repair the feature and run tests."),
    )
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    decide = FakeDecisions()
    tagger = DecisionsTraceTagger(decide, NAMER, cache=cache, maximum_concurrency=2)

    matrix, failed = _tag_units(tagger, units, bank)

    assert [row.trace_id for row in matrix.rows] == ["safe"]
    assert len(failed) == 1 and failed[0][0] == "unsafe"
    assert "credential boundary" in failed[0][1]
    assert decide.states == [units[1].text]
    assert len(cache.records.list_ids()) == 1


def test_recovery_keeps_remaining_calls_bounded_and_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed initial pass must not turn uncached recovery calls into serial work."""
    bank = _recovery_bank()
    units = (Unit("bad", "observations", "bad"),) + tuple(
        Unit(f"valid-{i}", "observations", f"valid-{i}") for i in range(4)
    )
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active = 0
    peak = 0
    calls: list[str] = []

    def decide(request: DecisionRequest) -> JsonObject:
        """Require two recovery calls to overlap and count actual in-flight work."""
        nonlocal active, peak
        state = request.state
        assert isinstance(state, str)
        if state == "bad":
            raise TraceTaggingError("invalid source")
        with lock:
            active += 1
            peak = max(peak, active)
            calls.append(state)
        try:
            barrier.wait()
            return _answers(state, request.questions, drop=None)
        finally:
            with lock:
                active -= 1

    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    tagger = DecisionsTraceTagger(decide, NAMER, cache=cache, maximum_concurrency=2)
    original_tag = tagger.tag
    initial = True

    def tag(items: Sequence[Unit], questions: QuestionBank) -> TagMatrix:
        """Make the first pass fail before testing real per-unit recovery."""
        nonlocal initial
        if initial:
            initial = False
            raise TraceTaggingError("initial pass failed")
        return original_tag(items, questions)

    monkeypatch.setattr(tagger, "tag", tag)
    matrix, failed = _tag_units(tagger, units, bank)

    assert [row.trace_id for row in matrix.rows] == [unit.trace_id for unit in units[1:]]
    assert len(failed) == 1 and failed[0][0] == "bad"
    assert sorted(calls) == sorted(unit.text for unit in units[1:])
    assert peak == 2


@pytest.mark.parametrize("after_first_failure", [False, True])
@pytest.mark.parametrize("budget_refusal", [False, True])
def test_artifact_refusals_abort_tagging_and_recovery(
    tmp_path: Path, after_first_failure: bool, budget_refusal: bool
) -> None:
    """Output, storage and spend refusals must not become skipped input rows."""
    refusal = (
        SpendCeilingReached(1, 1, 2)
        if budget_refusal
        else ArtifactStoreError("provider output was refused")
    )
    units = ((Unit("bad", "observations", "bad"),) if after_first_failure else ()) + (
        Unit("safe", "observations", "safe"),
    )

    def decide(request: DecisionRequest) -> JsonObject:
        """Optionally fail one input, then raise a fatal output refusal."""
        if request.state == "bad":
            raise TraceTaggingError("invalid source")
        raise refusal

    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    tagger = DecisionsTraceTagger(decide, NAMER, cache=cache, maximum_concurrency=1)

    with pytest.raises(type(refusal)) as caught:
        _tag_units(tagger, units, _recovery_bank())

    assert caught.value is refusal

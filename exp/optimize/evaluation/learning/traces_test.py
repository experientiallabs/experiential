"""The learning pass: every trace by default, measured test-then-train, budgeted and resumable."""

import functools
import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models import ModelSnapshot, load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input, write_project_config
from exp.common.tasks.facts import InstanceFacts
from exp.common.tasks.facts_test import fact
from exp.optimize.evaluation.audit import panel_test as held_out_fidelity_test
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _at,
    _obj,
    _tool_reply,
    build_project,
    build_report,
    episode,
    held_out_customers,
    held_out_lineages,
    package_rows,
    write_catalog,
)
from exp.optimize.evaluation.fidelity import replay
from exp.optimize.evaluation.fidelity.runner import FidelityMemory, FidelitySpec
from exp.optimize.evaluation.learning import lineage as plan_lineage
from exp.optimize.evaluation.learning import traces
from exp.optimize.evaluation.learning.profile import learned_evaluation_memory
from exp.optimize.evaluation.learning.traces import (
    SCOPES,
    LearningPlan,
    Scope,
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_spec,
    recorded_plan,
    reserve_held_out,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
)
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.retrieval.evidence_reading import ReadingPolicy
from exp.simulation.world_model import dreaming
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    consumed_task_lineages,
    is_reserved,
    knowledge_identity,
    memory_head,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


def test_serving_change_reuses_bank_but_pins_new_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A top-k edit retains learned knowledge and policy without replaying the old predictor."""
    episodes = [episode(i) for i in range(2)]
    provider = SimulatedProvider(episodes)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    original = learning_plan(store)
    fit_original = learning_plan(store, "jev", "fit", record=True)
    assert reserve_held_out(store, fit_original, code_revision=REVISION) > 0
    held = held_out_lineages(store)
    head = memory_head(store.artifacts, original.lineage)
    assert head is not None
    config = store.load_project()
    assert config.build is not None
    world = load_grounded_world_model_artifact(store.artifacts, original.world_model)
    changed = persist_grounded_world_model(
        store.artifacts,
        serving_rag=world.serving_rag,
        model_alias=world.model_alias,
        model=world.model,
        created_at=world.created_at,
        code_revision=world.code_revision,
        top_k=world.top_k + 1,
        knowledge=world.knowledge,
        selection_policy_sha256=world.selection_policy_sha256,
    )
    changed_ref = artifact_input(changed.manifest)
    write_project_config(
        store.paths,
        config.model_copy(
            update={
                "build": config.build.model_copy(update={"world_model": changed_ref}),
            }
        ),
    )
    assert recorded_plan(store, "all"), "compatible memory must remain discoverable"
    before = store.artifacts.list_ids()
    updated = learning_plan(store, scope="all")
    assert store.artifacts.list_ids() == before, "preflight must not publish a new plan"
    assert updated.lineage == original.lineage
    assert updated.world_model == changed_ref != original.world_model
    assert updated.recall == original.recall
    memory = learned_evaluation_memory(store, load_model_catalog(store.model_catalog_path))
    assert memory is not None and memory.version == head.version
    assert memory.lineage == original.lineage and memory.world_model == changed_ref
    case = learning_cases(store, "all")[0]
    deadline = datetime.now(UTC) + timedelta(hours=1)
    old_run = learning_spec(store, original, case, deadline=deadline)
    new_run = learning_spec(store, updated, case, deadline=deadline)
    assert replay.content_run_id(
        old_run, "learn", case.trace_id, code_revision=None
    ) != replay.content_run_id(new_run, "learn", case.trace_id, code_revision=None)
    assert old_run.world_model != new_run.world_model
    saved = learning_plan(store, scope="all", record=True)
    assert saved == updated
    assert len(set(store.artifacts.list_ids()) - set(before)) == 1
    assert memory_head(store.artifacts, updated.lineage) == head
    fit_updated = learning_plan(store, None, "fit")
    assert fit_updated.lineage == fit_original.lineage != updated.lineage
    assert fit_updated.world_model != fit_original.world_model
    assert all(is_reserved(store.artifacts, fit_updated.lineage, lineage) for lineage in held)
    with pytest.raises(ValueError, match="recall alias"):
        learning_plan(store, "different", "all")
    different = persist_grounded_world_model(
        store.artifacts,
        serving_rag=world.serving_rag,
        model_alias=world.model_alias,
        model=world.model,
        created_at=world.created_at,
        code_revision=world.code_revision,
        top_k=world.top_k,
        knowledge=SimulationKnowledge(rules="Refuse every cancellation.", authored=True),
    )
    write_project_config(
        store.paths,
        config.model_copy(
            update={
                "build": config.build.model_copy(
                    update={"world_model": artifact_input(different.manifest)}
                ),
            }
        ),
    )
    assert not recorded_plan(store, "all"), "different knowledge cannot inherit the bank"
    with pytest.raises(ValueError, match="no learned memory"):
        learning_plan(store, scope="all")
    assert learning_plan(store, "jev", "all").lineage != original.lineage


def test_obsolete_plan_refuses_silent_memory_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prior plan schema must request fresh setup, never appear to be empty memory."""
    episodes = [episode(0), episode(1)]
    provider = SimulatedProvider(episodes)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    plan = learning_plan(store, "jev", "all")
    store.artifacts.write_json(
        artifact_id="trace-learning-plan-obsolete",
        artifact_type="trace-learning-plan",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="old-plan"
        ),
        files={"plan.json": plan.model_dump(mode="json", exclude={"bank_protocol"})},
    )
    before = len(provider.requests)
    with pytest.raises(ValueError, match="incompatible bank contract.*fresh project"):
        recorded_plan(store, "all")
    with pytest.raises(ValueError, match="incompatible bank contract.*fresh project"):
        learning_plan(store, "jev", "all", record=True)
    assert len(provider.requests) == before


def test_concurrent_bank_creation_has_one_recall_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two bank writers cannot publish conflicting recall policies under one lineage."""
    episodes = [episode(0), episode(1)]
    provider = SimulatedProvider(episodes)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    gate = Barrier(2)

    def start(alias: str) -> LearningPlan | str:
        """Race registration only, without constructing or calling either provider."""
        gate.wait(timeout=10)
        try:
            return learning_plan(store, alias, "all", record=True)
        except ValueError as error:
            return str(error)

    before = len(provider.requests)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, ["jev", "other"]))
    plans = [row for row in results if isinstance(row, LearningPlan)]
    errors = [row for row in results if isinstance(row, str)]
    assert len(plans) == len(errors) == 1 and "recall alias" in errors[0]
    assert learning_plan(store, scope="all") == plans[0]
    assert len(provider.requests) == before


class ReadingProvider(SimulatedProvider):
    """Exercise the real CLI and dispatch ledger with deterministic wire responses."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        questions = payload.get("questions")
        if isinstance(questions, dict) and "support" in questions:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            return JsonHttpResponse(
                status_code=200,
                body={
                    "answers": {
                        "support": {"type": "choice", "choice": "whole"},
                        "contradiction": {"type": "choice", "choice": "none"},
                    }
                },
            )
        if not url.endswith(("/embeddings", "/systemone")):
            schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
            if "read_more" in schema:
                self.requests.append(RecordedRequest(url, dict(headers), payload))
                name = str(_at(payload, "tool_choice", "function", "name"))
                return _tool_reply(
                    "world-v1",
                    name,
                    {
                        "question": "Which recorded fields establish the response format?",
                        "read_more": True,
                    },
                )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def test_cli_evidence_reading_is_pinned_metered_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = [episode(i) for i in range(3)]
    provider = ReadingProvider(episodes, wrong=frozenset({"get_user_details"}))
    _, store, _ = build_project(
        tmp_path, monkeypatch, provider, "--evidence-reading", learn=True, episodes=episodes
    )
    plan = learning_plan(store)
    assert plan.reading is not None
    assert plan.recall.memory_input_bytes == 24000
    report = build_report(store)  # the build keeps its terminal short; the preflight is saved
    assert "objective-driven investigations" in report
    assert "canonical dataset order" in report
    assert "in export order" not in report
    assert any(
        isinstance(questions := row.payload.get("questions"), dict) and "support" in questions
        for row in provider.requests
    )
    case = learning_cases(store, "all")[0]
    spec = learning_spec(store, plan, case, deadline=datetime.now(UTC) + timedelta(hours=1))
    control = learning_spec(
        store, plan.model_copy(update={"reading": None}), case, deadline=spec.deadline
    )
    assert spec.memory_selection_identity == plan.selector_identity
    assert spec.retrieval is not None and spec.retrieval.memory_input_bytes == 24000
    assert spec.maximum_physical_attempts > control.maximum_physical_attempts
    assert replay.content_run_id(
        spec, "learn", case.trace_id, code_revision=None
    ) != replay.content_run_id(control, "learn", case.trace_id, code_revision=None)
    assert memory_head(store.artifacts, plan.lineage) is not None
    before = len(provider.requests)
    outcome = learn_from_traces(
        store,
        recall_alias="jev",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert outcome.incomplete == 0
    assert len(provider.requests) == before
    with pytest.raises(ValueError, match="differs from saved memory"):
        learning_plan(store, "jev", "all", evidence_reading=False)


def test_fit_only_reader_never_receives_held_out_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = [episode(i) for i in range(USERS)]
    provider = ReadingProvider(episodes, wrong=frozenset({"get_user_details"}))
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    plan = learning_plan(store, "jev", "fit", evidence_reading=True)
    cases = learning_cases(store, "fit")
    ceiling = sum(
        learning_spec(
            store, plan, case, deadline=datetime.now(UTC) + timedelta(hours=1)
        ).maximum_physical_attempts
        for case in cases[:2]
    )
    outcome = learn_from_traces(
        store,
        recall_alias="jev",
        scope="fit",
        maximum_attempts=ceiling,
        evidence_reading=True,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )
    assert outcome.learned >= 2 and outcome.incomplete == 0
    forbidden = held_out_lineages(store) | held_out_customers(store)
    reader_requests = [
        row
        for row in provider.requests
        if isinstance(questions := row.payload.get("questions"), dict) and "support" in questions
    ]
    assert reader_requests
    for row in reader_requests:
        assert not any(lineage in json.dumps(row.payload) for lineage in forbidden)


def test_budget_stops_between_traces_and_a_rerun_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A trace starts only when its whole ceiling fits; the next run picks up after it."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider)

    def learn(budget: int):  # noqa: ANN202 - local shorthand
        """Run the pass with one budget."""
        return learn_from_traces(
            store,
            recall_alias="jev",
            maximum_attempts=budget,
            code_revision=REVISION,
            clock=lambda: datetime.now(UTC),
        )

    nothing = learn(1)
    assert nothing.learned == 0 and nothing.stopped is not None and not provider.requests[1:]
    assert learning_curve(store) is None
    one = learning_spec(
        store,
        learning_plan(store, "jev"),
        learning_cases(store, "all")[0],
        deadline=datetime.now(UTC) + timedelta(hours=1),
    ).maximum_physical_attempts
    first = learn(one + one // 2)
    assert 0 < first.learned < first.traces and first.head_version == first.learned
    rest = learn(10_000)
    assert rest.stopped is None and rest.learned == rest.traces == len(learning_cases(store, "all"))
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == rest.traces
    assert [w.last for w in curve.halves][-1] == rest.traces


def test_default_learning_consumes_every_trace_and_reports_test_then_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plain build learns every trace, reserves nothing, and retests its own misses."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, output = build_project(tmp_path, monkeypatch, provider, learn=True)
    plan = learning_plan(store)
    assert plan.scope == "all"
    assert plan.recall.jev_alias == "jev", "the catalog's only decision model is chosen"
    head = memory_head(store.artifacts, plan.lineage)
    assert head is not None and head.version == USERS
    held = held_out_lineages(store)
    assert held and held <= consumed_task_lineages(store.artifacts, head)
    assert not any(is_reserved(store.artifacts, plan.lineage, lineage) for lineage in held)
    assert f"learned {USERS}/{USERS} traces · memory v{USERS}" in output
    exact = 3 * USERS - 1  # only the first trace's learnable turn is wrong
    headline = f"Learning progress (test-then-train, recorded actions) {exact}/{3 * USERS} replies"
    # The terminal gets the result lines; the report file keeps the full learning output.
    assert f"replies exact while learning 97% ({exact}/{3 * USERS}), last 3 traces 100%" in output
    assert "next: exp eval retail --audit N" in output
    assert re.search(r"Complete in (under 1s|\d+(s|m\d\ds|h\d\dm))", output), "time taken"
    report = build_report(store)
    assert headline in report
    # Traces 10-12 learn on nearly final memory; all their turns are exact.
    estimate = "latest unseen-task estimate 100% replies exact (9/9) over the last 3 trace(s)"
    assert estimate in report
    # The build ends by retesting its one missed trace; that retest is the only judge call.
    assert "Miss retest" in report and "fixed 1 of 1 missed turns" in output
    judging = [r for r in provider.requests if "axes" in json.dumps(r.payload.get("tools"))]
    assert len(judging) == 1, "learning is scored by exact replies; only the retest is judged"
    audit = CliRunner().invoke(
        app,
        ["eval", PROJECT, "--root", str(root), "--audit", "3", "--dry-run", "--non-interactive"],
    )
    assert audit.exit_code == 0, audit.output
    assert "learns a separate memory from fit traces only" in " ".join(audit.output.split())


def test_learning_hides_earlier_traces_of_the_same_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repeat of a learned task is predicted without the memory of its earlier trace."""
    repeat: JsonObject = {**episode(1), "repeat": "-again"}
    order = "#W2001"
    repeat["instruction"] = f"I am user_01. Please cancel order {order} if it has not shipped."
    repeat["exchanges"] = [
        ("find_user_id_by_email", {"email": "user_01@example.test"}, {"user_id": "user_01"}),
        ("get_order_details", {"order_id": order}, {"order_id": order, "status": "pending"}),
        ("cancel_pending_order", {"order_id": order}, {"order_id": order, "status": "cancelled"}),
    ]
    episodes = [*(episode(i) for i in range(USERS)), repeat]
    provider = SimulatedProvider(episodes)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    cases = learning_cases(store, "all")
    positions = [
        index
        for index, case in enumerate(cases)
        if case.task.initial_context.get("task_lineage") == "user_01"
    ]
    assert len(positions) == 2, "both traces of the task are learned"
    assert order in cases[positions[1]].task.instruction, "the repeat is learned second"
    sibling = cases[positions[0]].task.lineage_group_id  # names the task in every memory row
    repeats = [
        prediction
        for prediction in provider.predictions()
        if order in json.dumps(_at(prediction, "task", "instruction"))
    ]
    assert repeats, "the repeated task was predicted"
    others = [
        prediction
        for prediction in provider.predictions()
        if order not in json.dumps(_at(prediction, "task", "instruction"))
        and sibling in memory_delivered(prediction)
    ]
    assert others, "other tasks do read the earlier trace's memory"
    for prediction in repeats:
        assert sibling not in memory_delivered(prediction)
        contexts = _at(prediction, "grounding_contexts")
        assert "user_01" not in json.dumps(contexts), "retrieval also excludes the task"


def test_an_identical_repeat_trace_learns_as_an_already_consumed_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second trace with exactly the same content is predicted, then adds no memory version."""
    episodes = [*(episode(i) for i in range(USERS)), {**episode(1), "repeat": "-again"}]
    provider = SimulatedProvider(episodes, learnable=True)
    _, store, output = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    assert len(learning_cases(store, "all")) == USERS + 1
    head = memory_head(store.artifacts, learning_plan(store).lineage)
    assert head is not None and head.version == USERS
    assert f"learned {USERS + 1}/{USERS + 1} traces · memory v{USERS}" in output


def memory_delivered(prediction: JsonObject) -> str:
    """Everything memory delivered to one prediction: recalled records and the overview."""
    return json.dumps([prediction.get("memory_observations"), prediction.get("research_history")])


def test_eligibility_hides_one_task_lineage() -> None:
    """The hidden lineage is withheld like a reserved one; other lineages stay readable."""
    rule = Eligibility(hidden_task_lineage="user_01")
    assert not rule.admits({"task_lineage": "user_01"})
    assert rule.admits({"task_lineage": "user_02"})


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"mode": "frozen", "checkpoint": 1, "hide_task_lineage": True}, "only while learning"),
        ({"mode": "learn", "withhold_own": False}, "unreserved frozen retest"),
        (
            {"mode": "frozen", "checkpoint": 1, "reserved": True, "withhold_own": False},
            "unreserved frozen retest",
        ),
    ],
)
def test_memory_options_apply_only_where_they_mean_something(
    fields: dict[str, object], message: str
) -> None:
    """Hiding is a learning option; reading one's own trajectory is an unreserved retest."""
    with pytest.raises(ValueError, match=message):
        FidelityMemory.model_validate({"lineage": "learn-a", **fields})


def test_a_trace_is_predicted_whole_before_its_own_feedback_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first trace misses the learnable tool twice: its own investigation cannot help."""
    episodes = []
    for index in range(USERS):
        item = episode(index)
        user = f"user_{index:02d}"
        again = ("get_user_details", {"user_id": user, "fields": "payment"}, {"payment": "card"})
        exchanges = item["exchanges"]
        assert isinstance(exchanges, list)
        item["exchanges"] = [*exchanges, again]
        episodes.append(item)
    provider = SimulatedProvider(episodes, learnable=True)
    _, store, output = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == USERS
    first = curve.points[0]
    assert (first.exact, first.steps) == (2, 4), "both learnable turns miss on pinned memory"
    assert all(point.exact == point.steps for point in curve.points[1:])
    for prediction in provider.predictions():
        assert not _obj(prediction.get("research_history") or {}).get("session")
    assert f"learned {USERS}/{USERS} traces" in output


def _relearn(store: ProjectStore) -> None:
    """Run the build's learning pass again, as a rebuild does."""
    learn_from_traces(
        store,
        recall_alias="jev",
        scope="all",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
    )


def test_a_judge_change_replays_learning_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Learning never judges, so another judge leaves its runs and memory as they are."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    lineage, purchased = learning_plan(store, scope="all").lineage, len(provider.requests)
    catalog = load_model_catalog(store.model_catalog_path)
    judge = catalog.models["judge"]
    assert judge.capabilities is not None
    smaller = judge.capabilities.model_copy(update={"maximum_output_tokens": 8_192})
    write_model_catalog(
        store.model_catalog_path,
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    "judge": judge.model_copy(update={"capabilities": smaller}),
                }
            }
        ),
    )
    _relearn(store)
    assert len(provider.requests) == purchased, "no learning run was bought again"
    assert learning_plan(store, scope="all").lineage == lineage


def test_a_protocol_change_learns_a_fresh_memory_trace_by_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed prompt starts a new memory; no trace is predicted on memory of later traces."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    before = learning_plan(store, scope="all").lineage

    def changed(reading: ReadingPolicy | None = None, *, native_artifacts: bool = False) -> str:
        """Stand in for a protocol edit across both supported request projections."""
        return "changed-prompts"

    monkeypatch.setattr(plan_lineage, "protocol_prompts_sha256", changed)
    monkeypatch.setattr(replay, "protocol_prompts_sha256", changed)
    _relearn(store)
    plan = learning_plan(store, scope="all")
    assert plan.lineage != before, "the edited protocol learns into a new memory"
    head = memory_head(store.artifacts, plan.lineage)
    assert head is not None and head.version == USERS
    curve = learning_curve(store, "all")
    assert curve is not None and len(curve.points) == USERS
    # The first trace is predicted on empty memory again, so it misses the learnable reply,
    # exactly as in the first pass; on the final memory it would have been exact.
    assert curve.points[0].exact == 2


@pytest.mark.parametrize("scope", ["all", "fit"])
def test_facts_delivery_keys_the_learning_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: Scope
) -> None:
    """Delivered facts are learning inputs: adding, changing or removing them re-keys memory.

    Verifier-private facts never reach a simulator request, so they do not key the lineage,
    and a pass that delivers no facts keeps the identity earlier memories were learned under.
    """
    supplied = [episode(i) for i in range(USERS)]
    for item in supplied:
        item["facts"] = {"facts": [fact("input.xlsx/workbook", {"A1": 1})]}
    _, store, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider(supplied), episodes=supplied
    )
    delivered = learning_plan(store, "jev", scope)
    cases = learning_cases(store, scope)
    assert cases and all(case.facts is not None for case in cases)
    assert learning_plan(store, "jev", scope).lineage == delivered.lineage

    def lineage(*rows: JsonObject) -> str:
        """The lineage when every learned trace delivers ``rows`` instead (none: no facts)."""
        facts = InstanceFacts.model_validate({"facts": rows}) if rows else None
        replaced = tuple(case.model_copy(update={"facts": facts}) for case in cases)
        with monkeypatch.context() as patch:
            patch.setattr(traces, "learning_cases", lambda _project, _scope: replaced)
            return learning_plan(store, "jev", scope).lineage

    artifact = load_grounded_world_model_artifact(store.artifacts, delivered.world_model)
    protocol = {"protocol": "test-then-train", **plan_lineage.learning_protocol()}
    key = {"knowledge": knowledge_identity(artifact), "scope": scope, **protocol}
    fact_free = "learn-" + sha256_json(key)[:16]
    assert lineage() == fact_free, "a fact-free pass keeps the identity of earlier memories"
    changed = lineage(fact("input.xlsx/workbook", {"A1": 2}))
    assert len({fact_free, delivered.lineage, changed}) == 3
    private = fact("input.xlsx/expected", "x", visibility="verifier_private")
    assert lineage(private) == fact_free, "verifier-private facts are never delivered"
    assert lineage(fact("input.xlsx/workbook", {"A1": 1}), private) == delivered.lineage


def test_reading_the_recorded_plan_derives_each_lineage_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lineage reads every trace's facts, so one plan lookup derives it once per scope."""
    episodes = [episode(i) for i in range(USERS)]
    _, store, _ = build_project(tmp_path, monkeypatch, SimulatedProvider(episodes))
    for scope in SCOPES:
        learning_plan(store, "jev", scope, record=True)
    derived: list[Scope] = []
    lineage = traces._lineage  # noqa: SLF001

    def counted(project: ProjectStore, world_model: ArtifactInput, scope: Scope) -> str:
        """Count each lineage derivation."""
        derived.append(scope)
        return lineage(project, world_model, scope)

    monkeypatch.setattr(traces, "_lineage", counted)
    plan = learning_plan(store)
    assert sorted(derived) == sorted([*SCOPES, plan.scope]), derived
    derived.clear()
    assert recorded_plan(store, "fit") and derived == ["fit"]


def test_editing_the_workbook_or_facts_prompt_starts_a_fresh_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompts only some traces send are still part of the lineage that spans every trace."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    learned = learning_plan(store, scope="all")
    before, recall = learned.lineage, learned.recall.jev_alias
    monkeypatch.setattr(plan_lineage, "INSTANCE_FACTS_GUIDANCE", "edited facts guidance")
    facts_edited = learning_plan(store, recall, scope="all").lineage
    monkeypatch.setattr(replay, "NATIVE_ARTIFACT_PLAN_GUIDANCE", "edited workbook guidance")
    workbook_edited = learning_plan(store, recall, scope="all").lineage
    assert len({before, facts_edited, workbook_edited}) == 3


def test_learning_refuses_to_predict_a_learned_trace_on_later_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run input that drifts without a new lineage stops learning before any call."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    purchased = len(provider.requests)
    content_run_id = replay.content_run_id

    def drifted(
        spec: FidelitySpec,
        label: str,
        trace: str,
        generation: int = 1,
        *,
        code_revision: str | None,
    ) -> str:
        """Every run ID changes while the lineage stays the same."""
        return content_run_id(
            spec, label, trace + "-drift", generation, code_revision=code_revision
        )

    monkeypatch.setattr(replay, "content_run_id", drifted)
    with pytest.raises(ValueError, match="already learned into memory"):
        _relearn(store)
    assert len(provider.requests) == purchased, "refused before any call"


def test_a_world_model_without_tools_builds_without_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Learning needs a tool-capable world model; without one the build completes unlearned."""
    original = held_out_fidelity_test.write_catalog

    def tool_less(root: Path, *, recall: bool = True) -> None:
        """Write the fixture catalog with a world model that declares no tool support."""
        original(root, recall=recall)
        path = root / "models.toml"
        catalog = load_model_catalog(path)
        world = catalog.models["world"]
        assert world.capabilities is not None
        caps = world.capabilities.model_copy(update={"supports_tools": False})
        models = {**catalog.models, "world": world.model_copy(update={"capabilities": caps})}
        write_model_catalog(path, catalog.model_copy(update={"models": models}))

    monkeypatch.setattr(held_out_fidelity_test, "write_catalog", tool_less)
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, output = build_project(tmp_path, monkeypatch, provider, learn=True)
    assert "built without learning" in output and "'world' declares no tool support" in output
    assert not provider.predictions(), "no learning call was made"
    assert store.load_project().build is not None


def test_automation_without_yes_refuses_before_buying_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Learning's undefined cost is accepted up front, and the refusal names the full command."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root = tmp_path / ".exp"
    root.mkdir()
    write_catalog(root, recall=True)
    package = tmp_path / "package.jsonl"
    rows = package_rows([episode(i) for i in range(USERS)])
    package.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    result = CliRunner().invoke(
        app,
        ["build", PROJECT, "-t", str(package), "--source", "otel-genai", "--root", str(root)]
        + ["--non-interactive"],
    )
    output = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert not provider.requests, "no index is bought before the refusal"
    rerun = f"exp build {PROJECT} -t {package} --source otel-genai with --yes"
    assert "".join(rerun.split()) in "".join(output.split()), "long paths wrap in the terminal"


class _EmbeddingOutage(SimulatedProvider):
    """The routed fake with every embedding request rejected (a revoked key)."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer 401 to embeddings; route everything else normally."""
        if url.endswith("/embeddings"):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            return JsonHttpResponse(status_code=401, body={"error": {"message": "bad key"}})
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def test_an_embedding_failure_in_a_configured_build_stops_readably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exp build -t` reports a provider failure in one line and exits 1, not a traceback."""
    provider = _EmbeddingOutage([episode(i) for i in range(USERS)], learnable=True)
    root = tmp_path / ".exp"
    root.mkdir()
    write_catalog(root, recall=True)
    package = tmp_path / "package.jsonl"
    rows = package_rows([episode(i) for i in range(USERS)])
    package.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    result = CliRunner().invoke(
        app,
        ["build", PROJECT, "-t", str(package), "--source", "otel-genai", "--root", str(root)]
        + ["--yes", "--non-interactive"],
    )
    output = " ".join(result.output.split())
    assert result.exit_code == 1, result.output
    assert "a provider request failed" in output
    assert "Run the same exp build command again to resume" in output
    assert not isinstance(result.exception, ProviderTransportError | AssertionError)


def test_a_learning_request_that_runs_out_of_time_stops_the_build_readably(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`exp build` reports a provider deadline like other provider failures, not a traceback.

    The SpreadsheetBench wide pilot's simulator build printed a raw ProviderDeadlineExceeded
    traceback when one learning request ran out of time.
    """
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root = tmp_path / ".exp"
    root.mkdir()
    write_catalog(root, recall=True)
    package = tmp_path / "package.jsonl"
    rows = package_rows([episode(i) for i in range(USERS)])
    package.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )

    def out_of_time(*_args: object, **_kwargs: object) -> None:
        """Fail the way the pilot's learning request failed."""
        raise ProviderDeadlineExceeded("provider request deadline exceeded")

    monkeypatch.setattr("exp.cli.evaluation.learning.learn_from_traces", out_of_time)
    result = CliRunner().invoke(
        app,
        ["build", PROJECT, "-t", str(package), "--source", "otel-genai", "--root", str(root)]
        + ["--yes", "--non-interactive"],
    )
    output = " ".join(result.output.split())
    assert result.exit_code == 1, result.output
    assert "a provider request failed: provider request deadline exceeded" in output
    assert "run the same command with --retry-incomplete to resume it" in output
    assert not isinstance(result.exception, ProviderDeadlineExceeded | AssertionError)


def test_saved_dream_plan_pins_protocol_and_refuses_changed_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed learning lineage cannot silently adopt changed consolidation semantics."""
    provider = SimulatedProvider([episode(index) for index in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=False)
    plan = learning_plan(store, "jev", "all", dream_every=1, record=True)
    assert plan.dreaming is not None and plan.dream_protocol == plan.dreaming.identity
    unpinned = plan.model_dump(mode="json")
    del unpinned["dream_protocol"]
    with pytest.raises(ValueError, match="dream protocol"):
        LearningPlan.model_validate(unpinned)
    requests = len(provider.requests)
    monkeypatch.setattr(dreaming, "JOINT_CHECK_PROMPT", dreaming.JOINT_CHECK_PROMPT + " changed")
    with pytest.raises(ValueError, match="dream protocol"):
        learning_plan(store, "jev", "all")
    assert len(provider.requests) == requests


@pytest.mark.parametrize(
    "change",
    ["serving_model", "top_k", "selection_policy", "prediction_guidance", "knowledge"],
)
def test_lineage_keys_knowledge_not_serving_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """A serving or retrieval change keeps compatible memory; only a knowledge change re-keys it."""
    episodes = [episode(i) for i in range(2)]
    provider = SimulatedProvider(episodes)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes, learn=True)
    original = learning_plan(store, scope="all")
    world = load_grounded_world_model_artifact(store.artifacts, original.world_model)
    knowledge = SimulationKnowledge(rules="Answer from the records.", authored=True)

    def lineage(
        *,
        alias: str = world.model_alias,
        model: ModelSnapshot = world.model,
        top_k: int = world.top_k,
        served_knowledge: SimulationKnowledge = knowledge,
        selection_policy: str | None = world.selection_policy_sha256,
    ) -> str:
        """Serve a simulator with these settings and return the lineage it learns into."""
        served = persist_grounded_world_model(
            store.artifacts,
            serving_rag=world.serving_rag,
            model_alias=alias,
            model=model,
            created_at=world.created_at,
            code_revision=world.code_revision,
            top_k=top_k,
            knowledge=served_knowledge,
            selection_policy_sha256=selection_policy,
        )
        config = store.load_project()
        assert config.build is not None
        reference = artifact_input(served.manifest)
        write_project_config(
            store.paths,
            config.model_copy(
                update={"build": config.build.model_copy(update={"world_model": reference})}
            ),
        )
        plan = learning_plan(store, original.recall.jev_alias, "all")
        assert plan.world_model == reference
        return plan.lineage

    baseline = lineage()
    if change == "serving_model":
        changed = lineage(
            alias="another-simulator",
            model=world.model.model_copy(update={"model_id": "another-model"}),
        )
    elif change == "top_k":
        changed = lineage(top_k=world.top_k + 1)
    elif change == "selection_policy":
        changed = lineage(selection_policy="1" * 64)
    elif change == "prediction_guidance":
        changed = lineage(
            served_knowledge=knowledge.model_copy(
                update={"prediction_guidance": "Prefer the most recent record."}
            )
        )
    else:
        changed = lineage(
            served_knowledge=SimulationKnowledge(rules="Refuse every request.", authored=True)
        )
    if change == "knowledge":
        assert changed != baseline, "changed knowledge starts a fresh memory"
    else:
        assert changed == baseline, f"a {change} change keeps compatible memory"


class _InvestigationOutage(SimulatedProvider):
    """Reject research while preserving successful prediction and embedding responses."""

    fail = True

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Return an authentication failure only for discrepancy investigations."""
        if self.fail and not url.endswith(("/embeddings", "/systemone")):
            name = str(_at(payload, "tool_choice", "function", "name"))
            schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
            if name != "submit_transition" and "axes" not in schema and "summary" not in schema:
                self.requests.append(RecordedRequest(url, dict(headers), payload))
                return JsonHttpResponse(status_code=401, body={"error": "bad key"})
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def test_investigation_outage_resumes_without_redrawing_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected investigation leaves a resumable run, not a terminal unlearned report."""
    episodes = [episode(0)]
    provider = _InvestigationOutage(episodes, learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    with pytest.raises(ProviderTransportError):
        learn_from_traces(
            store,
            recall_alias="jev",
            maximum_attempts=10**6,
            code_revision=REVISION,
            clock=lambda: datetime.now(UTC),
        )
    predictions = provider.predictions()
    assert predictions
    assert memory_head(store.artifacts, learning_plan(store).lineage) is None
    provider.fail = False
    result = learn_from_traces(
        store,
        recall_alias="jev",
        maximum_attempts=10**6,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
        retry_incomplete=True,
    )
    assert result.learned == 1 and result.incomplete == 0
    assert provider.predictions() == predictions

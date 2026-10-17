"""The learning pass: every trace by default, measured test-then-train, budgeted and resumable."""

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _at,
    _obj,
    build_project,
    build_report,
    episode,
    held_out_lineages,
)
from exp.optimize.evaluation.fidelity.runner import FidelityMemory
from exp.optimize.evaluation.learning.traces import (
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_spec,
)
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    consumed_task_lineages,
    is_reserved,
    memory_head,
)


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

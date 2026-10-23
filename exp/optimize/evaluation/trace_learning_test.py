"""The learning pass: every trace by default, measured test-then-train, budgeted and resumable."""

import functools
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.models import load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation import held_out_fidelity_test, trace_learning, trace_replay
from exp.optimize.evaluation.fidelity import FidelityMemory, FidelitySpec
from exp.optimize.evaluation.held_out_fidelity_test import (
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
    package_rows,
    write_catalog,
)
from exp.optimize.evaluation.trace_learning import (
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_spec,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
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
    changed = lambda: "changed-prompts"  # noqa: E731 - one stand-in for an edited prompt
    monkeypatch.setattr(trace_learning, "protocol_prompts_sha256", changed)
    monkeypatch.setattr(trace_replay, "protocol_prompts_sha256", changed)
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


def test_learning_refuses_to_predict_a_learned_trace_on_later_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run input that drifts without a new lineage stops learning before any call."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, learn=True)
    purchased = len(provider.requests)
    content_run_id = trace_replay.content_run_id

    def drifted(spec: FidelitySpec, label: str, trace: str, generation: int = 1) -> str:
        """Every run ID changes while the lineage stays the same."""
        return content_run_id(spec, label, trace + "-drift", generation)

    monkeypatch.setattr(trace_replay, "content_run_id", drifted)
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

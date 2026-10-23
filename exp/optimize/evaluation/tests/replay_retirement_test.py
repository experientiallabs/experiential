"""Derived held-out and learning runs are replaced only when that buys nothing twice."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BillingSource,
    ModelCapabilities,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore
from exp.optimize.evaluation import trace_learning, trace_replay
from exp.optimize.evaluation.fidelity import execute_fidelity_evaluation
from exp.optimize.evaluation.fidelity_memory import DUPLICATE, saved_memory_receipt
from exp.optimize.evaluation.held_out_fidelity import register_held_out_fidelity
from exp.optimize.evaluation.held_out_fidelity_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    _arr,
    _at,
    _obj,
    _user_content,
    build_project,
    episode,
)
from exp.optimize.evaluation.miss_retest import register_miss_retest
from exp.optimize.evaluation.tests.fidelity_memory_test import memory_project
from exp.optimize.evaluation.trace_learning import (
    fit_calls,
    learn_from_traces,
    learning_curve,
    learning_plan,
    learning_progress,
)
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.retrieval import load_rag_index
from exp.simulation.world_model.observed_tools import observed_calls


def now() -> datetime:
    """Aware wall clock."""
    return datetime.now(UTC)


def add_unrelated_model(store: ProjectStore) -> None:
    """Edit the catalog without touching any alias the runs use."""
    catalog = load_model_catalog(store.model_catalog_path)
    spare = ModelRecord(
        connection="http",
        model="spare-v1",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities=ModelCapabilities(supports_tools=True, context_window_tokens=1000),
    )
    write_model_catalog(
        store.model_catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, "spare": spare}}),
    )


class Interrupting(SimulatedProvider):
    """Kills the process once, with a request already sent and its outcome unknown."""

    crash_at: int | None = None

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Interrupt the chosen request after it leaves the process."""
        if self.crash_at is not None and len(self.requests) == self.crash_at:
            self.crash_at = None
            raise KeyboardInterrupt("process killed mid-dispatch")
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def test_a_catalog_edit_replaces_a_registration_that_never_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executor refuses a changed catalog, so the unexecuted run is re-addressed."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = register_held_out_fidelity(
        store, 1, profile="source-rag", code_revision=REVISION, clock=now
    )
    add_unrelated_model(store)
    again = register_held_out_fidelity(
        store, 1, profile="source-rag", code_revision=REVISION, clock=now
    )
    assert again.retired == (first.prepared[0].spec.run_id,)
    report = execute_fidelity_evaluation(store, again.prepared[0])
    assert report.planned_steps > 0


def test_a_learning_pass_continues_after_a_catalog_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget stop leaves the next trace registered; editing the catalog must not wedge it."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    stopped = learn_from_traces(
        store, recall_alias="jev", maximum_attempts=1, code_revision=REVISION, clock=now
    )
    assert stopped.stopped is not None
    add_unrelated_model(store)
    done = learn_from_traces(
        store, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
    )
    assert done.stopped is None and done.learned == done.traces


def test_an_interrupted_run_that_bought_calls_is_not_bought_again_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --retry-incomplete the run stays and stops visibly; with it, a fresh attempt."""
    provider = Interrupting([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    panel = register_held_out_fidelity(
        store, 1, profile="source-rag", code_revision=REVISION, clock=now
    )
    provider.crash_at = len(provider.requests) + 2
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, panel.prepared[0])
    kept = register_held_out_fidelity(
        store, 1, profile="source-rag", code_revision=REVISION, clock=now
    )
    assert kept.retired == () and kept.prepared[0].id == panel.prepared[0].id
    sent = len(provider.requests)
    with pytest.raises(ValueError, match="--retry-incomplete"):
        execute_fidelity_evaluation(store, kept.prepared[0])
    assert len(provider.requests) == sent
    retried = register_held_out_fidelity(
        store,
        1,
        profile="source-rag",
        code_revision=REVISION,
        clock=now,
        retry_incomplete=True,
    )
    assert retried.retired == (panel.prepared[0].spec.run_id,)
    assert execute_fidelity_evaluation(store, retried.prepared[0]).planned_steps > 0


class FlakyInvestigator(SimulatedProvider):
    """The first ``failures`` investigations return output the completion owner refuses."""

    failures = 1

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one investigation with an argument the schema refuses."""
        response = super().post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        schema = payload.get("tools")
        investigating = (
            isinstance(schema, list)
            and schema
            and "hypothesis" in json.dumps(schema[0])
            and "axes" not in json.dumps(schema[0])
        )
        if investigating and self.failures > 0:
            self.failures -= 1
            body = json.loads(json.dumps(response.body))
            call = body["choices"][0]["message"]["tool_calls"][0]["function"]
            call["arguments"] = json.dumps({"unexpected": True})
            return JsonHttpResponse(status_code=200, body=body)
        return response


@pytest.mark.parametrize("failures", [1, 99])
def test_scores_keep_first_report_and_retests_require_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failures: int
) -> None:
    """Every trace is predicted whole; learning success decides learning, never scoring."""
    provider = FlakyInvestigator([episode(i) for i in range(USERS)], learnable=True)
    provider.failures = failures
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    outcome = learn_from_traces(
        store, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
    )
    plan = learning_plan(store)
    scored, unscored = learning_progress(store, plan)
    assert not unscored and len(scored) == USERS, "every trace is scored"
    assert sum(learned for *_, learned in scored) == outcome.learned
    for case, prepared, result, learned in scored:
        assert result.report.mechanical["valid_steps"] == result.steps, "whole trace predicted"
        attempts = trace_learning._saved_attempts(store, plan, case)
        assert prepared.id == attempts[0][0].id, "learning retries cannot replace a score"
        consumed = [
            receipt
            for attempt, _ in attempts
            if (receipt := saved_memory_receipt(store, attempt)) is not None
            and receipt.get("reason") in ("consumed", DUPLICATE)
        ]
        assert bool(consumed) == learned
    curve = learning_curve(store)
    assert curve is not None and len(curve.points) == USERS
    assert curve.incomplete == outcome.incomplete
    if failures == 99:
        # Every investigation is refused, so no trace is learned, yet every trace is scored.
        assert outcome.learned == 0 and curve.incomplete == USERS
        assert f"{USERS} trace(s) not learned" in curve.headline()
        with pytest.raises(ValueError, match="nothing has been learned"):
            register_miss_retest(store, 99, code_revision=REVISION, clock=now)
        return
    assert curve.incomplete == 0
    case, first_attempt, _, _ = scored[0]
    attempts = trace_learning._saved_attempts(store, plan, case)
    assert len(attempts) > 1 and attempts[0][0].id == first_attempt.id, "first attempt stays scored"
    panel = register_miss_retest(store, 99, code_revision=REVISION, clock=now)
    assert panel.learned == USERS
    assert panel.before[0].run_id == first_attempt.spec.run_id


class RedrawnPrediction(FlakyInvestigator):
    """An identical first draw misses; a subsequent draw of that tool happens to be correct."""

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Track prediction draws independently of investigation success."""
        super().__init__(episodes)
        self.first_task: str | None = None
        self.draws = 0

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Miss the first task's first user-details draw regardless of investigator behavior."""
        result = super()._transition(payload)
        body = _user_content(payload)
        task = str(_at(body, "task", "initial_context", "task_lineage"))
        if self.first_task is None:
            self.first_task = task
        tool = _at(body, "candidate_response", "tool_calls", 0, "name")
        if task == self.first_task and tool == "get_user_details":
            self.draws += 1
            if self.draws == 1:
                _obj(_arr(result["observations"])[0])["reply"] = {"error": "first draw missed"}
        return result


@pytest.mark.parametrize("investigation_failures", [0, 1])
def test_investigation_failure_cannot_select_a_better_prediction_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, investigation_failures: int
) -> None:
    """Retrying learning must preserve the first measurement and its miss-retest baseline."""
    episodes = [episode(i) for i in range(4)]
    provider = RedrawnPrediction(episodes)
    provider.failures = investigation_failures
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    outcome = learn_from_traces(
        store, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
    )
    assert outcome.learned == 4 and outcome.incomplete == 0
    curve = learning_curve(store)
    assert curve is not None
    assert sum(point.exact for point in curve.points) == 11
    assert sum(point.steps for point in curve.points) == 12
    case = trace_learning.learning_cases(store, "all")[0]
    attempts = trace_learning._saved_attempts(store, learning_plan(store), case)
    assert len(attempts) == 1 + investigation_failures
    assert curve.points[0].run_id == attempts[0][0].spec.run_id
    panel = register_miss_retest(store, 4, code_revision=REVISION, clock=now)
    assert panel.before[0].run_id == attempts[0][0].spec.run_id


def test_a_world_model_without_tools_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structured transitions and investigations need tools; refuse clearly, buy nothing."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    catalog = load_model_catalog(store.model_catalog_path)
    world = catalog.roles.world_model
    assert world is not None
    record = catalog.models[world]
    assert record.capabilities is not None
    plain = record.model_copy(
        update={"capabilities": record.capabilities.model_copy(update={"supports_tools": False})}
    )
    write_model_catalog(
        store.model_catalog_path,
        catalog.model_copy(update={"models": {**catalog.models, world: plain}}),
    )
    sent = len(provider.requests)
    with pytest.raises(ValueError, match="tool-capable world model"):
        learn_from_traces(
            store, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
        )
    assert len(provider.requests) == sent


def test_editing_a_sent_prompt_addresses_a_new_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved results never replay under a system prompt they were not produced with."""
    _, base = memory_project(tmp_path)
    before = trace_replay.content_run_id(base, "held-out", "trace")
    monkeypatch.setattr(
        trace_replay, "MEMORY_CONTEXT_GUIDANCE", trace_replay.MEMORY_CONTEXT_GUIDANCE + " Edited."
    )
    assert trace_replay.content_run_id(base, "held-out", "trace") != before


def test_both_profiles_derive_tool_schemas_from_the_same_fit_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory-first and source-RAG see the same fit tool inventory, prefixes included."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    build = store.load_project().build
    assert build is not None
    fit = load_rag_index(store.artifacts, build.fit_rag.artifact_id)
    assert {name for name, _ in fit_calls(store)} == {
        name for name, _ in observed_calls(fit.transitions)
    }

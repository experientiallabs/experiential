"""An interrupted fidelity run resumes in place: saved work replays, unknown outcomes are flagged.

Every test kills the process (``KeyboardInterrupt``) at one point of a run, then continues it
with ``--retry-incomplete`` semantics. The same run identity continues from its first
uncommitted step; nothing with a saved response is requested again; a request that left the
process without a saved response is re-dispatched, and the trace is a trace re-attempt: the
headline is qualified with the held-out audit's label, names the re-dispatches by stage, and adds
unaffected-only and affected-only scores with their denominators.
"""

import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.audit.lifecycle import open_held_out_audit
from exp.optimize.evaluation.audit.panel import (
    HeldOutPanel,
    register_held_out_fidelity,
    saved_held_out_results,
)
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _at,
    _obj,
    audit_report,
    build_project,
    episode,
    learn_fit,
)
from exp.optimize.evaluation.fidelity.recovery import qualify
from exp.optimize.evaluation.fidelity.replay import DEADLINE, ReplayResult
from exp.optimize.evaluation.fidelity.runner import (
    execute_fidelity_evaluation,
    saved_fidelity_report,
    unknown_outcomes,
)
from exp.optimize.evaluation.learning import traces as trace_learning
from exp.optimize.evaluation.learning.traces import (
    learn_from_traces,
    learning_curve,
    learning_plan,
    learning_progress,
)
from exp.runtime.environments import Observation
from exp.runtime.models.providers.transport import JsonHttpResponse, RecordedRequest
from exp.runtime.sessions.store import SessionStore, StepClaim
from exp.simulation.world_model.durable import DurableWorldModelSession


def now() -> datetime:
    """Aware wall clock."""
    return datetime.now(UTC)


def stage_of(url: str, payload: JsonObject) -> str:
    """Name the role one provider request serves."""
    if url.endswith("/embeddings"):
        return "embedding"
    if url.endswith("/systemone"):
        return "decision"
    if _at(payload, "tool_choice", "function", "name") == "submit_transition":
        return "prediction"
    schema = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
    if "axes" in schema:
        return "judge"
    if "summary" in schema:
        return "compression"
    return "investigation"


class Killing(SimulatedProvider):
    """Kills the process once, right after one chosen request has left it.

    The request is recorded as sent and no response is returned, so its outcome is unknown:
    the provider may have answered, billed it, and logged the answer.

    Attributes:
        kill: Role of the request to interrupt, or None to answer everything.
        after_predictions: Interrupt only once this many predictions were sent.
    """

    kill: str | None = None
    after_predictions: int = 0
    _guard = threading.Lock()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Interrupt the first ``kill`` request once enough predictions were sent."""
        with self._guard:
            killed = (
                self.kill is not None
                and stage_of(url, payload) == self.kill
                and self.sent("prediction") >= self.after_predictions
            )
            if killed:
                self.kill = None
                self.requests.append(RecordedRequest(url, dict(headers), payload))
        if killed:
            raise KeyboardInterrupt("process killed with a request in flight")
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def sent(self, stage: str, since: int = 0) -> int:
        """Requests of one role sent since a request count."""
        return sum(stage_of(row.url, row.payload) == stage for row in self.requests[since:])


def panel(store: ProjectStore, count: int, clock: Callable[[], datetime] = now) -> HeldOutPanel:
    """Register a source-RAG held-out panel of ``count`` traces; registration never buys."""
    return register_held_out_fidelity(
        store, count, profile="source-rag", code_revision=REVISION, clock=clock
    )


def qualified(results: tuple[ReplayResult, ...]) -> tuple[str, ...]:
    """The qualification lines of a report over ``results``; it must be qualified."""
    qualification = qualify([(row.unknown_outcomes, row.exact, row.steps) for row in results])
    assert qualification is not None
    return qualification.lines()


def kill_before_operation(monkeypatch: pytest.MonkeyPatch, ordinal: int) -> None:
    """Kill the process once, before the ``ordinal``-th prediction step claims anything."""
    original = DurableWorldModelSession.step
    seen: list[str] = []

    def step(
        self: DurableWorldModelSession, actions: tuple[ToolCall, ...], *, operation_id: str
    ) -> tuple[Observation, ...]:
        """Count distinct steps; stop the process before the chosen one starts."""
        if operation_id not in seen:
            seen.append(operation_id)
            if len(seen) == ordinal:
                monkeypatch.setattr(DurableWorldModelSession, "step", original)
                raise KeyboardInterrupt("process killed between steps")
        return original(self, actions, operation_id=operation_id)

    monkeypatch.setattr(DurableWorldModelSession, "step", step)


def test_a_crash_after_committed_steps_resumes_without_buying_them_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committed steps replay; the same run continues from its first uncommitted step."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 1)
    steps = len(first.cases[0].steps)
    start = len(provider.requests)
    kill_before_operation(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[0])
    assert provider.sent("prediction", start) == 2, "two steps committed before the crash"
    # Resuming hours later: the run's deadline has passed, and it still continues in place.
    later = now() + DEADLINE + timedelta(minutes=30)
    resumed = panel(store, 1, lambda: later)
    assert resumed.retired == () and resumed.prepared[0].id == first.prepared[0].id
    report = execute_fidelity_evaluation(
        store, resumed.prepared[0], resume_until=later + DEADLINE, clock=lambda: later
    )
    assert report.planned_steps == steps and report.judge_status == "completed"
    assert report.mechanical["valid_steps"] == steps
    assert provider.sent("prediction", start) == steps, "each step predicted exactly once"
    assert provider.sent("judge", start) == 1
    assert unknown_outcomes(store, resumed.prepared[0]) == ()
    sent = len(provider.requests)
    assert execute_fidelity_evaluation(store, resumed.prepared[0]) == report
    assert len(provider.requests) == sent, "a finished run replays without provider calls"


def test_a_saved_response_is_never_dispatched_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prediction answered and saved, but not yet committed when the process died, replays."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 1)
    steps = len(first.cases[0].steps)
    original = SessionStore.record_proposal
    calls: list[int] = []

    def record_proposal(self: SessionStore, claim: StepClaim, proposal: JsonObject) -> None:
        """Die right after the second prediction's response was saved by the request ledger."""
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt("process killed after the response was saved")
        original(self, claim, proposal)

    monkeypatch.setattr(SessionStore, "record_proposal", record_proposal)
    start = len(provider.requests)
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[0])
    monkeypatch.setattr(SessionStore, "record_proposal", original)
    assert provider.sent("prediction", start) == 2
    report = execute_fidelity_evaluation(store, first.prepared[0], resume_until=now() + DEADLINE)
    assert report.mechanical["valid_steps"] == steps
    assert provider.sent("prediction", start) == steps, "the saved answer replayed"
    assert unknown_outcomes(store, first.prepared[0]) == ()


def test_without_retry_incomplete_an_interrupted_run_stops_visibly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resumption is explicit: the plain command names the flag and buys nothing."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 1)
    provider.kill, provider.after_predictions = "prediction", 1
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[0])
    kept = panel(store, 1)
    sent = len(provider.requests)
    with pytest.raises(ValueError, match="--retry-incomplete"):
        execute_fidelity_evaluation(store, kept.prepared[0])
    assert len(provider.requests) == sent


def test_a_claimed_prediction_without_a_response_is_retried_and_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lost prediction is dispatched again and the headline is qualified."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 2)
    steps = len(first.cases[0].steps)
    execute_fidelity_evaluation(store, first.prepared[0])
    start = len(provider.requests)
    provider.kill, provider.after_predictions = "prediction", provider.sent("prediction") + 1
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[1])
    resumed = panel(store, 2)
    assert resumed.prepared == first.prepared and resumed.retired == ()
    report = execute_fidelity_evaluation(store, resumed.prepared[1], resume_until=now() + DEADLINE)
    assert report.mechanical["valid_steps"] == steps
    # Step 0 committed once; step 1 was sent, lost, and sent again; step 2 once.
    assert provider.sent("prediction", start) == steps + 1
    retries = unknown_outcomes(store, resumed.prepared[1])
    assert [(row.stage, row.step, row.attempts) for row in retries] == [("prediction", 1, 2)]
    assert unknown_outcomes(store, resumed.prepared[0]) == ()

    results = saved_held_out_results(store, resumed)
    assert results[1].unknown_outcomes == retries
    stages, unaffected, affected = qualified(results)
    assert "prediction 1 · judge 0 · investigation 0 · 1 of 2 traces affected" in stages
    clean, lost = results
    assert f"Unaffected traces only  {clean.exact}/{clean.steps} replies exact" in unaffected
    assert f"Re-attempted traces only  {lost.exact}/{lost.steps} replies exact" in affected
    assert unaffected.endswith("· 1/1 traces") and affected.endswith("· 1/1 traces")
    assert "step 2 re-dispatched after an unknown outcome (prediction, 2 attempts)" in "\n".join(
        results[1].recovery_lines()
    )


def test_only_the_missing_judge_runs_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every prediction is committed; resuming dispatches the judge alone, flagged."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 1)
    provider.kill = "judge"
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[0])
    start = len(provider.requests)
    resumed = panel(store, 1)
    report = execute_fidelity_evaluation(store, resumed.prepared[0], resume_until=now() + DEADLINE)
    assert report.judge_status == "completed"
    assert [stage_of(row.url, row.payload) for row in provider.requests[start:]] == ["judge"]
    retries = unknown_outcomes(store, resumed.prepared[0])
    assert [(row.stage, row.step, row.attempts) for row in retries] == [("judge", None, 2)]
    stages, *_ = qualified(saved_held_out_results(store, resumed))
    assert "prediction 0 · judge 1 · investigation 0 · 1 of 1 traces affected" in stages


def test_a_panel_resumes_trace_by_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Finished traces are untouched, the interrupted one resumes, the rest run fresh."""
    provider = Killing([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    first = panel(store, 3)
    execute_fidelity_evaluation(store, first.prepared[0])
    finished = saved_fidelity_report(store, first.prepared[0])
    provider.kill, provider.after_predictions = "prediction", provider.sent("prediction") + 2
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(store, first.prepared[1])
    start = len(provider.requests)
    resumed = panel(store, 3)
    assert resumed.prepared == first.prepared and resumed.retired == ()
    window = now() + DEADLINE
    reports = [
        execute_fidelity_evaluation(store, prepared, resume_until=window)
        for prepared in resumed.prepared
    ]
    assert reports[0] == finished
    steps = [len(case.steps) for case in resumed.cases]
    # Trace 1 replays; trace 2 re-sends only its lost third step; trace 3 runs whole.
    assert provider.sent("prediction", start) == 1 + steps[2]
    assert provider.sent("judge", start) == 2
    assert [len(unknown_outcomes(store, p)) for p in resumed.prepared] == [0, 1, 0]
    results = saved_held_out_results(store, resumed)
    assert len(results) == 3
    assert "1 of 3 traces affected" in qualified(results)[0]


def test_a_missing_investigation_receipt_is_retried_and_saved_ones_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Learning resumes the same trace: lost investigations rerun flagged, saved ones replay."""
    episodes = [episode(i) for i in range(4)]
    provider = Killing(episodes, wrong=frozenset({"get_order_details"}), learnable=True)
    _, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    provider.kill = "investigation"
    with pytest.raises(KeyboardInterrupt):
        learn_from_traces(
            store, recall_alias="jev", maximum_attempts=10_000, code_revision=REVISION, clock=now
        )
    plan = learning_plan(store)
    case = trace_learning.learning_cases(store, "all")[0]
    start = len(provider.requests)
    investigated = provider.sent("investigation")
    assert investigated >= 2, "both misses of the first trace were sent before the crash"
    outcome = learn_from_traces(
        store,
        recall_alias="jev",
        maximum_attempts=10_000,
        code_revision=REVISION,
        clock=now,
        retry_incomplete=True,
    )
    assert outcome.learned == outcome.traces == 4
    attempts = trace_learning._saved_attempts(store, plan, case)
    assert len(attempts) == 1, "the interrupted trace resumed; no new generation"
    first_trace = [
        row
        for row in provider.requests[start:]
        if stage_of(row.url, row.payload) in ("prediction", "investigation")
    ]
    # The first trace's predictions all committed and replay; only its lost investigation is
    # sent again, before the second trace's first prediction.
    assert stage_of(first_trace[0].url, first_trace[0].payload) == "investigation"
    assert stage_of(first_trace[1].url, first_trace[1].payload) == "prediction"
    retries = unknown_outcomes(store, attempts[0][0])
    assert [(row.stage, row.attempts) for row in retries] == [("investigation", 2)]
    scored, _ = learning_progress(store, plan)
    assert scored[0][2].unknown_outcomes == retries
    curve = learning_curve(store)
    assert curve is not None
    headline, stages, unaffected, affected, *_ = curve.headline().splitlines()
    assert headline.endswith("· 4 traces · qualified: trace re-attempts")
    assert "prediction 0 · judge 0 · investigation 1 · 1 of 4 traces affected" in stages
    assert "Unaffected traces only" in unaffected and unaffected.endswith("· 3/3 traces")
    assert "Re-attempted traces only" in affected and affected.endswith("· 1/1 traces")


def test_the_cli_resumes_only_with_retry_incomplete_and_qualifies_its_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exp eval --audit`` refuses an interrupted panel, then resumes it in place."""
    provider = Killing([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    command = ["eval", PROJECT, "--root", str(root), "--audit", "3"]
    command += ["--yes", "--non-interactive"]
    provider.kill, provider.after_predictions = "prediction", provider.sent("prediction") + 4
    killed = CliRunner().invoke(app, command)
    assert killed.exit_code != 0 and provider.kill is None, "the process was killed mid-panel"
    sent = len(provider.requests)
    # The audit command binds its runs to the installed source.
    with open_held_out_audit(
        store, 3, profile="memory", code_revision=installed_execution_revision(), clock=now
    ) as audit:
        registrations = audit.panel.prepared
        committed = {
            prepared.id: SessionStore(
                store.artifacts.project_directory / (prepared.id + "-sessions.sqlite")
            )
            .read_session(prepared.id)
            .state["transitions"]
            for prepared in registrations
        }
    assert all(isinstance(rows, list) for rows in committed.values())
    missing_predictions = sum(
        len(prepared.spec.workload.actions) - len(committed[prepared.id])
        for prepared in registrations
        if isinstance(committed[prepared.id], list)
    )
    refused = CliRunner().invoke(app, command)
    assert refused.exit_code != 0 and "--retry-incomplete" in " ".join(refused.output.split())
    assert "cost of this command" not in refused.output, "refused before any spend consent"
    assert len(provider.requests) == sent, "the plain command buys nothing"
    resumed = CliRunner().invoke(app, [*command, "--retry-incomplete"])
    assert resumed.exit_code == 0, resumed.output
    output = " ".join(resumed.output.split()) + " " + audit_report(store)
    assert "1 interrupted trace(s) resume in place" in output
    assert provider.sent("prediction", sent) == missing_predictions, (
        "only uncommitted steps may dispatch; concurrent traces can stop at different boundaries"
    )
    for prepared in registrations:
        before = committed[prepared.id]
        after = (
            SessionStore(store.artifacts.project_directory / (prepared.id + "-sessions.sqlite"))
            .read_session(prepared.id)
            .state["transitions"]
        )
        assert isinstance(before, list) and isinstance(after, list)
        assert after[: len(before)] == before, "recovery changed a committed transition"
    # A11 by stage: the receipt's qualified headline over every trace, the re-dispatches by
    # stage, then the unaffected-only and affected-only scores with their denominators.
    assert "Held-out fidelity (independent, qualified: trace re-attempts) 9/9 replies" in output
    assert "· judge 4.0/5 (3/3 traces judged, 0 judge failures) · 3/3 traces" in output
    assert "per-lineage mean exact 100% (95% interval" in output and "3 lineages" in output
    assert (
        "re-dispatched after an unknown outcome prediction 1 · judge 0 · investigation 0 · "
        "1 of 3 traces affected"
    ) in output
    assert "Unaffected traces only 6/6 replies exact (100%)" in output
    assert "Re-attempted traces only 3/3 replies exact (100%)" in output
    assert "· 2/2 traces" in output and "· 1/1 traces" in output
    assert "Report:" in output
    assert "independent, qualified: trace re-attempts · this run: evaluation" in audit_report(store)
    # The receipt's attempt tree holds the in-place re-attempt; the detail names the step.
    purchased_before_replay = len(provider.requests)
    replayed = CliRunner().invoke(app, [*command, "--verbose"])
    assert replayed.exit_code == 0, replayed.output
    shown = " ".join(replayed.output.split())
    assert "Report:" in shown and len(provider.requests) == purchased_before_replay
    details = audit_report(store)
    assert "Saved results. No provider calls." in details
    assert "resumed in place" in details
    assert "re-dispatched after an unknown outcome (prediction, 2 attempts)" in details
    assert "Held-out fidelity (independent, qualified: trace re-attempts) 9/9 replies" in shown

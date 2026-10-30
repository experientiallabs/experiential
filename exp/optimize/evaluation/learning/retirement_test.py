"""A deterministically refused trace retires after the bound; outages never retire."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectStore
from exp.optimize.evaluation.audit.panel_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    _at,
    build_project,
    episode,
    held_out_lineages,
)
from exp.optimize.evaluation.fidelity.memory import trajectory_record
from exp.optimize.evaluation.fidelity.replay import DEADLINE
from exp.optimize.evaluation.learning.repair import TraceRetirementRequest, retire_trace
from exp.optimize.evaluation.learning.retirement import (
    REFUSAL_TYPE,
    RETIREMENT_FAILURES,
    RETIREMENT_TYPE,
    RefusalObservation,
    SystemicRefusalError,
    automatic_retirement_limit,
    refusals,
    retired_traces,
    retirement_of,
)
from exp.optimize.evaluation.learning.traces import (
    LearningOutcome,
    learn_from_traces,
    learning_cases,
    learning_curve,
    learning_plan,
    learning_spec,
)
from exp.runtime.models.providers.backpressure import ProviderAdmissionRefused
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
)
from exp.simulation.world_model.experience_memory import (
    is_reserved,
    memory_head,
)
from exp.simulation.world_model.experience_sources import (
    consumed_trajectory_digest,
)

STREAM_REFUSAL: JsonObject = {
    "error": {
        "code": "invalid_provider_stream",
        "message": "Chat reasoning content requires complete unique tool calls.",
        "type": "api_error",
    }
}
OUTAGE: JsonObject = {"error": {"code": "gateway_overloaded", "message": "try again later"}}


class RefusingProvider(SimulatedProvider):
    """The routed fake, answering every prediction of one task with a fixed HTTP 503.

    Attributes:
        task_id: Learning task whose predictions are refused, or None to refuse nothing.
        refuse_all: Refuse every prediction of every task.
        body: Error body of the refusal.
        refused: Every refused request body, in order (retries included).
    """

    def __init__(self, episodes: list[JsonObject]) -> None:
        """Route like the fake; refuse nothing until a task is named."""
        super().__init__(episodes, learnable=True)
        self.task_id: str | None = None
        self.refuse_all = False
        self.body: JsonObject = STREAM_REFUSAL
        self.refused: list[JsonObject] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Refuse the target task's predictions; route everything else normally."""
        if (
            url.endswith("/chat/completions")
            and _at(payload, "tool_choice", "function", "name") == "submit_transition"
            and (
                self.refuse_all
                or (self.task_id is not None and f'"{self.task_id}-' in json.dumps(payload))
            )
        ):
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            self.refused.append(payload)
            return JsonHttpResponse(status_code=503, body=self.body)
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def refusing_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: JsonObject
) -> tuple[ProjectStore, RefusingProvider, str]:
    """A built project whose second fit trace's predictions are always refused with ``body``."""
    provider = RefusingProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    target = learning_cases(store, "fit")[1]
    provider.task_id, provider.body = target.task.task_id, body
    return store, provider, target.trace_id


def learn(store: ProjectStore, *, retry_incomplete: bool = False) -> LearningOutcome:
    """One fit learning pass with an ample budget."""
    return learn_from_traces(
        store,
        recall_alias="jev",
        scope="fit",
        maximum_attempts=10**6,
        code_revision=REVISION,
        clock=lambda: datetime.now(UTC),
        retry_incomplete=retry_incomplete,
    )


def test_a_deterministic_refusal_retires_the_trace_and_the_pass_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fit-trace 31202 blocker: identical refused resends end in a recorded retirement.

    The first refusal stops the pass as any provider failure does. ``--retry-incomplete``
    resends the identical request until the registered bound, records the trace as not learned
    with every refusal as evidence, and continues: every other trace is learned, the memory
    lineage keeps its identity, held-out lineages stay reserved, and the retired trace is
    counted but never scored. A rerun replays without calls.
    """
    store, provider, trace_id = refusing_project(tmp_path, monkeypatch, STREAM_REFUSAL)
    lineage = learning_plan(store, "jev", "fit").lineage
    with pytest.raises(ProviderTransportError) as stopped:
        learn(store)
    assert stopped.value.error_code == "invalid_provider_stream"
    assert len(store.artifacts.list_ids(artifact_type=REFUSAL_TYPE)) == 1
    assert retirement_of(store, lineage, trace_id) is None, "one refusal is not deterministic"
    outcome = learn(store, retry_incomplete=True)
    cases = learning_cases(store, "fit")
    assert outcome.stopped is None and outcome.retired == 1
    assert outcome.learned == len(cases) - 1 and outcome.incomplete == 0
    retirement = retirement_of(store, lineage, trace_id)
    assert retirement is not None and retirement.cause == "deterministic-refusal"
    assert len(retirement.refusals) == RETIREMENT_FAILURES
    assert len({row.request_sha256 for row in retirement.refusals}) == 1, "identical request"
    assert "invalid_provider_stream (HTTP 503) on 3 identical requests" == retirement.reason
    refused = {json.dumps(body, sort_keys=True) for body in provider.refused}
    assert len(refused) == 1, "every resend carried the identical request body"
    plan = learning_plan(store, scope="fit")
    assert plan.lineage == lineage, "retirement never changes the memory identity"
    head = memory_head(store.artifacts, lineage)
    assert head is not None and head.version == len(cases) - 1
    target = next(case for case in cases if case.trace_id == trace_id)
    spec = learning_spec(store, plan, target, deadline=datetime.now(UTC) + DEADLINE)
    assert consumed_trajectory_digest(trajectory_record(spec)) not in head.consumed
    held = held_out_lineages(store)
    assert held and all(is_reserved(store.artifacts, lineage, group) for group in held)
    curve = learning_curve(store, "fit")
    assert curve is not None and curve.retired == 1 and curve.incomplete == 1
    assert len(curve.points) == len(cases) - 1, "a retired trace is never scored"
    assert "1 trace(s) not learned (1 retired, unscored)" in curve.headline()
    (row,) = retired_traces(store, lineage, tuple(case.trace_id for case in cases))
    assert row.trace_id == trace_id and row.record.artifact_id.startswith(RETIREMENT_TYPE)
    calls = len(provider.requests)
    again = learn(store, retry_incomplete=True)
    assert (again.learned, again.retired, again.attempts_used) == (len(cases) - 1, 1, 0)
    assert len(provider.requests) == calls, "a retired trace is never bought again"


def test_a_transport_outage_is_never_retired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overloaded gateway stays resumable however often the identical request fails.

    Repeated overload opens the host's provider breaker, so later passes are refused before
    dispatch rather than resent; either stop is resumable and none of them is a refusal.
    """
    store, _, trace_id = refusing_project(tmp_path, monkeypatch, OUTAGE)
    lineage = learning_plan(store, "jev", "fit").lineage
    with pytest.raises((ProviderTransportError, ProviderAdmissionRefused)):
        learn(store)
    for _ in range(RETIREMENT_FAILURES + 1):
        with pytest.raises(
            (ProviderTransportError, DispatchOutcomeUnknown, ProviderAdmissionRefused)
        ):
            learn(store, retry_incomplete=True)
    assert retirement_of(store, lineage, trace_id) is None
    assert not store.artifacts.list_ids(artifact_type=REFUSAL_TYPE)
    assert not store.artifacts.list_ids(artifact_type=RETIREMENT_TYPE)


def test_a_refusal_below_the_bound_stops_without_a_resumption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--retry-incomplete`` nothing is resent: each run stops, one refusal recorded."""
    store, provider, trace_id = refusing_project(tmp_path, monkeypatch, STREAM_REFUSAL)
    with pytest.raises(ProviderTransportError):
        learn(store)
    sent = len(provider.refused)
    with pytest.raises(ValueError, match="--retry-incomplete"):
        learn(store)
    assert len(provider.refused) == sent
    lineage = learning_plan(store, scope="fit").lineage
    assert retirement_of(store, lineage, trace_id) is None


def test_systemic_refusals_stop_the_pass_instead_of_retiring_every_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every trace refused: retirement stops at the registered bound and the rest stay resumable.

    An operator retirement does not count toward the bound. The stopping message names the
    code and the affected traces and says what to do; the trace that tripped the bound keeps
    its recorded refusals, is not retired, and learns once the provider recovers.
    """
    store, provider, _ = refusing_project(tmp_path, monkeypatch, STREAM_REFUSAL)
    provider.task_id, provider.refuse_all = None, True
    cases = learning_cases(store, "fit")
    assert automatic_retirement_limit(len(cases)) == 1 < len(cases) - 2
    lineage = learning_plan(store, "jev", "fit", record=True).lineage
    operator, automatic, tripped = (case.trace_id for case in cases[:3])
    retire_trace(
        store,
        TraceRetirementRequest(resolution="retire-trace", trace_id=operator, reason="known bad"),
        code_revision=REVISION,
    )
    with pytest.raises(ProviderTransportError):
        learn(store)
    with pytest.raises(SystemicRefusalError) as stopped:
        learn(store, retry_incomplete=True)
    message = str(stopped.value)
    for text in (
        "refusals look systemic",
        "invalid_provider_stream",
        f"affected traces: {automatic}, {tripped}",
        "provider or model settings",
        "repair-learning retire-trace",
    ):
        assert text in message, message
    first = retirement_of(store, lineage, automatic)
    assert first is not None and first.cause == "deterministic-refusal"
    assert retirement_of(store, lineage, tripped) is None, "nothing past the bound is retired"
    assert memory_head(store.artifacts, lineage) is None
    recorded = [
        row
        for identity in store.artifacts.list_ids(artifact_type=REFUSAL_TYPE)
        if (row := refusals_of(store, identity)).trace_id == tripped
    ]
    assert len(recorded) == RETIREMENT_FAILURES, "the tripping refusals stay recorded"
    assert refusals(store, lineage, tripped, recorded[0].run, recorded[0].request_sha256) == tuple(
        sorted(recorded, key=lambda row: row.ordinal)
    )
    assert len(store.artifacts.list_ids(artifact_type=RETIREMENT_TYPE)) == 2
    provider.refuse_all = False  # the provider recovers
    outcome = learn(store, retry_incomplete=True)
    assert (outcome.learned, outcome.retired, outcome.stopped) == (len(cases) - 2, 2, None)
    assert learning_plan(store, scope="fit").lineage == lineage


def refusals_of(store: ProjectStore, identity: str) -> RefusalObservation:
    """Read one recorded refusal."""
    return RefusalObservation.model_validate_json(
        store.artifacts.read_bytes(identity, "refusal.json")
    )

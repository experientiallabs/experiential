"""Tests for tagging over the typed decisions surface."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import BillingSource, ModelCapabilities, ModelSnapshot
from exp.common.project.paths import ProjectPaths
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS, DecisionRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import ResolvedModel
from exp.simulation.mining.decisions import (
    DecisionsTraceTagger,
    parse_decision_answers,
)
from exp.simulation.mining.questions import Question, QuestionBank, packaged_question_bank
from exp.simulation.mining.tagging import (
    DecisionCache,
    TraceTagger,
    TraceTaggingError,
    decision_key,
    question_chunks,
)
from exp.simulation.mining.views import Unit

_SNAPSHOT = ModelSnapshot(
    provider="fixture",
    model_id="decisions-fixture",
    billing_source=BillingSource.CUSTOMER_MANAGED,
    capabilities_sha256="a" * 64,
    connection_sha256="b" * 64,
)


def _bank(count: int = 3) -> QuestionBank:
    """Return a small bank whose questions share one family."""
    return QuestionBank(
        bank_id="fixture-v1",
        version=1,
        questions=tuple(
            Question(question_id=f"sit.q{index}", family="situation", text=f"Did thing {index}?")
            for index in range(count)
        ),
    )


def _unit(trace_id: str, text: str = "one episode") -> Unit:
    """Return a rendered observations unit."""
    return Unit(trace_id, "observations", text)


def _answers(
    bank: QuestionBank,
    *,
    cost: float = 0.001,
    drop: str | None = None,
    replace: tuple[str, object] | None = None,
) -> JsonObject:
    """Return a response answering the bank, optionally with one answer missing or malformed.

    Args:
        bank: Questions the response answers.
        cost: Reported spend.
        drop: Question identity to omit entirely.
        replace: Question identity and the malformed value to put in its place.

    Returns:
        The response body.
    """
    answers: dict[str, object] = {
        question.question_id: {"type": "noul", "noul": 0.1 + index / 100}
        for index, question in enumerate(bank.questions)
        if question.question_id != drop
    }
    if replace is not None:
        answers[replace[0]] = replace[1]
    return {
        "id": "decision_fixture",
        "model": "clef",
        "answers": answers,
        "usage": {"input_tokens": 9532, "output_tokens": 0, "cost": cost},
    }


@dataclass
class _Recorder:
    """A decision boundary that replays prepared bodies and records every request."""

    bodies: list[JsonObject]
    requests: list[DecisionRequest] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Record one request and return the next prepared body."""
        self.requests.append(request)
        if not self.bodies:
            raise AssertionError("test made an unexpected decision call")
        return self.bodies.pop(0)


def test_the_tagger_satisfies_the_shared_seam() -> None:
    """Nothing downstream may depend on which tagger answered."""
    tagger = DecisionsTraceTagger(_Recorder([]), _SNAPSHOT)

    assert isinstance(tagger, TraceTagger)


def test_the_whole_packaged_bank_fits_one_decision() -> None:
    """Repeating a 9,500-token episode per call is the dominant cost, so batching matters."""
    bank = packaged_question_bank()
    recorder = _Recorder([_answers(bank)])
    tagger = DecisionsTraceTagger(recorder, _SNAPSHOT)

    matrix = tagger.tag([_unit("trace-1")], bank)

    assert len(recorder.requests) == 1
    assert len(question_chunks(bank.questions)) == 1
    assert len(matrix.rows[0].values) == len(bank.questions)
    assert len(recorder.requests[0].questions) == len(bank.questions)


def test_questions_are_sent_as_propositions_with_their_exact_text() -> None:
    """The question text is part of the cache key, so it must reach the model unaltered."""
    bank = _bank()
    recorder = _Recorder([_answers(bank)])

    DecisionsTraceTagger(recorder, _SNAPSHOT).tag([_unit("trace-1")], bank)

    sent = recorder.requests[0]
    assert sent.state == "one episode"
    assert set(sent.questions) == {"sit.q0", "sit.q1", "sit.q2"}
    assert sent.questions["sit.q1"].type == "noul"
    assert sent.questions["sit.q1"].instructions == "Did thing 1?"


def test_rows_follow_input_order_and_carry_the_tagged_text() -> None:
    """A row binds to the exact text it was answered about, so re-rendering is detectable."""
    bank = _bank()
    units = [_unit("trace-b", "b"), _unit("trace-a", "a")]
    recorder = _Recorder([_answers(bank), _answers(bank)])

    matrix = DecisionsTraceTagger(recorder, _SNAPSHOT).tag(units, bank)

    assert [row.trace_id for row in matrix.rows] == ["trace-b", "trace-a"]
    assert matrix.rows[0].unit_hash == units[0].unit_hash
    assert matrix.values_for(units[1]) == matrix.rows[1].values


def test_two_traces_rendering_one_text_are_decided_once() -> None:
    """Identical episodes are the common case in a duplicated corpus and must not be paid twice."""
    bank = _bank()
    recorder = _Recorder([_answers(bank)])

    matrix = DecisionsTraceTagger(recorder, _SNAPSHOT).tag(
        [_unit("trace-1", "same"), _unit("trace-2", "same")], bank
    )

    assert len(recorder.requests) == 1
    assert len(matrix.rows) == 2
    assert matrix.rows[0].values == matrix.rows[1].values


def test_a_rerun_replays_the_cache_and_buys_nothing(tmp_path: Path) -> None:
    """Buy once and replay forever is what makes a paid corpus affordable to iterate on."""
    bank = _bank()
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    recorder = _Recorder([_answers(bank)])

    first = DecisionsTraceTagger(recorder, _SNAPSHOT, cache=cache).tag([_unit("trace-1")], bank)
    replayed = DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, cache=cache).tag(
        [_unit("trace-1")], bank
    )

    assert first.rows == replayed.rows


def test_a_reworded_question_is_a_new_purchase(tmp_path: Path) -> None:
    """Replaying an answer to a question nobody asked would be worse than paying again."""
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    original = _bank()
    reworded = QuestionBank(
        bank_id="fixture-v1",
        version=2,
        questions=(
            Question(question_id="sit.q0", family="situation", text="Did thing 0, precisely?"),
            *original.questions[1:],
        ),
    )
    DecisionsTraceTagger(_Recorder([_answers(original)]), _SNAPSHOT, cache=cache).tag(
        [_unit("trace-1")], original
    )

    recorder = _Recorder([_answers(reworded)])
    DecisionsTraceTagger(recorder, _SNAPSHOT, cache=cache).tag([_unit("trace-1")], reworded)

    assert len(recorder.requests) == 1


def test_a_missing_answer_raises_rather_than_reading_as_zero() -> None:
    """A dropped answer and a confident no are different evidence."""
    bank = _bank()

    with pytest.raises(TraceTaggingError, match="no answer for question sit.q1"):
        DecisionsTraceTagger(_Recorder([_answers(bank, drop="sit.q1")]), _SNAPSHOT).tag(
            [_unit("trace-7")], bank
        )


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "noul": "0.5"},
        {"type": "noul", "noul": True},
        {"type": "noul", "noul": 1.3},
        {"type": "noul", "noul": -0.1},
        {"type": "noul", "noul": None},
        {"type": "noul"},
        {"type": "choice", "choice": "yes"},
        "0.5",
    ],
)
def test_an_answer_that_is_not_a_probability_raises(answer: object) -> None:
    """Every one of these would otherwise enter the geometry as a fabricated coordinate.

    A bool matters especially: it is an int in Python, so an unguarded read would turn a malformed
    answer into a maximally confident one.
    """
    bank = _bank(1)
    body = _answers(bank, replace=("sit.q0", answer))

    with pytest.raises(TraceTaggingError):
        DecisionsTraceTagger(_Recorder([body]), _SNAPSHOT).tag([_unit("trace-7")], bank)


def test_a_response_with_no_answers_object_names_the_record() -> None:
    """A paid reply is saved even when it is unusable, so the failure must say what to remove."""
    bank = _bank()

    with pytest.raises(TraceTaggingError, match="trace-decisions records"):
        DecisionsTraceTagger(_Recorder([{"id": "decision_1"}]), _SNAPSHOT).tag(
            [_unit("trace-7")], bank
        )


def test_the_failure_names_the_trace_it_was_asking_about() -> None:
    """A corpus-wide run needs to point at the one episode that failed."""
    bank = _bank()
    body = _answers(bank, replace=("sit.q2", {"type": "noul", "noul": "x"}))

    with pytest.raises(TraceTaggingError, match="trace-42"):
        DecisionsTraceTagger(_Recorder([body]), _SNAPSHOT).tag([_unit("trace-42")], bank)


def test_parsing_reads_the_asked_order_not_the_response_order() -> None:
    """Column meaning comes from the bank, so a reordered response must still align."""
    bank = _bank()
    body: dict[str, object] = {
        "answers": {
            "sit.q2": {"type": "noul", "noul": 0.3},
            "sit.q0": {"type": "noul", "noul": 0.1},
            "sit.q1": {"type": "noul", "noul": 0.2},
        }
    }

    assert parse_decision_answers(body, bank.questions, "trace-1") == (0.1, 0.2, 0.3)


def test_a_bank_larger_than_one_request_is_split() -> None:
    """A bank that outgrows the surface's limit must still be answerable."""
    bank = _bank(5)
    recorder = _Recorder([_answers(_bank(2)), _answers(_bank(3))])

    tagger = DecisionsTraceTagger(recorder, _SNAPSHOT, batch_limit=2)
    with pytest.raises(TraceTaggingError):
        # The second prepared body answers sit.q0..q2, not the asked sit.q2..q4, so strict
        # alignment is what catches a chunking mistake instead of silently mixing columns.
        tagger.tag([_unit("trace-1")], bank)
    assert len(recorder.requests) == 2
    assert list(recorder.requests[0].questions) == ["sit.q0", "sit.q1"]


def test_a_batch_limit_the_surface_cannot_carry_is_refused() -> None:
    """Discovering this on dispatch would mean a request rejected after the episode was rendered."""
    with pytest.raises(TraceTaggingError, match=f"1 through {MAX_DECISION_QUESTIONS}"):
        DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, batch_limit=MAX_DECISION_QUESTIONS + 1)
    with pytest.raises(TraceTaggingError):
        DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, batch_limit=0)


@dataclass
class _Client:
    """A resolved client exposing one wire profile."""

    profile: GatewayWireProfile | None

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the prepared profile."""
        assert self.profile is not None
        return self.profile


def _resolved(client: object) -> ResolvedModel:
    """Wrap one client as a resolved alias."""
    return ResolvedModel(
        alias="clef-decisions",
        snapshot=_SNAPSHOT,
        capabilities=ModelCapabilities(),
        client=client,  # ty: ignore[invalid-argument-type]
        embedding_client=None,
    )


def test_an_alias_binds_its_own_endpoint_and_credential() -> None:
    """Nothing here reads a URL or an environment variable; the alias carries both."""
    profile = GatewayWireProfile(
        dialect="typesafe_systemone",
        url="https://example.test/v1/systemone",
        decisions_url="https://example.test/v1/systemone",
        headers={"Authorization": "Bearer token"},
        model_id="clef",
    )
    bank = _bank()
    transport = ScriptedJsonTransport([JsonHttpResponse(200, _answers(bank))])

    tagger = DecisionsTraceTagger.for_alias(_resolved(_Client(profile)), transport=transport)
    matrix = tagger.tag([_unit("trace-1")], bank)

    assert len(matrix.rows) == 1
    assert transport.requests[0].url == "https://example.test/v1/systemone"
    assert transport.requests[0].payload is not None
    assert transport.requests[0].payload["model"] == "clef"


def test_a_chat_only_alias_is_refused_with_the_fix() -> None:
    """The configured decision model serves no chat, and a chat alias serves no decisions."""

    class _ChatOnly:
        """A client with no wire profile at all."""

    with pytest.raises(TraceTaggingError, match="typesafe connection"):
        DecisionsTraceTagger.for_alias(_resolved(_ChatOnly()))

    other = GatewayWireProfile(dialect="openai_compatible", url="https://example.test/v1")
    with pytest.raises(TraceTaggingError, match="exposes no typed"):
        DecisionsTraceTagger.for_alias(_resolved(_Client(other)))


def test_a_failed_call_raises_rather_than_returning_an_empty_row() -> None:
    """A transport failure must stop the run, not produce a row of fabricated zeros."""
    profile = GatewayWireProfile(
        dialect="typesafe_systemone",
        url="https://example.test/v1/systemone",
        decisions_url="https://example.test/v1/systemone",
        model_id="clef",
    )
    transport = ScriptedJsonTransport([JsonHttpResponse(500, {"error": "boom"})])

    tagger = DecisionsTraceTagger.for_alias(
        _resolved(_Client(profile)),
        transport=transport,
        retry=RetryPolicy(maximum_attempts=1),
    )

    with pytest.raises(ProviderTransportError, match="HTTP 500"):
        tagger.tag([_unit("trace-1")], _bank())


def test_a_replayed_run_does_not_report_spend_it_did_not_incur(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cached answer carries the original call's usage, so the two must be told apart."""
    bank = _bank()
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    DecisionsTraceTagger(_Recorder([_answers(bank, cost=0.25)]), _SNAPSHOT, cache=cache).tag(
        [_unit("trace-1")], bank
    )

    with caplog.at_level("INFO", logger="exp.simulation.mining.decisions"):
        DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, cache=cache).tag([_unit("trace-1")], bank)

    assert "bought 0 of them (0 input tokens, 0.0000 USD)" in caplog.text


def test_a_bought_run_reports_what_it_paid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Spend has to be visible without the module naming a provider."""
    bank = _bank()
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))

    with caplog.at_level("INFO", logger="exp.simulation.mining.decisions"):
        DecisionsTraceTagger(_Recorder([_answers(bank, cost=0.25)]), _SNAPSHOT, cache=cache).tag(
            [_unit("trace-1")], bank
        )

    assert "bought 1 of them (9532 input tokens, 0.2500 USD)" in caplog.text


def _state_answers(
    request: DecisionRequest, value: float, *, cost: float = 0.001, tokens: int = 100
) -> JsonObject:
    """Answer every asked question with one value, so a row is traceable to its own call.

    Args:
        request: The request being answered, read for the exact questions it asked.
        value: Probability returned for every question in it.
        cost: Reported spend for this call.
        tokens: Reported input tokens for this call.

    Returns:
        The response body.
    """
    return {
        "id": "decision_fixture",
        "answers": {
            question_id: {"type": "noul", "noul": value} for question_id in request.questions
        },
        "usage": {"input_tokens": tokens, "output_tokens": 0, "cost": cost},
    }


def _episode_text(request: DecisionRequest) -> str:
    """Return the rendered episode one request asked about, as the fixture wrote it."""
    state = request.state
    assert isinstance(state, str)
    return state


def _episode_index(request: DecisionRequest) -> int:
    """Return which fixture episode one request asked about."""
    return int(_episode_text(request).removeprefix("episode "))


def _episodes(count: int) -> list[Unit]:
    """Return units whose rendered text, and therefore cache key, is distinct per trace."""
    return [_unit(f"trace-{index}", f"episode {index}") for index in range(count)]


def _episode_value(unit: Unit) -> float:
    """Return the one probability this fixture answers about one episode."""
    return (int(unit.text.removeprefix("episode ")) + 1) / 100


@dataclass
class _Overlapping:
    """A decision boundary that only answers once every expected caller is inside it.

    A barrier is the proof that calls really overlap: it cannot trip unless that many callers
    are waiting in it at the same moment, so a serial pass fails this fixture by timing out
    rather than by an assertion about a parameter.

    Attributes:
        arrived: Barrier every call waits at before answering.
        cost: Reported spend per call.
        tokens: Reported input tokens per call.
        states: Episode text of every call made, in completion order.
    """

    arrived: threading.Barrier
    cost: float = 0.001
    tokens: int = 100
    states: list[str] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Wait for every sibling call, then answer this one."""
        self.arrived.wait()
        self.states.append(_episode_text(request))
        value = (_episode_index(request) + 1) / 100
        return _state_answers(request, value, cost=self.cost, tokens=self.tokens)


@dataclass
class _Delayed:
    """A decision boundary whose calls finish in the reverse of the order they were made.

    Attributes:
        delay_seconds: Longest wait, taken by the first episode and shrinking from there.
        order: Episode text of every call, in the order the calls completed.
    """

    delay_seconds: float = 0.05
    order: list[str] = field(default_factory=list)

    def __call__(self, request: DecisionRequest) -> JsonObject:
        """Answer one request after a wait that falls with the episode index."""
        index = _episode_index(request)
        time.sleep(max(0.0, self.delay_seconds - index * 0.01))
        self.order.append(_episode_text(request))
        return _state_answers(request, (index + 1) / 100)


def test_the_configured_concurrency_really_overlaps_calls(tmp_path: Path) -> None:
    """Six hours of serial waiting is the whole problem, so overlap is the thing to pin.

    The cache is in play on purpose. Its project lock covers the lookup and the commit around
    each call and never the call itself, so eight workers can be inside the endpoint at once.
    Holding that lock across dispatch would make this fixture's barrier time out.
    """
    bank = _bank()
    units = _episodes(8)
    boundary = _Overlapping(threading.Barrier(8, timeout=10.0))
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))

    matrix = DecisionsTraceTagger(boundary, _SNAPSHOT, cache=cache, maximum_concurrency=8).tag(
        units, bank
    )

    assert len(boundary.states) == 8
    assert [row.trace_id for row in matrix.rows] == [unit.trace_id for unit in units]
    for unit, row in zip(units, matrix.rows, strict=True):
        assert row.values == (_episode_value(unit),) * len(bank.questions)


def test_a_serial_pass_and_a_parallel_pass_return_the_same_matrix(tmp_path: Path) -> None:
    """Concurrency is an execution knob: it may not change one value of the result."""
    bank = _bank()
    units = _episodes(6)
    serial = DecisionsTraceTagger(
        _Delayed(delay_seconds=0.0),
        _SNAPSHOT,
        cache=DecisionCache(ProjectPaths(root=tmp_path / "serial", project_id="fixture")),
        maximum_concurrency=1,
    ).tag(units, bank)
    parallel = DecisionsTraceTagger(
        _Delayed(),
        _SNAPSHOT,
        cache=DecisionCache(ProjectPaths(root=tmp_path / "parallel", project_id="fixture")),
        maximum_concurrency=8,
    ).tag(units, bank)

    assert serial == parallel


def test_rows_follow_the_supplied_units_whatever_finishes_first() -> None:
    """Row order is an input property, so a slow early episode may not move its own row."""
    bank = _bank()
    units = _episodes(6)
    boundary = _Delayed()

    matrix = DecisionsTraceTagger(boundary, _SNAPSHOT, maximum_concurrency=6).tag(units, bank)

    assert boundary.order != [unit.text for unit in units]
    assert [row.trace_id for row in matrix.rows] == [unit.trace_id for unit in units]
    for unit, row in zip(units, matrix.rows, strict=True):
        assert row.values == (_episode_value(unit),) * len(bank.questions)


def test_a_parallel_rerun_replays_every_answer_and_buys_nothing(tmp_path: Path) -> None:
    """Buy once and replay forever has to hold at every concurrency, not just at one."""
    bank = _bank()
    units = _episodes(8)
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    first = DecisionsTraceTagger(
        _Overlapping(threading.Barrier(8, timeout=10.0)),
        _SNAPSHOT,
        cache=cache,
        maximum_concurrency=8,
    ).tag(units, bank)

    replayed = DecisionsTraceTagger(
        _Recorder([]), _SNAPSHOT, cache=cache, maximum_concurrency=8
    ).tag(units, bank)

    assert replayed == first
    assert [name.split("/")[0] for name in cache.records.list_ids()] == ["decision"] * 8


def test_parallel_accounting_reports_exactly_what_the_pass_paid(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Totals are accumulated across workers, so they must not race or depend on order."""
    bank = _bank()
    units = _episodes(8)
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    boundary = _Overlapping(threading.Barrier(8, timeout=10.0), cost=0.25, tokens=9532)

    with caplog.at_level("INFO", logger="exp.simulation.mining.decisions"):
        DecisionsTraceTagger(boundary, _SNAPSHOT, cache=cache, maximum_concurrency=8).tag(
            units, bank
        )

    assert "bought 8 of them (76256 input tokens, 2.0000 USD)" in caplog.text


def test_a_partly_replayed_parallel_pass_pays_only_for_what_is_missing(tmp_path: Path) -> None:
    """A corpus grows an episode at a time, and the common rerun is almost entirely replay."""
    bank = _bank()
    units = _episodes(8)
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    DecisionsTraceTagger(
        _Overlapping(threading.Barrier(8, timeout=10.0)),
        _SNAPSHOT,
        cache=cache,
        maximum_concurrency=8,
    ).tag(units, bank)

    grown = [*units, *_episodes(10)[8:]]
    boundary = _Overlapping(threading.Barrier(2, timeout=10.0))
    matrix = DecisionsTraceTagger(boundary, _SNAPSHOT, cache=cache, maximum_concurrency=8).tag(
        grown, bank
    )

    assert sorted(boundary.states) == ["episode 8", "episode 9"]
    assert [row.trace_id for row in matrix.rows] == [unit.trace_id for unit in grown]


def test_the_same_failure_is_raised_however_the_calls_race(tmp_path: Path) -> None:
    """A corpus-wide run must name the same episode on every attempt, not whichever lost."""
    bank = _bank()
    units = _episodes(6)

    def boundary(request: DecisionRequest) -> JsonObject:
        """Fail the last episode immediately and the second one after a wait."""
        index = _episode_index(request)
        if index == 5:
            raise ProviderTransportError("the last episode failed first")
        if index == 1:
            time.sleep(0.05)
            raise ProviderTransportError("the earliest failure")
        return _state_answers(request, (index + 1) / 100)

    for attempt in range(5):
        with pytest.raises(ProviderTransportError, match="the earliest failure"):
            DecisionsTraceTagger(
                boundary,
                _SNAPSHOT,
                cache=DecisionCache(ProjectPaths(root=tmp_path, project_id=f"attempt-{attempt}")),
                maximum_concurrency=6,
            ).tag(units, bank)


def test_answers_bought_before_a_failure_are_still_committed(tmp_path: Path) -> None:
    """Those calls were billed, so the next run must replay them instead of buying again."""
    bank = _bank()
    units = _episodes(4)
    cache = DecisionCache(ProjectPaths(root=tmp_path, project_id="fixture"))
    released = threading.Event()

    def boundary(request: DecisionRequest) -> JsonObject:
        """Fail the first episode only after its siblings have answered."""
        index = _episode_index(request)
        if index == 0:
            released.wait(10.0)
            raise ProviderTransportError("the first episode failed")
        answer = _state_answers(request, (index + 1) / 100)
        if index == 3:
            released.set()
        return answer

    with pytest.raises(ProviderTransportError, match="the first episode failed"):
        DecisionsTraceTagger(boundary, _SNAPSHOT, cache=cache, maximum_concurrency=4).tag(
            units, bank
        )

    identity = _SNAPSHOT.model_dump(mode="json")
    assert sorted(cache.records.list_ids()) == sorted(
        [f"intent/{decision_key(units[0], bank.questions, identity)}"]
        + [f"decision/{decision_key(unit, bank.questions, identity)}" for unit in units[1:]]
    )


def test_a_nonpositive_concurrency_is_refused_before_anything_is_rendered() -> None:
    """A tagger that can never dispatch is a configuration mistake, not a quiet no-op."""
    with pytest.raises(TraceTaggingError, match="at least one call in flight"):
        DecisionsTraceTagger(_Recorder([]), _SNAPSHOT, maximum_concurrency=0)

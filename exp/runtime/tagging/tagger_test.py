"""Tests for the SystemOne tagger against a fake decision endpoint."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.tags.bank import DEFAULT_TAG_BANK, TagBank, TagScope
from exp.common.tags.exchange import TagExchange, TagMessage, opening_key, tag_facts
from exp.common.tags.records import ExchangeTags
from exp.common.tags.windows import TagWindowBudget, turn_windows
from exp.runtime.models.providers.systemone import (
    SystemOneClient,
    SystemOneHttpError,
    SystemOneResponseError,
)
from exp.runtime.tagging.tagger import Tagger, TaggingError, conversation_key

_BUDGET = TagWindowBudget(window_chars=2_000, overlap_chars=0, header_chars=200)
_CONVERSATION_IDS = tuple(tag.id for tag in DEFAULT_TAG_BANK.for_scope(TagScope.CONVERSATION))
_TURN_IDS = tuple(tag.id for tag in DEFAULT_TAG_BANK.for_scope(TagScope.TURN))

Handler = Callable[[JsonObject], httpx.Response | None]


def _probability(tag_id: str) -> float:
    """Return the fake endpoint's distinct probability for one tag."""
    return (DEFAULT_TAG_BANK.ids.index(tag_id) + 1) / 100


def _answer(body: JsonObject, *, model: str = "clef-27b") -> httpx.Response:
    """Answer every question of a request, in reverse order, with its fake probability."""
    questions = body["questions"]
    assert isinstance(questions, dict)
    answers = {key: {"type": "noul", "noul": _probability(key)} for key in reversed(questions)}
    usage = {"input_tokens": len(str(body["state"])), "output_tokens": len(questions)}
    return httpx.Response(200, json={"model": model, "answers": answers, "usage": usage})


def _client(
    seen: list[JsonObject], handler: Handler | None = None, *, max_concurrency: int = 8
) -> SystemOneClient:
    """Build a client whose endpoint records each body and answers through ``handler``."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Record the body, then let the handler override the default answer."""
        body = json.loads(request.content)
        seen.append(body)
        return (handler(body) if handler else None) or _answer(body)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    return SystemOneClient(
        "http://127.0.0.1:30000/v1",
        "clef-27b",
        max_concurrency=max_concurrency,
        http_client=http_client,
    )


def _exchange(exchange_id: str, *, key: str | None = None, task: str = "Fix it") -> TagExchange:
    """Build an exchange whose three long response turns need several turn windows."""
    return TagExchange(
        exchange_id=exchange_id,
        conversation_key=key,
        messages=(
            TagMessage(role="system", content="You are a coding agent."),
            TagMessage(role="user", content=task),
        ),
        response=tuple(
            TagMessage(role="assistant", content=f"step {step} " + "x" * 900) for step in range(3)
        ),
    )


def _scope(body: JsonObject) -> TagScope:
    """Return which scope a recorded request asked."""
    questions = body["questions"]
    assert isinstance(questions, dict)
    return TagScope.CONVERSATION if "coding" in questions else TagScope.TURN


def test_tag_sends_one_request_per_window_and_keeps_bank_order() -> None:
    """Answers given in any order land in bank scope order, with stamps and facts."""
    seen: list[JsonObject] = []
    tagger = Tagger(_client(seen), budget=_BUDGET, max_in_flight=4)
    exchange = _exchange("ex-1")
    tags = asyncio.run(tagger.tag(exchange, opening_already_tagged=False))
    windows = turn_windows(exchange, _BUDGET)
    assert len(windows) > 1
    assert len(seen) == 1 + len(windows)
    assert tags.conversation is not None
    assert tags.conversation.scores == tuple(_probability(key) for key in _CONVERSATION_IDS)
    assert [(w.start_message, w.end_message) for w in tags.turn] == [
        (w.start_message, w.end_message) for w in windows
    ]
    assert all(w.scores == tuple(_probability(key) for key in _TURN_IDS) for w in tags.turn)
    assert (tags.model, tags.bank_id, tags.bank_version, tags.bank_digest) == (
        "clef-27b",
        DEFAULT_TAG_BANK.bank_id,
        DEFAULT_TAG_BANK.version,
        DEFAULT_TAG_BANK.digest,
    )
    assert tags.conversation_key == opening_key(exchange) == conversation_key(exchange)
    assert tags.facts == tag_facts(exchange)
    assert ExchangeTags.model_validate_json(tags.model_dump_json()) == tags
    opening = next(body for body in seen if _scope(body) is TagScope.CONVERSATION)
    coding = DEFAULT_TAG_BANK.get("coding")
    questions = opening["questions"]
    assert isinstance(questions, dict)
    assert list(questions) == list(_CONVERSATION_IDS)
    assert questions["coding"] == {
        "type": "noul",
        "instructions": coding.question,
        "criteria": {"true": coding.true_when, "false": coding.false_when},
    }


def test_opening_already_tagged_asks_only_turn_windows() -> None:
    """A conversation whose opening is tagged gets no opening request and no opening scores."""
    seen: list[JsonObject] = []
    tagger = Tagger(_client(seen), budget=_BUDGET, max_in_flight=4)
    tags = asyncio.run(tagger.tag(_exchange("ex-1", key="conv"), opening_already_tagged=True))
    assert tags.conversation is None
    assert tags.conversation_key == "conv"
    assert {_scope(body) for body in seen} == {TagScope.TURN}


def test_failed_window_fails_the_exchange_with_its_cause() -> None:
    """A failed turn decision raises a typed error naming the exchange; no record is made."""
    seen: list[JsonObject] = []

    def fail_turns(body: JsonObject) -> httpx.Response | None:
        """Fail every turn request with a server error."""
        return httpx.Response(503) if _scope(body) is TagScope.TURN else None

    tagger = Tagger(_client(seen, fail_turns), budget=_BUDGET, max_in_flight=1)
    with pytest.raises(TaggingError) as raised:
        asyncio.run(tagger.tag(_exchange("ex-9"), opening_already_tagged=False))
    error = raised.value
    assert (error.exchange_id, error.scope, error.window_index) == ("ex-9", TagScope.TURN, 0)
    assert isinstance(error.cause, SystemOneHttpError)
    assert error.__cause__ is error.cause
    assert "ex-9" in str(error)


def test_windows_answered_by_different_models_fail() -> None:
    """One record names one served model, so a mixed answer set is refused."""
    seen: list[JsonObject] = []

    def other_model(body: JsonObject) -> httpx.Response | None:
        """Answer turn requests from a different model."""
        return _answer(body, model="other") if _scope(body) is TagScope.TURN else None

    tagger = Tagger(_client(seen, other_model), budget=_BUDGET, max_in_flight=4)
    with pytest.raises(TaggingError) as raised:
        asyncio.run(tagger.tag(_exchange("ex-2"), opening_already_tagged=False))
    assert isinstance(raised.value.cause, SystemOneResponseError)


def _run_many(
    tagger: Tagger, exchanges: list[TagExchange], keys: frozenset[str] = frozenset()
) -> dict[str, ExchangeTags | TaggingError]:
    """Collect every ``tag_many`` outcome, returning records and errors by exchange id."""

    async def collect() -> dict[str, ExchangeTags | TaggingError]:
        """Drain the stream."""
        results: dict[str, ExchangeTags | TaggingError] = {}
        async for outcome in tagger.tag_many(exchanges, opening_tagged_keys=keys):
            results[outcome.exchange_id] = outcome
        return results

    return asyncio.run(collect())


def test_tag_many_tags_each_opening_once_and_yields_failures() -> None:
    """Openings are asked once per key, known keys are skipped and failures are yielded."""
    seen: list[JsonObject] = []

    def fail_marked(body: JsonObject) -> httpx.Response | None:
        """Fail requests about the task that asks to fail."""
        return httpx.Response(500) if "please fail" in str(body["state"]) else None

    tagger = Tagger(_client(seen, fail_marked), budget=_BUDGET, max_in_flight=2)
    exchanges = [
        _exchange("a1", key="conv-a"),
        _exchange("a2", key="conv-a"),
        _exchange("b1"),
        _exchange("b2"),
        _exchange("c1", key="conv-c", task="please fail"),
        _exchange("d1", key="conv-d"),
    ]
    results = _run_many(tagger, exchanges, frozenset({"conv-d"}))
    assert set(results) == {"a1", "a2", "b1", "b2", "c1", "d1"}
    assert isinstance(results["c1"], TaggingError)
    records = {key: value for key, value in results.items() if isinstance(value, ExchangeTags)}
    assert {key for key, value in records.items() if value.conversation is not None} == {
        "a1",
        "b1",
    }
    assert records["b1"].conversation_key == records["b2"].conversation_key


def test_failed_opener_hands_the_opening_to_the_next_exchange() -> None:
    """When the first exchange of a conversation fails, a later one tags the opening."""
    seen: list[JsonObject] = []
    calls = 0

    def fail_first(body: JsonObject) -> httpx.Response | None:
        """Fail the very first request only."""
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls == 1 else None

    tagger = Tagger(_client(seen, fail_first), budget=_BUDGET, max_in_flight=4)
    exchanges = [_exchange("x1", key="k"), _exchange("x2", key="k"), _exchange("x3", key="k")]
    results = _run_many(tagger, exchanges)
    assert isinstance(results["x1"], TaggingError)
    second, third = results["x2"], results["x3"]
    assert isinstance(second, ExchangeTags) and second.conversation is not None
    assert isinstance(third, ExchangeTags) and third.conversation is None


def test_decisions_in_flight_never_exceed_the_bound() -> None:
    """Across exchanges and windows, at most ``max_in_flight`` decisions run at once."""
    active = peak = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        """Hold each request briefly while counting concurrent requests."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return _answer(json.loads(request.content))

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = SystemOneClient("http://127.0.0.1:30000/v1", "clef-27b", http_client=http_client)
    tagger = Tagger(client, budget=_BUDGET, max_in_flight=3)
    results = _run_many(tagger, [_exchange(f"e{i}") for i in range(6)])
    assert all(isinstance(value, ExchangeTags) for value in results.values())
    assert peak == 3


def test_tagger_rejects_unusable_settings() -> None:
    """The bound must fit the client, and both scopes need tags."""
    client = _client([], max_concurrency=2)
    with pytest.raises(ValueError, match="max_concurrency"):
        Tagger(client, max_in_flight=3)
    turn_only = TagBank(
        bank_id="turn-only", version=1, tags=DEFAULT_TAG_BANK.for_scope(TagScope.TURN)
    )
    with pytest.raises(ValueError, match="no conversation tags"):
        Tagger(client, bank=turn_only, max_in_flight=1)

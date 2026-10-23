"""Tests for the resumable JSONL tagging helper."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from exp.common.tags.bank import DEFAULT_TAG_BANK, TagBank
from exp.common.tags.exchange import TagExchange, TagFunction, TagMessage, TagToolCall
from exp.common.tags.records import ExchangeTags
from exp.runtime.models.providers.systemone import SystemOneClient
from exp.runtime.tagging.jsonl import TagFailure, TagJsonlReport, tag_jsonl
from exp.runtime.tagging.tagger import Tagger


def _tagger(failing: set[str], requests: list[str]) -> Tagger:
    """Build a tagger whose endpoint fails requests mentioning any ``failing`` task."""

    def respond(request: httpx.Request) -> httpx.Response:
        """Answer every question, or fail when the state names a failing task."""
        body = json.loads(request.content)
        requests.append(body["state"])
        if any(task in body["state"] for task in failing):
            return httpx.Response(500)
        answers = {key: {"type": "noul", "noul": 0.25} for key in body["questions"]}
        usage = {"input_tokens": 10, "output_tokens": 1}
        return httpx.Response(200, json={"model": "clef", "answers": answers, "usage": usage})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = SystemOneClient("http://127.0.0.1:30000/v1", "clef", http_client=http_client)
    return Tagger(client, max_in_flight=1)


def _exchange(exchange_id: str, task: str, key: str = "conv") -> str:
    """Return one exchange as a JSON line."""
    exchange = TagExchange(
        exchange_id=exchange_id,
        conversation_key=key,
        messages=(TagMessage(role="user", content=task),),
        response=(TagMessage(role="assistant", content="Done."),),
    )
    return exchange.model_dump_json() + "\n"


def _run(tagger: Tagger, tmp_path: Path) -> TagJsonlReport:
    """Tag ``exchanges.jsonl`` into ``tags.jsonl`` and ``failures.jsonl``."""
    return asyncio.run(
        tag_jsonl(
            tagger,
            tmp_path / "exchanges.jsonl",
            output=tmp_path / "tags.jsonl",
            failures=tmp_path / "failures.jsonl",
        )
    )


def _records(path: Path) -> list[ExchangeTags]:
    """Read every record of an output file."""
    return [ExchangeTags.model_validate_json(line) for line in path.read_text().splitlines()]


def test_tag_jsonl_writes_results_and_failures_then_resumes(tmp_path: Path) -> None:
    """A rerun skips tagged ids, retries failures and keeps the conversation opening tagged."""
    lines = [_exchange("e1", "first task"), _exchange("e2", "broken task"), "\n"]
    (tmp_path / "exchanges.jsonl").write_text("".join([*lines, _exchange("e1", "first task")]))
    first = _run(_tagger({"broken"}, []), tmp_path)
    assert first == TagJsonlReport(tagged=1, failed=1, skipped=1)
    (failure,) = [
        TagFailure.model_validate_json(line)
        for line in (tmp_path / "failures.jsonl").read_text().splitlines()
    ]
    assert (failure.exchange_id, failure.error) == ("e2", "SystemOneHttpError")
    assert [record.exchange_id for record in _records(tmp_path / "tags.jsonl")] == ["e1"]

    requests: list[str] = []
    second = _run(_tagger(set(), requests), tmp_path)
    assert second == TagJsonlReport(tagged=1, failed=0, skipped=2)
    records = _records(tmp_path / "tags.jsonl")
    assert [record.exchange_id for record in records] == ["e1", "e2"]
    assert records[1].conversation is None
    assert len(requests) == 1
    assert len((tmp_path / "failures.jsonl").read_text().splitlines()) == 1


def test_an_exchange_that_cannot_be_tagged_fails_alone(tmp_path: Path) -> None:
    """An exchange whose windows cannot be built is one recorded failure; the rest are tagged."""
    call = TagToolCall(id="c", function=TagFunction(name="write", arguments="[" * 50_000))
    poisoned = TagExchange(
        exchange_id="deep",
        conversation_key="k2",
        messages=(TagMessage(role="user", content="Go"),),
        response=(TagMessage(role="assistant", tool_calls=(call,)),),
    )
    lines = [_exchange("e1", "a"), poisoned.model_dump_json() + "\n", _exchange("e2", "b")]
    (tmp_path / "exchanges.jsonl").write_text("".join(lines))
    assert _run(_tagger(set(), []), tmp_path) == TagJsonlReport(tagged=2, failed=1, skipped=0)
    assert [record.exchange_id for record in _records(tmp_path / "tags.jsonl")] == ["e1", "e2"]
    (line,) = (tmp_path / "failures.jsonl").read_text().splitlines()
    failure = TagFailure.model_validate_json(line)
    assert (failure.exchange_id, failure.scope, failure.window_index) == ("deep", None, None)
    assert failure.error == "RecursionError"


def test_an_interrupted_last_line_is_dropped_and_tagged_again(tmp_path: Path) -> None:
    """Unterminated output and failure lines left by a crash never merge with the next one."""
    (tmp_path / "exchanges.jsonl").write_text(_exchange("e1", "a") + _exchange("e2", "b", "k2"))
    _run(_tagger(set(), []), tmp_path)
    output = tmp_path / "tags.jsonl"
    complete, interrupted = output.read_text().splitlines()
    output.write_text(complete + "\n" + interrupted[:40])
    (tmp_path / "failures.jsonl").write_text('{"exchange_id": "e2", "sco')
    assert _run(_tagger({"b"}, []), tmp_path) == TagJsonlReport(tagged=0, failed=1, skipped=1)
    assert [record.exchange_id for record in _records(output)] == ["e1"]
    (failure,) = (tmp_path / "failures.jsonl").read_text().splitlines()
    assert TagFailure.model_validate_json(failure).exchange_id == "e2"


def test_invalid_lines_and_foreign_banks_stop_the_run(tmp_path: Path) -> None:
    """Bad input names its line without echoing it; aliased paths and other banks are refused."""
    (tmp_path / "exchanges.jsonl").write_text(_exchange("e1", "a") + '{"secret": "canary"}\n')
    with pytest.raises(ValueError, match=r"exchanges.jsonl:2 is not a valid TagExchange") as bad:
        _run(_tagger(set(), []), tmp_path)
    assert "canary" not in str(bad.value)
    assert [record.exchange_id for record in _records(tmp_path / "tags.jsonl")] == ["e1"]

    source = tmp_path / "exchanges.jsonl"
    before = source.read_bytes()
    (tmp_path / "alias.jsonl").symlink_to(source)
    with pytest.raises(ValueError, match="same file"):
        asyncio.run(
            tag_jsonl(
                _tagger(set(), []), source, output=tmp_path / "o", failures=tmp_path / "alias.jsonl"
            )
        )
    assert source.read_bytes() == before

    other = TagBank(bank_id="other", version=2, tags=DEFAULT_TAG_BANK.tags)
    client = _tagger(set(), []).client
    with pytest.raises(ValueError, match="different|not this tagger's bank"):
        _run(Tagger(client, bank=other, max_in_flight=1), tmp_path)

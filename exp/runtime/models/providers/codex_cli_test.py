"""The codex-cli transport with a scripted ``codex exec``: no real Codex process is started."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.codex_cli import (
    CODEX_FLAGS,
    DISABLED_FEATURES,
    CodexCliTransport,
    CodexRun,
    OfflineScopeRequired,
    drop_optional_nulls,
    offline_closed_model,
    parse_events,
    set_codex_concurrency,
    strict_schema,
)
from exp.runtime.models.providers.transport import ProviderTransportError, classify_retry

PAYLOAD: JsonObject = {
    "model": "gpt-6-astra",
    "messages": [
        {"role": "system", "content": "Grade the ticket."},
        {"role": "user", "content": '{"ticket": "refund"}'},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "submit_ticket",
                "description": "Answer the ticket.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "maxLength": 20},
                        "note": {"type": "string", "default": ""},
                    },
                    "required": ["label"],
                },
            },
        }
    ],
    "tool_choice": {"type": "function", "function": {"name": "submit_ticket"}},
}
CLEAN = [
    {"type": "thread.started", "thread_id": "t"},
    {"type": "turn.started"},
    {"type": "item.completed", "item": {"type": "reasoning", "text": "thinking"}},
    {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
    {"type": "turn.completed", "usage": {"input_tokens": 14000, "output_tokens": 30}},
]


def events(rows: Sequence[object]) -> str:
    """A JSONL event stream."""
    return "\n".join(json.dumps(row) for row in rows)


class Runner:
    """A scripted ``codex exec``: records each call and returns scripted runs in order."""

    def __init__(self, runs: Sequence[CodexRun | BaseException], delay: float = 0.0) -> None:
        """Script the runs; an exception is raised instead of returning."""
        self.runs = list(runs)
        self.calls: list[tuple[list[str], str, Path]] = []
        self.delay, self.active, self.peak = delay, 0, 0
        self.lock = threading.Lock()

    def __call__(self, argv: Sequence[str], stdin: str, cwd: Path, timeout: float) -> CodexRun:
        """Run one scripted call."""
        with self.lock:
            self.calls.append((list(argv), stdin, cwd))
            assert list(cwd.iterdir()) == []
            self.active += 1
            self.peak = max(self.peak, self.active)
            run = self.runs.pop(0) if len(self.runs) > 1 else self.runs[0]
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        if isinstance(run, BaseException):
            raise run
        return run


def answered(answer: JsonObject, stream: Sequence[object] = CLEAN) -> CodexRun:
    """A clean run that wrote ``answer`` to the output file."""
    return CodexRun(0, events(stream), "", json.dumps(answer))


def transport(runner: Runner, rate_limit_retries: int = 3) -> CodexCliTransport:
    """A transport over the scripted runner, version pinned to 0.166.0."""
    return CodexCliTransport(
        model="gpt-6-astra", effort="high", version="0.166.0", runner=runner,
        version_of=lambda binary: "0.166.0", sleep=lambda seconds: None,
        rate_limit_retries=rate_limit_retries,
    )  # fmt: skip


def post(codex: CodexCliTransport) -> Any:  # noqa: ANN401
    """Post the payload inside the offline scope and return the body."""
    with offline_closed_model("test"):
        return codex.post(
            "https://codex-cli.invalid/v1/chat/completions", headers={}, payload=PAYLOAD,
            timeout_seconds=10,
        ).body  # fmt: skip


def test_a_structured_answer_becomes_the_forced_tool_call() -> None:
    """The answer file is the tool's arguments; flags disable every tool; the cwd is empty."""
    runner = Runner([answered({"label": "refund", "note": None})])
    body = post(transport(runner))
    choice = body["choices"][0]
    call = choice["message"]["tool_calls"][0]["function"]
    assert call["name"] == "submit_ticket" and json.loads(call["arguments"]) == {"label": "refund"}
    assert body["usage"] == {"prompt_tokens": 14000, "completion_tokens": 30, "total_tokens": 14030}
    argv, stdin, cwd = runner.calls[0]
    assert argv[1:4] == ["exec", "-m", "gpt-6-astra"] and argv[-1] == "-"
    for flag in ("--ephemeral", "--skip-git-repo-check", "--ignore-user-config", "--json"):
        assert flag in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert 'model_reasoning_effort="high"' in argv and "--output-schema" in argv
    assert all(name in argv for name in DISABLED_FEATURES) and "shell_tool" in DISABLED_FEATURES
    assert tuple(argv[4:6]) == ("-c", 'model_reasoning_effort="high"')
    assert all(flag in argv for flag in CODEX_FLAGS)
    assert "Grade the ticket." in stdin and '{"ticket": "refund"}' in stdin
    assert not cwd.exists()


def test_strict_schema_closes_objects_and_nulls_only_optional_fields() -> None:
    """Every property required, optional ones nullable, unsupported keywords dropped."""
    parameters: Any = cast(Any, PAYLOAD)["tools"][0]["function"]["parameters"]
    strict, optional = strict_schema(parameters)
    strict = cast(Any, strict)
    assert strict["required"] == ["label", "note"] and strict["additionalProperties"] is False
    assert strict["properties"]["note"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "maxLength" not in json.dumps(strict) and optional == {"": {"note"}}
    assert drop_optional_nulls({"label": None, "note": None}, optional) == {"label": None}


@pytest.mark.parametrize(
    "item",
    ["command_execution", "file_change", "mcp_tool_call", "web_search", "todo_list"],
)
def test_an_output_with_a_tool_event_is_refused(item: str) -> None:
    """Any non-message item refuses the run, even with a valid answer file."""
    stream = [*CLEAN[:2], {"type": "item.started", "item": {"type": item}}, *CLEAN[2:]]
    with pytest.raises(ProviderTransportError) as refused:
        post(transport(Runner([answered({"label": "x"}, stream)])))
    assert refused.value.error_code == "codex_tool_use_refused"
    assert not classify_retry(refused.value).retryable
    assert parse_events(events(stream)).tools == [f"item.started:{item}"]
    assert parse_events(events([{"type": "exec_command_begin"}])).tools == ["exec_command_begin"]


def test_startup_error_items_are_warnings_not_tools() -> None:
    """The real smoke call's stream: three error notices before the turn, then the answer."""
    stream = [
        CLEAN[0],
        *({"type": "item.completed", "item": {"type": "error", "message": "notice"}},) * 3,
        *CLEAN[1:],
    ]
    summary = parse_events(events(stream))
    assert summary.tools == [] and summary.errors == [] and len(summary.warnings) == 3
    body = post(transport(Runner([answered({"label": "x"}, stream)])))
    assert body["model"] == "gpt-6-astra"


def test_timeouts_and_exits_are_transient_and_rate_limits_back_off() -> None:
    """A timeout and a non-zero exit classify as retryable; a rate limit reruns, then 429s."""
    with pytest.raises(TimeoutError) as timed_out:
        post(transport(Runner([TimeoutError("codex exec exceeded 900s")])))
    assert classify_retry(timed_out.value).retryable
    with pytest.raises(ProviderTransportError) as exited:
        post(transport(Runner([CodexRun(1, events(CLEAN[:2]), "boom", None)])))
    assert classify_retry(exited.value).reason == "transport"
    limited = CodexRun(1, events([{"type": "error", "message": "Rate limit reached"}]), "", None)
    slept: list[float] = []
    codex = CodexCliTransport(
        model="gpt-6-astra", effort="high", version=None, rate_limit_retries=2,
        backoff_seconds=5, runner=Runner([limited, limited, answered({"label": "ok"})]),
        sleep=slept.append,
    )  # fmt: skip
    assert post(codex)["model"] == "gpt-6-astra" and slept == [5, 10]
    always = Runner([limited])
    with pytest.raises(ProviderTransportError) as throttled:
        post(transport(always, rate_limit_retries=1))
    assert throttled.value.status_code == 429 and len(always.calls) == 2


def test_outside_the_offline_scope_nothing_is_spawned_and_a_changed_version_is_refused() -> None:
    """No offline scope: refused before any process; a different codex version: refused."""
    runner = Runner([answered({"label": "x"})])
    with pytest.raises(OfflineScopeRequired):
        transport(runner).post("u", headers={}, payload=PAYLOAD, timeout_seconds=1)
    assert runner.calls == []
    moved = CodexCliTransport(
        model="gpt-6-astra", effort=None, version="0.166.0", runner=runner,
        version_of=lambda binary: "0.170.0",
    )  # fmt: skip
    with pytest.raises(ProviderTransportError, match="pinned"):
        post(moved)
    assert runner.calls == []


def test_the_concurrency_limit_is_respected(tmp_path: Path) -> None:
    """Ten concurrent calls never run more than the configured limit at once."""
    set_codex_concurrency(3, tmp_path / "slots")
    try:
        runner = Runner([answered({"label": "x"})], delay=0.05)
        codex = transport(runner)
        threads = [threading.Thread(target=post, args=(codex,)) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(runner.calls) == 10 and runner.peak == 3
    finally:
        set_codex_concurrency(4)

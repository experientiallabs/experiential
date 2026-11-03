"""Offline tests for per-trajectory rule extraction over scripted provider replies."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.candidates_test import _reply
from exp.optimize.evaluation.rules.extraction import (
    EXTRACTION_TOOL,
    draw_identity,
    extract_rules_in_parallel,
    extract_trajectory_rules,
    extraction_request,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.rules.contracts import parse_tool_contracts
from exp.simulation.rules.contracts_test import refund_trace


class RoutedTransport(ScriptedJsonTransport):
    """Thread-safe scripted provider answering each request by the trajectory it carries.

    Concurrent draws reach the provider in any order, so replies are keyed by trace identity
    rather than consumed in sequence. Each trace's reply is served at most once.
    """

    def __init__(self, replies: Mapping[str, JsonHttpResponse]) -> None:
        """Store one reply per trace identity.

        Args:
            replies: Replies keyed by the trace identity whose locator the request carries.
        """
        super().__init__()
        self._replies = dict(replies)
        self._lock = threading.Lock()

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one POST and return the reply for the trajectory it carries.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers sent by the caller.
            payload: JSON request object sent by the caller.
            timeout_seconds: Bounded per-attempt timeout, ignored by the fake.

        Returns:
            The reply scripted for that trajectory.

        Raises:
            AssertionError: No unused reply matches the request.
        """
        del timeout_seconds
        text = json.dumps(payload)
        with self._lock:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            for trace_id in sorted(self._replies):
                if f"trace:{trace_id}:span" in text:
                    return self._replies.pop(trace_id)
        raise AssertionError("test made an unexpected provider request")


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ProjectStore, AuthorSettings]:
    """Configure a project with a tool-capable author alias and frozen settings."""
    project = ProjectStore(tmp_path, "case")
    write_catalog(tmp_path)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    catalog = load_model_catalog(project.model_catalog_path)
    settings = AuthorSettings(
        model_alias="judge",
        model=RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=1024,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    return project, settings


def _answer(trace_id: str, evidence: str | None = None) -> JsonObject:
    """One valid extraction answer citing the trajectory's own steps."""
    return {
        "trace_id": trace_id,
        "rules": [
            {
                "tool": "refund",
                "kind": "precondition",
                "statement": "A refund larger than the paid amount is refused.",
                "predicate": {
                    "op": "le",
                    "left": {"scope": "action", "path": "/arguments/amount"},
                    "right": {"value": 40},
                },
                "evidence": [evidence or f"trace:{trace_id}:span:r2"],
                "counter_evidence": [],
                "confidence": 0.6,
            }
        ],
        "questions": [
            {
                "tool": "refund",
                "question": "Is a partial refund allowed?",
                "why_it_matters": "Decides whether the amount precondition is an upper bound.",
                "evidence": [f"trace:{trace_id}:span:c3"],
            }
        ],
        "limitations": ["Only one refused refund was observed."],
    }


def _tool_reply(arguments: JsonObject) -> JsonHttpResponse:
    """One scripted reply making the forced extraction call."""
    return _reply(
        "judge-v1",
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call",
                    "function": {"name": EXTRACTION_TOOL, "arguments": json.dumps(arguments)},
                }
            ],
        },
        "tool_calls",
    )


def _wire(monkeypatch: pytest.MonkeyPatch, transport: ScriptedJsonTransport) -> None:
    """Route author calls to a scripted transport."""
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
    )


def test_one_trajectory_yields_validated_rules_and_replays_without_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid answer is parsed once; a second extraction replays the saved response."""
    project, settings = _setup(tmp_path, monkeypatch)
    trace = refund_trace("t1")
    contracts = parse_tool_contracts((trace,))
    transport = ScriptedJsonTransport((_tool_reply(_answer("t1")),))
    _wire(monkeypatch, transport)

    reference, rules = extract_trajectory_rules(
        project, settings, trace, contracts, identity="extract-one", code_revision="test"
    )

    assert len(transport.requests) == 1
    assert rules.trace_id == "t1"
    assert rules.rules[0].predicate is not None and rules.rules[0].kind == "precondition"
    assert rules.questions[0].evidence == ("trace:t1:span:c3",)

    replay = extract_trajectory_rules(
        project, settings, trace, contracts, identity="extract-one", code_revision="test"
    )

    assert replay == (reference, rules)
    assert len(transport.requests) == 1


def test_the_request_supplies_only_this_trajectorys_locators() -> None:
    """Contract examples from other traces are withheld, and the call is forced."""
    traces = (refund_trace("t1"), refund_trace("t2"))
    request, locators = extraction_request(
        traces[0], parse_tool_contracts(traces), output_tokens=512
    )

    assert all(locator.startswith("trace:t1") for locator in locators)
    assert "trace:t2" not in str(request.messages[1].content)
    assert request.tool_choice is not None and request.tools[0].name == EXTRACTION_TOOL
    assert "untrusted data" in str(request.messages[0].content)


def test_an_invented_locator_is_refused_and_kept_as_a_failure_without_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A citation outside the supplied inventory fails the draw, which replay never rebuys."""
    project, settings = _setup(tmp_path, monkeypatch)
    trace = refund_trace("t1")
    contracts = parse_tool_contracts((trace,))
    transport = ScriptedJsonTransport(
        (_tool_reply(_answer("t1", evidence="trace:t1:span:invented")),)
    )
    _wire(monkeypatch, transport)

    first = extract_rules_in_parallel(
        project,
        settings,
        (trace,),
        contracts,
        identity="run-bad",
        code_revision="test",
        concurrency=2,
    )
    replay = extract_rules_in_parallel(
        project,
        settings,
        (trace,),
        contracts,
        identity="run-bad",
        code_revision="test",
        concurrency=2,
    )

    assert len(transport.requests) == 1
    assert first == replay
    (draw,) = first
    assert draw.rules is None and draw.response is not None
    assert draw.failure is not None and "trace:t1:span:invented" in draw.failure
    assert draw.identity == draw_identity("run-bad", "t1")
    with pytest.raises(ValueError, match="not supplied"):
        extract_trajectory_rules(
            project, settings, trace, contracts, identity=draw.identity, code_revision="test"
        )
    assert len(transport.requests) == 1


def test_a_refused_answer_in_a_run_keeps_its_saved_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong trace identity or unknown tool is a failure record pointing at the response."""
    project, settings = _setup(tmp_path, monkeypatch)
    trace = refund_trace("alpha")
    contracts = parse_tool_contracts((trace,))
    _wire(monkeypatch, RoutedTransport({"alpha": _tool_reply(_answer("beta"))}))

    (draw,) = extract_rules_in_parallel(
        project, settings, (trace,), contracts, identity="run", code_revision="test", concurrency=1
    )

    assert draw.response is not None and draw.rules is None
    assert draw.failure is not None and "not 'alpha'" in draw.failure


def test_parallel_run_draws_each_trajectory_once_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent draws get distinct identities, keep a failure, and replay with no call."""
    project, settings = _setup(tmp_path, monkeypatch)
    names = ("alpha", "beta", "gamma", "delta")
    traces: tuple[Trace, ...] = tuple(refund_trace(name) for name in names)
    contracts = parse_tool_contracts(traces)
    replies = {name: _tool_reply(_answer(name)) for name in names[:3]}
    replies["delta"] = _tool_reply(_answer("delta", evidence="trace:alpha:span:r2"))
    transport = RoutedTransport(replies)
    _wire(monkeypatch, transport)

    draws = extract_rules_in_parallel(
        project,
        settings,
        traces,
        contracts,
        identity="run-many",
        code_revision="test",
        concurrency=4,
    )

    assert [draw.trace_id for draw in draws] == list(names)
    assert len({draw.identity for draw in draws}) == 4
    assert len(transport.requests) == 4
    assert all(
        draw.rules is not None and draw.rules.trace_id == draw.trace_id for draw in draws[:3]
    )
    assert draws[3].rules is None and draws[3].failure is not None
    assert "trace:alpha:span:r2" in draws[3].failure

    replay = extract_rules_in_parallel(
        project,
        settings,
        traces,
        contracts,
        identity="run-many",
        code_revision="test",
        concurrency=2,
    )

    assert replay == draws
    assert len(transport.requests) == 4


def test_repeated_traces_and_nonpositive_concurrency_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two draws for one trace would share an identity, so the run refuses before any call."""
    project, settings = _setup(tmp_path, monkeypatch)
    trace = refund_trace("t1")
    contracts = parse_tool_contracts((trace,))

    with pytest.raises(ValueError, match="repeats"):
        extract_rules_in_parallel(
            project,
            settings,
            (trace, trace),
            contracts,
            identity="r",
            code_revision="t",
            concurrency=2,
        )
    with pytest.raises(ValueError, match="positive"):
        extract_rules_in_parallel(
            project, settings, (trace,), contracts, identity="r", code_revision="t", concurrency=0
        )

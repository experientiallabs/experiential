"""Tests for facet extraction (signatures, digests, LLM parsing + fallback)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.scenarios.mining.facets import FacetExtractor, Outcome, tool_signature, trace_digest


class FakeProvider:
    """Returns a canned completion text; records the last prompt for assertions."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _tool_step(name: str, content: str = "ok", *, task: str | None = None) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name=name, arguments={"q": 1}),
        observation=Observation(content=content),
        task=task,
    )


def _trace(names: list[str], *, task: str = "book a flight") -> Trace:
    steps = [_tool_step(name, task=task if i == 0 else None) for i, name in enumerate(names)]
    return Trace(trace_id="t1", steps=steps)


def test_tool_signature_collapses_consecutive_repeats() -> None:
    trace = _trace(["search", "search", "book", "search"])
    assert tool_signature(trace) == "search>book>search"


def test_tool_signature_skips_messages() -> None:
    trace = _trace(["search"])
    trace.steps.append(
        Step(
            action=Action(kind=ActionKind.MESSAGE, content="done"),
            observation=Observation(content=""),
        )
    )
    assert tool_signature(trace) == "search"


def test_trace_digest_includes_task_and_elides_middle() -> None:
    trace = _trace([f"tool{i}" for i in range(50)])
    digest = trace_digest(trace, max_steps=10)
    assert digest.startswith("TASK: book a flight")
    assert "steps elided" in digest
    assert "tool0" in digest and "tool49" in digest
    assert "tool25" not in digest


def test_extract_parses_facet_json() -> None:
    reply = (
        '{"task_summary": "Cancel a flight and get a refund", '
        '"outcome": "failure", "failure_category": "Policy Violation"}'
    )
    facet = FacetExtractor(FakeProvider(reply)).extract(_trace(["cancel"]))
    assert facet.task_summary == "Cancel a flight and get a refund"
    assert facet.outcome is Outcome.FAILURE
    assert facet.failure_category == "policy_violation"
    assert facet.tool_signature == "cancel"


def test_extract_ignores_failure_category_on_success() -> None:
    reply = '{"task_summary": "Book a hotel", "outcome": "success", "failure_category": "noise"}'
    facet = FacetExtractor(FakeProvider(reply)).extract(_trace(["book"]))
    assert facet.outcome is Outcome.SUCCESS
    assert facet.failure_category is None


def test_extract_falls_back_to_recorded_task_on_garbage() -> None:
    facet = FacetExtractor(FakeProvider("not json at all")).extract(_trace(["search"]))
    assert facet.task_summary == "book a flight"
    assert facet.outcome is Outcome.UNKNOWN


def test_extract_normalizes_unknown_outcome_values() -> None:
    reply = '{"task_summary": "Do a thing", "outcome": "partial"}'
    facet = FacetExtractor(FakeProvider(reply)).extract(_trace(["go"]))
    assert facet.outcome is Outcome.UNKNOWN


def test_embed_text_includes_domain_and_tools() -> None:
    reply = '{"task_summary": "Review a return", "outcome": "success"}'
    trace = _trace(["review", "refund"])
    trace.metadata["domain"] = "retail"
    facet = FacetExtractor(FakeProvider(reply)).extract(trace)
    assert facet.domain == "retail"
    assert facet.embed_text() == "[retail] Review a return | tools: review>refund"


def test_embed_text_without_domain_or_tools_is_the_summary() -> None:
    reply = '{"task_summary": "Review a return", "outcome": "success"}'
    message_step = Step(
        action=Action(kind=ActionKind.MESSAGE, content="hello"),
        observation=Observation(content="ok"),
        task="review",
    )
    trace = Trace(trace_id="t1", steps=[message_step])
    facet = FacetExtractor(FakeProvider(reply)).extract(trace)
    assert facet.domain is None
    assert facet.embed_text() == "Review a return"


def test_extract_populates_domain_on_fallback_and_ignores_blank_domain() -> None:
    trace = _trace(["search"])
    trace.metadata["domain"] = "  telecom  "
    facet = FacetExtractor(FakeProvider("not json at all")).extract(trace)
    assert facet.domain == "telecom"
    blank = _trace(["search"])
    blank.metadata["domain"] = "   "
    assert FacetExtractor(FakeProvider("not json at all")).extract(blank).domain is None

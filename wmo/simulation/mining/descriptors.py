"""Leakage-safe request descriptors and mining-only coverage descriptors."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.core.text import normalize_durable_text
from wmo.common.routing.features import (
    RouterFeatureExtractor,
    RouterFeatureRecord,
)
from wmo.common.tasks import ToolSchema
from wmo.common.traces import Trace

_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)


class DescriptorEmbedder(Protocol):
    """Embeds deterministic request-visible descriptor text for mining and later routing.

    The caller supplies the configured embedding client in production. Tests may use a deterministic
    fake. This protocol deliberately has no provider, credential, or runtime dependency.
    """

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one finite vector for every supplied descriptor text.

        Args:
            texts: Canonical request-visible descriptor strings.

        Returns:
            One vector per input text in the same order.
        """


@dataclass(frozen=True)
class RoutingDescriptor:
    """Only facts a live request can expose to the router.

    Args:
        trace_id: Source trace identifier used only for audit and selection joins.
        intent: Initial user-visible instruction.
        initial_context: Request-visible starting context allowed by project policy.
        tools: Tools available when the request starts.
        tags: Customer tags that a live caller also supplies.
    """

    trace_id: str
    intent: str
    initial_context: JsonObject
    tools: tuple[ToolSchema, ...]
    tags: tuple[str, ...]

    def canonical_payload(self) -> JsonObject:
        """Return the deterministic request-visible payload used for hashing and embedding.

        Returns:
            JSON-safe descriptor content with no outcome, later action, or trace-length fields.
        """
        return RouterFeatureRecord(
            initial_user_intent=self.intent,
            initial_context=self.initial_context,
            tools=tuple(sorted(self.tools, key=lambda tool: tool.name)),
            allowed_tags={"tags": list(self.tags)},
        ).model_dump(mode="json")

    def embedding_text(self) -> str:
        """Return canonical UTF-8 text that an embedding client may consume.

        Returns:
            Stable serialized request-visible descriptor content.
        """
        record = RouterFeatureRecord.model_validate(self.canonical_payload())
        return RouterFeatureExtractor().render(record)

    def fingerprint(self) -> str:
        """Return the exact duplicate fingerprint of request-visible content."""
        return hashlib.sha256(self.embedding_text().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CoverageDescriptor:
    """Mining-only episode facts that are forbidden from router features.

    Args:
        trace_id: Source trace identifier used for audit joins.
        tool_transitions: Later observed tool transition sequence.
        outcome_statuses: Terminal trace outcome values represented by this descriptor.
        is_escalation: Whether source evidence marks an escalation or handoff.
        span_counts: Complete episode span counts represented by this descriptor.
        domains: Source domain tags used only for coverage reporting.
    """

    trace_id: str
    tool_transitions: tuple[str, ...]
    outcome_statuses: tuple[str, ...]
    is_escalation: bool
    span_counts: tuple[int, ...]
    domains: tuple[str, ...]

    @property
    def has_failure(self) -> bool:
        """Return whether any represented source trace ended in failure."""
        return "failure" in self.outcome_statuses


@dataclass(frozen=True)
class HashingDescriptorEmbedder:
    """A deterministic no-network embedder for tests and explicit local fixtures.

    It is intentionally a lightweight hash vector, not a substitute for the configured production
    embedding model. W3 model access can supply a real ``DescriptorEmbedder`` without changing
    mining behavior or persisted task contracts.
    """

    dimensions: int = 64

    def __post_init__(self) -> None:
        """Reject unusably small deterministic vector spaces."""
        if self.dimensions < 8:
            raise ValueError("hashing descriptor embedder needs at least 8 dimensions")

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed request-visible descriptor text without network or model calls.

        Args:
            texts: Canonical descriptor text.

        Returns:
            Unit-normalized signed hashing vectors in input order.
        """
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        """Hash normalized content tokens into one fixed-width unit vector."""
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        if not tokens:
            tokens = ["empty"]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return tuple(value / norm for value in vector)


def routing_descriptor(trace: Trace) -> RoutingDescriptor:
    """Extract a router-safe descriptor from canonical source evidence.

    Args:
        trace: Canonical trace whose initial request is being represented.

    Returns:
        A descriptor that excludes every later span, outcome, transition, and length fact.
    """
    return RoutingDescriptor(
        trace_id=trace.trace_id,
        intent=normalize_durable_text(trace.task),
        initial_context=trace.initial_context,
        tools=trace.tools,
        tags=_request_tags(trace),
    )


def coverage_descriptor(trace: Trace) -> CoverageDescriptor:
    """Extract mining-only coverage facts from a canonical source trace.

    Args:
        trace: Canonical trace whose full episode may inform coverage.

    Returns:
        A descriptor for selection and reporting that must never reach router feature extraction.
    """
    transitions = []
    escalation = False
    for span in trace.spans:
        tool_name = span.attributes.get("gen_ai.tool.name")
        if isinstance(tool_name, str) and tool_name.strip():
            transitions.append(normalize_durable_text(tool_name.strip()))
        escalation = escalation or _span_marks_escalation(span.attributes)
    domain = _domain(trace)
    return CoverageDescriptor(
        trace_id=trace.trace_id,
        tool_transitions=tuple(_collapse_runs(transitions)),
        outcome_statuses=() if trace.outcome is None else (trace.outcome.status,),
        is_escalation=escalation,
        span_counts=(len(trace.spans),),
        domains=() if domain is None else (domain,),
    )


def normalized_vectors(
    embedder: DescriptorEmbedder,
    descriptors: Sequence[RoutingDescriptor],
) -> tuple[tuple[float, ...], ...]:
    """Validate and unit-normalize one embedding vector per routing descriptor.

    Args:
        embedder: Injected deterministic or configured embedding implementation.
        descriptors: Request-visible descriptors to embed in order.

    Returns:
        Finite, equal-width, unit-normalized vectors in descriptor order.

    Raises:
        ValueError: The injected embedder returns the wrong count, shape, or non-finite values.
    """
    vectors = embedder.embed(tuple(descriptor.embedding_text() for descriptor in descriptors))
    if len(vectors) != len(descriptors):
        raise ValueError("descriptor embedder returned a vector count different from its inputs")
    normalized: list[tuple[float, ...]] = []
    dimensions: int | None = None
    for index, vector in enumerate(vectors):
        values = tuple(float(value) for value in vector)
        if not values:
            raise ValueError(f"descriptor embedder returned an empty vector at index {index}")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"descriptor embedder returned a non-finite vector at index {index}")
        if dimensions is None:
            dimensions = len(values)
        elif len(values) != dimensions:
            raise ValueError("descriptor embedder returned vectors with inconsistent dimensions")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            raise ValueError(f"descriptor embedder returned a zero vector at index {index}")
        normalized.append(tuple(value / norm for value in values))
    return tuple(normalized)


def _request_tags(trace: Trace) -> tuple[str, ...]:
    """Read tags from the initial request span, never later episode evidence."""
    ordered = sorted(trace.spans, key=lambda span: (span.started_at, span.span_id))
    request_span = next(
        (span for span in ordered if _initial_prompt_text(span.attributes) is not None),
        ordered[0],
    )
    value = _json_value(request_span.attributes.get("wmo.request.tags"))
    if isinstance(value, str) and value.strip():
        return (normalize_durable_text(value.strip()),)
    if isinstance(value, list):
        tags: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("wmo.request.tags must contain non-empty text values")
            tags.add(normalize_durable_text(item.strip()))
        return tuple(sorted(tags))
    if value is not None:
        raise ValueError("wmo.request.tags must be text or a JSON array of text")
    return ()


def _initial_prompt_text(attributes: JsonObject) -> str | None:
    """Return initial user-prompt text when one canonical span supplies it."""
    messages = _json_value(attributes.get("gen_ai.input.messages"))
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if not isinstance(role, str) or role.casefold() not in {"user", "human"}:
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return normalize_durable_text(content.strip())
            if isinstance(content, list):
                texts = [
                    item["text"].strip()
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item["text"].strip()
                ]
                if texts:
                    return normalize_durable_text("\n".join(texts))
    prompt = attributes.get("gen_ai.prompt")
    if isinstance(prompt, str) and prompt.strip():
        return normalize_durable_text(prompt.strip())
    return None


def _domain(trace: Trace) -> str | None:
    """Select one source domain for coverage strata without router leakage."""
    context_domain = trace.initial_context.get("domain")
    if isinstance(context_domain, str) and context_domain.strip():
        return normalize_durable_text(context_domain.strip())
    for tag in _request_tags(trace):
        if tag.startswith("domain:") and len(tag) > len("domain:"):
            return tag.removeprefix("domain:")
    return None


def _span_marks_escalation(attributes: JsonObject) -> bool:
    """Return whether explicit mining-only source evidence marks an escalation."""
    for key in ("wmo.outcome.escalated", "wmo.escalation", "wmo.coverage.escalation"):
        value = attributes.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.casefold() in {"true", "escalated", "handoff"}:
            return True
    return False


def _collapse_runs(values: Sequence[str]) -> list[str]:
    """Collapse consecutive duplicate tool transitions for compact coverage reporting."""
    collapsed: list[str] = []
    for value in values:
        if not collapsed or collapsed[-1] != value:
            collapsed.append(value)
    return collapsed


def _json_value(value: JsonValue | None) -> JsonValue | None:
    """Decode a JSON-encoded source attribute when it is valid JSON."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

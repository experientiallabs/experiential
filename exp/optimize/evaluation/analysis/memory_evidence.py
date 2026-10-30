"""Passive, reference-only inspection of canonical requests and physical call receipts.

This reader performs no model call and never writes memory. Canonical intent establishes the
framed input, not provider receipt or cognitive use. Missing joins stay unknown.
"""

import json
from typing import Literal

from pydantic import Field, TypeAdapter, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    JsonValue,
    Sha256,
    sha256_json,
)
from exp.common.models import ModelRequest
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.models.call_evidence import Consumer


class EvidenceLocator(ContractModel):
    """An exact JSON value already retained in an immutable source artifact.

    Attributes:
        source: Verified original artifact, not a copied prompt or reasoning transcript.
        file: Artifact-relative JSON file.
        path: Object keys and array indices; $json decodes a JSON string before traversal.
        sha256: Canonical digest of the located value.
    """

    source: ArtifactInput
    file: str
    path: tuple[str | int, ...]
    sha256: Sha256


class InputFragment(ContractModel):
    """One retained input channel, without a claim that the model considered it.

    Attributes:
        channel: Original message, tools or explicitly framed simulator memory channel.
        location: Exact value in the final canonical request.
    """

    channel: str
    location: EvidenceLocator


class ReasoningEvidence(ContractModel):
    """Provider-exposed diagnostic evidence; absence never means memory was ignored.

    Attributes:
        availability: What the raw provider response actually retained.
        locations: Exact source values, never hidden reasoning requested or reconstructed.
    """

    availability: Literal[
        "exposed_text",
        "provider_summary",
        "opaque_carrier",
        "usage_only",
        "not_exposed",
        "not_captured",
        "incomplete",
    ]
    locations: tuple[EvidenceLocator, ...] = ()


class AttemptEvidence(ContractModel):
    """One physically admitted attempt, including unknown and rejected outcomes.

    Attributes:
        intent: Saved dispatch intent with the canonical parent.
        result: Retained terminal receipt, if any.
        status: Recorded outcome, not inference from an absent result.
        reasoning: Diagnostic availability, separate from task correctness or utility.
    """

    intent: ArtifactInput
    result: ArtifactInput | None
    status: Literal["response_saved", "not_dispatched", "outcome_unknown", "response_rejected"]
    reasoning: tuple[ReasoningEvidence, ...]


class CallExposure(ContractModel):
    """One requested canonical call, including calls without physical receipt coverage.

    Attributes:
        intent: Original canonical call artifact.
        consumer: Explicit execution-owner role; old or unlabelled calls remain unknown.
        logical_id: Owner coordinate, if it was captured.
        request_sha256: Final framed canonical request digest.
        fragments: Located input values, without duplicating their content.
        attempts: All linked physical intents, including failures.
        coverage: Whether a physical join exists; its absence does not prove non-dispatch.
        disposition: No model statement of use is inferred by this mechanical reader.
    """

    intent: ArtifactInput
    consumer: Consumer
    logical_id: str | None
    request_sha256: Sha256
    fragments: tuple[InputFragment, ...]
    attempts: tuple[AttemptEvidence, ...]
    coverage: Literal["linked_attempts", "physical_join_unavailable"]
    disposition: Literal["not_assessed"] = "not_assessed"


class MemoryInspection(ContractModel):
    """Explicit calls or a saved result's complete declared dependency graph to inspect offline.

    Attributes:
        intents: Every canonical call in the requested window, in caller-declared order.
        roots: Exact result artifacts to traverse, including ancestral evidence dependencies.
        maximum_artifacts: Hard traversal bound; exceeding it refuses instead of truncating.
    """

    intents: tuple[ArtifactInput, ...] = ()
    roots: tuple[ArtifactInput, ...] = ()
    maximum_artifacts: int = Field(default=10000, ge=1, le=100000, strict=True)

    @model_validator(mode="after")
    def one_inventory(self) -> "MemoryInspection":
        """Require exactly one inventory mode; neither an empty request nor a hidden union."""
        if bool(self.intents) == bool(self.roots):
            raise ValueError("memory inspection requires either explicit intents or result roots")
        if len(set(self.roots)) != len(self.roots):
            raise ValueError("memory inspection repeats a result root")
        return self


class MemoryExposureReport(ContractModel):
    """Passive input and dispatch evidence, with no inferred causal or learning authority.

    Attributes:
        calls: Every requested call, including unsuccessful execution.
        scope: Boundaries on interpretation and on the supplied inventory.
        inventory: Exact caller inventory, including traversal limits for reproducible rechecks.
        dependency_count: Verified graph size for automatic inventories, absent for explicit calls.
    """

    calls: tuple[CallExposure, ...]
    inventory: MemoryInspection
    dependency_count: int | None = None
    scope: str = (
        "explicit call inventory, not proof of complete run coverage; canonical input is not "
        "proof of provider receipt, attention, correctness or causal utility; no memory mutation"
    )

    def sources(self) -> tuple[ArtifactInput, ...]:
        """Retain roots and all original input/output receipts used by this inspection."""
        sources = {*self.inventory.roots, *self.inventory.intents}
        for call in self.calls:
            sources.add(call.intent)
            sources.update(fragment.location.source for fragment in call.fragments)
            for attempt in call.attempts:
                sources.add(attempt.intent)
                if attempt.result is not None:
                    sources.add(attempt.result)
                sources.update(
                    location.source
                    for reasoning in attempt.reasoning
                    for location in reasoning.locations
                )
        return tuple(sorted(sources, key=lambda reference: reference.artifact_id))


def located_value(store: ArtifactStore, location: EvidenceLocator) -> JsonValue:
    """Verify the artifact and exact value digest before returning diagnostic evidence."""
    if artifact_input(store.read(location.source.artifact_id).manifest) != location.source:
        raise ValueError("evidence source identity differs")
    value = json.loads(store.read_bytes(location.source.artifact_id, location.file))
    for part in location.path:
        if part == "$json" and isinstance(value, str):
            value = json.loads(value)
            continue
        if isinstance(part, int) and not isinstance(part, bool) and isinstance(value, list):
            if part < 0 or part >= len(value):
                raise ValueError("evidence array locator is outside the source")
            value = value[part]
        elif isinstance(part, str) and isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ValueError("evidence locator does not exist")
    if sha256_json(value) != location.sha256:
        raise ValueError("evidence content digest differs")
    return value


def _location(
    source: ArtifactInput, file: str, path: tuple[str | int, ...], value: JsonValue
) -> EvidenceLocator:
    """Bind a value already read from verified source bytes."""
    return EvidenceLocator(source=source, file=file, path=path, sha256=sha256_json(value))


def _reasoning(store: ArtifactStore, result: ArtifactInput) -> tuple[ReasoningEvidence, ...]:
    """Recognize explicit provider fields; unsupported response layouts remain not captured."""
    receipt = TypeAdapter(JsonObject).validate_json(
        store.read_bytes(result.artifact_id, "receipt.json")
    )
    body = receipt.get("response_body")
    if not isinstance(body, dict):
        return (ReasoningEvidence(availability="not_captured"),)
    found: list[ReasoningEvidence] = []
    incomplete = body.get("status") in {"incomplete", "failed"}

    def add(kind: str, path: tuple[str | int, ...], value: JsonValue) -> None:
        """Retain a locator only when an explicit reasoning field has content."""
        if value:
            found.append(
                ReasoningEvidence.model_validate(
                    {
                        "availability": kind,
                        "locations": [
                            _location(result, "receipt.json", ("response_body", *path), value)
                        ],
                    }
                )
            )

    recognized = False
    choices = body.get("choices")
    if isinstance(choices, list):
        recognized = True
        for i, choice in enumerate(choices):
            if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                incomplete = True
            message = choice.get("message") if isinstance(choice, dict) else None
            if not isinstance(message, dict):
                continue
            for key in ("reasoning_content", "reasoning"):
                text = message.get(key)
                if isinstance(text, str):
                    add("exposed_text", ("choices", i, "message", key), text)
    for collection in ("content", "output"):
        blocks = body.get(collection)
        if not isinstance(blocks, list):
            continue
        recognized = True
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            thinking = block.get("thinking")
            if block.get("type") == "thinking" and isinstance(thinking, str):
                add("exposed_text", (collection, i, "thinking"), thinking)
            if block.get("type") == "redacted_thinking":
                add("opaque_carrier", (collection, i), block)
            if block.get("type") == "reasoning":
                summaries = block.get("summary")
                if isinstance(summaries, list):
                    for j, summary in enumerate(summaries):
                        text = summary.get("text") if isinstance(summary, dict) else None
                        if isinstance(text, str):
                            add(
                                "provider_summary",
                                (collection, i, "summary", j, "text"),
                                text,
                            )
                add(
                    "opaque_carrier",
                    (collection, i, "encrypted_content"),
                    block.get("encrypted_content"),
                )
    usage = body.get("usage")
    if isinstance(usage, dict):
        for details in ("completion_tokens_details", "output_tokens_details"):
            counts = usage.get(details)
            if isinstance(counts, dict) and counts.get("reasoning_tokens"):
                add(
                    "usage_only", ("usage", details, "reasoning_tokens"), counts["reasoning_tokens"]
                )
    if incomplete:
        found.append(ReasoningEvidence(availability="incomplete"))
    return tuple(found) or (
        ReasoningEvidence(availability="not_exposed" if recognized else "not_captured"),
    )


def _inventory(
    store: ArtifactStore, spec: MemoryInspection
) -> tuple[tuple[ArtifactInput, ...], int | None]:
    """Traverse exact manifest edges, rejecting broken links or an incomplete bounded walk."""
    if spec.intents:
        return spec.intents, None
    pending, seen, intents = list(spec.roots), {}, []
    while pending:
        reference = pending.pop()
        if reference.artifact_id in seen:
            if seen[reference.artifact_id] != reference:
                raise ValueError("memory dependency graph contains conflicting source identities")
            continue
        if len(seen) >= spec.maximum_artifacts:
            raise ValueError("memory dependency graph exceeds maximum_artifacts; raise the bound")
        manifest = store.read(reference.artifact_id).manifest
        if artifact_input(manifest) != reference:
            raise ValueError("memory dependency graph source digest differs")
        seen[reference.artifact_id] = reference
        if manifest.artifact_type in {"structured-call-intent", "continuation-actor-request"}:
            intents.append(reference)
        pending.extend(manifest.inputs)
    return tuple(sorted(intents, key=lambda ref: ref.artifact_id)), len(seen)


def inspect_memory(store: ArtifactStore, spec: MemoryInspection) -> MemoryExposureReport:
    """Read canonical and physical artifacts only; never call providers or publish memory."""
    inventory, dependency_count = _inventory(store, spec)
    if len(set(inventory)) != len(inventory):
        raise ValueError("canonical call inventory repeats an intent")
    physical = [
        store.read(identity)
        for identity in store.list_ids(artifact_type="provider-physical-attempt")
    ]
    children_by_parent = {}
    for artifact in physical:
        if len(artifact.manifest.inputs) == 1:
            children_by_parent.setdefault(artifact.manifest.inputs[0], []).append(artifact)
    calls = []
    for reference in inventory:
        saved = store.read(reference.artifact_id)
        if artifact_input(saved.manifest) != reference or saved.manifest.artifact_type not in {
            "structured-call-intent",
            "continuation-actor-request",
        }:
            raise ValueError("inspection requires exact canonical call intents")
        payload = json.loads(store.read_bytes(reference.artifact_id, "request.json"))
        continuation = saved.manifest.artifact_type == "continuation-actor-request"
        body = payload if continuation else payload["request"]
        prefix = () if continuation else ("request",)
        request = ModelRequest.model_validate(body)
        consumer = "actor" if continuation else payload.get("consumer", "unknown")
        fragments = [
            InputFragment(
                channel=f"message:{message.role}",
                location=_location(
                    reference,
                    "request.json",
                    (*prefix, "messages", i),
                    body["messages"][i],
                ),
            )
            for i, message in enumerate(request.messages)
        ]
        if request.tools:
            fragments.append(
                InputFragment(
                    channel="tool_schemas",
                    location=_location(
                        reference,
                        "request.json",
                        (*prefix, "tools"),
                        body["tools"],
                    ),
                )
            )
        if consumer == "simulator":
            for i, message in enumerate(request.messages):
                if message.role != "user" or not message.content:
                    continue
                try:
                    framed = json.loads(message.content)
                except ValueError:
                    continue
                if not isinstance(framed, dict):
                    continue
                base = ("request", "messages", i, "content", "$json")
                for key in (
                    "memory_observations",
                    "behavioral_knowledge",
                    "library_originals",
                    "prediction_evidence",
                ):
                    if key in framed:
                        fragments.append(
                            InputFragment(
                                channel=key,
                                location=_location(
                                    reference,
                                    "request.json",
                                    (*base, key),
                                    framed[key],
                                ),
                            )
                        )
                research = framed.get("research_history")
                if isinstance(research, dict):
                    for key, value in research.items():
                        fragments.append(
                            InputFragment(
                                channel="research_history." + key,
                                location=_location(
                                    reference,
                                    "request.json",
                                    (*base, "research_history", key),
                                    value,
                                ),
                            )
                        )
        attempts = []
        for intent in children_by_parent.get(reference, ()):
            receipt = json.loads(intent.payloads["receipt.json"])
            if receipt.get("canonical_request") != reference.model_dump(mode="json"):
                raise ValueError("physical intent canonical parent differs")
            parent = artifact_input(intent.manifest)
            children = children_by_parent.get(parent, ())
            if len(children) > 1:
                raise ValueError("physical attempt has conflicting terminal receipts")
            result = artifact_input(children[0].manifest) if children else None
            terminal = json.loads(children[0].payloads["receipt.json"]) if children else {}
            status = "outcome_unknown"
            if "response_body" in terminal:
                status = "response_saved"
            elif terminal.get("dispatch") == "not_started":
                status = "not_dispatched"
            elif terminal.get("outcome") == "response_not_retained_credential_boundary":
                status = "response_rejected"
            attempts.append(
                AttemptEvidence.model_validate(
                    {
                        "intent": parent,
                        "result": result,
                        "status": status,
                        "reasoning": _reasoning(store, result)
                        if result and status == "response_saved"
                        else (ReasoningEvidence(availability="incomplete"),),
                    }
                )
            )
        calls.append(
            CallExposure(
                intent=reference,
                consumer=consumer,
                logical_id=reference.artifact_id if continuation else payload.get("logical_id"),
                request_sha256=sha256_json(body),
                fragments=tuple(fragments),
                attempts=tuple(attempts),
                coverage="linked_attempts" if attempts else "physical_join_unavailable",
            )
        )
    scope = MemoryExposureReport.model_fields["scope"].default
    if spec.roots:
        scope = (
            "complete declared dependency graph at read time, including ancestral source calls; "
            "undeclared calls and missing physical joins remain unknown; canonical input is not "
            "proof of provider receipt, attention, correctness or causal utility; "
            "no memory mutation"
        )
    return MemoryExposureReport(
        calls=tuple(calls), inventory=spec, dependency_count=dependency_count, scope=scope
    )


class EmittedSpan(ContractModel):
    """An exact substring in an explicitly exposed response field, without copying its text.

    Attributes:
        location: Original exposed text or provider summary, not opaque or usage evidence.
        start: Inclusive character offset in the saved text.
        end: Exclusive character offset.
        sha256: Canonical string digest of the exact substring.
    """

    location: EvidenceLocator
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    sha256: Sha256


class MemoryDisposition(ContractModel):
    """A diagnostic interpretation with verifiable input and output citations.

    Attributes:
        intent: Exact consumer call, not another model's context.
        fragment: Delivered input fragment being discussed.
        statement: Analyst's interpretation, never a verified behavioral or causal assertion.
        spans: Required emitted evidence; no citation means no supported interpretation.
    """

    intent: ArtifactInput
    fragment: EvidenceLocator
    statement: Literal["explicitly_considered", "explicitly_rejected", "explicitly_deferred"]
    spans: tuple[EmittedSpan, ...] = Field(min_length=1)


def validate_disposition(
    store: ArtifactStore,
    report: MemoryExposureReport,
    assessment: MemoryDisposition,
) -> None:
    """Check evidence identity and exact spans; do not certify the analyst's semantics.

    Passing establishes citation integrity only. Independent rubric calibration remains necessary
    before treating a statement as reliable, and no statement establishes causal utility.
    """
    verified = inspect_memory(store, report.inventory)
    if verified != report:
        raise ValueError("assessment exposure inventory changed or was not verified")
    call = next((row for row in report.calls if row.intent == assessment.intent), None)
    if call is None or call.consumer == "unknown":
        raise ValueError("assessment lacks a verified consumer boundary")
    if assessment.fragment not in tuple(row.location for row in call.fragments):
        raise ValueError("assessment input was not in this consumer's canonical request")
    located_value(store, assessment.fragment)
    permitted = tuple(
        location
        for attempt in call.attempts
        for evidence in attempt.reasoning
        if evidence.availability in {"exposed_text", "provider_summary"}
        for location in evidence.locations
    )
    for span in assessment.spans:
        if span.location not in permitted:
            raise ValueError("assessment span is not exposed text from this consumer")
        value = located_value(store, span.location)
        if not isinstance(value, str) or not span.start < span.end <= len(value):
            raise ValueError("assessment span is outside the exposed text")
        if sha256_json(value[span.start : span.end]) != span.sha256:
            raise ValueError("assessment span digest differs")

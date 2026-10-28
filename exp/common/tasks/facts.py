"""Instance facts: validated, provenance-bearing statements about one task instance's world.

Learned memory describes how an environment behaves across instances. Instance facts describe
this instance's world (its workbook cells, orders or tickets) at an exact state version. They
are a separate channel with separate authority, and they enter a simulator request only through
:class:`ConditioningFacts`.

State versions count applied actions: version 0 is the starting state before the first action,
and version ``n`` is the state after ``n`` actions. A prediction for the action at index ``i``
sees state version ``i``, so a fact read after that action (version ``i + 1`` or later) can
never condition it or any earlier prediction.

Two visibility classes exist. ``conditioning`` facts may condition simulation.
``verifier_private`` facts are evidence for verifiers only: no simulator or actor request ever
receives them, and :class:`ConditioningFacts` refuses them.

Environment adapters implement :class:`InstanceFactsProvider`, and a backend that decodes its own
declared input files implements :class:`InstanceFactsDeriver`. Domain decoding (spreadsheet
XML, database snapshots, API reads) belongs in the adapter; this module only validates and
carries the decoded facts.
"""

from __future__ import annotations

from typing import Literal, Protocol, Self

from pydantic import Field, JsonValue, ValidationError, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256
from exp.common.tasks.task import TaskCase

# Request-context entry through which a trace package supplies instance facts.
INSTANCE_FACTS_CONTEXT_KEY = "instance_facts"


class InstanceFactsError(ValueError):
    """Instance facts are malformed, unvalidated, or not allowed where they were supplied."""


class FactValidation(ContractModel):
    """How a fact was checked against its source.

    Attributes:
        status: ``validated`` facts may be used; ``unvalidated`` and ``failed`` facts are
            carried only so they can be refused with a clear error.
        method: How the fact was independently checked; required when validated.
    """

    status: Literal["validated", "unvalidated", "failed"]
    method: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _require_method(self) -> Self:
        """A validated fact states how it was validated."""
        if self.status == "validated" and self.method is None:
            raise ValueError("a validated fact must state its validation method")
        return self


class FactProvenance(ContractModel):
    """Where one fact came from and which state it describes.

    Attributes:
        adapter: Source adapter name and version that decoded the fact.
        source: Source artifact or snapshot the fact was read from, such as a file name,
            snapshot ID, or API resource.
        source_sha256: Content digest of the source artifact, when it is a file.
        derived_from_traces: Recorded traces whose content the fact was derived from; empty
            when the source is independent of any recorded trace. A fact derived from a
            held-out trace inherits that trace's reservation and exposure rules.
        validation: How the fact was checked against its source.
        state_version: State version the fact describes: 0 is the starting state, ``n`` the
            state after ``n`` actions.
    """

    adapter: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=2048)
    source_sha256: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)
    derived_from_traces: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    validation: FactValidation
    state_version: int = Field(ge=0, strict=True)


class InstanceFact(ContractModel):
    """One keyed value of an instance's world, with visibility and provenance.

    Attributes:
        key: Path or key naming the value, such as ``input.xlsx/Sheet1/A2`` or
            ``orders/#W123/status``.
        value: Decoded JSON value. ``null`` is an observed empty value, distinct from a value
            that is simply absent from the facts.
        visibility: ``conditioning`` (may condition simulation) or ``verifier_private``
            (verifier evidence only, never in any simulator or actor request).
        provenance: Source, validation, and described state version.
    """

    key: str = Field(min_length=1, max_length=1024)
    value: JsonValue
    visibility: Literal["conditioning", "verifier_private"]
    provenance: FactProvenance

    def rendered(self) -> JsonObject:
        """The fact as one simulator-visible record, with its authority and provenance."""
        provenance = self.provenance
        record: JsonObject = {
            "key": self.key,
            "value": self.value,
            "state_version": provenance.state_version,
            "source": provenance.source,
            "adapter": provenance.adapter,
            "validated_by": provenance.validation.method,
        }
        return record


class InstanceFacts(ContractModel):
    """Every fact one adapter supplies for one task instance.

    Attributes:
        schema_version: Facts contract version.
        facts: Facts in adapter order; a key appears at most once per state version.
    """

    schema_version: Literal[1] = 1
    facts: tuple[InstanceFact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_keys(self) -> Self:
        """One value per key and state version."""
        seen: set[tuple[str, int]] = set()
        for fact in self.facts:
            identity = (fact.key, fact.provenance.state_version)
            if identity in seen:
                raise ValueError(
                    f"instance fact {fact.key!r} appears twice at state version "
                    f"{fact.provenance.state_version}; supply one value per key and version"
                )
            seen.add(identity)
        return self

    def require_validated(self, origin: str) -> Self:
        """Refuse the whole set when any fact is not validated.

        Args:
            origin: Who supplied the facts, named in the error.

        Returns:
            These facts, unchanged.

        Raises:
            InstanceFactsError: A fact is unvalidated or failed validation.
        """
        refused = [fact for fact in self.facts if fact.provenance.validation.status != "validated"]
        if refused:
            listed = ", ".join(
                f"{fact.key} ({fact.provenance.validation.status})" for fact in refused[:5]
            )
            more = f" and {len(refused) - 5} more" if len(refused) > 5 else ""
            raise InstanceFactsError(
                f"{origin} supplies instance facts that are not validated: {listed}{more}. "
                "Only validated facts may be used; check each fact against its source in the "
                "adapter and mark it validated with the method, or omit it"
            )
        return self

    def conditioning(self, state_version: int) -> ConditioningFacts | None:
        """The facts that may condition a prediction made at ``state_version``.

        Args:
            state_version: State version the prediction starts from: the number of actions
                applied before the predicted action.

        Returns:
            Validated conditioning facts describing that version or an earlier one, or None
            when there are none. Verifier-private facts are never included.
        """
        allowed = tuple(
            fact
            for fact in self.facts
            if fact.visibility == "conditioning" and fact.provenance.state_version <= state_version
        )
        if not allowed:
            return None
        return ConditioningFacts(state_version=state_version, facts=allowed)


class ConditioningFacts(ContractModel):
    """The only form in which instance facts reach a simulator request.

    Attributes:
        state_version: State version of the prediction these facts condition.
        facts: Validated conditioning facts describing that version or an earlier one.
    """

    state_version: int = Field(ge=0, strict=True)
    facts: tuple[InstanceFact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_admissible(self) -> Self:
        """Refuse private, unvalidated, or later-state facts."""
        for fact in self.facts:
            if fact.visibility != "conditioning":
                raise ValueError(
                    f"instance fact {fact.key!r} is {fact.visibility}; it never enters a "
                    "simulator request"
                )
            if fact.provenance.validation.status != "validated":
                raise ValueError(f"instance fact {fact.key!r} is not validated")
            if fact.provenance.state_version > self.state_version:
                raise ValueError(
                    f"instance fact {fact.key!r} describes state version "
                    f"{fact.provenance.state_version}, after the predicted action at state "
                    f"version {self.state_version}; it cannot condition this prediction"
                )
        return self

    def rendered(self) -> JsonObject:
        """The simulator-visible ``instance_facts`` channel."""
        return {
            "state_version": self.state_version,
            "facts": [fact.rendered() for fact in self.facts],
        }


class InstanceFactsProvider(Protocol):
    """An environment adapter that knows the starting world of a task instance."""

    def initial_facts(self, task: TaskCase) -> InstanceFacts | None:
        """Facts about the instance ``task`` runs in, or None when the adapter has none.

        Args:
            task: Task whose instance the facts describe.

        Returns:
            Facts with provenance; validation status is checked by the caller.
        """
        ...


class InstanceFactsDeriver(Protocol):
    """An environment backend's decoder from one declared input file to instance facts.

    A backend that can read its own starting files (a workbook, a database snapshot) declares
    one deriver. The deriver decodes the exact bytes, validates the decoded facts against the
    backend's own reader, and refuses with :class:`InstanceFactsError` any content it cannot
    represent exactly, so a caller never receives facts with silent gaps.
    """

    @property
    def kind(self) -> str:
        """Stable deriver name, as declared in a derivation spec."""
        ...

    def derive(self, path: str, payload: bytes) -> InstanceFacts:
        """Validated starting-state facts about one input file.

        Args:
            path: The file's canonical path inside the task's world, such as ``input.xlsx``.
                It names the facts and their provenance source; host paths never appear.
            payload: Exact file bytes.

        Returns:
            Validated conditioning facts at state version 0.

        Raises:
            InstanceFactsError: The file has content the deriver cannot represent exactly,
                exceeds the deriver's bounds, or fails validation.
        """
        ...


def parse_instance_facts(value: JsonValue, origin: str) -> InstanceFacts:
    """Validate supplied facts JSON and refuse anything that is not validated.

    Args:
        value: Supplied facts object.
        origin: Who supplied the facts, named in errors.

    Returns:
        Validated facts.

    Raises:
        InstanceFactsError: The value is not a facts object, or a fact is not validated.
    """
    try:
        facts = InstanceFacts.model_validate(value)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in row['loc']) or 'facts'}: {row['msg']}"
            for row in error.errors()[:5]
        )
        raise InstanceFactsError(
            f"{origin} supplies invalid instance facts: {problems}. Each fact needs a key, a "
            "value, a visibility (conditioning or verifier_private) and provenance (adapter, "
            "source, validation, state_version)"
        ) from error
    return facts.require_validated(origin)

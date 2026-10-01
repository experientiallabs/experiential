"""Project-scoped provider receipts and accounting in the shared content database."""

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal, Self, cast
from uuid import uuid4

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_key_secret_free,
    assert_prose_secret_free,
    assert_text_secret_free,
    canonical_json_bytes,
    stable_id,
)
from exp.common.project.errors import ArtifactStoreError
from exp.common.project.manifests import artifact_input
from exp.common.project.records import ProjectRecords
from exp.common.project.store import ProjectStore
from exp.common.release_revision import installed_release_revision


class _Response(ContractModel):
    """Exact encoded provider output retained as data rather than credential configuration.

    Attributes:
        payload: Original encoded response, without normalization or redaction.
    """

    payload: str


def _validate_response(payload: str) -> None:
    """Validate encoded output using its matching structured or plain-text boundary.

    Args:
        payload: Encoded response, which may be arbitrary text or serialized JSON.

    Raises:
        ArtifactStoreError: The response violates the credential boundary or decoder depth limit.
    """
    try:
        try:
            value = _response_json(payload)
        except json.JSONDecodeError:
            assert_text_secret_free(payload)
        else:
            _validate_response_values(value)
    except SecretBoundaryError as exc:
        raise ArtifactStoreError(
            "provider response violates the secret boundary (credential-like content). "
            "Remove credential-bearing fields, assignments, and secret values from the output, "
            "then start a fresh evaluation. The rejected response was not saved."
        ) from exc
    except RecursionError as exc:
        raise ArtifactStoreError(
            "provider response exceeds the JSON nesting supported by the decoder. "
            "Simplify the nested JSON output, then start a fresh evaluation. "
            "The rejected response was not saved."
        ) from exc


def _response_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    """Reject duplicate keys before parsing can discard any retained response content."""
    value: dict[str, JsonValue] = {}
    for key, nested in pairs:
        if key in value:
            raise ArtifactStoreError(
                "provider response violates the secret boundary (duplicate JSON keys). "
                "Remove duplicate keys from the encoded output, then start a fresh evaluation. "
                "The rejected response was not saved."
            )
        value[key] = nested
    return value


def _response_json(payload: str) -> JsonValue:
    """Decode fields and string content without discarding duplicate object members.

    Integer literals stay as digit strings in this validation-only view, avoiding Python's
    integer-conversion limit. Persistence retains the original response, including number types.
    """
    return cast(JsonValue, json.loads(payload, object_pairs_hook=_response_object, parse_int=str))


def _validate_response_values(value: JsonValue) -> None:
    """Check credential fields and assignments, including JSON encoded inside string leaves."""
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, nested in current.items():
                assert_key_secret_free(key)
                pending.append(nested)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, str):
            assert_prose_secret_free(current)
            try:
                nested = _response_json(current)
            except json.JSONDecodeError:
                continue
            if nested != current:
                pending.append(nested)


class RequestReceipt(ContractModel):
    """One request reservation or completed provider response.

    Attributes:
        fingerprint: Exact request and pricing identity.
        charge: Settled cost or conservative unresolved reservation, in USD.
        response: Immutable response artifact, present only after completion.
        state: Whether dispatch is pending, complete, unresolved, or certified wholly unpaid.
        unbilled_attempts: Exact positive attempt count for a terminal wholly unpaid request.
            Absent for existing receipts and all other states.
    """

    fingerprint: str
    charge: float = Field(ge=0, allow_inf_nan=False)
    response: ArtifactInput | None = None
    state: Literal["pending", "complete", "unknown", "unbilled"]
    unbilled_attempts: int = Field(
        default=0, ge=0, strict=True, exclude_if=lambda value: value == 0
    )

    @model_validator(mode="after")
    def _validate_unbilled_failure(self) -> Self:
        """Only explicit wholly unpaid failures may retain a certified positive count."""
        if self.state == "unbilled":
            if self.charge != 0 or self.response is not None or self.unbilled_attempts == 0:
                raise ValueError("unbilled receipt requires positive attempt proof and zero charge")
        elif self.unbilled_attempts:
            raise ValueError("unbilled attempt proof requires a terminal unbilled receipt")
        return self


class _Total(ContractModel):
    """Atomic accounting total, including unresolved provider reservations.

    Attributes:
        charge: Sum of all request charges in this execution namespace.
    """

    charge: float = Field(ge=0, allow_inf_nan=False)


class RequestBudgetStore:
    """Bind accounting and response artifacts to one project and execution identity."""

    def __init__(self, project: ProjectStore, identity: str) -> None:
        """Select an execution namespace without creating state.

        Args:
            project: Owner of accounting records and immutable response artifacts.
            identity: Digest binding the approved execution's inputs and prices.
        """
        self._project = project
        self._identity = identity
        self._records = ProjectRecords(
            project.paths.root, project.paths.project_id, f"request-budget/{identity}"
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize short admission or settlement mutations, excluding provider calls."""
        with self._records.transaction():
            yield

    def authorize(self, limit: float | None) -> None:
        """Append the approved aggregate limit, or null for uncapped accounting, to history."""
        self._records.append(
            str(uuid4()),
            canonical_json_bytes({"limit_usd": limit, "created_at": datetime.now(UTC).isoformat()}),
        )

    def read(self, key: str) -> RequestReceipt | None:
        """Read an exact request coordinate with verified metadata."""
        payload = self._records.read(f"request/{key}")
        return None if payload is None else RequestReceipt.model_validate_json(payload)

    def total(self) -> float:
        """Read the accounting total committed atomically with every request mutation."""
        payload = self._records.read("total")
        return 0.0 if payload is None else _Total.model_validate_json(payload).charge

    def write(self, key: str, receipt: RequestReceipt) -> None:
        """Commit a receipt and its accounting delta together.

        Args:
            key: Exact scope, role, and ordinal digest within this execution.
            receipt: Reservation or settlement replacing the previous charge atomically.
        """
        with self.transaction():
            previous = self.read(key)
            total = math.fsum((self.total(), -(previous.charge if previous else 0), receipt.charge))
            self._records.write(f"request/{key}", canonical_json_bytes(receipt))
            self._records.write("total", canonical_json_bytes(_Total(charge=max(0.0, total))))

    def complete(self, key: str, cost: float, payload: str) -> None:
        """Atomically bind paid response bytes and replace the conservative reservation.

        Args:
            key: Request coordinate with a durable pending reservation.
            cost: Validated finite, nonnegative settled charge in USD.
            payload: Encoded response to retain for exact replay.

        Raises:
            ValueError: The coordinate has no pending reservation.
        """
        with self.transaction():
            previous = self.read(key)
            if previous is None or previous.state != "pending":
                raise ValueError("provider request does not have a pending reservation")
            _validate_response(payload)
            manifest = self._project.artifacts.write(
                artifact_id=stable_id("request-response", {"budget": self._identity, "key": key}),
                artifact_type="provider-response",
                envelope=ArtifactEnvelope(
                    schema_version=2,
                    created_at=datetime.now(UTC),
                    code_revision=installed_release_revision(),
                ),
                files={"response.json": canonical_json_bytes(_Response(payload=payload))},
            )
            self.write(
                key,
                previous.model_copy(
                    update={
                        "charge": cost,
                        "response": artifact_input(manifest),
                        "state": "complete",
                    }
                ),
            )

    def response(self, receipt: RequestReceipt) -> str:
        """Verify the bound response manifest and payload before exact replay.

        Args:
            receipt: Completed request with an exact immutable response pointer.

        Returns:
            Original encoded response after manifest and content digest verification.

        Raises:
            ValueError: The response pointer is absent or its manifest changed.
        """
        pointer = receipt.response
        if pointer is None:
            raise ValueError("completed provider request has no response receipt")
        stored = self._project.artifacts.read(pointer.artifact_id)
        if artifact_input(stored.manifest) != pointer:
            raise ValueError("saved provider response identity changed")
        if stored.manifest.schema_version == 1:
            return self._project.artifacts.read_bytes(pointer.artifact_id, "response.txt").decode(
                "utf-8"
            )
        if stored.manifest.schema_version != 2:
            raise ValueError("saved provider response schema is unsupported")
        payload = self._project.artifacts.read_bytes(pointer.artifact_id, "response.json")
        return _Response.model_validate_json(payload).payload

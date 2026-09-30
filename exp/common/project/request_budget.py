"""Project-scoped provider receipts and accounting in the shared content database."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import Field, JsonValue, TypeAdapter, ValidationError

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    SecretBoundaryError,
    assert_secret_free,
    assert_text_secret_free,
    canonical_json_bytes,
    stable_id,
)
from exp.common.project.errors import ArtifactStoreError
from exp.common.project.manifests import artifact_input
from exp.common.project.records import ProjectRecords
from exp.common.project.store import ProjectStore
from exp.common.release_revision import installed_release_revision

_JSON_VALUE = TypeAdapter(JsonValue)


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
        ArtifactStoreError: The response contains a secret value or credential field.
    """
    try:
        try:
            value = _JSON_VALUE.validate_json(payload)
        except ValidationError:
            assert_text_secret_free(payload)
        else:
            _validate_response_keys(value)
            assert_secret_free(value)
    except SecretBoundaryError as exc:
        raise ArtifactStoreError("provider response violates the secret boundary") from exc


def _validate_response_keys(value: JsonValue) -> None:
    """Reject credential references and environment-variable names in nested JSON keys."""
    if isinstance(value, dict):
        for key, nested in value.items():
            assert_text_secret_free(key)
            _validate_response_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _validate_response_keys(nested)


class RequestReceipt(ContractModel):
    """One request reservation or completed provider response.

    Attributes:
        fingerprint: Exact request and pricing identity.
        charge: Settled cost or conservative unresolved reservation, in USD.
        response: Immutable response artifact, present only after completion.
        state: Whether dispatch is pending, complete, or unresolved.
    """

    fingerprint: str
    charge: float = Field(ge=0, allow_inf_nan=False)
    response: ArtifactInput | None = None
    state: Literal["pending", "complete", "unknown"]


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

    def authorize(self, limit: float) -> None:
        """Append the explicitly approved limit to the durable execution history."""
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

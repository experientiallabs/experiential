"""Tests for admission-time rung deadness in the native control plane."""

from __future__ import annotations

import logging

import pytest

from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayFailureClass,
)
from exp.runtime.gateway.native_dead_rungs import (
    UNAVAILABLE_AT_ADMISSION,
    admission_dead_failure,
    dead_rung,
    escalation_failure,
    log_dead_rungs,
)
from exp.runtime.models import ModelConnectionError
from exp.runtime.models.credentials import MissingModelCredentialError, ModelCredentialError
from exp.runtime.models.providers.errors import ProviderCapabilityError


def _deployment(deployment_id: str) -> ExactModelDeployment:
    """Build one exact deployment."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="vertex",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(),
    )


def _authorization() -> AuthorizationSnapshot:
    """Build one accepted request's authorization snapshot."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )


def test_admission_dead_failure_mirrors_runtime_circuit_classes() -> None:
    """A dead-at-admission rung feeds the circuit like the runtime failure it mirrors."""
    # A missing credential opens the circuit at once, like a runtime 401.
    for credential_error in (
        ModelCredentialError("no key"),
        MissingModelCredentialError("TEST_KEY", connection_id="conn"),
    ):
        failure = admission_dead_failure(credential_error)
        assert failure.failure_class == GatewayFailureClass.PROVIDER_AUTHENTICATION

    # A connection or capability drift is operational, like a runtime transport failure.
    assert (
        admission_dead_failure(ModelConnectionError("drifted")).failure_class
        == GatewayFailureClass.TRANSPORT
    )
    assert (
        admission_dead_failure(ProviderCapabilityError(capability="reasoning")).failure_class
        == GatewayFailureClass.TRANSPORT
    )


def test_a_credential_cause_never_carries_its_message() -> None:
    """A credential error names only its type; a drift keeps its bounded message."""
    secret = dead_rung(0, _deployment("one"), ModelCredentialError("sk-secret-value"))
    assert secret.cause == "ModelCredentialError"
    drift = dead_rung(1, _deployment("two"), ModelConnectionError("x" * 400))
    assert drift.cause.startswith("ModelConnectionError: x")
    assert len(drift.cause) == 300
    assert drift.failure.failure_class == GatewayFailureClass.TRANSPORT


def test_an_all_dead_route_logs_every_rung_cause(caplog: pytest.LogCaptureFixture) -> None:
    """Only a route with no live rung logs the per-rung causes."""
    dead = (
        dead_rung(0, _deployment("one"), ModelConnectionError("swap")),
        dead_rung(1, _deployment("two"), ModelCredentialError("no key")),
    )
    with caplog.at_level(logging.WARNING):
        log_dead_rungs(_authorization(), dead, fallback_available=False)
    (record,) = [r for r in caplog.records if r.__dict__.get("operation") == "admission_dead_route"]
    assert record.__dict__["dead_rungs"] == [
        {"deployment_id": "one", "provider": "vertex", "cause": "ModelConnectionError: swap"},
        {"deployment_id": "two", "provider": "vertex", "cause": "ModelCredentialError"},
    ]

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        log_dead_rungs(_authorization(), dead[:1], fallback_available=True)
    assert [r.getMessage() for r in caplog.records] == [
        "gateway admission skipped the lead rung for alias 'public-model': served off a "
        "fallback because deployment 'one' (provider 'vertex') was dead at admission"
    ]


def test_escalation_failure_records_a_dead_route_as_unavailable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The all-dead reason is a retryable unavailable row; any other stays internal."""
    with caplog.at_level(logging.WARNING):
        dead = escalation_failure(_authorization(), UNAVAILABLE_AT_ADMISSION)
        other = escalation_failure(_authorization(), "host policy forbids this route")
    assert dead.failure_class == GatewayFailureClass.UNAVAILABLE
    assert other.failure_class == GatewayFailureClass.INTERNAL
    reasons = [r.__dict__["reason"] for r in caplog.records]
    assert reasons == [UNAVAILABLE_AT_ADMISSION, "host policy forbids this route"]

"""Tests for the honest later-provider certification artifact."""

from __future__ import annotations

from wmo.common.core.artifacts import assert_secret_free
from wmo.runtime.gateway.provider_certification import (
    PROVIDER_CERTIFICATION_MATRIX,
    ProviderCapability,
    ProviderCertificationMatrix,
    ProviderCertificationResult,
)


def test_provider_matrix_is_complete_deterministic_and_secret_free() -> None:
    """Every later-provider cell is labeled and the artifact round-trips exactly."""
    matrix = PROVIDER_CERTIFICATION_MATRIX

    assert len(matrix.cells) == 4 * len(ProviderCapability)
    assert matrix.identity_sha256() == PROVIDER_CERTIFICATION_MATRIX.identity_sha256()
    assert ProviderCertificationMatrix.model_validate_json(matrix.model_dump_json()) == matrix
    assert_secret_free(matrix.model_dump(mode="json"))


def test_provider_matrix_does_not_claim_unperformed_live_credentials() -> None:
    """Credential-gated status remains explicitly unrun until a dated live lane executes."""
    live_cells = tuple(
        cell
        for cell in PROVIDER_CERTIFICATION_MATRIX.cells
        if cell.capability is ProviderCapability.CREDENTIAL_GATED_LIVE
    )

    assert {cell.provider for cell in live_cells} == {
        "azure",
        "bedrock",
        "gemini",
        "openrouter",
    }
    assert all(
        cell.result is ProviderCertificationResult.NOT_RUN_REQUIRES_CREDENTIALS
        for cell in live_cells
    )


def test_gemini_raw_argument_limit_is_explicit_while_bedrock_is_certified() -> None:
    """Structured Gemini arguments are not mislabeled as byte-incremental streaming."""
    cells = {(cell.provider, cell.capability): cell for cell in PROVIDER_CERTIFICATION_MATRIX.cells}

    assert (
        cells[("gemini", ProviderCapability.TOOL_ARGUMENT_STREAM)].result
        is ProviderCertificationResult.UNSUPPORTED
    )
    assert (
        cells[("bedrock", ProviderCapability.TOOL_ARGUMENT_STREAM)].result
        is ProviderCertificationResult.PROVIDER_FIXTURE_PASS
    )

"""Tests for the honest launch-provider certification artifact."""

from __future__ import annotations

from datetime import UTC, datetime

from exp.common.core.artifacts import assert_secret_free
from exp.common.models import GATEWAY_EXCLUDED_PROVIDERS
from exp.runtime.gateway.provider_certification import (
    PROVIDER_CERTIFICATION_MATRIX,
    ProviderCapability,
    ProviderCertificationMatrix,
    ProviderCertificationResult,
)
from exp.runtime.models import SUPPORTED_PROVIDERS


def test_provider_matrix_is_complete_deterministic_and_secret_free() -> None:
    """Every launch-provider cell is labeled and the artifact round-trips exactly."""
    matrix = PROVIDER_CERTIFICATION_MATRIX

    assert len(matrix.cells) == 8 * len(ProviderCapability)
    assert {cell.provider for cell in matrix.cells} == {
        "anthropic",
        "azure",
        "bedrock",
        "gemini",
        "openai",
        "openai-compatible",
        "openrouter",
        "vertex",
    }
    assert {cell.client_sdk for cell in matrix.cells} == {"openai==3.0.0"}
    assert {cell.evaluated_at for cell in matrix.cells} == {datetime(2026, 8, 25, tzinfo=UTC)}
    assert {cell.gateway_api_surfaces for cell in matrix.cells} == {
        ("chat.completions", "responses")
    }
    assert all(cell.provider_api_surface for cell in matrix.cells)
    assert matrix.identity_sha256() == PROVIDER_CERTIFICATION_MATRIX.identity_sha256()
    assert ProviderCertificationMatrix.model_validate_json(matrix.model_dump_json()) == matrix
    assert_secret_free(matrix.model_dump(mode="json"))


def test_provider_matrix_tracks_every_gateway_servable_runtime_provider() -> None:
    """A new runtime provider must gain explicit certification before gateway release."""
    certified = {cell.provider for cell in PROVIDER_CERTIFICATION_MATRIX.cells}

    assert certified == SUPPORTED_PROVIDERS - GATEWAY_EXCLUDED_PROVIDERS


def test_provider_matrix_does_not_claim_unperformed_live_credentials() -> None:
    """Credential-gated status remains explicitly unrun until a dated live lane executes."""
    live_cells = tuple(
        cell
        for cell in PROVIDER_CERTIFICATION_MATRIX.cells
        if cell.capability is ProviderCapability.CREDENTIAL_GATED_LIVE
    )

    assert {cell.provider for cell in live_cells} == {
        "anthropic",
        "azure",
        "bedrock",
        "gemini",
        "openai",
        "openai-compatible",
        "openrouter",
        "vertex",
    }
    assert all(
        cell.result is ProviderCertificationResult.NOT_RUN_REQUIRES_CREDENTIALS
        for cell in live_cells
    )


def test_google_raw_argument_limit_is_explicit_while_bedrock_is_certified() -> None:
    """Structured Google arguments are not mislabeled as byte-incremental streaming."""
    cells = {(cell.provider, cell.capability): cell for cell in PROVIDER_CERTIFICATION_MATRIX.cells}

    assert (
        cells[("gemini", ProviderCapability.TOOL_ARGUMENT_STREAM)].result
        is ProviderCertificationResult.UNSUPPORTED
    )
    assert (
        cells[("vertex", ProviderCapability.TOOL_ARGUMENT_STREAM)].result
        is ProviderCertificationResult.UNSUPPORTED
    )
    assert (
        cells[("bedrock", ProviderCapability.TOOL_ARGUMENT_STREAM)].result
        is ProviderCertificationResult.PROVIDER_FIXTURE_PASS
    )


def test_anthropic_text_stream_cell_names_exact_native_fixture() -> None:
    """Anthropic text certification is bound to native golden-fixture evidence."""
    cells = {(cell.provider, cell.capability): cell for cell in PROVIDER_CERTIFICATION_MATRIX.cells}
    cell = cells[("anthropic", ProviderCapability.TEXT_STREAM)]

    assert cell.result is ProviderCertificationResult.PROVIDER_FIXTURE_PASS
    assert (
        "exp/runtime/gateway/native_bridge_test.py::"
        "test_rust_messages_sse_frames_match_the_committed_golden"
    ) in cell.evidence


def test_vertex_cells_name_native_profile_and_shared_stream_fixtures() -> None:
    """Vertex certification binds distinct routing plus shared Gemini parsing evidence."""
    cells = {(cell.provider, cell.capability): cell for cell in PROVIDER_CERTIFICATION_MATRIX.cells}
    cell = cells[("vertex", ProviderCapability.TEXT_STREAM)]

    assert cell.result is ProviderCertificationResult.PROVIDER_FIXTURE_PASS
    assert "exp/runtime/models/providers/vertex_test.py" in cell.evidence
    assert "exp/runtime/models/providers/gemini_streaming_test.py" in cell.evidence

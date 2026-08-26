"""Dated, secret-free certification matrix for launch gateway provider adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json


class ProviderCapability(StrEnum):
    """Provider adapter behaviors tracked by the launch certification matrix."""

    TEXT_STREAM = "text_stream"
    TOOL_ARGUMENT_STREAM = "tool_argument_stream"
    USAGE = "usage"
    CANCELLATION = "cancellation"
    REFUSAL = "refusal"
    CREDENTIAL_GATED_LIVE = "credential_gated_live"


class ProviderCertificationResult(StrEnum):
    """Evidence level for one provider and capability cell."""

    PROVIDER_FIXTURE_PASS = "provider_fixture_pass"
    INHERITED_COMPATIBLE_FIXTURE_PASS = "inherited_compatible_fixture_pass"
    NOT_RUN_REQUIRES_CREDENTIALS = "not_run_requires_credentials"
    UNSUPPORTED = "unsupported"


class ProviderCertificationCell(ContractModel):
    """One dated provider-capability result with reviewable local evidence."""

    provider: str = Field(min_length=1, max_length=64)
    provider_api_surface: str = Field(min_length=1, max_length=128)
    client_sdk: str = Field(min_length=1, max_length=64)
    gateway_api_surfaces: tuple[str, ...] = Field(min_length=1)
    capability: ProviderCapability
    result: ProviderCertificationResult
    evaluated_at: datetime
    evidence: tuple[str, ...] = Field(min_length=1)
    limitation: str | None = Field(default=None, min_length=1, max_length=512)


class ProviderCertificationMatrix(ContractModel):
    """Complete later-provider matrix that cannot omit an unsupported capability."""

    schema_version: int = Field(default=1, frozen=True)
    cells: tuple[ProviderCertificationCell, ...]

    @model_validator(mode="after")
    def _require_complete_unique_matrix(self) -> ProviderCertificationMatrix:
        """Require exactly one result for every tracked provider-capability pair.

        Returns:
            The validated complete matrix.

        Raises:
            ValueError: A provider-capability pair repeats or is absent.
        """
        providers = {
            "anthropic",
            "azure",
            "bedrock",
            "gemini",
            "openai",
            "openai-compatible",
            "openrouter",
            "vertex",
        }
        expected = {
            (provider, capability) for provider in providers for capability in ProviderCapability
        }
        actual = {(cell.provider, cell.capability) for cell in self.cells}
        if len(actual) != len(self.cells):
            raise ValueError("provider certification cells must not repeat")
        if actual != expected:
            raise ValueError("provider certification matrix must label every tracked cell")
        return self

    def identity_sha256(self) -> Sha256:
        """Return the deterministic digest of the dated certification artifact."""
        return sha256_json(self)


_EVALUATED_AT = datetime(2026, 8, 25, tzinfo=UTC)
_CLIENT_SDK = "openai==3.0.0"
_GATEWAY_API_SURFACES = ("chat.completions", "responses")
_PROVIDER_API_SURFACES = {
    "anthropic": "messages SSE",
    "azure": "chat.completions SSE",
    "bedrock": "ConverseStream EventStream",
    "gemini": "streamGenerateContent SSE",
    "openai": "responses SSE",
    "openai-compatible": "chat.completions SSE",
    "openrouter": "chat.completions SSE",
    "vertex": "streamGenerateContent SSE",
}
_GEMINI_EVIDENCE = ("exp/runtime/models/providers/gemini_streaming_test.py",)
_VERTEX_EVIDENCE = (
    "exp/runtime/models/providers/vertex_test.py",
    "exp/runtime/models/providers/gemini_streaming_test.py",
)
_BEDROCK_EVIDENCE = ("exp/runtime/models/providers/bedrock_streaming_test.py",)
_OPENAI_EVIDENCE = (
    "exp/runtime/models/providers/native_test.py",
    "exp/runtime/models/providers/streaming_test.py",
)
_ANTHROPIC_EVIDENCE = (
    "exp/runtime/models/providers/native_test.py",
    "exp/runtime/models/providers/streaming_test.py",
    "exp/runtime/models/providers/streaming_test.py::"
    "test_anthropic_text_stream_emits_text_usage_and_completion",
)
_OPENAI_COMPATIBLE_EVIDENCE = (
    "exp/runtime/models/providers/openai_compatible_test.py",
    "exp/runtime/models/providers/streaming_test.py",
)
_COMPATIBLE_EVIDENCE = (
    "exp/runtime/models/providers/later_provider_streaming_test.py",
    "exp/runtime/models/providers/streaming_test.py",
)


def _cell(
    provider: str,
    capability: ProviderCapability,
    result: ProviderCertificationResult,
    evidence: tuple[str, ...],
    *,
    limitation: str | None = None,
) -> ProviderCertificationCell:
    """Build one consistently dated secret-free matrix cell.

    Args:
        provider: Stable gateway provider identifier.
        capability: Adapter behavior represented by this cell.
        result: Honest evidence level reached by the current tree.
        evidence: Repository paths containing deterministic verification.
        limitation: Optional explicit boundary on the claimed behavior.

    Returns:
        Typed certification cell.
    """
    return ProviderCertificationCell(
        provider=provider,
        provider_api_surface=_PROVIDER_API_SURFACES[provider],
        client_sdk=_CLIENT_SDK,
        gateway_api_surfaces=_GATEWAY_API_SURFACES,
        capability=capability,
        result=result,
        evaluated_at=_EVALUATED_AT,
        evidence=evidence,
        limitation=limitation,
    )


def _native_provider_cells(
    provider: str,
    evidence: tuple[str, ...],
) -> tuple[ProviderCertificationCell, ...]:
    """Return native provider fixture results plus the unrun live cell.

    Args:
        provider: Gemini, Vertex, or Bedrock provider identifier.
        evidence: Provider-specific deterministic fixture path.

    Returns:
        Complete capability cells for the native adapter.
    """
    cells: list[ProviderCertificationCell] = []
    for capability in ProviderCapability:
        if capability is ProviderCapability.CREDENTIAL_GATED_LIVE:
            cells.append(
                _cell(
                    provider,
                    capability,
                    ProviderCertificationResult.NOT_RUN_REQUIRES_CREDENTIALS,
                    evidence,
                    limitation="No provider credential was available in the deterministic lane.",
                )
            )
            continue
        if (
            provider in {"gemini", "vertex"}
            and capability is ProviderCapability.TOOL_ARGUMENT_STREAM
        ):
            cells.append(
                _cell(
                    provider,
                    capability,
                    ProviderCertificationResult.UNSUPPORTED,
                    evidence,
                    limitation=(
                        "Gemini supplies a structured complete function argument object, not "
                        "provider-byte incremental argument fragments."
                    ),
                )
            )
            continue
        cells.append(
            _cell(
                provider,
                capability,
                ProviderCertificationResult.PROVIDER_FIXTURE_PASS,
                evidence,
            )
        )
    return tuple(cells)


def _compatible_provider_cells(provider: str) -> tuple[ProviderCertificationCell, ...]:
    """Return inherited compatible-stream results without claiming a live run.

    Args:
        provider: Azure or OpenRouter provider identifier.

    Returns:
        Complete capability cells for the compatible adapter.
    """
    cells: list[ProviderCertificationCell] = []
    for capability in ProviderCapability:
        if capability is ProviderCapability.CREDENTIAL_GATED_LIVE:
            result = ProviderCertificationResult.NOT_RUN_REQUIRES_CREDENTIALS
            limitation = "No provider credential was available in the deterministic lane."
        else:
            result = ProviderCertificationResult.INHERITED_COMPATIBLE_FIXTURE_PASS
            limitation = "Behavior is inherited from the generic compatible adapter fixtures."
        cells.append(
            _cell(
                provider,
                capability,
                result,
                _COMPATIBLE_EVIDENCE,
                limitation=limitation,
            )
        )
    return tuple(cells)


def _launch_provider_cells(
    provider: str,
    evidence: tuple[str, ...],
) -> tuple[ProviderCertificationCell, ...]:
    """Return direct launch-provider fixture results plus the unrun live cell.

    Args:
        provider: OpenAI, Anthropic, or generic OpenAI-compatible identifier.
        evidence: Deterministic provider-specific fixture paths.

    Returns:
        Complete capability cells for the launch adapter.
    """
    cells: list[ProviderCertificationCell] = []
    for capability in ProviderCapability:
        if capability is ProviderCapability.CREDENTIAL_GATED_LIVE:
            result = ProviderCertificationResult.NOT_RUN_REQUIRES_CREDENTIALS
            limitation = "No provider credential was available in the deterministic lane."
        else:
            result = ProviderCertificationResult.PROVIDER_FIXTURE_PASS
            limitation = None
        cells.append(_cell(provider, capability, result, evidence, limitation=limitation))
    return tuple(cells)


PROVIDER_CERTIFICATION_MATRIX = ProviderCertificationMatrix(
    cells=tuple(
        sorted(
            (
                *_launch_provider_cells("openai", _OPENAI_EVIDENCE),
                *_launch_provider_cells("anthropic", _ANTHROPIC_EVIDENCE),
                *_launch_provider_cells(
                    "openai-compatible",
                    _OPENAI_COMPATIBLE_EVIDENCE,
                ),
                *_native_provider_cells("gemini", _GEMINI_EVIDENCE),
                *_native_provider_cells("vertex", _VERTEX_EVIDENCE),
                *_native_provider_cells("bedrock", _BEDROCK_EVIDENCE),
                *_compatible_provider_cells("azure"),
                *_compatible_provider_cells("openrouter"),
            ),
            key=lambda cell: (cell.provider, cell.capability.value),
        )
    )
)

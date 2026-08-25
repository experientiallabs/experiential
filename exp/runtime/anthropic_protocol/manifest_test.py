"""Tests for the Anthropic Messages compatibility manifest."""

from __future__ import annotations

from exp.runtime.anthropic_protocol.manifest import MESSAGES_MANIFEST
from exp.runtime.gateway.contracts import CompatibilityDisposition, GatewayApiSurface


def test_manifest_binds_the_messages_surface_with_unique_fields() -> None:
    """The manifest is a closed, non-repeating field-decision contract."""
    assert MESSAGES_MANIFEST.surface == GatewayApiSurface.MESSAGES
    paths = [field.field_path for field in MESSAGES_MANIFEST.fields]
    assert len(paths) == len(set(paths))


def test_required_protocol_and_sampling_fields_are_supported() -> None:
    """Core Anthropic fields, including documented top-k, stay installed."""
    decisions = {field.field_path: field.disposition for field in MESSAGES_MANIFEST.fields}
    for path in ("model", "messages", "max_tokens", "system", "stream", "stop_sequences"):
        assert decisions[path] == CompatibilityDisposition.SUPPORTED
    assert decisions["tools"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert decisions["thinking"] == CompatibilityDisposition.UNSUPPORTED
    assert decisions["output_config"] == CompatibilityDisposition.UNSUPPORTED
    assert decisions["top_k"] == CompatibilityDisposition.SUPPORTED

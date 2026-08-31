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
    assert decisions["thinking"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert decisions["output_config"] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED
    assert decisions["top_k"] == CompatibilityDisposition.SUPPORTED


def _decided(new_fields: frozenset[str] | set[str], surface: str) -> str:
    """Render the drift-gate failure text naming exactly what must be decided."""
    listed = ", ".join(sorted(new_fields))
    return (
        f"The installed anthropic SDK carries {surface} fields this gateway has never "
        f"classified: {listed}. Decide each one now: add it to MESSAGES_MANIFEST or the "
        "tool-field decision tables in exp/runtime/anthropic_protocol/manifest.py with a "
        "written rationale. Silent drift here is how paying customers become the first "
        "detector (a live Claude Code session was the first detector for "
        "tools.eager_input_streaming)."
    )


def test_every_official_sdk_top_level_field_is_consciously_classified() -> None:
    """The SDK-surface drift gate: an ``anthropic`` bump may add request fields.

    Every top-level field the official GA and beta Messages request types
    carry must appear in the manifest, in exactly one disposition bucket, so
    a dependency bump turns new Anthropic surface into a loud decision
    instead of a silent customer-facing 400.
    """
    from anthropic.types import message_create_params
    from anthropic.types.beta import message_create_params as beta_message_create_params

    sdk_fields = (
        set(message_create_params.MessageCreateParamsBase.__annotations__)
        | set(beta_message_create_params.MessageCreateParamsBase.__annotations__)
        | {"stream"}
    )
    decided = {field.field_path for field in MESSAGES_MANIFEST.fields}
    undecided = sdk_fields - decided
    assert not undecided, _decided(undecided, "Messages")


def test_every_official_sdk_tool_definition_field_is_consciously_classified() -> None:
    """Custom tool-definition fields are part of the drift gate.

    Claude Code sends tool annotations conditionally (a production session
    sent ``eager_input_streaming`` and was answered with a 400), so every
    field on the official custom tool types must be a conscious decision:
    accepted or rejected by name, never a silent 400.
    """
    from anthropic.types import tool_param
    from anthropic.types.beta import beta_tool_param

    from exp.runtime.anthropic_protocol.manifest import (
        MESSAGES_TOOL_FIELDS_ACCEPTED,
        MESSAGES_TOOL_FIELDS_REJECTED,
    )

    assert not MESSAGES_TOOL_FIELDS_ACCEPTED & MESSAGES_TOOL_FIELDS_REJECTED
    sdk_fields = set(tool_param.ToolParam.__annotations__) | set(
        beta_tool_param.BetaToolParam.__annotations__
    )
    undecided = sdk_fields - MESSAGES_TOOL_FIELDS_ACCEPTED - MESSAGES_TOOL_FIELDS_REJECTED
    assert not undecided, _decided(undecided, "custom tool")

    # The accepted table is executable, not aspirational: the strict wire
    # model must accept exactly the accepted fields.
    from exp.runtime.anthropic_protocol import requests as anthropic_requests

    wire_fields = set(anthropic_requests._Tool.model_fields)
    assert wire_fields == set(MESSAGES_TOOL_FIELDS_ACCEPTED)


def test_route_identity_and_delegation_fields_are_recorded_rejections() -> None:
    """Fallbacks, body-borne betas, and third-party attribution stay rejected."""
    decisions = {field.field_path: field.disposition for field in MESSAGES_MANIFEST.fields}
    for path in ("fallbacks", "fallback_credit_token", "betas", "user_profile_id"):
        assert decisions[path] == CompatibilityDisposition.UNSUPPORTED
    for path in ("cache_control", "inference_geo"):
        assert decisions[path] == CompatibilityDisposition.CONDITIONALLY_SUPPORTED

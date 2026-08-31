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


def test_every_official_sdk_tool_union_type_is_consciously_classified() -> None:
    """The tool-type union is part of the drift gate.

    A production Claude Code WebSearch request 400ed (engine 0.7.10) because
    the tool decode classified custom-tool fields but not the typed
    server-tool union members, so every ``type`` literal on the official GA
    and beta tool unions must be a recorded decision.
    """
    import typing

    from anthropic.types.beta.beta_tool_union_param import BetaToolUnionParam
    from anthropic.types.tool_union_param import ToolUnionParam

    from exp.runtime.anthropic_protocol.manifest import (
        MESSAGES_SERVER_TOOL_TYPES_ACCEPTED,
        MESSAGES_SERVER_TOOL_TYPES_REJECTED,
    )

    def type_literals(union: object) -> set[str]:
        """Collect every ``type`` literal across one tool union."""
        values: set[str] = set()

        def walk(node: object) -> None:
            """Accumulate string literals depth-first across wrappers."""
            if typing.get_origin(node) is typing.Literal:
                values.update(arg for arg in typing.get_args(node) if isinstance(arg, str))
                return
            for argument in typing.get_args(node):
                walk(argument)

        for member in typing.get_args(union):
            walk(typing.get_type_hints(member).get("type"))
        return values

    assert not MESSAGES_SERVER_TOOL_TYPES_ACCEPTED & MESSAGES_SERVER_TOOL_TYPES_REJECTED
    sdk_types = type_literals(ToolUnionParam) | type_literals(BetaToolUnionParam)
    undecided = (
        sdk_types
        - {"custom"}
        - MESSAGES_SERVER_TOOL_TYPES_ACCEPTED
        - MESSAGES_SERVER_TOOL_TYPES_REJECTED
    )
    assert not undecided, _decided(undecided, "tool-union type")


def test_every_official_sdk_content_block_type_is_consciously_classified() -> None:
    """Caller content-block types are part of the drift gate."""
    import typing

    from anthropic.types.content_block_param import ContentBlockParam

    from exp.runtime.anthropic_protocol.manifest import (
        MESSAGES_CONTENT_BLOCKS_REJECTED,
        MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED,
    )

    values: set[str] = set()

    def walk(node: object) -> None:
        """Accumulate string literals depth-first across wrappers."""
        if typing.get_origin(node) is typing.Literal:
            values.update(arg for arg in typing.get_args(node) if isinstance(arg, str))
            return
        for argument in typing.get_args(node):
            walk(argument)

    for member in typing.get_args(ContentBlockParam):
        walk(typing.get_type_hints(member).get("type"))
    translated = {"text", "tool_use", "tool_result", "thinking", "redacted_thinking"}
    undecided = (
        values
        - translated
        - MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED
        - MESSAGES_CONTENT_BLOCKS_REJECTED
    )
    assert not undecided, _decided(undecided, "content-block type")
    assert not MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED & MESSAGES_CONTENT_BLOCKS_REJECTED


def test_server_tool_block_tables_match_the_wire_and_native_sets() -> None:
    """The accepted history-block set is executable in both engines.

    The strict wire literal and the native normalizer's opaque-carry set
    must both equal the manifest table, so every block the gateway can emit
    is a block it accepts back and vice versa.
    """
    import re
    import typing
    from pathlib import Path

    from exp.runtime.anthropic_protocol import requests as anthropic_requests
    from exp.runtime.anthropic_protocol.manifest import (
        MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED,
    )

    wire_literal = set(
        typing.get_args(typing.get_type_hints(anthropic_requests._ServerToolHistoryBlock)["type"])
    )
    assert wire_literal == set(MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED)

    dialects_source = (
        Path(__file__).resolve().parents[1] / "gateway" / "native" / "src" / "dialects.rs"
    ).read_text(encoding="utf-8")
    constant = re.search(
        r"ANTHROPIC_SERVER_TOOL_BLOCK_TYPES: &\[&str\] = &\[(.*?)\];",
        dialects_source,
        re.DOTALL,
    )
    assert constant is not None
    native_set = set(re.findall(r'"([a-z_]+)"', constant.group(1)))
    assert native_set == set(MESSAGES_SERVER_TOOL_RESULT_BLOCKS_ACCEPTED)

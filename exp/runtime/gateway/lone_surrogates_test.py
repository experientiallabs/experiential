"""Lone surrogate repair and refusal over decoded request JSON."""

from __future__ import annotations

import json
import time
import tracemalloc
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject, JsonValue
from exp.runtime.gateway.lone_surrogates import (
    MAX_DISCLOSED_PATH_CHARS,
    MAX_DISCLOSED_PATHS,
    SURROGATE_ESCAPE,
    LoneSurrogateRefusal,
    has_lone_surrogate,
    refuse_lone_surrogates,
    repaired_escapes,
    repaired_text,
    replace_lone_surrogates,
)
from exp.runtime.models.providers.json_prefix import incomplete_json_object_prefix


def _replace(payload: JsonObject) -> tuple[str, ...]:
    """Repair ``payload`` with the decoder's own cut-off object predicate."""
    return replace_lone_surrogates(payload, object_prefix=incomplete_json_object_prefix)


def _all_strings(value: JsonValue) -> list[str]:
    """Every string and key in ``value``, plus those of any JSON text in an argument."""
    found: list[str] = []
    pending: list[JsonValue] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            found.append(item)
            try:
                pending.append(json.loads(item))
            except json.JSONDecodeError:
                pass
        elif isinstance(item, dict):
            found.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return found


@pytest.mark.parametrize(
    ("text", "expected", "changed"),
    [
        ("cut \ud83d", "cut \ufffd", True),
        ("\ude00 cut", "\ufffd cut", True),
        ("\ude00\ud83d", "\ufffd\ufffd", True),
        ("\U0001f600 whole", "\U0001f600 whole", False),
        ("ascii", "ascii", False),
    ],
)
def test_repaired_text_replaces_each_lone_surrogate_once(
    text: str, expected: str, changed: bool
) -> None:
    """One U+FFFD per lone surrogate; a scalar and ASCII are untouched."""
    assert repaired_text(text) == (expected, changed)
    assert has_lone_surrogate(text) is changed


@pytest.mark.parametrize(
    ("json_text", "expected"),
    [
        ('{"q": "cut \\ud83d"}', '{"q": "cut \\ufffd"}'),
        ('{"q": "\\ude00 cut"}', '{"q": "\\ufffd cut"}'),
        ('{"q": "\\ud83d\\ude00"}', '{"q": "\\ud83d\\ude00"}'),
        ('{"q": "\\ude00\\ud83d"}', '{"q": "\\ufffd\\ufffd"}'),
        ('{"q": "\\ud83d"}', '{"q": "\\ufffd"}'),
        ('{"q": "\\\\ud83d"}', '{"q": "\\\\ud83d"}'),
        ('{"q": "\\\\\\ud83d"}', '{"q": "\\\\\\ufffd"}'),
        ('{ "q" :  "\\ud83dx" ,"r":1}', '{ "q" :  "\\ufffdx" ,"r":1}'),
    ],
)
def test_repaired_escapes_rewrites_only_unpaired_escapes_and_keeps_every_other_byte(
    json_text: str, expected: str
) -> None:
    """An escaped backslash is not an escape; a pair is kept; spacing survives."""
    repaired, changed = repaired_escapes(json_text)
    assert repaired == expected
    assert changed is (repaired != json_text)
    assert not any(has_lone_surrogate(text) for text in _all_strings(json.loads(repaired)))


def test_the_gate_matches_both_decoded_and_nested_escapes() -> None:
    """One raw-body search finds a lone escape at either JSON level."""
    assert SURROGATE_ESCAPE.search('{"a":"\\uD83D"}')
    assert SURROGATE_ESCAPE.search(json.dumps({"arguments": json.dumps({"q": "\ud83d"})}))
    assert not SURROGATE_ESCAPE.search('{"a":"\\u00e9 \\uE000"}')


def _payload() -> JsonObject:
    """Return a minimal Chat body the surrogate tests mutate."""
    return {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys \ud83d"},
            {"role": "user", "name": "Jo \ud83d", "content": [{"type": "text", "text": "a\ude00"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": json.dumps({"q": "x\ud83d"})},
                    }
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {"id": {"title": "\ud83d"}}},
                },
            }
        ],
        "metadata": {"name": "label \ud83d"},
    }


def test_content_is_repaired_in_place_with_one_disclosure_per_field_path() -> None:
    """List positions collapse, caller-keyed objects stop the path, and nothing is left."""
    payload = _payload()
    assert _replace(payload) == (
        "messages.content->replaced(lone_surrogate)",
        "messages.content.text->replaced(lone_surrogate)",
        "messages.name->replaced(lone_surrogate)",
        "messages.tool_calls.function.arguments->replaced(lone_surrogate)",
        "metadata->replaced(lone_surrogate)",
        "tools.function.parameters->replaced(lone_surrogate)",
    )
    assert not any(has_lone_surrogate(text) for text in _all_strings(payload))
    messages = cast(list[JsonObject], payload["messages"])
    assert messages[0] == {"role": "system", "content": "sys \ufffd"}
    calls = cast(list[JsonObject], messages[2]["tool_calls"])
    function = cast(JsonObject, calls[0]["function"])
    assert json.loads(cast(str, function["arguments"])) == {"q": "x\ufffd"}


def test_a_body_without_lone_surrogates_is_untouched_and_undisclosed() -> None:
    """Nothing to replace: the payload is byte-identical and nothing is disclosed."""
    payload: JsonObject = {"model": "m", "input": "\U0001f600", "metadata": {"k": "v"}}
    snapshot = json.dumps(payload)
    assert _replace(payload) == ()
    assert json.dumps(payload) == snapshot


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"model": "m\ud83d"}, "model"),
        ({"model": "m", "tools": [{"function": {"name": "f\ud83d"}}]}, "tools.0.function.name"),
        (
            {"model": "m", "messages": [{"role": "tool", "tool_call_id": "c\ud83d"}]},
            "messages.0.tool_call_id",
        ),
        (
            {"model": "m", "messages": [{"role": "tool", "name": "f\ud83d", "content": "r"}]},
            "messages.0.name",
        ),
        (
            {"model": "m", "input": [{"type": "function_call", "call_id": "c\udc00"}]},
            "input.0.call_id",
        ),
        (
            {
                "model": "m",
                "messages": [{"content": [{"type": "thinking", "signature": "\ud83d"}]}],
            },
            "messages.0.content.0.signature",
        ),
    ],
)
def test_identifiers_are_refused_by_field(payload: JsonObject, param: str) -> None:
    """A name is a typed refusal that names its field and never echoes its text."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param
    assert param in refused.value.message
    assert not has_lone_surrogate(refused.value.message)


def test_names_inside_caller_keyed_objects_are_content() -> None:
    """A tool call's ``input`` holds the caller's arguments, so an ``id`` there is text."""
    payload: JsonObject = {
        "messages": [{"content": [{"type": "tool_use", "id": "t", "input": {"id": "\ud83d"}}]}]
    }
    assert _replace(payload) == ("messages.content.input->replaced(lone_surrogate)",)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"\ud83d": 1}, "body"),
        ({"messages": [{"role": "user", "\ud83d": 1}]}, "messages.0"),
        ({"tools": [{"parameters": {"properties": {"q\ud83d": {}}}}]}, "tools.0.parameters"),
    ],
)
def test_object_keys_are_refused_without_echoing_caller_keys(
    payload: JsonObject, param: str
) -> None:
    """Keys are names; the refusal stops at the caller-keyed object that holds them."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param
    assert "properties" not in refused.value.message


def test_deep_nesting_is_walked_without_recursion() -> None:
    """A body nested far past the recursion limit is repaired, not a RecursionError."""
    deep: JsonValue = "cut \ud83d"
    for _ in range(5_000):
        deep = [deep]
    payload: JsonObject = {"input": deep}
    assert _replace(payload) == ("input->replaced(lone_surrogate)",)


def test_refusing_surfaces_name_the_first_lone_surrogate() -> None:
    """Surfaces with no disclosure channel refuse content and keys alike."""
    refuse_lone_surrogates({"model": "e", "input": ["ok"]})
    with pytest.raises(LoneSurrogateRefusal) as value:
        refuse_lone_surrogates({"model": "e", "input": ["ok", "cut \ud83d"]})
    assert value.value.param == "input.1"
    with pytest.raises(LoneSurrogateRefusal) as key:
        refuse_lone_surrogates({"model": "e", "metadata": {"\ud83d": "v"}})
    assert key.value.param == "metadata"


@pytest.mark.parametrize(
    "arguments",
    ['{"city\\ud83d": "Paris"}', '{"a": [{"b\\udc00": 1}]}', '{"city\ud83d": "Paris"}'],
)
def test_lone_surrogate_keys_inside_tool_arguments_are_refused(arguments: str) -> None:
    """A key in a tool call's JSON arguments names a parameter, so it is not rewritten."""
    payload: JsonObject = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "f", "arguments": arguments}}],
            }
        ]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "messages.0.tool_calls.0.function.arguments"
    assert "city" not in refused.value.message
    assert not has_lone_surrogate(refused.value.message)


def test_truncated_tool_arguments_are_repaired_textually() -> None:
    """Arguments that do not parse have no keys to judge; their escapes are content."""
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": '{"q": "x\\ud83d'}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    assert cast(list[JsonObject], payload["input"])[0]["arguments"] == '{"q": "x\\ufffd'


def test_a_wide_body_is_walked_without_materializing_every_member() -> None:
    """One lone escape in a wide body costs no per-member bookkeeping."""
    wide: list[JsonValue] = [{"k": index} for index in range(100_000)]
    payload: JsonObject = {"input": [*wide, "cut \ud83d"]}
    tracemalloc.start()
    try:
        assert _replace(payload) == ("input->replaced(lone_surrogate)",)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000


def test_a_tool_namespace_is_refused_as_a_name() -> None:
    """A Responses call's ``namespace`` must round-trip, so it is never rewritten."""
    payload: JsonObject = {
        "input": [{"type": "function_call", "namespace": "ns\ud83d", "arguments": "{}"}]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.namespace"


def test_double_encoded_arguments_are_repaired_at_the_inner_layer() -> None:
    """A JSON string encoding the arguments object hides its escape one layer deeper."""
    arguments = json.dumps(json.dumps({"q": "x\ud83d"}))
    assert "\\\\ud83d" in arguments
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(list[JsonObject], payload["input"])[0]["arguments"]
    assert json.loads(json.loads(cast(str, repaired))) == {"q": "x�"}


def test_double_encoded_argument_keys_are_refused() -> None:
    """A lone surrogate in a key of double-encoded arguments still names a parameter."""
    arguments = json.dumps(json.dumps({"city\ud83d": "Paris"}))
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.arguments"


def test_paired_and_literal_escapes_in_double_encoded_arguments_are_kept() -> None:
    """A paired emoji one layer down is valid, so the arguments are untouched."""
    arguments = json.dumps(json.dumps({"q": "\U0001f600"}))
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ()
    assert cast(list[JsonObject], payload["input"])[0]["arguments"] == arguments


def test_escape_repair_does_not_collect_every_escape() -> None:
    """Many ordinary escapes beside one lone escape are scanned, not retained."""
    json_text = '{"q": "' + "\\n" * 200_000 + '\\ud83d"}'
    tracemalloc.start()
    try:
        repaired, changed = repaired_escapes(json_text)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert changed
    assert repaired.endswith('\\ufffd"}')
    assert peak < 3 * len(json_text)


def test_an_arguments_property_inside_caller_keyed_input_is_plain_content() -> None:
    """Only a tool call's own ``arguments`` is JSON text; a caller's property is not."""
    payload: JsonObject = {
        "messages": [{"content": [{"type": "tool_use", "input": {"arguments": "\\ud83d"}}]}]
    }
    assert _replace(payload) == ()
    content = cast(list[JsonObject], cast(list[JsonObject], payload["messages"])[0]["content"])
    assert content[0]["input"] == {"arguments": "\\ud83d"}


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {"input": [{"type": "function_call", "caller": {"id": "p", "x": "\ud83d"}}]},
            "input.0.caller",
        ),
        ({"user": "u\ud83d"}, "user"),
        ({"prompt_cache_key": "k\ud83d"}, "prompt_cache_key"),
        ({"safety_identifier": "s\ud83d"}, "safety_identifier"),
        ({"input": [{"type": "input_file", "file_id": "f\ud83d"}]}, "input.0.file_id"),
        ({"metadata": {"user_id": "u\ud83d"}}, "metadata.user_id"),
    ],
)
def test_attribution_and_handles_are_refused_as_names(payload: JsonObject, param: str) -> None:
    """Opaque caller attribution, end-user ids, cache keys and file handles are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_other_metadata_values_stay_content() -> None:
    """Only the listed identifier inside ``metadata`` is a name."""
    payload: JsonObject = {"metadata": {"note": "n\ud83d"}}
    assert _replace(payload) == ("metadata->replaced(lone_surrogate)",)


def test_disclosures_are_bounded_by_the_body() -> None:
    """Past the bound, repairs are disclosed by their top-level field only."""
    many: JsonObject = {f"k{index}": {"deep": "\ud83d"} for index in range(MAX_DISCLOSED_PATHS * 2)}
    disclosures = _replace(many)
    assert len(disclosures) == MAX_DISCLOSED_PATHS + MAX_DISCLOSED_PATHS
    assert sum(".deep" in item for item in disclosures) == MAX_DISCLOSED_PATHS


def test_a_truncated_argument_key_is_refused_before_recovery() -> None:
    """A cut-off object the decoder later closes still names its parameters."""
    payload: JsonObject = {
        "input": [{"type": "function_call", "arguments": '{"city\\ud83d": "Paris"'}]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.arguments"


def test_argument_values_that_look_like_keys_stay_content() -> None:
    """A colon inside a string value does not make the value a key."""
    payload: JsonObject = {
        "input": [{"type": "function_call", "arguments": '{"q": "a\\ud83d:", "r": "b:"}'}]
    }
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"promptCacheKey": "k\ud83d"}, "promptCacheKey"),
        ({"tools": [{"type": "mcp", "server_label": "s\ud83d"}]}, "tools.0.server_label"),
        ({"context_management": {"edits": [{"x": "\ud83d"}]}}, "context_management"),
        ({"diagnostics": {"previous_message_id": "m\ud83d"}}, "diagnostics"),
        ({"safeguards": {"mode": "\ud83d"}}, "safeguards"),
    ],
)
def test_aliases_labels_and_verbatim_objects_are_refused(payload: JsonObject, param: str) -> None:
    """Cache-key aliases, server labels and verbatim provider objects are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def _timed_repair(depth: int, leaves: int) -> float:
    """Seconds to repair ``leaves`` one-leaf lists nested under ``depth`` lists."""
    inner: list[JsonValue] = [["\ud800"] for _ in range(leaves)]
    body: JsonValue = inner
    for _ in range(depth):
        body = [body]
    payload: JsonObject = {"input": body}
    started = time.perf_counter()
    _replace(payload)
    return time.perf_counter() - started


def test_repair_cost_does_not_grow_with_nesting_depth() -> None:
    """Classifying a repair reads its container's frame, never its full path."""
    shallow = _timed_repair(depth=1, leaves=20_000)
    deep = _timed_repair(depth=400, leaves=20_000)
    assert deep < shallow * 5 + 0.05


@pytest.mark.parametrize(
    "arguments",
    ['not-json "field\\ud83d": 1', '{"q": 1, "k\\ud83d":', '["k\\ud83d", {"a": 1}]'],
)
def test_arguments_the_decoder_serves_verbatim_have_no_keys(arguments: str) -> None:
    """Invalid text, a key whose value never started, and a non-object are content."""
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)


def test_object_valued_arguments_hold_caller_content() -> None:
    """Arguments sent as an object are the caller's own keys and values."""
    payload: JsonObject = {
        "input": [{"type": "function_call", "name": "f", "arguments": {"name": "n\ud83d"}}]
    }
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {"messages": [{"content": [{"type": "tool_reference", "tool_name": "t\ud83d"}]}]},
            "messages.0.content.0.tool_name",
        ),
        ({"output_config": {"effort": "\ud83d"}}, "output_config"),
        (
            {"input": [{"type": "input_image", "image_url": "https://x/\ud83d"}]},
            "input.0.image_url",
        ),
        ({"provider": {"order": ["p\ud83d"]}}, "provider"),
    ],
)
def test_handles_and_routing_names_are_refused(payload: JsonObject, param: str) -> None:
    """Deferred tool names, verbatim controls, media URLs and routing labels name things."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"client_metadata": {"thread_id": "t\ud83d"}}, "client_metadata"),
        (
            {"messages": [{"content": [{"citations": [{"encrypted_index": "\ud83d"}]}]}]},
            "messages.0.content.0.citations",
        ),
        ({"tools": [{"name": "t", "allowed_callers": ["c\ud83d"]}]}, "tools.0.allowed_callers.0"),
    ],
)
def test_telemetry_citations_and_caller_allowlists_are_refused(
    payload: JsonObject, param: str
) -> None:
    """Client telemetry, provider citation indexes and caller allowlists round-trip verbatim."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_tool_input_examples_hold_caller_content() -> None:
    """Example tool inputs are the caller's own keys and values."""
    payload: JsonObject = {"tools": [{"name": "t", "input_examples": [{"name": "n\ud83d"}]}]}
    assert _replace(payload) == ("tools.input_examples->replaced(lone_surrogate)",)


def test_a_unicode_spelled_backslash_in_double_encoded_arguments_is_repaired() -> None:
    """``\\u005c`` one layer down spells the backslash of the innermost escape."""
    arguments = json.dumps(json.dumps({"q": "x\ud83d"})).replace("\\\\", "\\u005c")
    assert "\\\\ud83d" not in arguments
    assert json.loads(json.loads(arguments)) == {"q": "x\ud83d"}
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)


def test_keys_of_arguments_too_deep_to_parse_are_still_refused() -> None:
    """Valid arguments past the parser's nesting limit are judged on their key tokens."""
    arguments = '{"k\\ud83d": ' + "[" * 100_000 + "]" * 100_000 + "}"
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    with pytest.raises(LoneSurrogateRefusal):
        _replace(payload)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"tools": [{"type": "mcp", "server_url": "https://x/\ud83d"}]}, "tools.0.server_url"),
        ({"inference_geo": "us\ud83d"}, "inference_geo"),
    ],
)
def test_endpoints_and_regions_are_refused(payload: JsonObject, param: str) -> None:
    """An MCP server address and a processing region select something; never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {"input": [{"type": "function_call", "caller": {"metadata": {"t": "\ud83d"}}}]},
            "input.0.caller",
        ),
        (
            {"web_search_options": {"include_domains": ["a\ud83d.com"]}},
            "web_search_options.include_domains.0",
        ),
        (
            {"tools": [{"type": "web_search", "allowed_domains": ["\ud83d"]}]},
            "tools.0.allowed_domains.0",
        ),
    ],
)
def test_verbatim_objects_and_domain_filters_are_refused(payload: JsonObject, param: str) -> None:
    """A caller-keyed name below a verbatim object does not make it content; domains are names."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_a_caller_object_inside_caller_metadata_is_content() -> None:
    """A caller's own ``caller`` key inside metadata is the caller's content."""
    payload: JsonObject = {"metadata": {"caller": {"x": "\ud83d"}}}
    assert _replace(payload) == ("metadata->replaced(lone_surrogate)",)


def test_an_mcp_approval_request_id_is_refused() -> None:
    """An approval answers one request id; it is never rewritten."""
    payload: JsonObject = {
        "input": [{"type": "mcp_approval_response", "approval_request_id": "a\ud83d"}]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.approval_request_id"


def test_top_level_fallback_disclosures_are_capped_too() -> None:
    """Past twice the bound, every further repair is disclosed as the whole body."""
    many: JsonObject = {f"k{index}": "\ud83d" for index in range(MAX_DISCLOSED_PATHS * 10)}
    disclosures = _replace(many)
    assert len(disclosures) == 2 * MAX_DISCLOSED_PATHS + 1
    assert "body->replaced(lone_surrogate)" in disclosures


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"stop": ["x\ud83d"]}, "stop.0"),
        ({"stop_sequences": ["\ud83d"]}, "stop_sequences.0"),
        (
            {"input": [{"type": "code_interpreter_call", "container_id": "c\ud83d"}]},
            "input.0.container_id",
        ),
        (
            {"tools": [{"type": "web_search", "user_location": {"city": "\ud83d"}}]},
            "tools.0.user_location",
        ),
        ({"tools": [{"type": "mcp", "allowed_tools": ["t\ud83d"]}]}, "tools.0.allowed_tools.0"),
    ],
)
def test_controls_containers_locations_and_allowlists_are_refused(
    payload: JsonObject, param: str
) -> None:
    """Stop controls, container handles, search locations and MCP allowlists are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_a_vector_store_id_is_refused() -> None:
    """A native file search names its vector stores; they are never rewritten."""
    payload: JsonObject = {"tools": [{"type": "file_search", "vector_store_ids": ["v\ud83d"]}]}
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "tools.0.vector_store_ids.0"


def test_an_oversized_field_name_is_disclosed_as_the_body() -> None:
    """A disclosure never repeats a long caller key; past the bound it names ``body``."""
    wide_key = "k" * (MAX_DISCLOSED_PATH_CHARS * 100)
    payload: JsonObject = {wide_key: {f"c{index}": "\ud83d" for index in range(64)}}
    assert _replace(payload) == ("body->replaced(lone_surrogate)",)


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {"tools": [{"type": "code_interpreter", "container": {"file_ids": ["f\ud83d"]}}]},
            "tools.0.container.file_ids.0",
        ),
        ({"tools": [{"type": "mcp", "tunnel_id": "t\ud83d"}]}, "tools.0.tunnel_id"),
    ],
)
def test_native_tool_resource_ids_are_refused(payload: JsonObject, param: str) -> None:
    """File handles in a native container and an MCP tunnel are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"tools": [{"type": "mcp", "authorization": "Bearer \ud83d"}]}, "tools.0.authorization"),
        (
            {"tools": [{"type": "file_search", "filters": {"key": "k\ud83d", "value": "v"}}]},
            "tools.0.filters",
        ),
    ],
)
def test_mcp_credentials_and_file_search_filters_are_refused(
    payload: JsonObject, param: str
) -> None:
    """A credential and a metadata filter select what the provider reaches; never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_double_encoded_repair_keeps_every_other_outer_byte() -> None:
    """Only the lone inner escape changes; the outer spelling of everything else stays."""
    arguments = '"{\\"q\\": \\"a\\\\/b \\\\ud83d \\\\ud83d\\\\ude00\\" }"'
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(str, cast(list[JsonObject], payload["input"])[0]["arguments"])
    assert repaired == arguments.replace("\\\\ud83d \\\\ud83d", "\\\\ufffd \\\\ud83d", 1)
    assert json.loads(json.loads(repaired)) == {"q": "a/b � \U0001f600"}


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        (
            {"tools": [{"type": "mcp", "headers": {"Authorization": "Bearer \ud83d"}}]},
            "tools.0.headers",
        ),
        (
            {
                "tools": [
                    {
                        "type": "shell",
                        "network_policy": {"domain_secrets": [{"value": "s\ud83d"}]},
                    }
                ]
            },
            "tools.0.network_policy.domain_secrets",
        ),
    ],
)
def test_native_tool_secrets_are_refused(payload: JsonObject, param: str) -> None:
    """Authentication headers and domain secrets are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_a_literal_backslash_u_one_layer_down_is_not_an_escape() -> None:
    """An even run of inner backslashes before ``ud83d`` is literal text and is kept."""
    inner = json.dumps({"lit": "\\ud83d", "cut": "x\ud83d"})
    arguments = json.dumps(inner)
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(str, cast(list[JsonObject], payload["input"])[0]["arguments"])
    assert json.loads(json.loads(repaired)) == {"lit": "\\ud83d", "cut": "x�"}
    assert repaired == arguments.replace("x\\\\ud83d", "x\\\\ufffd")


def test_a_unicode_spelled_inner_backslash_keeps_the_other_outer_bytes() -> None:
    """``\\u005c`` before ``ud83d`` is repaired in place; ``\\u0020`` elsewhere is kept."""
    arguments = '"{\\"q\\":\\"\\u005cud83d\\",\\"keep\\":\\u0020 1}"'
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(str, cast(list[JsonObject], payload["input"])[0]["arguments"])
    assert repaired == arguments.replace("\\u005cud83d", "\\\\ufffd")


def test_a_program_fingerprint_is_refused() -> None:
    """A replayed program's fingerprint round-trips verbatim; it is never rewritten."""
    payload: JsonObject = {"input": [{"type": "program", "fingerprint": "f\ud83d"}]}
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.fingerprint"


def test_a_key_refusal_stops_at_a_verbatim_object() -> None:
    """A lone key deep in a verbatim object names only the object, never its other keys."""
    payload: JsonObject = {"client_metadata": {"tenant/acme": {"\ud83d": 1}}}
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "client_metadata"
    assert "tenant" not in refused.value.message


def test_a_unicode_spelled_inner_u_keeps_the_other_outer_bytes() -> None:
    """``\\u0075`` spelling the ``u`` of an inner escape is repaired in place, pairs kept."""
    arguments = (
        '"{\\"q\\":\\"\\\\\\u0075d83d \\\\\\u0075d83d\\\\\\u0075de00\\",\\"keep\\":\\u0020 1}"'
    )
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(str, cast(list[JsonObject], payload["input"])[0]["arguments"])
    assert repaired == arguments.replace("\\\\\\u0075d83d ", "\\\\ufffd ", 1)
    assert json.loads(json.loads(repaired)) == {"q": "� \U0001f600", "keep": 1}


def test_provider_bookkeeping_fields_are_caller_content() -> None:
    """``provider_specific_fields`` is dropped bookkeeping, so its strings are content."""
    payload: JsonObject = {
        "messages": [{"role": "assistant", "provider_specific_fields": {"name": "x\ud83d"}}]
    }
    assert _replace(payload) == ("messages.provider_specific_fields->replaced(lone_surrogate)",)


def test_a_key_refusal_inside_caller_content_never_names_caller_keys() -> None:
    """A verbatim-looking key inside caller content does not extend the refusal path."""
    payload: JsonObject = {"metadata": {"caller": {"\ud83d": 1}}}
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "metadata"


def test_unicode_spelled_inner_hex_digits_keep_the_other_outer_bytes() -> None:
    """Inner escape digits spelled as outer escapes are still repaired in place."""
    arguments = '"{\\"q\\":\\"\\\\u\\u006483d\\",\\"keep\\":\\u0020 1}"'
    payload: JsonObject = {"input": [{"type": "function_call", "arguments": arguments}]}
    assert _replace(payload) == ("input.arguments->replaced(lone_surrogate)",)
    repaired = cast(str, cast(list[JsonObject], payload["input"])[0]["arguments"])
    assert repaired == arguments.replace("\\\\u\\u006483d", "\\\\ufffd")


@pytest.mark.parametrize(
    ("payload", "param"),
    [
        ({"tools": [{"type": "mcp", "connector_id": "c\ud83d"}]}, "tools.0.connector_id"),
        ({"tools": [{"type": "code_interpreter", "container": "c\ud83d"}]}, "tools.0.container"),
    ],
)
def test_connector_and_container_handles_are_refused(payload: JsonObject, param: str) -> None:
    """A connector id and a string container id are never rewritten."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == param


def test_an_apply_patch_path_is_refused() -> None:
    """A replayed patch operation names its file; that path is never rewritten."""
    payload: JsonObject = {
        "input": [
            {"type": "apply_patch_call", "operation": {"type": "create_file", "path": "a\ud83d"}}
        ]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(payload)
    assert refused.value.param == "input.0.operation.path"


@pytest.mark.parametrize(
    ("item", "param"),
    [
        (
            {"type": "local_shell_call", "action": {"working_directory": "/w\ud83d"}},
            "input.0.action.working_directory",
        ),
        ({"type": "image_generation_call", "result": "aGk\ud83d"}, "input.0.result"),
        ({"type": "agent_message", "author": "a\ud83d"}, "input.0.author"),
        ({"type": "agent_message", "recipient": "r\ud83d"}, "input.0.recipient"),
        ({"type": "agent_message", "agent": {"agent_name": "n\ud83d"}}, "input.0.agent.agent_name"),
        (
            {"type": "configuration_update", "reasoning": {"effort": "hi\ud83d"}},
            "input.0.reasoning.effort",
        ),
        (
            {"type": "local_shell_call", "action": {"env": {"TOKEN": "t\ud83d"}}},
            "input.0.action.env",
        ),
        (
            {"type": "computer_call", "action": {"keys": ["ctrl", "k\ud83d"]}},
            "input.0.action.keys.1",
        ),
    ],
)
def test_replayed_item_paths_payloads_and_agents_are_refused(item: JsonObject, param: str) -> None:
    """Shell directories, image results and agent identities in replayed items stay verbatim."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace({"input": [item]})
    assert refused.value.param == param


def test_shell_output_json_is_repaired_at_the_inner_layer() -> None:
    """A local shell output is JSON text: its lone escape is repaired, its keys judged."""
    output = '{"stdout": "x\\ud83d", "code": 0}'
    payload: JsonObject = {"input": [{"type": "local_shell_call_output", "output": output}]}
    assert _replace(payload) == ("input.output->replaced(lone_surrogate)",)
    repaired = cast(list[JsonObject], payload["input"])[0]["output"]
    assert repaired == output.replace("\\ud83d", "\\ufffd")
    with pytest.raises(LoneSurrogateRefusal):
        _replace({"input": [{"type": "local_shell_call_output", "output": '{"k\\ud83d": 1}'}]})


def test_an_output_field_elsewhere_is_plain_content() -> None:
    """Only a local shell output is JSON text; other outputs keep literal escape text."""
    payload: JsonObject = {"input": [{"type": "function_call_output", "output": "\\ud83d"}]}
    assert _replace(payload) == ()


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({"speed": "fast\ud83d"}, "speed"),
        ({"thinking": {"type": "enabled", "display": "sum\ud83d"}}, "thinking.display"),
    ],
)
def test_messages_pass_through_controls_are_refused(body: JsonObject, param: str) -> None:
    """Evolving provider selectors forwarded verbatim are refused, never repaired."""
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(body)
    assert refused.value.param == param


def test_a_shell_skill_id_is_refused() -> None:
    """A stored skill reference on a shell tool declaration is a resource selector."""
    body: JsonObject = {
        "tools": [{"type": "shell", "environment": {"skills": [{"skill_id": "s\ud83d"}]}}]
    }
    with pytest.raises(LoneSurrogateRefusal) as refused:
        _replace(body)
    assert refused.value.param == "tools.0.environment.skills.0.skill_id"


def test_a_namespace_output_schema_is_caller_keyed_schema_content() -> None:
    """Keys below ``output_schema`` are the caller's schema: repaired, never refused."""
    body: JsonObject = {
        "tools": [{"type": "namespace", "output_schema": {"$defs": {"id": {"title": "x\ud83d"}}}}]
    }
    assert _replace(body) == ("tools.output_schema->replaced(lone_surrogate)",)

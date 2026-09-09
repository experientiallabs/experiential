"""Calibration and behaviour tests for the per-attempt input-token estimate."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import tiktoken

from exp.common.core.artifacts import JsonObject, canonical_json_bytes
from exp.common.models.content import (
    AudioContentPart,
    DocumentContentPart,
    ImageContentPart,
    TextContentPart,
    VideoContentPart,
)
from exp.common.models.model import ToolCall
from exp.runtime.gateway.attempt_tokens import (
    AUDIO_BYTES_PER_TOKEN,
    DOCUMENT_BYTES_PER_TOKEN,
    DOCUMENT_TOKENS,
    IMAGE_TOKENS,
    INPUT_TOKEN_HEADROOM_PERCENT,
    MESSAGE_FRAMING_TOKENS,
    OPAQUE_BYTES_PER_TOKEN,
    TOOLS_PRESENT_TOKENS,
    VIDEO_BYTES_PER_TOKEN,
    VIDEO_TOKENS,
    worst_case_input_tokens,
)
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
    StructuredTextFormat,
)
from exp.runtime.gateway.embeddings_contracts import EmbeddingsRequest
from exp.runtime.gateway.images_contracts import ImagesRequest
from exp.runtime.gateway.reasoning_blocks import EncryptedReasoningBlock
from exp.runtime.gateway.replay_identity import provider_replay_authority
from exp.runtime.gateway.reservation_tokenizer import RESERVATION_ENCODING, reservation_encoder

GATEWAY_DIR = Path(__file__).resolve().parent

# The headroom the estimate is designed to carry over a provider's own count.
# The lower edge is the tokenizer-drift allowance; the upper edge keeps the
# reservation from throttling free-tier callers the way the byte bound did.
CALIBRATION_LOW = 1.05
CALIBRATION_HIGH = 1.40


def _tool(
    name: str, description: str, properties: JsonObject, required: list[str]
) -> GatewayToolDefinition:
    """Build one strict-object tool definition."""
    return GatewayToolDefinition(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


def _coding_agent_tools() -> tuple[GatewayToolDefinition, ...]:
    """Fifteen tool schemas shaped like a coding agent's toolbox."""
    string = {"type": "string"}
    integer = {"type": "integer"}
    boolean = {"type": "boolean"}
    return (
        _tool(
            "read_file",
            "Read a file from the local filesystem. Returns the contents with line numbers; "
            "use offset and limit for large files.",
            {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line to start from"},
                "limit": {"type": "integer", "description": "Maximum lines to return"},
            },
            ["file_path"],
        ),
        _tool(
            "write_file",
            "Write content to a file, creating it if needed and overwriting otherwise.",
            {"file_path": string, "content": {"type": "string", "description": "Full content"}},
            ["file_path", "content"],
        ),
        _tool(
            "edit_file",
            "Replace an exact string in a file with another string. The old string must be unique.",
            {
                "file_path": string,
                "old_string": string,
                "new_string": string,
                "replace_all": {"type": "boolean", "default": False},
            },
            ["file_path", "old_string", "new_string"],
        ),
        _tool(
            "bash",
            "Run a shell command and return stdout and stderr. Commands time out after the given "
            "number of milliseconds.",
            {
                "command": {"type": "string", "description": "The command to run"},
                "timeout": {"type": "integer", "minimum": 0, "maximum": 600_000},
                "description": {"type": "string", "description": "What the command does"},
            },
            ["command"],
        ),
        _tool(
            "grep",
            "Search file contents with a regular expression, with include globs and context.",
            {
                "pattern": string,
                "path": string,
                "glob": string,
                "context": integer,
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                },
            },
            ["pattern"],
        ),
        _tool(
            "glob", "Find files by glob pattern.", {"pattern": string, "path": string}, ["pattern"]
        ),
        _tool(
            "web_search",
            "Search the web and return the top results with snippets and URLs.",
            {
                "query": string,
                "allowed_domains": {"type": "array", "items": string},
                "max_results": {"type": "integer", "default": 5},
            },
            ["query"],
        ),
        _tool(
            "web_fetch",
            "Fetch a URL and convert its content to markdown.",
            {"url": {"type": "string", "format": "uri"}, "prompt": string},
            ["url"],
        ),
        _tool(
            "todo_write",
            "Create or update the task list for the current session.",
            {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": string,
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "id": string,
                        },
                        "required": ["content", "status", "id"],
                    },
                }
            },
            ["todos"],
        ),
        _tool(
            "ask_user",
            "Ask the user a clarifying question and wait for the answer.",
            {"question": string, "options": {"type": "array", "items": string}},
            ["question"],
        ),
        _tool(
            "run_tests",
            "Run the project's test suite, optionally filtered to a path or expression.",
            {"path": string, "expression": string, "verbose": boolean},
            [],
        ),
        _tool(
            "git_diff",
            "Show the working tree diff or the diff between two refs.",
            {"base": string, "head": string, "stat_only": boolean},
            [],
        ),
        _tool(
            "git_commit",
            "Create a commit with the given message from the staged changes.",
            {"message": {"type": "string", "minLength": 1}, "all": boolean},
            ["message"],
        ),
        _tool(
            "list_directory",
            "List the entries of a directory with sizes and types.",
            {
                "path": string,
                "recursive": boolean,
                "max_depth": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            ["path"],
        ),
        _tool(
            "notebook_edit",
            "Replace, insert, or delete a cell in a Jupyter notebook.",
            {
                "notebook_path": string,
                "cell_id": string,
                "new_source": string,
                "cell_type": {"type": "string", "enum": ["code", "markdown"]},
                "edit_mode": {"type": "string", "enum": ["replace", "insert", "delete"]},
            },
            ["notebook_path", "new_source"],
        ),
    )


def _chat(
    messages: tuple[GatewayMessage, ...],
    tools: tuple[GatewayToolDefinition, ...] = (),
    structured_text: StructuredTextFormat | None = None,
) -> GatewayRequest:
    """Build one chat request with a fixed output ceiling."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=messages,
        tools=tools,
        structured_text=structured_text,
        maximum_output_tokens=4_096,
    )


def _read_call(call_id: str, path: str) -> GatewayMessage:
    """One assistant turn calling ``read_file``."""
    return GatewayMessage(
        role="assistant",
        content=None,
        tool_calls=(ToolCall(call_id=call_id, name="read_file", arguments={"file_path": path}),),
    )


def _prose_chat() -> GatewayRequest:
    """A short multi-turn prose conversation."""
    return _chat(
        (
            GatewayMessage(
                role="system", content="You are a helpful assistant that answers concisely."
            ),
            GatewayMessage(
                role="user",
                content=(
                    "Can you explain, in a few paragraphs, why continual learning systems suffer "
                    "from catastrophic forgetting and what distillation-based approaches do "
                    "about it?"
                ),
            ),
            GatewayMessage(
                role="assistant",
                content=(
                    "Catastrophic forgetting happens because gradient updates for a new task "
                    "overwrite the parameters that encoded earlier tasks. "
                )
                * 6,
            ),
            GatewayMessage(
                role="user",
                content=(
                    "Great. Now compare that with replay buffers and parameter isolation, and say "
                    "which one you would pick for an on-device assistant."
                ),
            ),
        )
    )


def _coding_agent_request() -> GatewayRequest:
    """A tool-heavy coding-agent turn: fifteen tools and several whole-file tool results.

    The tool results are this package's own sources, so the fixture is a
    realistic mix of code, docstrings, and JSON-ish text at roughly fifty
    thousand tokens.
    """
    sources = [
        (GATEWAY_DIR / name).read_text()
        for name in (
            "budgets.py",
            "native_accounting.py",
            "contracts.py",
            "native_bridge.py",
            "replay_identity.py",
            "budgets_test.py",
        )
    ]
    messages: list[GatewayMessage] = [
        GatewayMessage(
            role="system",
            content=(
                "You are a careful coding agent working inside a git repository. Prefer reading "
                "files before editing them, keep edits minimal and reversible, run the tests after "
                "every change, and explain what you did in plain language."
            ),
        ),
        GatewayMessage(
            role="user",
            content="Please refactor the budget reservation code so the tier check is explicit.",
        ),
    ]
    for index, source in enumerate(sources):
        call_id = f"call_{index}"
        messages.append(_read_call(call_id, f"/repo/exp/runtime/gateway/file_{index}.py"))
        messages.append(GatewayMessage(role="tool", tool_call_id=call_id, content=source))
    messages.append(GatewayMessage(role="user", content="Looks good, go ahead."))
    return _chat(tuple(messages), _coding_agent_tools())


def _json_heavy_request() -> GatewayRequest:
    """Structured extraction over a large JSON record set with a strict output schema."""
    records = [
        {
            "id": index,
            "name": f"customer {index}",
            "email": f"user{index}@example.com",
            "plan": "pro" if index % 3 else "free",
            "spend_usd": round(index * 1.37, 2),
            "tags": ["alpha", "beta"][index % 2 :],
        }
        for index in range(400)
    ]
    return _chat(
        (
            GatewayMessage(
                role="system", content="Extract structured data from the user's records."
            ),
            GatewayMessage(role="user", content=json.dumps(records, indent=2)),
        ),
        structured_text=StructuredTextFormat(
            name="summary",
            json_schema={
                "type": "object",
                "properties": {
                    "total_spend": {"type": "number"},
                    "plans": {"type": "object", "additionalProperties": {"type": "integer"}},
                    "top_customers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}, "name": {"type": "string"}},
                            "required": ["id", "name"],
                        },
                    },
                },
                "required": ["total_spend", "plans", "top_customers"],
                "additionalProperties": False,
            },
        ),
    )


def _multilingual_request() -> GatewayRequest:
    """Chinese, Russian, and Japanese prose in one message."""
    return _chat(
        (
            GatewayMessage(
                role="user",
                content=(
                    "深度学习模型在持续学习任务中面临灾难性遗忘问题，我们提出了一种基于蒸馏的方法来缓解这一问题。"
                    * 30
                    + "Модель непрерывного обучения должна усваивать новые навыки, "
                    "сохраняя старые. " * 30 + "これは日本語のテキストです。継続学習において、"
                    "モデルは新しい知識を獲得しながら古い知識を保持する必要があります。" * 30
                ),
            ),
        )
    )


def _provider_count(request: GatewayRequest) -> int:
    """Independently render the prompt the way a chat provider counts it.

    Follows OpenAI's published chat accounting (three framing tokens per
    message plus three priming the reply), tool calls as ``name(arguments)``,
    tool definitions and output schemas as compact JSON, all tokenized with
    the same published BPE but without any of the estimator's chunking,
    framing constants, or headroom.
    """
    encoder = tiktoken.get_encoding(RESERVATION_ENCODING)
    total = 3
    for message in request.messages:
        total += 3
        text = (message.content or "") + "".join(
            f"{call.name}({json.dumps(call.arguments)})" for call in message.tool_calls
        )
        total += len(encoder.encode(text, disallowed_special=()))
    for tool in request.tools:
        rendered = json.dumps(
            {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
        )
        total += len(encoder.encode(rendered))
    if request.structured_text is not None:
        total += len(encoder.encode(json.dumps(request.structured_text.json_schema)))
    return total


@pytest.mark.parametrize(
    "build",
    [_prose_chat, _coding_agent_request, _json_heavy_request, _multilingual_request],
    ids=["prose_chat", "coding_agent_15_tools", "json_heavy", "multilingual"],
)
def test_estimate_is_calibrated_within_headroom_of_a_provider_count(
    build: Callable[[], GatewayRequest],
) -> None:
    """Realistic fixtures reserve between +5% and +40% over the provider's count, never under.

    The byte bound these fixtures replaced reserved three to ten times the
    provider count (production medians ran 4.2x to 6.6x), which is what made
    free-tier hourly windows bind at a fraction of their allowance.
    """
    request = build()
    expected = _provider_count(request)
    estimate = worst_case_input_tokens(request)
    ratio = estimate / expected
    assert estimate >= expected
    assert CALIBRATION_LOW <= ratio <= CALIBRATION_HIGH, (expected, estimate, ratio)
    byte_bound = len(canonical_json_bytes(request))
    assert byte_bound / expected > 3


def test_hello_world_arithmetic_is_framing_plus_headroom() -> None:
    """``Hello, world!`` is four o200k tokens; one message adds framing, then headroom."""
    request = _chat((GatewayMessage(role="user", content="Hello, world!"),))
    counted = 4 + MESSAGE_FRAMING_TOKENS
    assert (
        worst_case_input_tokens(request)
        == (counted * (100 + INPUT_TOKEN_HEADROOM_PERCENT) + 99) // 100
    )


def test_tools_add_their_schemas_and_the_tool_use_preamble() -> None:
    """Declaring tools costs the fixed preamble plus every rendered schema."""
    bare = _chat((GatewayMessage(role="user", content="List the repo."),))
    tooled = _chat(bare.messages, _coding_agent_tools())
    schema_tokens = sum(
        len(reservation_encoder().encode_ordinary(json.dumps(tool.parameters)))
        for tool in tooled.tools
    )
    difference = worst_case_input_tokens(tooled) - worst_case_input_tokens(bare)
    assert difference > TOOLS_PRESENT_TOKENS + schema_tokens
    assert difference < 2 * (TOOLS_PRESENT_TOKENS + schema_tokens)


def test_inline_media_reserve_planning_constants_not_their_base64_length() -> None:
    """A 600 KB PNG reserves about one image, not six hundred thousand tokens."""
    png = base64.b64encode(bytes(600_000)).decode()
    image = _chat(
        (
            GatewayMessage(
                role="user",
                content="What is in this screenshot?",
                content_parts=(
                    TextContentPart(text="What is in this screenshot?"),
                    ImageContentPart(media_type="image/png", data=png),
                ),
            ),
        )
    )
    estimate = worst_case_input_tokens(image)
    assert IMAGE_TOKENS <= estimate < IMAGE_TOKENS * 1.3
    assert estimate < len(png) // 100

    pdf = base64.b64encode(bytes(1_000_000)).decode()
    inline_document = _chat(
        (
            GatewayMessage(
                role="user",
                content="Summarize.",
                content_parts=(
                    TextContentPart(text="Summarize."),
                    DocumentContentPart(data=pdf, name="report.pdf"),
                ),
            ),
        )
    )
    remote_document = _chat(
        (
            GatewayMessage(
                role="user",
                content="Summarize.",
                content_parts=(
                    TextContentPart(text="Summarize."),
                    DocumentContentPart(url="https://example.com/report.pdf"),
                ),
            ),
        )
    )
    inline_estimate = worst_case_input_tokens(inline_document)
    assert 1_000_000 // DOCUMENT_BYTES_PER_TOKEN <= inline_estimate < 1_000_000 // 20
    assert DOCUMENT_TOKENS <= worst_case_input_tokens(remote_document) < DOCUMENT_TOKENS * 1.3

    clip = base64.b64encode(bytes(2_000_000)).decode()
    video = _chat(
        (
            GatewayMessage(
                role="user",
                content="Describe.",
                content_parts=(
                    TextContentPart(text="Describe."),
                    VideoContentPart(media_type="video/mp4", data=clip),
                ),
            ),
        )
    )
    remote_video = _chat(
        (
            GatewayMessage(
                role="user",
                content="Describe.",
                content_parts=(
                    TextContentPart(text="Describe."),
                    VideoContentPart(url="https://example.com/clip.mp4"),
                ),
            ),
        )
    )
    assert 2_000_000 // VIDEO_BYTES_PER_TOKEN <= worst_case_input_tokens(video) < 2_000_000 // 64
    assert VIDEO_TOKENS <= worst_case_input_tokens(remote_video) < VIDEO_TOKENS * 1.3

    audio = _chat(
        (
            GatewayMessage(
                role="user",
                content="Transcribe.",
                content_parts=(
                    TextContentPart(text="Transcribe."),
                    AudioContentPart(media_type="audio/mpeg", data=clip),
                ),
            ),
        )
    )
    assert 2_000_000 // AUDIO_BYTES_PER_TOKEN <= worst_case_input_tokens(audio) < 2_000_000 // 128


def test_opaque_replay_carriers_count_by_decoded_length() -> None:
    """Encrypted reasoning reserves its decoded prose, not its base64 text.

    The carrier is excluded from the plain serialization, so it must still
    reach the estimate through the replay envelope; but tokenizing base64
    would charge several times what the provider counts after decrypting.
    """
    plain = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="Plan the migration in three steps."),
            GatewayMessage(role="assistant", content="Inventory, dual-write, cut over."),
            GatewayMessage(role="user", content="Expand step two."),
        ),
        maximum_output_tokens=4_096,
    )
    carrier_bytes = 6_000
    encrypted = base64.b64encode(bytes(range(256)) * (carrier_bytes // 256)).decode()
    replayed = plain.model_copy(
        update={
            "messages": (
                plain.messages[0],
                plain.messages[1].model_copy(
                    update={
                        "provider_reasoning": (
                            EncryptedReasoningBlock(
                                id="rs_1", encrypted_content=encrypted, output_index=0
                            ),
                        )
                    }
                ),
                plain.messages[2],
            )
        }
    )
    assert provider_replay_authority(plain) is None
    assert provider_replay_authority(replayed) is not None
    added = worst_case_input_tokens(replayed) - worst_case_input_tokens(plain)
    decoded_tokens = len(encrypted) * 3 // 4 // OPAQUE_BYTES_PER_TOKEN
    assert decoded_tokens <= added < decoded_tokens * 1.4
    assert added < len(reservation_encoder().encode_ordinary(encrypted)) // 2


def test_retained_raw_arguments_are_not_counted_twice() -> None:
    """A verbatim provider argument string replaces, never doubles, its parsed twin."""
    arguments: JsonObject = {"file_path": "/repo/README.md", "note": "x " * 500}
    parsed = _chat(
        (
            GatewayMessage(role="user", content="Read it."),
            GatewayMessage(
                role="assistant",
                content=None,
                tool_calls=(ToolCall(call_id="call_1", name="read_file", arguments=arguments),),
            ),
        )
    )
    verbatim = parsed.model_copy(
        update={
            "messages": (
                parsed.messages[0],
                parsed.messages[1].model_copy(
                    update={
                        "tool_calls": (
                            parsed.messages[1]
                            .tool_calls[0]
                            .model_copy(update={"raw_arguments": json.dumps(arguments)}),
                        )
                    }
                ),
            )
        }
    )
    assert provider_replay_authority(verbatim) is not None
    once = worst_case_input_tokens(parsed)
    # The envelope adds its metadata (indices, ids, null slots), never the payload again.
    assert once <= worst_case_input_tokens(verbatim) < once + 150


def test_embeddings_and_image_prompts_count_their_text() -> None:
    """Non-completion surfaces estimate their text inputs with the same headroom."""
    embeddings = EmbeddingsRequest(inputs=("hello world", "the quick brown fox " * 50))
    estimate = worst_case_input_tokens(embeddings)
    counted = len(reservation_encoder().encode_ordinary("\n".join(embeddings.inputs)))
    assert counted <= estimate <= (counted * (100 + INPUT_TOKEN_HEADROOM_PERCENT) + 99) // 100
    assert estimate < len(canonical_json_bytes(embeddings)) // 2

    images = ImagesRequest(prompt="a watercolor cat on a windowsill")
    assert 0 < worst_case_input_tokens(images) < 20


def test_encoder_is_loaded_once_and_the_estimate_stays_cheap() -> None:
    """One cached BPE; a fifty-thousand-token, fifteen-tool request estimates in milliseconds.

    The bound is loose (measured about ten milliseconds on a laptop) because
    it guards against tokenizing twice or re-loading the table per call, not
    against machine speed.
    """
    assert reservation_encoder() is reservation_encoder()
    request = _coding_agent_request()
    worst_case_input_tokens(request)
    started = time.perf_counter()
    for _ in range(5):
        worst_case_input_tokens(request)
    per_call = (time.perf_counter() - started) / 5
    assert per_call < 0.25

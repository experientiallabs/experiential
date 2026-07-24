"""A thin wmh-owned seam over tinker-cookbook chat renderers.

The Tinker provider needs five things from a renderer: turn a structured chat
request into prompt token ids, render only the NEW messages of a growing
conversation as a token suffix (so multi-turn prompts can extend the episode's
own token history instead of re-rendering it), expose the model's stop
sequences, decode token ids for plain-text callers, and parse sampled token
ids back into an assistant message (text plus tool calls). `ChatRendering`
captures that contract; `CookbookChatRendering` implements it on top of a
tinker-cookbook `Renderer` so cookbook churn stays contained in this module.

`build_renderer` resolves the cookbook renderer for a base model the same way
the cookbook's own LiteLLM provider does (`get_recommended_renderer_name`).
tinker-cookbook is an optional extra and is imported lazily
(`uv sync --extra distill`), mirroring the e2b extra's contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, cast

from llm_waterfall.types import ChatFunctionCall, ChatMessage, ChatTool, ChatToolCall
from pydantic import BaseModel, Field, JsonValue

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer
    from tinker_cookbook.renderers.base import Message as RendererMessage
    from tinker_cookbook.renderers.base import RenderedMessage, ToolSpec
    from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)

MISSING_DISTILL_EXTRA = (
    "the tinker-cookbook SDK is not installed; run `uv sync --extra distill` "
    "to use the Tinker distillation provider"
)


class RendererTokenizer(Protocol):
    """The tokenizer slice cookbook renderers rely on.

    HuggingFace `PreTrainedTokenizer` (what `tinker.SamplingClient.get_tokenizer`
    returns) satisfies this structurally; tests supply small deterministic fakes.
    """

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...


class ParsedAssistantMessage(BaseModel):
    """One sampled assistant turn parsed out of student token ids."""

    text: str = ""
    """Assistant text, with any thinking rendered inline as <think>...</think>."""

    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    """Parsed tool calls in OpenAI chat format."""

    stopped: bool
    """True when generation terminated cleanly (stop sequence or EOS), False on truncation."""


class ChatRendering(Protocol):
    """What the Tinker provider needs from a chat renderer."""

    @property
    def stop_sequences(self) -> list[str] | list[int]:
        """Stop strings or stop token ids that end one assistant turn."""
        ...

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        """Render chat messages (and optional tool schemas) into prompt token ids."""
        ...

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        """Render `messages[delta_start:]` plus the generation header as a token suffix.

        The suffix extends an incrementally built prompt (previous prompt ids
        plus the raw sampled ids of the previous assistant turn) so the next
        prompt is a verbatim token-prefix extension of the episode so far.
        When `previous_sampled_ids` does not already end with the template's
        end-of-turn framing (a max_tokens truncation), that framing is
        prepended so the delta messages start on a properly closed turn.
        """
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text."""
        ...

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        """Parse sampled token ids into an assistant message (text plus tool calls)."""
        ...


def _text_content(content: JsonValue) -> str:
    """Flatten llm_waterfall message content (None, string, or part list) to text.

    OpenAI-format content parts (`{"type": "text", "text": ...}`) are joined in
    order; non-text parts are rejected loudly rather than dropped silently.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError(f"unsupported message content part: {part!r}")
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    "the tinker provider is text-only; message content parts must be "
                    f"text parts, got {part.get('type')!r}"
                )
            fragments.append(text)
        return "".join(fragments)
    raise ValueError(f"unsupported message content: {content!r}")


def renderer_messages_from_chat(messages: list[ChatMessage]) -> list[RendererMessage]:
    """Convert llm_waterfall chat messages into cookbook renderer messages.

    Mirrors the cookbook's `openai_messages_to_tinker`: roles pass through,
    content flattens to text, and tool-result linkage (`tool_call_id`, `name`)
    plus assistant `tool_calls` are preserved.

    Raises:
        ImportError: If tinker-cookbook is not installed (distill extra).
        ValueError: If a message carries non-text content parts.
    """
    try:
        from tinker_cookbook.renderers.base import ToolCall
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
        raise ImportError(MISSING_DISTILL_EXTRA) from exc

    out: list[RendererMessage] = []
    for msg in messages:
        renderer_msg: RendererMessage = {
            "role": msg.role,
            "content": _text_content(msg.content),
        }
        if msg.tool_call_id is not None:
            renderer_msg["tool_call_id"] = msg.tool_call_id
        name = (msg.model_extra or {}).get("name")
        if isinstance(name, str):
            renderer_msg["name"] = name
        if msg.tool_calls:
            renderer_msg["tool_calls"] = [
                ToolCall(
                    id=tc.id,
                    function=ToolCall.FunctionBody(
                        name=tc.function.name, arguments=tc.function.arguments
                    ),
                )
                for tc in msg.tool_calls
            ]
        out.append(renderer_msg)
    return out


def tool_specs_from_chat(tools: list[ChatTool]) -> list[ToolSpec]:
    """Convert llm_waterfall tool definitions into cookbook ToolSpec dicts."""
    return [
        {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": dict(tool.function.parameters),
        }
        for tool in tools
    ]


class CookbookChatRendering:
    """`ChatRendering` backed by a tinker-cookbook `Renderer`.

    Construct via `build_renderer`. Tool schemas are injected through the
    renderer's `create_conversation_prefix_with_tools`, folding any leading
    system message into the tool prefix the way the cookbook's LiteLLM
    provider does.
    """

    def __init__(self, renderer: Renderer) -> None:
        self._renderer = renderer

    @property
    def stop_sequences(self) -> list[str] | list[int]:
        """The renderer's stop strings or stop token ids."""
        return self._renderer.get_stop_sequences()

    def _effective_messages(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None
    ) -> tuple[list[RendererMessage], int]:
        """The renderer message list a full render sees, plus the index shift.

        With tools, the leading system message (when present) is folded into
        the renderer's tool-prefix messages, exactly as `build_generation_prompt`
        does. The returned shift maps an index into `messages` to the same
        message's index in the returned list.
        """
        renderer_messages = renderer_messages_from_chat(messages)
        shift = 0
        if tools:
            system_prompt = ""
            dropped = 0
            if renderer_messages and renderer_messages[0]["role"] == "system":
                first_content = renderer_messages[0]["content"]
                system_prompt = first_content if isinstance(first_content, str) else ""
                renderer_messages = renderer_messages[1:]
                dropped = 1
            prefix = self._renderer.create_conversation_prefix_with_tools(
                tool_specs_from_chat(tools), system_prompt
            )
            renderer_messages = list(prefix) + renderer_messages
            shift = len(prefix) - dropped
        return renderer_messages, shift

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        """Render chat messages (and optional tool schemas) into prompt token ids."""
        renderer_messages, _ = self._effective_messages(messages, tools)
        return self._renderer.build_generation_prompt(renderer_messages).to_ints()

    @staticmethod
    def _chunk_tokens(rendered: RenderedMessage) -> list[int]:
        """Flatten one rendered message's header and output chunks to token ids.

        Mirrors the base `Renderer.build_generation_prompt` composition: header
        tokens first, then each output chunk's tokens (`stop_overlap` is
        ignored there too). Non-text chunks are rejected loudly; the tinker
        provider is text-only.
        """
        tokens: list[int] = []
        if rendered.header is not None:
            tokens.extend(rendered.header.tokens)
        for chunk in rendered.output:
            chunk_tokens = getattr(chunk, "tokens", None)
            if chunk_tokens is None:
                raise ValueError(
                    "the renderer produced a non-text chunk while rendering a prompt "
                    "suffix; the tinker provider is text-only"
                )
            tokens.extend(chunk_tokens)
        return tokens

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        """Render `messages[delta_start:]` plus the generation header as a token suffix.

        Composes per-message segments exactly the way the base cookbook
        `Renderer.build_generation_prompt` does (same `RenderContext` for each
        absolute position, computed over the full effective message list so
        position-sensitive renderers frame the delta identically to a full
        render), but emits only the delta messages and the trailing generation
        header. Verified against the live qwen3_5 sink: for the recorded
        episodes this composition reproduces the full render's delta tokens
        byte for byte.

        The end-of-turn framing is derived from the renderer by rendering an
        empty assistant message and taking its output tokens (for Qwen that is
        exactly `<|im_end|>`); when `previous_sampled_ids` does not already end
        with it (max_tokens truncation), it is prepended so the spliced turn is
        properly closed before the next message begins.

        Args:
            messages: The FULL incoming message list of the next call.
            delta_start: Index of the first message not covered by the
                previous prompt plus the previous sampled turn (the message
                right after the caller's echo of that turn).
            tools: Tool schemas for this call; must match the ones the shared
                prefix was rendered with.
            previous_sampled_ids: The raw sampled ids being spliced in ahead
                of this suffix, used only to decide end-of-turn completion.

        Returns:
            Token ids to append after the previous prompt plus sampled ids.

        Raises:
            ImportError: If tinker-cookbook is not installed (distill extra).
            ValueError: If `delta_start` is out of range or a delta message
                renders to non-text chunks.
        """
        try:
            from tinker_cookbook.renderers.base import RenderContext
        except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
            raise ImportError(MISSING_DISTILL_EXTRA) from exc

        if not 1 <= delta_start <= len(messages):
            raise ValueError(
                f"delta_start {delta_start} is out of range for {len(messages)} message(s); "
                "it must point past the shared history (at least 1) and at most one past "
                "the final message"
            )
        effective, shift = self._effective_messages(messages, tools)
        start = delta_start + shift
        last_user_index = max(
            (idx for idx, msg in enumerate(effective) if msg["role"] == "user"),
            default=-1,
        )
        tokens: list[int] = []
        end_of_turn = self._end_of_turn_tokens(effective, start, last_user_index)
        if end_of_turn and previous_sampled_ids[-len(end_of_turn) :] != end_of_turn:
            tokens.extend(end_of_turn)
        for idx in range(start, len(effective)):
            ctx = RenderContext(
                idx=idx,
                is_last=(idx == len(effective) - 1),
                prev_message=effective[idx - 1] if idx > 0 else None,
                last_user_index=last_user_index,
            )
            tokens.extend(self._chunk_tokens(self._renderer.render_message(effective[idx], ctx)))
        suffix_ctx = RenderContext(
            idx=len(effective),
            is_last=True,
            prev_message=effective[-1] if effective else None,
            last_user_index=last_user_index,
        )
        # Private cookbook seam, mirrored from the base build_generation_prompt;
        # cookbook churn is contained here by design (module docstring).
        tokens.extend(self._renderer._get_generation_suffix("assistant", suffix_ctx))  # noqa: SLF001
        return tokens

    def _end_of_turn_tokens(
        self, effective: list[RendererMessage], start: int, last_user_index: int
    ) -> list[int]:
        """The template's end-of-turn framing, derived from the renderer.

        Rendering an empty assistant message at the previous turn's position
        yields output chunks of exactly the end-of-turn framing (for Qwen,
        `<|im_end|>`); header tokens are excluded because the sampled turn's
        header was already part of the previous generation prompt.
        """
        from tinker_cookbook.renderers.base import RenderContext

        ctx = RenderContext(
            idx=start - 1,
            is_last=False,
            prev_message=effective[start - 2] if start >= 2 else None,
            last_user_index=last_user_index,
        )
        empty: RendererMessage = {"role": "assistant", "content": ""}
        rendered = self._renderer.render_message(empty, ctx)
        tokens: list[int] = []
        for chunk in rendered.output:
            chunk_tokens = getattr(chunk, "tokens", None)
            if chunk_tokens is None:  # pragma: no cover - empty content renders text-only
                raise ValueError(
                    "the renderer produced a non-text chunk for an empty assistant "
                    "message; cannot derive end-of-turn framing"
                )
            tokens.extend(chunk_tokens)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids back to text with the renderer's tokenizer."""
        return str(self._renderer.tokenizer.decode(token_ids))

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        """Parse sampled token ids into text plus OpenAI-format tool calls.

        Thinking parts are rendered back inline as `<think>...</think>` so the
        parsed text round-trips the decoded sample exactly (the cookbook's
        `parse_content_blocks` preserves whitespace). Tool calls the renderer
        could not parse are dropped with a warning, matching the cookbook's
        LiteLLM provider.
        """
        message, termination = self._renderer.parse_response(sampled_ids)
        content = message["content"]
        if isinstance(content, str):
            text = content
        else:
            fragments: list[str] = []
            for part in content:
                if part["type"] == "thinking":
                    fragments.append("<think>" + part["thinking"] + "</think>")
                elif part["type"] == "text":
                    fragments.append(part["text"])
                else:
                    raise ValueError(
                        "sampled response contained a non-text content part "
                        f"({part['type']!r}); the tinker provider is text-only"
                    )
            text = "".join(fragments)
        tool_calls = [
            ChatToolCall(
                id=tc.id or f"call_{index}",
                function=ChatFunctionCall(name=tc.function.name, arguments=tc.function.arguments),
            )
            for index, tc in enumerate(message.get("tool_calls") or [])
        ]
        unparsed = message.get("unparsed_tool_calls") or []
        if unparsed:
            logger.warning(
                "dropping %d tool call(s) the renderer could not parse: %s",
                len(unparsed),
                "; ".join(item.error for item in unparsed),
            )
        # is_clean, not is_stop_sequence: some renderers (e.g. role_colon) report a
        # clean end-of-turn via the model's EOS token, which must not read as truncation.
        return ParsedAssistantMessage(
            text=text, tool_calls=tool_calls, stopped=termination.is_clean
        )


def build_renderer(base_model: str, tokenizer: RendererTokenizer) -> CookbookChatRendering:
    """Build the cookbook-backed rendering for a base model.

    The renderer name comes from the cookbook's own model catalog
    (`get_recommended_renderer_name`), exactly how its LiteLLM provider picks
    renderers, so wmh never maintains a parallel model-to-renderer table.

    Args:
        base_model: Base model name in `org/model` form (e.g. `Qwen/Qwen3-8B`).
        tokenizer: Tokenizer for that base model, typically from
            `tinker.SamplingClient.get_tokenizer()`.

    Returns:
        The cookbook-backed `ChatRendering` implementation.

    Raises:
        ImportError: If tinker-cookbook is not installed (distill extra).
        ValueError: If the cookbook has no renderer mapping for `base_model`.
    """
    try:
        from tinker_cookbook.exceptions import ConfigurationError
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
        raise ImportError(MISSING_DISTILL_EXTRA) from exc

    try:
        renderer_name = get_recommended_renderer_name(base_model)
    except (ConfigurationError, KeyError, ValueError) as exc:
        raise ValueError(
            f"no cookbook renderer is known for base model {base_model!r} ({exc}); "
            "use a model family listed in tinker_cookbook.model_info"
        ) from exc
    # The cookbook types its tokenizer as HF PreTrainedTokenizer but treats it as
    # Any at runtime; RendererTokenizer is the slice it actually calls.
    renderer = get_renderer(renderer_name, cast("Tokenizer", tokenizer), model_name=base_model)
    return CookbookChatRendering(renderer)

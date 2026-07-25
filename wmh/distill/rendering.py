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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, cast

from llm_waterfall.types import ChatFunctionCall, ChatMessage, ChatTool, ChatToolCall
from pydantic import BaseModel, Field, JsonValue

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer
    from tinker_cookbook.renderers.base import Message as RendererMessage
    from tinker_cookbook.renderers.base import RenderedMessage, ToolSpec
    from tinker_cookbook.renderers.base import ToolCall as RendererToolCall
    from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)

MISSING_DISTILL_EXTRA = (
    "the tinker-cookbook SDK is not installed; run `uv sync --extra distill` "
    "to use the Tinker distillation provider"
)


class RendererTokenizer(Protocol):
    """The tokenizer slice cookbook renderers (and this seam's decodes) rely on.

    HuggingFace `PreTrainedTokenizer` (what `tinker.SamplingClient.get_tokenizer`
    returns) satisfies this structurally; tests supply small deterministic fakes.
    """

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token ids."""
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = ...) -> str:
        """Decode token ids back to text (HF's `skip_special_tokens` contract)."""
        ...


class ParsedAssistantMessage(BaseModel):
    """One sampled assistant turn parsed out of student token ids."""

    text: str = ""
    """Assistant text, with any thinking rendered inline as <think>...</think>."""

    tool_calls: list[ChatToolCall] = Field(default_factory=list)
    """Parsed tool calls in OpenAI chat format."""

    stopped: bool
    """True when generation terminated cleanly (stop sequence or EOS), False on truncation."""

    unparsed_errors: list[str] = Field(default_factory=list)
    """Parser complaints for tool calls that could not be decoded and were not salvaged.

    Surfaced (not just logged) so the agent scaffold can feed the complaint back to the model as an
    observation. A turn whose only tool call failed to parse is NOT a completion: reporting it as
    plain prose is how a truncated `write_file` became a "submitted" episode with reward 0."""

    salvaged_tool_calls: int = 0
    """How many of `tool_calls` were recovered from a truncated/unterminated emission."""

    def to_chat_message(self) -> ChatMessage:
        """This turn as an llm_waterfall assistant message.

        The single place a parsed sample becomes a canonical `ChatMessage`, so
        the response the agent sees and the assistant turn a conversation
        replay reconstructs (`wmh.distill.tokens.reconstruct_conversation`)
        cannot drift apart. Empty text and an empty tool-call list collapse to
        None, matching the OpenAI-shaped messages the agent sends back.
        """
        return ChatMessage(
            role="assistant", content=self.text or None, tool_calls=self.tool_calls or None
        )


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

    def decode_with_specials(self, token_ids: list[int]) -> str:
        """Decode token ids to text KEEPING special tokens (the template framing)."""
        ...

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        """Parse sampled token ids into an assistant message (text plus tool calls)."""
        ...


_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def salvage_truncated_tool_call(text: str) -> str | None:
    """Close an unterminated trailing tool-call block so a parser can read it.

    A turn cut off at the output-token cap ends mid-emission, so the template's closers are simply
    absent (`</parameter></function></tool_call>`) and every parser sees plain prose. The reference
    terminus-2 agent recovers the action instead of discarding the turn; this is the same repair,
    restricted to appending the closers the emission opened. Nothing is rewritten or guessed: a
    block that is malformed for any other reason (two `<function=` blocks, a stray `</think>`) still
    fails to parse afterwards and becomes an explicit parse error the model is told about.

    Args:
        text: The sampled turn's decoded text.

    Returns:
        The repaired text, or None when there is no unterminated tool-call block to repair (no
        opener, the last opener is already closed, or no function name was emitted yet).
    """
    open_at = text.rfind(_TOOL_CALL_OPEN)
    if open_at < 0 or text.rfind(_TOOL_CALL_CLOSE) > open_at:
        return None
    block = text[open_at:].rstrip()
    if "<function=" not in block:
        return None
    if block.rfind("<parameter=") > block.rfind("</parameter>"):
        block += "\n</parameter>"
    if block.rfind("<function=") > block.rfind("</function>"):
        block += "\n</function>"
    return text[:open_at] + block + f"\n{_TOOL_CALL_CLOSE}"


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


def _chat_tool_calls(calls: Sequence[RendererToolCall]) -> list[ChatToolCall]:
    """Cookbook renderer tool calls as llm_waterfall chat tool calls."""
    return [
        ChatToolCall(
            id=tc.id or f"call_{index}",
            function=ChatFunctionCall(name=tc.function.name, arguments=tc.function.arguments),
        )
        for index, tc in enumerate(calls)
    ]


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

    def decode_with_specials(self, token_ids: list[int]) -> str:
        """Raw decode preserving special tokens, for human-readable episode logs.

        `decode` stays the parsing/serving path and rides the tokenizer's
        default cleanup, which may strip or normalize special tokens
        depending on the tokenizer's configuration; this variant pins
        `skip_special_tokens=False` so the chat template's framing
        (`<|im_start|>`, think blocks, tool-call markers) survives verbatim
        into the text.
        """
        return str(self._renderer.tokenizer.decode(token_ids, skip_special_tokens=False))

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        """Parse sampled token ids into text plus OpenAI-format tool calls.

        Thinking parts are rendered back inline as `<think>...</think>` so the
        parsed text round-trips the decoded sample exactly (the cookbook's
        `parse_content_blocks` preserves whitespace).

        A turn whose tool call the renderer could not read is NOT reported as
        plain prose. First the truncation repair runs
        (`salvage_truncated_tool_call`, re-parsed through this same renderer, so
        an emission cut off at the output cap still yields its action); when that
        recovers nothing, the parser's own complaints are surfaced on
        `unparsed_errors` for the scaffold to feed back to the model. The
        RECORDED token span is never touched by any of this: the repair exists
        only to read an action out of the turn, so the training data stays a
        verbatim prefix of the episode.
        """
        message, termination = self._renderer.parse_response(sampled_ids)
        text = self._message_text(message)
        tool_calls = _chat_tool_calls(message.get("tool_calls") or [])
        unparsed_errors = [item.error for item in message.get("unparsed_tool_calls") or []]
        salvaged = 0
        if not tool_calls:
            tool_calls, unparsed_errors, salvaged = self._salvage(text, unparsed_errors)
        if unparsed_errors and not tool_calls:
            logger.warning(
                "no tool call could be read from a sampled turn (%d parser complaint(s): %s); "
                "the scaffold feeds this back to the model instead of ending the episode",
                len(unparsed_errors),
                "; ".join(unparsed_errors),
            )
        # is_clean, not is_stop_sequence: some renderers (e.g. role_colon) report a
        # clean end-of-turn via the model's EOS token, which must not read as truncation.
        return ParsedAssistantMessage(
            text=text,
            tool_calls=tool_calls,
            stopped=termination.is_clean,
            unparsed_errors=unparsed_errors,
            salvaged_tool_calls=salvaged,
        )

    def _message_text(self, message: RendererMessage) -> str:
        """The parsed turn's text, with thinking rendered back inline."""
        content = message["content"]
        if isinstance(content, str):
            return content
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
        return "".join(fragments)

    def _salvage(
        self, text: str, unparsed_errors: list[str]
    ) -> tuple[list[ChatToolCall], list[str], int]:
        """Recover a tool call from an unterminated emission, through this renderer.

        Args:
            text: The parsed turn's text (an unclosed `<tool_call>` block survives here as text,
                which is exactly the emission to repair).
            unparsed_errors: Parse complaints the first pass already produced.

        Returns:
            The recovered calls, the errors to report, and how many calls were salvaged. When
            nothing is recoverable the errors are returned unchanged, except that an unterminated
            block that still fails to parse contributes its own complaint.
        """
        repaired = salvage_truncated_tool_call(text)
        if repaired is None:
            return [], unparsed_errors, 0
        tokenizer = self._renderer.tokenizer
        try:
            # The end-of-turn framing is part of the repair: renderers skip block parsing
            # entirely on a turn they read as truncated (`if not termination.is_clean:
            # return`), so without it the re-parse hands the whole emission back as prose.
            repaired_ids = [
                *tokenizer.encode(repaired, add_special_tokens=False),
                *self._end_of_turn_ids(),
            ]
            message, _ = self._renderer.parse_response(repaired_ids)
        except Exception as exc:  # noqa: BLE001 - salvage is best effort; report, never raise
            return [], [*unparsed_errors, f"truncated tool call could not be salvaged: {exc}"], 0
        salvaged_calls = _chat_tool_calls(message.get("tool_calls") or [])
        if salvaged_calls:
            logger.info(
                "salvaged %d tool call(s) from a tool-call emission that was cut off before its "
                "closing tags; the episode continues instead of ending as a completion",
                len(salvaged_calls),
            )
            return salvaged_calls, [], len(salvaged_calls)
        errors = [item.error for item in message.get("unparsed_tool_calls") or []] or [
            "tool-call block was left unterminated and could not be repaired"
        ]
        return [], [*unparsed_errors, *errors], 0

    def _end_of_turn_ids(self) -> list[int]:
        """Token ids that mark one assistant turn as cleanly ended, per the renderer.

        Taken from the renderer's own stop sequences (token ids for the Qwen/Nemotron families,
        strings for the plainer templates), so no template framing is hardcoded here.
        """
        stops = self._renderer.get_stop_sequences()
        if not stops:
            return []
        first = stops[0]
        if isinstance(first, int):
            return [first]
        return list(self._renderer.tokenizer.encode(first, add_special_tokens=False))


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

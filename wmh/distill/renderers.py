"""wmh-owned tinker-cookbook renderers that keep a sampled assistant turn verbatim.

Distillation rollouts run harbor's own terminus-2 agent, and terminus-2 keeps
only `parse_response(...)["content"]` in its chat history. Every REASONING
renderer in the lineup these runs train against (`nemotron3`,
`nemotron3_ultra`, and `qwen3_5`, which is the auto-discovered renderer for
both Qwen3.5 and Qwen3.6) returns that content as a LIST of thinking/text
parts, and strips the thinking block when it re-renders a turn that is no
longer the last message. Two measured consequences:

- harbor's `TerminusJSONPlainParser.parse_response` runs `re` over the content
  and raises `TypeError: expected string or bytes-like object, got 'list'`
  (`terminus_json_plain_parser.py:339`); the XML parser raises
  `AttributeError: 'list' object has no attribute 'find'`
  (`terminus_xml_plain_parser.py:199`). Every trial dies before it grades
  anything.
- the prefix property breaks at every turn boundary, because turn N+1's prompt
  re-renders turn N's assistant message with its thinking removed. The episode
  becomes one training datum PER TURN instead of one, and the waste is
  QUADRATIC in turn count: 2.7x the tokens at 6 turns, 7.8x at 20, 15.1x at 40.

The renderers here fix both without patching harbor, by splitting what the
AGENT reads from what the MODEL re-reads:

- `parse_response` hands the harbor parser a plain `str` view of the TEXT parts
  only (the action payload it already knows how to parse), with the turn's
  exact sampled token ids riding along inside a `str` subclass;
- `render_message` re-renders that historical turn from those ids, under the
  generation header the turn was actually sampled behind, so the reasoning
  survives in the token stream and prompt(N+1) extends prompt(N) plus
  sampled(N) verbatim.

The ids ride the path `parse_response -> LLMResponse.content ->
Chat._messages -> message_history -> TinkerLLM.call ->
renderer.build_generation_prompt`, every step of which passes `content`
through untouched. No harbor code is modified.

Keeping the reasoning in the token stream is not free, and the module ships
BOTH sides of the trade (`VerbatimHistoryMixin` and `StripHistoryMixin`, one
renderer family each). A turn's thinking is in its sampled ids either way, so
the model reasons every turn and the loss covers every reasoning token in both;
what differs is whether a LATER prompt still contains it:

- carry it (verbatim): the episode is one datum, but the context grows with
  every turn's reasoning. Measured on a 53-trial TB2 teacher baseline, this put
  27 of 53 episodes over the window, and terminus-2 turns a context overflow
  into a FAILED trial rather than a finished episode -- so those trials left the
  denominator instead of scoring zero, and the reported solve rate was 95.8% of
  the survivors.
- drop it (strip history): later prompts stay small (that same set: final
  context p50 23,631 -> 8,682, max 49,117 -> 40,162), which is also what the
  stock renderers and the published terminus-2 setups do. The cost is the
  prefix property, so the episode fragments into one datum per turn and teacher
  prefill rises ~5.1x cold, ~2.6x once the shared prefix bills cached.

Pick by whether the episode fits the window: an overflow costs a whole trial
AND biases the measurement, whereas fragmentation only costs money.

Only a CLEANLY terminated turn is carried verbatim. A turn that hit the output
cap has no stop token, so re-rendering it from its ids would splice an
unterminated assistant message into the context; those fall back to the base
renderer's behavior and show up as a prefix break plus a
`RolloutStats.truncated_spans` count.

Unlike the rest of `wmh.distill`, this module cannot be imported without the
`distill` extra: it subclasses tinker-cookbook renderer classes at module
scope. Importers reach it lazily at the point of use (see
`wmh.distill.rollouts.terminus_2_agent_kwargs`, the single call site that
registers these renderers for every rollout path).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import tinker
from tinker_cookbook.exceptions import RendererError
from tinker_cookbook.image_processing_utils import ImageProcessor
from tinker_cookbook.renderers import (
    get_renderer,
    is_renderer_registered,
    register_renderer,
)
from tinker_cookbook.renderers.base import (
    Message,
    ParseTermination,
    RenderContext,
    RenderedMessage,
    Renderer,
    get_text_content,
)
from tinker_cookbook.renderers.nemotron3 import Nemotron3Renderer, Nemotron3UltraRenderer
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer
from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer


class VerbatimContent(str):
    """The action text a turn parsed to, carrying that turn's sampled token ids.

    A `str` subclass, so every consumer that treats message content as text
    (regex, `.find`, `json`, logging, pydantic coercion) behaves exactly as it
    would with a plain string; the ids ride in the instance `__dict__` and only
    `VerbatimHistoryMixin.render_message` looks for them. Ordinary string
    operations (`.strip()`, f-strings, slicing) return plain `str` and drop the
    ids, which is the safe direction: a turn whose ids were lost re-renders
    through the base renderer instead of emitting the wrong tokens.
    """

    ids: list[int]
    """The turn's sampled ids, truncated at (and including) the stop token."""

    def __new__(cls, action_text: str, ids: Sequence[int] = ()) -> VerbatimContent:
        """Build the content view, copying `ids` so later mutation cannot alias it."""
        content = super().__new__(cls, action_text)
        content.ids = list(ids)
        return content

    # The canonical idiom for an immutable subclass with extra __new__ arguments
    # (CPython's pickle docs): a wider tuple than `str.__getnewargs__` returns, which
    # is what feeds those arguments back to __new__.
    def __getnewargs__(self) -> tuple[str, list[int]]:  # ty: ignore[invalid-method-override]
        """Keep the ids across copy, deepcopy, and pickle of the content object."""
        return (str(self), self.ids)


class VerbatimHistoryMixin(Renderer):
    """Re-renders a sampled assistant turn from its exact token ids.

    Mix in FIRST (`class Foo(VerbatimHistoryMixin, SomeRenderer)`) so both
    overrides win over the base renderer's. It inherits `Renderer` only so the
    base surface it calls (`tokenizer`, `_end_message_token`,
    `_get_generation_suffix`) is declared; it implements no stop sequences of
    its own and is never instantiated directly.
    """

    @property
    def has_extension_property(self) -> bool:
        """True: a historical turn re-renders to the ids it was sampled as.

        The base reasoning renderers report False because they strip thinking
        from history. These do not, so successive generation prompts really are
        prefixes of one another and the episode merges into one datum.
        """
        return True

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse a sampled turn into text-only content that carries its ids.

        The base renderer does the family-specific work (thinking/tool-call
        block splitting, separator-newline trimming, XML tool-call conversion);
        this narrows the result to the text parts so harbor's terminus-2
        parsers see the `str` they expect, and attaches the sampled ids.

        Args:
            response: The turn's sampled token ids, exactly as the sampler
                returned them.

        Returns:
            The parsed message and its termination. Content is a
            `VerbatimContent` only when the turn terminated cleanly; a
            truncated or malformed turn keeps the base renderer's content, so
            it re-renders through the base path rather than splicing an
            unterminated message into the next prompt.
        """
        message, termination = super().parse_response(response)
        if not termination.is_clean:
            return message, termination
        sampled = list(response)
        stop = self._end_message_token
        if stop in sampled:
            sampled = sampled[: sampled.index(stop) + 1]
        message["content"] = VerbatimContent(get_text_content(message), sampled)
        return message, termination

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render one message, replaying a verbatim assistant turn from its ids.

        The header is the generation suffix rebuilt under the context that was
        live when the turn was sampled: the turn WAS the last message, and the
        last user message was the one immediately before it. That is exactly
        what `build_generation_prompt` emitted then, so replaying it now
        reproduces the earlier prompt token for token.

        Args:
            message: The message to render.
            ctx: Its position in the conversation being rendered.

        Returns:
            The rendered header and output. Any message without verbatim ids
            (every user/system/tool turn, plus assistant turns that did not
            terminate cleanly) falls through to the base renderer.
        """
        content = message.get("content")
        if message["role"] != "assistant" or not isinstance(content, VerbatimContent):
            return super().render_message(message, ctx)
        sampled_ctx = RenderContext(
            idx=ctx.idx,
            is_last=True,
            prev_message=ctx.prev_message,
            last_user_index=ctx.idx - 1,
        )
        header = self._get_generation_suffix("assistant", sampled_ctx)
        return RenderedMessage(
            header=tinker.EncodedTextChunk(tokens=list(header)),
            output=[tinker.EncodedTextChunk(tokens=list(content.ids))],
        )


class StripHistoryMixin(Renderer):
    """Parses a sampled turn to plain text and lets the base renderer drop thinking.

    Mix in FIRST, as with `VerbatimHistoryMixin`. This is the OTHER side of the
    trade that module docstring describes, and the two cannot be combined:

    - `VerbatimHistoryMixin` replays a turn's exact ids, so reasoning survives
      into later prompts and the episode is ONE datum. The cost is context:
      every turn's thinking accumulates.
    - this mixin keeps only the action text, so later prompts carry no prior
      reasoning and stay small. The cost is the prefix property: turn N's
      sampled ids (which include its thinking) are no longer a substring of
      turn N+1's prompt, so the episode fragments into one datum PER TURN.

    Both keep the model REASONING and both train on those reasoning tokens --
    thinking is in every sampled turn and inside the loss mask either way. What
    differs is only whether the model is shown its own earlier reasoning, which
    is a serving choice; the stock reasoning renderers, and the published
    terminus-2 setups, do not show it.

    Measured on the 53-trial Qwen3.6-27B teacher baseline that motivated this:
    carrying reasoning forward pushed 27 of 53 episodes (51%) past the context
    budget, and a `ContextLengthExceededError` fails the whole trial rather
    than ending it, so those episodes left the denominator entirely. Fragmented
    datums cost more teacher prefill; overflowed episodes cost the measurement.
    """

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse a sampled turn down to its text parts, as a plain `str`.

        The base reasoning renderers return content as a LIST of thinking/text
        parts, which is what makes harbor's terminus-2 parsers raise (see the
        module docstring). Narrowing to the text parts hands them the `str`
        they expect, and -- because the thinking parts are simply gone from the
        message -- the base `render_message` has nothing to carry forward.

        Args:
            response: The turn's sampled token ids, exactly as the sampler
                returned them.

        Returns:
            The parsed message with plain-text content, and its termination.
        """
        message, termination = super().parse_response(response)
        message["content"] = get_text_content(message)
        return message, termination


class Qwen3_5StripHistoryRenderer(StripHistoryMixin, Qwen3_5Renderer):
    """Qwen3.5 and Qwen3.6 with thinking on, dropped from history after each turn."""


class Qwen3VerbatimRenderer(VerbatimHistoryMixin, Qwen3Renderer):
    """Qwen3 with thinking on and a verbatim-id history."""


class Qwen3_5VerbatimRenderer(VerbatimHistoryMixin, Qwen3_5Renderer):
    """Qwen3.5 and Qwen3.6 with thinking on and a verbatim-id history."""


class Nemotron3VerbatimRenderer(VerbatimHistoryMixin, Nemotron3Renderer):
    """Nemotron-3 Nano/Super with reasoning on and a verbatim-id history."""


class Nemotron3UltraVerbatimRenderer(VerbatimHistoryMixin, Nemotron3UltraRenderer):
    """Nemotron-3 Ultra with reasoning on and a verbatim-id history."""


QWEN3_VERBATIM = "wmh/qwen3_verbatim"
QWEN3_5_VERBATIM = "wmh/qwen3_5_verbatim"
NEMOTRON3_VERBATIM = "wmh/nemotron3_verbatim"
NEMOTRON3_ULTRA_VERBATIM = "wmh/nemotron3_ultra_verbatim"
QWEN3_5_STRIP_HISTORY = "wmh/qwen3_5_strip_history"


def _build_qwen3(tokenizer: Tokenizer, image_processor: ImageProcessor | None = None) -> Renderer:
    """Build the Qwen3 verbatim renderer (text only, no image processor)."""
    return Qwen3VerbatimRenderer(tokenizer)


def _build_qwen3_5(tokenizer: Tokenizer, image_processor: ImageProcessor | None = None) -> Renderer:
    """Build the Qwen3.5/Qwen3.6 verbatim renderer."""
    return Qwen3_5VerbatimRenderer(tokenizer, image_processor=image_processor)


def _build_nemotron3(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Build the Nemotron-3 Nano/Super verbatim renderer."""
    return Nemotron3VerbatimRenderer(tokenizer, image_processor=image_processor)


def _build_nemotron3_ultra(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Build the Nemotron-3 Ultra verbatim renderer."""
    return Nemotron3UltraVerbatimRenderer(tokenizer, image_processor=image_processor)


def _build_qwen3_5_strip_history(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None = None
) -> Renderer:
    """Build the Qwen3.5/Qwen3.6 renderer that drops prior turns' reasoning."""
    return Qwen3_5StripHistoryRenderer(tokenizer, image_processor=image_processor)


VERBATIM_RENDERERS = {
    QWEN3_VERBATIM: _build_qwen3,
    QWEN3_5_VERBATIM: _build_qwen3_5,
    NEMOTRON3_VERBATIM: _build_nemotron3,
    NEMOTRON3_ULTRA_VERBATIM: _build_nemotron3_ultra,
}
"""The renderers that carry a sampled turn forward as its exact ids.

One datum per episode, at the cost of a context that grows with every turn's
reasoning. Prefer these only when the episode fits the model's window.
"""

STRIP_HISTORY_RENDERERS = {
    QWEN3_5_STRIP_HISTORY: _build_qwen3_5_strip_history,
}
"""The renderers that drop prior turns' reasoning from later prompts.

Small contexts, at the cost of per-turn datums. Only the Qwen3.5/3.6 family is
built out, because that is the pair these runs train; the mixin is family-
agnostic, so adding a Nemotron variant is a subclass and a factory.
"""

WMH_RENDERERS = {**VERBATIM_RENDERERS, **STRIP_HISTORY_RENDERERS}
"""Every wmh renderer name, mapped to its cookbook factory.

Name these from `[rollout.renderers]` in a run config, keyed by the base model
whose rollouts should use them (`student.base_model`, `teacher.model`).
"""


def register_wmh_renderers() -> None:
    """Register every wmh renderer with the cookbook's global registry.

    Idempotent: registration is a dict assignment keyed by name, so repeated
    calls leave the registry in the same state. Call it before anything
    resolves a renderer by name (terminus-2 resolves `llm_kwargs.renderer_name`
    inside its own `TinkerLLM`, in this process).
    """
    for name, factory in WMH_RENDERERS.items():
        register_renderer(name, factory)


class _RendererNameProbe:
    """The tokenizer slice a renderer constructor touches, for name validation only.

    `get_renderer` is the only authority on which names the cookbook knows, and
    it answers by CONSTRUCTING the renderer, which needs a tokenizer. Loading a
    real one would mean a model download at config-load time, so name checks
    build against this stand-in and throw the renderer away.
    """

    name_or_path = "wmh-renderer-name-probe"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """One token per call: enough for the constructors that resolve special tokens."""
        return [0]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Never reached during construction; present so the slice is complete."""
        return ""


def is_known_renderer(name: str) -> bool:
    """Whether the cookbook can build a renderer called `name`.

    Covers the wmh verbatim renderers, anything else registered through
    `tinker_cookbook.renderers.register_renderer`, and every built-in cookbook
    name. Use it to reject a typo where it is cheap (config load) instead of
    where it is expensive (the first rollout of a paid run).

    Args:
        name: The renderer name to check.

    Returns:
        True when `get_renderer` would accept the name.
    """
    if name in WMH_RENDERERS or is_renderer_registered(name):
        return True
    try:
        get_renderer(name, cast("Tokenizer", _RendererNameProbe()))
    except RendererError:
        return False
    return True

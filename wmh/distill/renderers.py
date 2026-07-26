"""wmh-owned tinker-cookbook renderers: verbatim content, ephemeral reasoning.

Distillation rollouts run harbor's own terminus-2 agent, and terminus-2 keeps
only `parse_response(...)["content"]` in its chat history. Every REASONING
renderer in the lineup these runs train against (`nemotron3`,
`nemotron3_ultra`, and `qwen3_5`, which is the auto-discovered renderer for
both Qwen3.5 and Qwen3.6) returns that content as a LIST of thinking/text
parts, so harbor's `TerminusJSONPlainParser.parse_response` runs `re` over a
list and raises `TypeError: expected string or bytes-like object, got 'list'`
(`terminus_json_plain_parser.py:339`); the XML parser raises
`AttributeError: 'list' object has no attribute 'find'`
(`terminus_xml_plain_parser.py:199`). Every trial dies before it grades
anything. Fixing that is what the `str`-subclass content view here is for.

The second half of this module is the history policy, and it is a deliberate
reversal of the one these classes were introduced with. A historical assistant
turn replays only its POST-REASONING content: the model still thinks fully on
every turn, that reasoning is still in the turn's own sampled ids, still
trained on and still scored by the teacher, and it is simply absent from every
LATER turn's prompt. Reasoning is generated once and paid for once instead of N
times.

Why, measured on a real TerminalBench-2 run of this exact agent:

- 9 of 19 trials exceeded 60,000 tokens of context against a usable ceiling of
  65,530, and 8 of 17 holdout trials died with `ContextLengthExceededError`.
  A dead trial is excluded from the solve rate as an infrastructure failure,
  which silently biases the score toward short tasks: the tasks that need the
  most turns are exactly the ones that cannot be measured.
- dropping reasoning from history brings all 9 under the ceiling:
  `merge-diff-arc-agi-task` 64,228 -> 31,411, `schemelike-metacircular-eval`
  63,774 -> 33,967, `dna-assembly` 64,969 -> 40,257.
- it is insufficient alone for exactly one trial, `make-mips-interpreter`
  (64,228 -> 62,086), whose context is terminal OUTPUT rather than reasoning.
  That is what the separate per-observation clip
  (`RolloutConfig.observation_clip_tokens`, applied here) is the margin for.

What it costs, stated plainly because nothing downstream will: the prefix
property is gone (`has_extension_property` is False and says why), so an
episode becomes one training datum PER TURN and shared context is re-prefilled
per turn. That waste is quadratic in turn count (2.7x the tokens at 6 turns,
7.8x at 20, 15.1x at 40). The trade is deliberate: a re-prefilled token is a
cost, a trial killed by the context ceiling is a measurement that does not
exist.

Three mechanisms, none of which patch harbor:

- `parse_response` hands the harbor parser a plain `str` view of the TEXT parts
  only (the action payload it already knows how to parse), with the turn's
  exact sampled token ids riding along inside a `str` subclass, plus the offset
  where its post-reasoning content begins;
- `render_message` replays a historical turn from those ids alone, so no text
  is ever re-encoded and the ids in the context are the ids the sampler issued;
- `render_message` also clips a per-command terminal-output observation to
  `observation_clip_tokens` tokens (head plus tail, with a marker naming what
  was dropped), which is the other half of the context margin.

Both edits happen where the SAMPLING prompt is built, which is the only place
they can happen safely: the recorded `prompt_token_ids` are what training
consumes, so the prompt the sampler saw and the prompt trained on are the same
object by construction. Applying either edit only when assembling training
data would leave `sampled_logprobs` describing a sequence that no longer
exists, and nothing would error.

The ids ride the path `parse_response -> LLMResponse.content ->
Chat._messages -> message_history -> TinkerLLM.call ->
renderer.build_generation_prompt`, every step of which passes `content`
through untouched.

Only a CLEANLY terminated turn is carried verbatim. A turn that hit the output
cap has no stop token, so re-rendering it from its ids would splice an
unterminated assistant message into the context; those fall back to the base
renderer's behavior and are counted by `RolloutStats.truncated_spans`.

Unlike the rest of `wmh.distill`, this module cannot be imported without the
`distill` extra: it subclasses tinker-cookbook renderer classes at module
scope. Importers reach it lazily at the point of use (see
`wmh.distill.rollouts.terminus_2_agent_kwargs`, the single call site that
registers these renderers for every rollout path).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
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

from wmh.distill.rendering import CLOSE_THINK

logger = logging.getLogger(__name__)

OBSERVATION_CLIP_MARKER = "\n[... {dropped} tokens of terminal output omitted ...]\n"
"""What replaces the middle of a clipped observation, so the model is told."""


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

    content_start: int
    """Index into `ids` where the turn's post-reasoning content begins.

    0 means the turn has no reasoning block to drop (no `</think>` token in the
    sampled ids), in which case `ids` replay whole and the turn's prompt IS
    reproduced verbatim. Anything greater is the position just past the first
    `</think>` token: everything before it is the reasoning, which history does
    not carry.
    """

    def __new__(
        cls, action_text: str, ids: Sequence[int] = (), content_start: int = 0
    ) -> VerbatimContent:
        """Build the content view, copying `ids` so later mutation cannot alias it."""
        content = super().__new__(cls, action_text)
        content.ids = list(ids)
        content.content_start = content_start
        return content

    @property
    def content_ids(self) -> list[int]:
        """The ids a historical turn replays: everything after the reasoning block."""
        return self.ids[self.content_start :]

    # The canonical idiom for an immutable subclass with extra __new__ arguments
    # (CPython's pickle docs): a wider tuple than `str.__getnewargs__` returns, which
    # is what feeds those arguments back to __new__.
    def __getnewargs__(self) -> tuple[str, list[int], int]:  # ty: ignore[invalid-method-override]
        """Keep the ids across copy, deepcopy, and pickle of the content object."""
        return (str(self), self.ids, self.content_start)


def _subsequence_end(haystack: Sequence[int], needle: Sequence[int]) -> int:
    """Index just past the first occurrence of `needle` in `haystack`; 0 if absent.

    Args:
        haystack: The token ids to search.
        needle: The token ids to find (empty means "no boundary").

    Returns:
        The end offset of the first match, or 0 when there is none. 0 is
        unambiguous as "no match" because a match ending at 0 is impossible for
        a non-empty needle.
    """
    if not needle:
        return 0
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            return start + width
    return 0


class VerbatimHistoryMixin(Renderer):
    """Replays a sampled assistant turn from its ids, minus its reasoning.

    Mix in FIRST (`class Foo(VerbatimHistoryMixin, SomeRenderer)`) so both
    overrides win over the base renderer's. It inherits `Renderer` only so the
    base surface it calls (`tokenizer`, `_end_message_token`,
    `_get_generation_suffix`) is declared; it implements no stop sequences of
    its own and is never instantiated directly.
    """

    observation_clip_tokens: int = 0
    """Per-observation token clip; 0 (the class default) disables it entirely.

    Set per instance by the factories in `VERBATIM_RENDERERS` from
    `RolloutConfig.observation_clip_tokens`, so one config value reaches every
    rollout path through the single registration call site.
    """

    _close_think_ids: list[int] | None = None
    """Cached `</think>` token ids; resolved once per renderer instance."""

    @property
    def has_extension_property(self) -> bool:
        """False: turn N's reasoning is absent from turn N+1's prompt.

        The extension property means every successive generation prompt is a
        token PREFIX of the next (prompt(N) plus sampled(N) is the start of
        prompt(N+1)), which is what lets a whole episode merge into one
        training datum and one KV cache.

        This class was introduced to provide exactly that, and now deliberately
        gives it up: a historical turn replays without its reasoning, so
        sampled(N) does not appear in prompt(N+1) and the sequences diverge at
        every turn boundary that had a `</think>`. What that buys is the whole
        point of the reversal (module docstring): 9 of 19 measured trials sat
        over 60,000 tokens against a 65,530 ceiling and 8 of 17 holdout trials
        died of it, and dropping reasoning from history brings them to roughly
        half their context. The cost is one training datum per turn and
        re-prefilled shared context, quadratic in turn count.

        Reported honestly rather than left True, because the cookbook's own
        merge and supervised paths gate on this flag; claiming a property that
        does not hold would silently merge sequences that are not extensions.
        """
        return False

    @property
    def close_think_ids(self) -> list[int]:
        """The token ids of `</think>` under this model's tokenizer.

        The reasoning boundary is located by TOKEN ID, in the sampled ids, so no
        step of the drop ever decodes and re-encodes content (which could
        retokenize it into different ids than the sampler issued). For every
        model in this lineup this is a single special-token id; a tokenizer that
        splits it into pieces still works, since the search is over the id
        subsequence.
        """
        if self._close_think_ids is None:
            self._close_think_ids = list(
                self.tokenizer.encode(CLOSE_THINK, add_special_tokens=False)
            )
        return self._close_think_ids

    def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
        """Parse a sampled turn into text-only content that carries its ids.

        The base renderer does the family-specific work (thinking/tool-call
        block splitting, separator-newline trimming, XML tool-call conversion);
        this narrows the result to the text parts so harbor's terminus-2
        parsers see the `str` they expect, and attaches the sampled ids plus the
        offset where the turn's post-reasoning content begins.

        The boundary is the FIRST `</think>` token, which is where the parsers
        close the block too (`parse_content_blocks` matches
        `<think>(.*?)</think>` non-greedily). For Qwen3.5 and Nemotron-3 the
        OPENING `<think>` is prefilled by the generation header and is not part
        of the sampled ids at all, so the sampled ids for a reasoning turn start
        inside the block and the first `</think>` closes it.

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
        message["content"] = VerbatimContent(
            get_text_content(message), sampled, _subsequence_end(sampled, self.close_think_ids)
        )
        return message, termination

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render one message: replay an assistant turn, clip an observation.

        An assistant turn whose reasoning was dropped renders under the header
        the base renderer gives a HISTORICAL assistant message, which is the
        framing the family's own template uses for a turn whose thinking is not
        shown (`<|im_start|>assistant\\n` for Qwen3.5, plus an empty
        `<think></think>` for Nemotron-3). Its output is the post-reasoning
        sampled ids, untouched.

        A turn with no reasoning block to drop keeps the previous behavior
        exactly: the header is the generation suffix rebuilt under the context
        that was live when the turn was sampled (the turn WAS the last message,
        and the last user message was the one before it), so nothing was
        dropped and that turn's prompt is still reproduced token for token.

        Both branches are stable across later turns, which is what makes the
        recorded sampling prompt reproducible: an assistant message's index
        never changes and, once a later user message exists, it stays
        historical, so re-rendering it at turn N+5 emits the same tokens it did
        at turn N+1.

        Args:
            message: The message to render.
            ctx: Its position in the conversation being rendered.

        Returns:
            The rendered header and output. A message without verbatim ids
            (every system turn, plus assistant turns that did not terminate
            cleanly) falls through to the base renderer, and an observation
            message may be clipped (`_clipped_observation`).
        """
        content = message.get("content")
        if message["role"] != "assistant" or not isinstance(content, VerbatimContent):
            return self._clipped_observation(message, ctx)
        if content.content_start == 0:
            sampled_ctx = RenderContext(
                idx=ctx.idx,
                is_last=True,
                prev_message=ctx.prev_message,
                last_user_index=ctx.idx - 1,
            )
            header = tinker.EncodedTextChunk(
                tokens=list(self._get_generation_suffix("assistant", sampled_ctx))
            )
        else:
            # The base render's HEADER only (its output re-encodes the text view,
            # which is exactly what must not reach the context); taking it from
            # the base renderer keeps the family-specific framing in one place.
            header = super().render_message(message, ctx).header
        return RenderedMessage(
            header=header,
            output=[tinker.EncodedTextChunk(tokens=content.content_ids)],
        )

    def _clipped_observation(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render a non-assistant message, clipping a long command observation.

        Under terminus-2 every user message that FOLLOWS an assistant turn is
        one command batch's terminal output, so that is the message this clips;
        the conversation's opening user message (the task instruction plus the
        initial terminal state) is never clipped, because losing the
        instruction would change the task. The clip keeps the head and the tail
        (the tail carries the shell prompt the model reads state from, and the
        template's end-of-turn framing) and puts a marker naming the dropped
        count in the middle.

        Measured over one TerminalBench-2 run: 770 observations, median 349
        tokens, p90 1,222, max 6,096. Harbor's own per-observation cap
        (`Terminus2._limit_output_length`) is 10,000 BYTES, which is far too
        loose in token terms to bound that tail, so this is the clip that
        binds. It exists for the one measured trial that the reasoning drop
        alone does not rescue (`make-mips-interpreter`, 64,228 -> 62,086
        tokens), whose context is output rather than reasoning.

        Args:
            message: The message to render.
            ctx: Its position in the conversation being rendered.

        Returns:
            The base render, with the output token stream clipped when this is
            an observation over budget. Anything else is the base render
            unchanged.
        """
        rendered = super().render_message(message, ctx)
        budget = self.observation_clip_tokens
        previous = ctx.prev_message
        if (
            budget <= 0
            or message["role"] not in ("user", "tool")
            or previous is None
            or previous["role"] != "assistant"
        ):
            return rendered
        tokens: list[int] = []
        for chunk in rendered.output:
            chunk_tokens = getattr(chunk, "tokens", None)
            if chunk_tokens is None:
                # A non-text chunk (an image) has no token stream to clip; the
                # tinker provider is text-only, so this is unreachable for a
                # terminus-2 rollout and is left alone rather than guessed at.
                return rendered
            tokens.extend(chunk_tokens)
        if len(tokens) <= budget:
            return rendered
        head = budget // 2
        tail = budget - head
        marker = self.tokenizer.encode(
            OBSERVATION_CLIP_MARKER.format(dropped=len(tokens) - budget),
            add_special_tokens=False,
        )
        logger.info(
            "clipped a %d-token observation to %d tokens plus a marker "
            "(rollout.observation_clip_tokens = %d)",
            len(tokens),
            budget,
            budget,
        )
        return RenderedMessage(
            header=rendered.header,
            output=[
                tinker.EncodedTextChunk(
                    tokens=[*tokens[:head], *marker, *tokens[len(tokens) - tail :]]
                )
            ],
        )


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


VerbatimBuilder = Callable[[Tokenizer, "ImageProcessor | None", int], Renderer]
"""Builds one verbatim renderer: (tokenizer, image processor, observation clip)."""


def _build_qwen3(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None, observation_clip_tokens: int
) -> Renderer:
    """Build the Qwen3 verbatim renderer (text only, no image processor)."""
    renderer = Qwen3VerbatimRenderer(tokenizer)
    renderer.observation_clip_tokens = observation_clip_tokens
    return renderer


def _build_qwen3_5(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None, observation_clip_tokens: int
) -> Renderer:
    """Build the Qwen3.5/Qwen3.6 verbatim renderer."""
    renderer = Qwen3_5VerbatimRenderer(tokenizer, image_processor=image_processor)
    renderer.observation_clip_tokens = observation_clip_tokens
    return renderer


def _build_nemotron3(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None, observation_clip_tokens: int
) -> Renderer:
    """Build the Nemotron-3 Nano/Super verbatim renderer."""
    renderer = Nemotron3VerbatimRenderer(tokenizer, image_processor=image_processor)
    renderer.observation_clip_tokens = observation_clip_tokens
    return renderer


def _build_nemotron3_ultra(
    tokenizer: Tokenizer, image_processor: ImageProcessor | None, observation_clip_tokens: int
) -> Renderer:
    """Build the Nemotron-3 Ultra verbatim renderer."""
    renderer = Nemotron3UltraVerbatimRenderer(tokenizer, image_processor=image_processor)
    renderer.observation_clip_tokens = observation_clip_tokens
    return renderer


VERBATIM_RENDERERS: dict[str, VerbatimBuilder] = {
    QWEN3_VERBATIM: _build_qwen3,
    QWEN3_5_VERBATIM: _build_qwen3_5,
    NEMOTRON3_VERBATIM: _build_nemotron3,
    NEMOTRON3_ULTRA_VERBATIM: _build_nemotron3_ultra,
}
"""Every wmh verbatim renderer name, mapped to its builder.

Name these from `[rollout.renderers]` in a run config, keyed by the base model
whose rollouts should use them (`student.base_model`, `teacher.model`).
"""


def register_verbatim_renderers(observation_clip_tokens: int = 0) -> None:
    """Register every wmh verbatim renderer with the cookbook's global registry.

    Idempotent: registration is a dict assignment keyed by name, so repeated
    calls leave the registry in the same state. Call it before anything
    resolves a renderer by name (terminus-2 resolves `llm_kwargs.renderer_name`
    inside its own `TinkerLLM`, in this process).

    Args:
        observation_clip_tokens: The per-observation token clip every renderer
            built from this registry applies (`RolloutConfig
            .observation_clip_tokens`); 0 disables it. Passed here rather than
            read from a global, because the cookbook builds renderers from a
            name alone and this is the only place a run's config can reach that
            construction. It MUST be the same value in every arm of a
            comparison (student-before baseline, teacher baseline, training,
            post-training eval): a clip that changes between arms changes the
            observations the model reads, so the arms would no longer be
            measuring the same task. `wmh.distill.rollouts
            .terminus_2_agent_kwargs` is the single call site every arm's
            rollouts pass through, which is what makes that hold.
    """
    for name, builder in VERBATIM_RENDERERS.items():
        register_renderer(name, _clip_bound(builder, observation_clip_tokens))


def _clip_bound(
    builder: VerbatimBuilder, observation_clip_tokens: int
) -> Callable[[Tokenizer, ImageProcessor | None], Renderer]:
    """Bind an observation clip into a builder, giving the cookbook's factory shape."""

    def factory(tokenizer: Tokenizer, image_processor: ImageProcessor | None = None) -> Renderer:
        """Build the renderer the cookbook asked for, with this run's clip."""
        return builder(tokenizer, image_processor, observation_clip_tokens)

    return factory


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
    if name in VERBATIM_RENDERERS or is_renderer_registered(name):
        return True
    try:
        get_renderer(name, cast("Tokenizer", _RendererNameProbe()))
    except RendererError:
        return False
    return True

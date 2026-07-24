"""Tinker sampling provider: serves the pi agent from a Tinker LoRA student.

During distillation rollouts the student model lives on Tinker. This provider
renders each structured chat request to token ids with the base model's
cookbook renderer, samples from a Tinker sampling client, and parses the
sampled tokens back into an OpenAI-style chat response for the pi bridge.

Every successful completion records a `TokenSpan` (exact prompt ids, sampled
ids, and per-token logprobs) into an optional `TokenRecorder`. Downstream
training consumes ONLY these recorded ids (tokens-in tokens-out); text is
never re-encoded, and a sample without per-token logprobs fails loudly.

Multi-turn prompts are built incrementally from the episode's own token
history: agents re-serialize earlier assistant turns (reformatted tool-call
JSON, collapsed think framing), so re-rendering the full history never
byte-matches the tokens actually sampled and every turn would fragment into
its own training datum. Instead the next prompt is (previous prompt + raw
sampled ids + a rendered suffix of only the new messages); a genuine history
edit falls back to a full re-render and is counted on the recorder.

`config.model_type` carries the base model name (renderer and tokenizer
identity); `config.model` carries either a `tinker://` sampler-weights path or
a base model name for an untrained student. The tinker SDK is an optional
extra imported lazily (`uv sync --extra distill`), same contract as e2b.

Every SDK call is deadline-bounded (`wmh.distill.deadlines`): a wedged
session raises a retryable `TinkerDeadlineError` instead of hanging, and the
provider drops its lazily built sampling client on expiry so the retry
wrapper's next attempt heals through a fresh session.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from llm_waterfall.types import ChatChoice, ChatMessage, ChatTool, ChatUsage
from pydantic import BaseModel, model_validator

from wmh.distill.deadlines import TinkerDeadlineError, call_with_deadline, wait_with_deadline
from wmh.distill.rendering import (
    ChatRendering,
    ParsedAssistantMessage,
    RendererTokenizer,
    build_renderer,
)
from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    ChatRequest,
    ChatResponse,
    Completion,
    Message,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    import tinker

logger = logging.getLogger(__name__)

TINKER_API_KEY_ENV = "TINKER_API_KEY"

_MISSING_TINKER_EXTRA = (
    "the tinker SDK is not installed; run `uv sync --extra distill` to use the tinker provider"
)

# The tinker SamplingParams default; used when a structured request carries no
# temperature (pi normally stamps one on every request).
_DEFAULT_CHAT_TEMPERATURE = 1.0


class TokenSpan(BaseModel):
    """The exact tokens one sampling call consumed and produced.

    This is the tokens-in-tokens-out ground truth for one completion: training
    data is assembled from these ids verbatim, never from re-encoded text.
    """

    call_index: int
    """0-based index of the successful completion within one episode."""

    prompt_token_ids: list[int]
    sampled_token_ids: list[int]
    sampled_logprobs: list[float]
    """Sampler-assigned logprob for each sampled token, aligned one to one."""

    @model_validator(mode="after")
    def _check_alignment(self) -> TokenSpan:
        if len(self.sampled_logprobs) != len(self.sampled_token_ids):
            raise ValueError(
                f"sampled_logprobs length {len(self.sampled_logprobs)} does not match "
                f"sampled_token_ids length {len(self.sampled_token_ids)}"
            )
        return self


class TokenRecorder:
    """Collects the `TokenSpan`s of ONE episode/trial.

    Ownership contract: one recorder per episode, driven by a single thread
    (the pi agent issues its completions sequentially). Create a fresh
    recorder (and a fresh sink file) per episode so `call_index` starts at 0.

    Args:
        jsonl_path: Optional sink; every recorded span is appended and flushed
            immediately, so a killed trial still leaves its captured spans on
            disk.
    """

    def __init__(self, jsonl_path: Path | None = None) -> None:
        self._spans: list[TokenSpan] = []
        self._jsonl_path = jsonl_path
        self._fallbacks = 0

    def __len__(self) -> int:
        return len(self._spans)

    def record(self, span: TokenSpan) -> None:
        """Append one span, writing through to the jsonl sink when configured."""
        self._spans.append(span)
        if self._jsonl_path is not None:
            with self._jsonl_path.open("a", encoding="utf-8") as sink:
                sink.write(span.model_dump_json() + "\n")
                sink.flush()

    def spans(self) -> list[TokenSpan]:
        """A snapshot copy of the spans recorded so far."""
        return list(self._spans)

    @property
    def fallback_count(self) -> int:
        """How many prompts fell back to a full re-render (each one fragments)."""
        return self._fallbacks

    def record_fallback(self) -> None:
        """Count one incremental-prompt fallback (genuine mid-episode history edit)."""
        self._fallbacks += 1


class SampledSequenceLike(Protocol):
    """The slice of a sampled sequence the provider consumes."""

    @property
    def tokens(self) -> list[int]:
        """Sampled token ids."""
        ...

    @property
    def logprobs(self) -> list[float] | None:
        """Per-token logprobs aligned with `tokens`, or None if unavailable."""
        ...


class TinkerSampler(Protocol):
    """The sampling call the provider makes, in token-id terms.

    `wmh.distill.fake_tinker.FakeSamplingClient` satisfies this directly;
    real `tinker.SamplingClient`s are adapted via `SdkSampler`.
    """

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> SampledSequenceLike:
        """Sample one sequence conditioned on the prompt token ids."""
        ...


@runtime_checkable
class _TokenizerSource(Protocol):
    """A sampling client that can supply the base model's tokenizer."""

    def get_tokenizer(self) -> RendererTokenizer: ...


class SdkSampler:
    """Adapts a real `tinker.SamplingClient` to the `TinkerSampler` seam.

    The provider's lazy path builds this itself; callers that already hold a
    sampling client (e.g. the distill loop refreshing student weights via
    `save_weights_and_get_sampling_client`) wrap it in this before injecting.
    """

    def __init__(self, client: tinker.SamplingClient) -> None:
        self._client = client

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> SampledSequenceLike:
        """Run one deadline-bounded sample and return the single sampled sequence.

        Raises:
            TinkerDeadlineError: If the sample deadline expires (the session
                is likely wedged; the caller should retry with a fresh one).
        """
        import tinker

        future = self._client.sample(
            prompt=tinker.ModelInput.from_ints(prompt_token_ids),
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=max_tokens, temperature=temperature, stop=stop
            ),
        )
        return wait_with_deadline("sample", future).sequences[0]

    def get_tokenizer(self) -> RendererTokenizer:
        """The HF tokenizer for the client's base model (deadline-bounded fetch)."""
        # HF stubs type decode as `str | list[str]` depending on the input
        # shape; for the list[int] calls renderers make it is always str.
        return cast("RendererTokenizer", call_with_deadline("connect", self._client.get_tokenizer))


class _SampledTurn(BaseModel):
    """One successful render-sample-parse round trip (internal)."""

    prompt_token_ids: list[int]
    sampled_token_ids: list[int]
    parsed: ParsedAssistantMessage


@dataclass
class _PromptState:
    """The provider's last successful call, for incremental prompt extension.

    Shares the recorder's single-episode ownership: one provider serves one
    episode sequentially, so the next call's history normally extends this
    call's message list by the assistant echo plus new tool/user messages.
    """

    messages: list[ChatMessage]
    """Snapshot of the message list the last prompt was built from."""

    tool_signature: str | None
    """Normalized digest of the tool schemas the last prompt rendered with."""

    prompt_tokens: list[int]
    """The exact prompt ids sent on the last call (incremental or full)."""

    sampled_tokens: list[int]
    """The raw sampled ids of the last call, including any end-of-turn token."""


def _tool_signature(tools: list[ChatTool] | None) -> str | None:
    """A normalized digest of tool schemas; None when no tools are rendered."""
    if not tools:
        return None
    return json.dumps([tool.model_dump(mode="json") for tool in tools], sort_keys=True)


def _first_incompatible_index(
    previous: list[ChatMessage], incoming: list[ChatMessage]
) -> int | None:
    """Where `incoming` stops being a tolerant extension of `previous`, or None.

    The shared region must match role for role and count for count. Assistant
    turns are compared by role only: the agent re-serializes the provider's
    own turns (parsed and reformatted tool calls, collapsed think framing), so
    their text never byte-matches and the provider's token history is the
    ground truth for them. System, user, and tool messages must match exactly
    by content and tool linkage. When `incoming` is longer, the first new
    message must be the assistant echo of the provider's last sampled turn.

    Returns:
        None when compatible; otherwise the index of the first message that
        breaks the extension (a genuine history edit or compaction).
    """
    if len(incoming) < len(previous):
        return len(incoming)
    for index, (prev, cur) in enumerate(zip(previous, incoming, strict=False)):
        if prev.role != cur.role:
            return index
        if prev.role == "assistant":
            continue
        if prev.content != cur.content:
            return index
        if prev.tool_call_id != cur.tool_call_id:
            return index
        if (prev.model_extra or {}).get("name") != (cur.model_extra or {}).get("name"):
            return index
    if len(incoming) > len(previous) and incoming[len(previous)].role != "assistant":
        return len(previous)
    return None


class TinkerChatProvider:
    """Serves completions from a Tinker-hosted LoRA student during distillation.

    Args:
        config: Provider config; `model_type` is the base model name and
            `model` is the `tinker://` sampler-weights path (or a base model
            name for an untrained student).
        sampling_client: Optional injected sampler (tests use the fakes in
            `wmh.distill.fake_tinker`; wrap a real `tinker.SamplingClient` in
            `SdkSampler`). When None, a real client is built lazily from
            `config.model` on first use and dropped after a
            `TinkerDeadlineError` so the next attempt rebuilds a fresh
            session; an injected client is never dropped.
        renderer: Optional injected rendering. When None, it is built lazily
            from the base model name and the sampling client's tokenizer.
        recorder: Optional per-episode span recorder; when present, every
            successful completion records exactly one `TokenSpan`.
        api_key: Accepted for `get_provider`'s explicit-credential channel and
            REJECTED when set. The Tinker SDK reads `TINKER_API_KEY` from the
            process environment when it builds a `ServiceClient`; it takes no
            per-client credential, so a pool entry's key cannot be honored
            here. Failing loudly beats silently sampling on a different
            account than the pool entry named.

    Raises:
        ValueError: If `api_key` is set.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        sampling_client: TinkerSampler | None = None,
        renderer: ChatRendering | None = None,
        recorder: TokenRecorder | None = None,
        api_key: str | None = None,
    ) -> None:
        # get_provider promises the backend authenticates with exactly this key
        # (wmh/providers/registry.py). Tinker structurally cannot, so refuse
        # rather than break that contract quietly.
        if api_key is not None:
            raise ValueError(
                "the tinker provider does not accept an explicit api_key: the SDK reads "
                f"{TINKER_API_KEY_ENV} from the process environment, so a per-entry key "
                f"cannot be honored; drop api_key_env from the pool entry and export "
                f"{TINKER_API_KEY_ENV} instead"
            )
        self.config = config
        self._sampler = sampling_client
        # Only a client the provider built itself may be dropped and rebuilt
        # after a deadline expiry; an injected one cannot be reconstructed.
        self._owns_sampler = sampling_client is None
        self._rendering = renderer
        self._recorder = recorder
        self._prompt_state: _PromptState | None = None

    def _base_model_name(self) -> str:
        base = self.config.model_type or self.config.model
        if base.startswith("tinker://"):
            if self.config.model_type:
                raise ValueError(
                    "config.model_type is a tinker:// weights path; set model_type to "
                    "the base model name (e.g. 'Qwen/Qwen3-8B') so the renderer and "
                    "tokenizer can be resolved (weights paths belong in config.model)"
                )
            raise ValueError(
                "config.model is a tinker:// weights path and config.model_type is "
                "unset; set model_type to the base model name (e.g. 'Qwen/Qwen3-8B') "
                "so the renderer and tokenizer can be resolved"
            )
        return base

    def _get_sampler(self) -> TinkerSampler:
        if self._sampler is None:
            self._sampler = self._build_sdk_sampler()
        return self._sampler

    def _build_sdk_sampler(self) -> TinkerSampler:
        # Lazy: don't import the SDK or read the key env var until first use.
        try:
            import tinker
        except ImportError as exc:
            raise ImportError(_MISSING_TINKER_EXTRA) from exc
        if not os.environ.get(TINKER_API_KEY_ENV):
            raise RuntimeError(
                f"{TINKER_API_KEY_ENV} is not set in the environment; set it to "
                "your Tinker API key to use the tinker provider"
            )

        def build() -> tinker.SamplingClient:
            service = tinker.ServiceClient()
            if self.config.model.startswith("tinker://"):
                return service.create_sampling_client(model_path=self.config.model)
            return service.create_sampling_client(base_model=self.config.model)

        return SdkSampler(call_with_deadline("connect", build))

    def _drop_wedged_sampler(self) -> None:
        """Forget a lazily built sampling client after a deadline expiry.

        A wedged session keeps timing out while a freshly built one heals
        (observed live), so dropping here makes the retry wrapper's next
        attempt rebuild through `_get_sampler`. An injected client is never
        dropped: the provider cannot rebuild what it did not build.
        """
        if self._owns_sampler and self._sampler is not None:
            logger.warning(
                "dropping the tinker sampling client after a deadline expiry; "
                "the next attempt builds a fresh session"
            )
            self._sampler = None

    def _get_rendering(self) -> ChatRendering:
        if self._rendering is None:
            base_model = self._base_model_name()
            sampler = self._get_sampler()
            if not isinstance(sampler, _TokenizerSource):
                raise RuntimeError(
                    "the injected sampling client exposes no get_tokenizer(); pass "
                    "renderer= explicitly when constructing TinkerChatProvider with "
                    "a custom sampling client"
                )
            self._rendering = build_renderer(base_model, sampler.get_tokenizer())
        return self._rendering

    def _build_prompt_tokens(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None
    ) -> list[int]:
        """Build the prompt ids, extending the episode's own token history when possible.

        When the previous call's message list is a tolerant prefix of the
        incoming one (see `_first_incompatible_index`) and the tool schemas
        are unchanged, the prompt is the previous prompt plus the raw sampled
        ids plus a rendered suffix of only the NEW messages, so it extends
        (previous prompt + previous sample) verbatim as a token prefix and the
        episode merges into one training datum. An identical-length compatible
        history (the caller discarded the last turn and re-asks) reuses the
        previous prompt unchanged. Anything else (a genuine history edit or
        compaction) falls back to a full re-render, which is counted on the
        recorder because every fallback fragments the episode's datums.
        """
        rendering = self._get_rendering()
        state = self._prompt_state
        if state is None:
            return rendering.build_generation_prompt(messages, tools)
        signature = _tool_signature(tools)
        mismatch = _first_incompatible_index(state.messages, messages)
        if mismatch is None and signature == state.tool_signature:
            if len(messages) == len(state.messages):
                return list(state.prompt_tokens)
            suffix = rendering.render_suffix(
                messages,
                len(state.messages) + 1,
                tools,
                previous_sampled_ids=state.sampled_tokens,
            )
            return state.prompt_tokens + state.sampled_tokens + suffix
        if self._recorder is not None:
            self._recorder.record_fallback()
        if mismatch is not None:
            logger.info(
                "incremental prompt fallback: incoming message %d does not extend the "
                "previous call's history (genuine edit or compaction); re-rendering the "
                "full prompt, which fragments this episode's training datums",
                mismatch,
            )
        else:
            logger.info(
                "incremental prompt fallback: the tool schemas changed since the previous "
                "call; re-rendering the full prompt, which fragments this episode's "
                "training datums"
            )
        return rendering.build_generation_prompt(messages, tools)

    def _sample_turn(
        self,
        messages: list[ChatMessage],
        tools: list[ChatTool] | None,
        *,
        temperature: float,
        max_tokens: int,
    ) -> _SampledTurn:
        """Render, sample, parse, and (on success) record exactly one span."""
        try:
            rendering = self._get_rendering()
            prompt_ids = self._build_prompt_tokens(messages, tools)
            sequence = self._get_sampler().sample(
                prompt_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=rendering.stop_sequences,
            )
        except TinkerDeadlineError:
            # The session is likely wedged; drop it so the retry wrapper's
            # next attempt rebuilds fresh. No span was recorded (recording
            # happens only after the whole completion succeeds, below).
            self._drop_wedged_sampler()
            raise
        sampled_ids = list(sequence.tokens)
        logprobs = sequence.logprobs
        if logprobs is None or len(logprobs) != len(sampled_ids):
            got = (
                "no logprobs"
                if logprobs is None
                else f"{len(logprobs)} logprobs for {len(sampled_ids)} tokens"
            )
            raise RuntimeError(
                f"tinker sampling returned {got}; per-token logprobs are required "
                "for tokens-in-tokens-out training and are never fabricated"
            )
        parsed = rendering.parse_response(sampled_ids)
        # Update the incremental state and record only after the whole completion
        # succeeded, so a failure that an outer retry wrapper re-invokes never
        # leaves a span (or stale prompt state) behind.
        self._prompt_state = _PromptState(
            messages=list(messages),
            tool_signature=_tool_signature(tools),
            prompt_tokens=prompt_ids,
            sampled_tokens=sampled_ids,
        )
        if self._recorder is not None:
            self._recorder.record(
                TokenSpan(
                    call_index=len(self._recorder),
                    prompt_token_ids=prompt_ids,
                    sampled_token_ids=sampled_ids,
                    sampled_logprobs=list(logprobs),
                )
            )
        return _SampledTurn(
            prompt_token_ids=prompt_ids, sampled_token_ids=sampled_ids, parsed=parsed
        )

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        """Serve one structured agent request from the student sampler."""
        temperature = (
            request.temperature if request.temperature is not None else _DEFAULT_CHAT_TEMPERATURE
        )
        max_tokens = request.max_completion_tokens or request.max_tokens or DEFAULT_MAX_TOKENS
        if (request.model_extra or {}).get("stop") is not None:
            logger.debug(
                "ignoring request-supplied stop sequences; the renderer's stop "
                "sequences are authoritative for token-exact sampling"
            )
        # tool_choice is honored where token sampling can express it ("none" renders
        # without tool schemas) and rejected loudly where it cannot ("required" or a
        # named function would need constrained decoding the sampler does not offer).
        tools = request.tools
        choice = request.tool_choice
        if choice == "none":
            tools = None
        elif choice is not None and choice != "auto":
            raise ValueError(
                f"unsupported tool_choice {choice!r}: the tinker provider samples raw "
                "tokens and cannot force the student to call a tool; use 'auto', "
                "'none', or omit tool_choice"
            )
        turn = self._sample_turn(
            request.messages, tools, temperature=temperature, max_tokens=max_tokens
        )
        parsed = turn.parsed
        if parsed.tool_calls:
            finish_reason = "tool_calls"
        elif parsed.stopped:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        message = ChatMessage(
            role="assistant",
            content=parsed.text or None,
            tool_calls=parsed.tool_calls or None,
        )
        return ChatResponse(
            choices=[ChatChoice(index=0, message=message, finish_reason=finish_reason)],
            usage=ChatUsage(
                prompt_tokens=len(turn.prompt_token_ids),
                completion_tokens=len(turn.sampled_token_ids),
            ),
            model=self.config.model,
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Completion:
        """Plain-text completion through the same render-sample-parse machinery."""
        chat_messages: list[ChatMessage] = []
        if system:
            chat_messages.append(ChatMessage(role="system", content=system))
        chat_messages.extend(ChatMessage(role=m.role, content=m.content) for m in messages)
        turn = self._sample_turn(
            chat_messages, None, temperature=temperature, max_tokens=max_tokens
        )
        return Completion(
            text=turn.parsed.text,
            usage=TokenUsage(
                input_tokens=len(turn.prompt_token_ids),
                output_tokens=len(turn.sampled_token_ids),
            ),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Tinker has no embeddings API; configure a dedicated embedder instead."""
        raise ValueError(
            "the tinker provider has no embeddings API; configure a dedicated "
            "embedder (hashing, openai, azure, or bedrock) for retrieval instead"
        )

    def verify(self) -> VerifyResult:
        """One-token sample through the real render+sample path (never recorded)."""
        return verify_via_ping(self, ping=self._ping)

    def _ping(self) -> None:
        try:
            rendering = self._get_rendering()
            prompt_ids = rendering.build_generation_prompt(
                [ChatMessage(role="user", content="ping")]
            )
            self._get_sampler().sample(
                prompt_ids, max_tokens=1, temperature=0.0, stop=rendering.stop_sequences
            )
        except TinkerDeadlineError:
            self._drop_wedged_sampler()
            raise

"""Gateway protocol capability declarations for one provider deployment.

Split from :mod:`exp.common.models.catalog` for the module line budget: catalog
loading still owns connections and records, while this declaration can evolve
with the gateway protocol without invalidating frozen router artifacts.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.common.models.model import ReasoningEffort


class GatewayDeploymentCapabilities(ContractModel):
    """Gateway protocol capabilities declared for one provider deployment.

    These fields are intentionally separate from ``ModelCapabilities``. The latter participates
    in frozen optimizer and runtime identities, while this declaration can evolve with the
    gateway protocol without invalidating existing router artifacts.
    """

    supports_developer_messages: bool = False
    supports_streaming: bool = False
    supports_streaming_tool_arguments: bool = False
    supports_strict_tools: bool = False
    supports_parallel_tool_calls: bool = False
    supports_custom_tools: bool = False
    """Whether this deployment's relevant native wire can preserve free-form custom tools.

    False means the capability is not declared. Public Chat still refuses
    custom tools even when a Responses-native deployment declares this,
    because Chat accepts function tools only. Read the flag with the parity
    row's dialect and the caller's public API surface.
    """
    supports_grammar_tools: bool = False
    """Whether grammar-constrained custom tools can be preserved on this deployment.

    This requires ``supports_custom_tools``. False means the capability is
    not declared; it does not describe Chat, which still refuses custom and
    grammar tools.
    """
    supports_tool_call_limit: bool = False
    """Whether a caller's Responses ``max_tool_calls`` limit can be preserved.

    False means the capability is not declared. The public Responses
    surface currently refuses the field, so no authored catalog should set
    this until a route honors the cap.
    """
    supports_structured_text: bool = False
    supports_stop_sequences: bool = False
    supports_image_input: bool = False
    """Whether this deployment's wire and model can carry caller image parts.

    Image input is declaration-driven and never assumed: a route that does
    not declare it rejects an image request at admission, so a picture is
    never dropped and answered from the surrounding text alone.
    """
    supports_image_url_input: bool = False
    """Whether this route's provider fetches a caller image URL itself.

    Inline base64 rides every image-capable wire, but only some wires accept a
    remote URL. A route that does not declare this rejects a URL image at
    admission, which lets a waterfall narrow to a rung that can carry it.
    """
    supports_video_input: bool = False
    """Whether this deployment's wire and model can carry caller video parts.

    Video is narrower than images: only the Gemini, Bedrock Converse, and
    OpenAI-compatible ``video_url`` wires define a video carrier, and only
    some models on those wires accept one. Like images the declaration is
    never assumed, so a route without it rejects a video at admission rather
    than answering from the surrounding text.
    """
    supports_video_url_input: bool = False
    """Whether this route's provider fetches a caller video URL itself.

    Bedrock accepts inline bytes (or an S3 location the gateway does not
    author) only; Gemini and the OpenAI-compatible video wires fetch an
    http(s) URL on the caller's behalf.
    """
    supports_audio_input: bool = False
    """Whether this deployment's wire and model can carry caller audio parts.

    Audio is the narrowest attachment: only the OpenAI-compatible Chat
    ``input_audio`` wire and the Gemini ``inline_data`` wire carry a clip a
    model serves, and on those wires only specific models (the gpt-audio
    family, audio-capable Gemini models) accept one. The declaration is never
    assumed, so a route without it rejects audio at admission rather than
    answering from the surrounding text. Audio has no remote URL carrier on
    any public surface, so there is no separate URL declaration.
    """
    supports_pdf_input: bool = False
    """Whether this deployment's wire and model can carry caller PDF documents.

    Like image input this is declaration-driven and never assumed: a route
    that does not declare it rejects a document request at admission, so a
    PDF is never dropped and answered from the surrounding text alone.
    """
    supports_pdf_url_input: bool = False
    """Whether this route's provider fetches a caller PDF URL itself.

    Only the OpenAI Responses (``file_url``) and Anthropic Messages (``url``
    source) wires fetch a remote document; Chat Completions ``file`` parts,
    Gemini, and Bedrock accept inline bytes only.
    """
    supports_prompt_cache_boundaries: bool = False
    """Whether explicit caller-selected prompt-cache boundaries can be preserved.

    This covers Chat ``prompt_cache_options`` and ``prompt_cache_retention``.
    False means those explicit boundaries are not declared as preserved; it
    does not mean implicit prefix caching is absent.
    """
    supports_media_handle_input: bool = False
    """Whether this route forwards handles to media the caller uploaded to its provider.

    A handle (an OpenAI or Anthropic ``file_id``, a Gemini Files URI, a
    ``gs://`` object on Vertex, an ``s3://`` object on Bedrock) is scoped to
    the provider that minted it and never portable, so admission requires
    both this declaration and a handle provider equal to the route's
    provider. Providers whose inference wire defines no uploaded-media
    reference (Fireworks, OpenRouter) never declare it.
    """
    maximum_stop_sequences: int | None = Field(default=None, ge=1)
    """Largest stop-sequence count this route accepts, when the provider caps it.

    ``None`` leaves the count unbounded (only ``supports_stop_sequences`` gates the
    field). A concrete value lets admission reject an over-limit list locally with a
    named parameter error instead of forwarding it and surfacing the provider's
    opaque 4xx (e.g. Gemini caps ``stopSequences`` at 5)."""
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    """Exact caller values this deployment can preserve without normalization.

    An empty tuple means the gateway should use its maintained provider-family
    contract. OpenRouter and other catalog-driven providers declare the exact
    ordered set here because their supported values vary by model.
    """
    reasoning_default_effort: ReasoningEffort | None = None
    """Explicit provider default used only when the wire requires this field."""
    reasoning_effort_required: bool = False
    """Whether this deployment requires an explicit reasoning effort on its wire."""
    reports_refusals: bool = False
    reports_cached_input_tokens: bool = False
    reports_reasoning_tokens: bool = False
    reports_model_status: bool = False
    """Whether provider model-status metadata is preserved in the normalized response.

    False means the capability is not declared. Gemini ``modelStatus`` is
    currently unpreserved, so no authored catalog should set this until the
    response contract carries that field.
    """
    supports_async_tools: bool = False
    """Whether a tool may be flagged ``async`` so the model keeps generating
    while the caller runs it, with the result returned later on the tool call's
    ORIGINAL ``call_id`` (GPT-6 Astra Responses). Declaration-driven and off
    until the decoder + turn lifecycle honor it; a route that declares it must
    not drop an async tool call. See the platform's astra_responses helpers."""
    supports_mid_turn_steering: bool = False
    """Whether the caller may inject additional input over the Responses
    WebSocket WHILE the model is working, folded into a continuation that
    preserves completed work (GPT-6 Astra). Off until the WS transport accepts
    inbound mid-turn frames."""
    supports_reasoning_effort_update: bool = False
    """Whether a ``configuration_update`` input item may change reasoning effort
    mid-conversation without invalidating the cached prompt prefix -- the
    request-level ``reasoning.effort`` stays fixed (GPT-6 Astra). Off until the
    decoder recognizes the item (it must not hit the unknown-item reject path)
    and applies the effort forward."""
    time_to_first_byte_base_seconds: float | None = Field(default=None, gt=0)
    """Deployment override for the lane's flat time-to-first-byte allowance.

    ``None`` uses the serving configuration's default. The effective bound on
    the wait for a provider's response headers is this base plus the
    input-scaled allowance below, so very large prompts are not misread as a
    dead lane.
    """
    time_to_first_byte_seconds_per_million_input_tokens: float | None = Field(default=None, ge=0)
    """Deployment override for the input-scaled time-to-first-byte allowance.

    Seconds added per million approximate input tokens (the request body's
    bytes divided by four; an allowance heuristic, never a billing quantity).
    ``None`` uses the serving configuration's default; ``0`` disables scaling
    for this deployment.
    """

    @property
    def declares_reasoning_contract(self) -> bool:
        """Whether this metadata overrides provider-family reasoning behavior."""
        return bool(
            self.supported_reasoning_efforts
            or self.reasoning_default_effort is not None
            or self.reasoning_effort_required
        )

    @model_validator(mode="after")
    def _require_custom_tools_for_grammar(self) -> GatewayDeploymentCapabilities:
        """Reject grammar-tool support that is not backed by custom-tool support.

        Returns:
            The validated declaration.

        Raises:
            ValueError: ``supports_grammar_tools`` is true while
                ``supports_custom_tools`` is false.
        """
        if self.supports_grammar_tools and not self.supports_custom_tools:
            raise ValueError("supports_grammar_tools requires supports_custom_tools=true")
        return self

    @model_validator(mode="after")
    def _require_valid_reasoning_contract(self) -> GatewayDeploymentCapabilities:
        """Reject ambiguous or non-canonical reasoning declarations."""
        order = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
        indexes = tuple(order.index(effort) for effort in self.supported_reasoning_efforts)
        if len(set(self.supported_reasoning_efforts)) != len(self.supported_reasoning_efforts):
            raise ValueError("supported_reasoning_efforts cannot repeat values")
        if indexes != tuple(sorted(indexes)):
            raise ValueError("supported_reasoning_efforts must use canonical order")
        if (
            self.reasoning_default_effort is not None
            and self.reasoning_default_effort not in self.supported_reasoning_efforts
        ):
            raise ValueError(
                "reasoning_default_effort must be one of the supported reasoning efforts"
            )
        if self.reasoning_effort_required and not self.supported_reasoning_efforts:
            raise ValueError(
                "reasoning_effort_required needs at least one supported reasoning effort"
            )
        if self.reasoning_effort_required and self.reasoning_default_effort is None:
            raise ValueError("reasoning_effort_required needs reasoning_default_effort")
        return self

"""Sampling-only runtime adapter for cataloged Tinker model handles."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict, cast, runtime_checkable

from exp.common.core.artifacts import ContractModel
from exp.common.models import (
    AssistantAction,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    Usage,
)
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.openai_compatible import parse_openai_wire_tool_call

if TYPE_CHECKING:
    import tinker
    from tinker_cookbook.renderers import Message as CookbookMessage
    from tinker_cookbook.renderers import Renderer as CookbookRenderer
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall
    from tinker_cookbook.renderers import ToolSpec as CookbookToolSpec


_OPTIONAL_DEPENDENCY_GUIDANCE = (
    "install the Tinker sampling dependencies with `uv sync --extra sft` or "
    "`pip install 'experiential[sft]'`"
)
_LOGGER = logging.getLogger(__name__)


class TinkerPromptMessage(TypedDict):
    """One cookbook-shaped chat message assembled for Tinker prompt rendering."""

    role: str
    content: str
    tool_calls: NotRequired[list[CookbookToolCall]]
    tool_call_id: NotRequired[str]


class TinkerOptionalDependencyError(ImportError):
    """Tinker sampling was requested without the optional sampling dependencies installed."""


class TinkerSamplingError(ValueError):
    """A completed Tinker handle could not represent a typed EXP sampling request."""


class TinkerSample(ContractModel):
    """One completed sampled Tinker turn, independent of training lifecycle state."""

    output: AssistantAction
    usage: Usage | None = None
    served_model_id: str | None = None


@runtime_checkable
class TinkerSampler(Protocol):
    """The narrow completed-handle sampling operation EXP needs at runtime."""

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Sample one complete assistant action from the completed trained handle.

        Args:
            request: Typed EXP request to render for the trained model.

        Returns:
            Parsed output and any observed token accounting.
        """


class TinkerSdkSampler:
    """Lazily sample a cataloged Tinker handle through the installed SDK.

    Construction verifies that the optional SDK and cookbook are installed, but creates no
    Tinker session. The first completion owns the SDK session and sampling request. This keeps
    capability preflight local while preserving the one runtime-owned composition path.
    """

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str | None,
    ) -> None:
        """Bind one catalog identity and credential to a future sampling session.

        Args:
            model: Resolved Tinker handle or base-model identity from the local catalog.
            api_key: Credential read from the configured environment variable.
            base_url: Optional explicit Tinker endpoint for the local connection.
        """
        if not api_key:
            raise ValueError("Tinker sampling requires a non-empty API key")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._sdk_sampler: tinker.SamplingClient | None = None
        self._renderer: CookbookRenderer | None = None

    def sample(self, request: ModelRequest) -> TinkerSample:
        """Render and sample one complete request from the trained handle.

        Args:
            request: Typed visible messages and optional tool definitions.

        Returns:
            Parsed assistant action and any token accounting Tinker exposes.

        Raises:
            TinkerSamplingError: The request or Tinker response cannot preserve the EXP contract.
        """
        if request.tool_choice is not None and request.tool_choice != "auto":
            raise TinkerSamplingError(
                "Tinker sampling supports automatic tool selection only; omit tool_choice or use "
                "'auto'"
            )
        sampler = self._get_sdk_sampler()
        renderer = self._get_renderer(sampler)
        prompt = _tinker_prompt(request, renderer)
        import tinker

        ignored = tuple(
            name
            for name, value in (
                ("top_p", request.top_p),
                ("top_k", request.top_k),
                ("logprobs", request.logprobs),
                ("top_logprobs", request.top_logprobs),
                ("reasoning_effort", request.reasoning_effort),
            )
            if value is not None
        )
        if ignored:
            _LOGGER.info(
                "omitting unsupported optional generation parameters for Tinker route: %s",
                ", ".join(ignored),
            )

        response = sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=request.maximum_output_tokens,
                temperature=request.temperature if request.temperature is not None else 1.0,
                stop=renderer.get_stop_sequences(),
            ),
        ).result()
        if len(response.sequences) != 1:
            raise TinkerSamplingError(
                "Tinker sampling returned an unexpected number of sequences for one completion"
            )
        parsed, termination = renderer.parse_response(list(response.sequences[0].tokens))
        if not termination.is_clean:
            raise TinkerSamplingError("Tinker sampling ended without a clean renderer termination")
        return TinkerSample(
            output=_assistant_action(renderer.to_openai_message(parsed)),
            served_model_id=self._model.model_id,
        )

    def _get_sdk_sampler(self) -> tinker.SamplingClient:
        """Create the SDK sampler only when a real completion is requested."""
        if self._sdk_sampler is not None:
            return self._sdk_sampler
        import tinker

        service = (
            tinker.ServiceClient(api_key=self._api_key)
            if self._base_url is None
            else tinker.ServiceClient(api_key=self._api_key, base_url=self._base_url)
        )
        if self._model.model_id.startswith("tinker://"):
            self._sdk_sampler = service.create_sampling_client(model_path=self._model.model_id)
        else:
            self._sdk_sampler = service.create_sampling_client(base_model=self._model.model_id)
        return self._sdk_sampler

    def _get_renderer(self, sampler: tinker.SamplingClient) -> CookbookRenderer:
        """Resolve the sampler's authoritative base-model renderer on first use."""
        if self._renderer is not None:
            return self._renderer
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer

        base_model = sampler.get_base_model()
        if not base_model:
            raise TinkerSamplingError("Tinker sampling did not report a base model for its handle")
        try:
            renderer_name = get_recommended_renderer_name(base_model)
            self._renderer = get_renderer(
                renderer_name,
                sampler.get_tokenizer(),
                model_name=base_model,
            )
        except (KeyError, ValueError) as exc:
            raise TinkerSamplingError(
                f"Tinker sampling has no supported cookbook renderer for base model {base_model!r}"
            ) from exc
        return self._renderer


class TinkerSamplingClient:
    """Adapts a completed Tinker sampler to the common non-streaming model protocol."""

    def __init__(self, *, model: ModelSnapshot, sampler: TinkerSampler) -> None:
        """Bind one completed trained-model identity to its sampling handle.

        Args:
            model: Catalog identity, typically with a ``tinker://`` model handle.
            sampler: Already-created sampling handle. This adapter never creates, trains, saves,
                promotes, or deploys a Tinker model.
        """
        self._model = model
        self._sampler = sampler

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Sample one action and attach local observed latency to the shared response.

        Args:
            request: Typed non-streaming request for the completed Tinker handle.

        Returns:
            A shared response carrying the served model identity and observed latency.
        """
        started_at = time.monotonic()
        sample = self._sampler.sample(request)
        return ModelResponse.completed(
            output=sample.output,
            configured_model=self._model,
            served_model_id=sample.served_model_id,
            usage=sample.usage,
            latency_seconds=time.monotonic() - started_at,
        )


def create_tinker_sampler(
    *,
    model: ModelSnapshot,
    api_key: str,
    base_url: str | None,
) -> TinkerSampler:
    """Construct EXP's default lazy sampler for one cataloged Tinker record.

    Args:
        model: Resolved handle or base-model identity from the local catalog.
        api_key: Credential already read from the record's configured environment variable.
        base_url: Optional Tinker API base URL.

    Returns:
        A sampling-only adapter that opens its session on the first completion.

    Raises:
        TinkerOptionalDependencyError: The optional SDK or cookbook is not installed.
    """
    _require_tinker_dependencies()
    return TinkerSdkSampler(model=model, api_key=api_key, base_url=base_url)


def _require_tinker_dependencies() -> None:
    """Fail clearly before runtime composition when optional Tinker dependencies are absent."""
    try:
        import tinker  # noqa: F401
        import tinker_cookbook  # noqa: F401
    except ImportError as exc:
        raise TinkerOptionalDependencyError(
            f"Tinker sampling is optional; {_OPTIONAL_DEPENDENCY_GUIDANCE}"
        ) from exc


def _tinker_prompt(request: ModelRequest, renderer: CookbookRenderer) -> tinker.ModelInput:
    """Convert the EXP request to one cookbook-rendered Tinker model input."""
    from tinker_cookbook.renderers import ToolCall as CookbookToolCall

    messages: list[TinkerPromptMessage] = []
    for message in request.messages:
        payload: TinkerPromptMessage = {
            "role": message.role,
            "content": _message_content(message.content, message.assistant_action),
        }
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id or ""
        if message.assistant_action is not None and message.assistant_action.tool_calls:
            payload["tool_calls"] = [
                CookbookToolCall(
                    id=call.call_id,
                    function=CookbookToolCall.FunctionBody(
                        name=call.name,
                        arguments=call.arguments_json(sort_keys=True),
                    ),
                )
                for call in message.assistant_action.tool_calls
            ]
        messages.append(payload)
    if not request.tools:
        return renderer.build_generation_prompt(cast("list[CookbookMessage]", messages))
    from tinker_cookbook.renderers import ToolSpec

    tool_specs: list[CookbookToolSpec] = [
        ToolSpec(
            name=tool.name,
            description=tool.description,
            parameters=tool.input_schema,
        )
        for tool in request.tools
    ]
    system_prompt = ""
    if messages and messages[0]["role"] == "system":
        system_prompt = messages.pop(0)["content"]
    return renderer.build_generation_prompt(
        renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)
        + cast("list[CookbookMessage]", messages)
    )


def _message_content(content: str | None, action: AssistantAction | None) -> str:
    """Return visible text while tool calls stay in the renderer's structured fields."""
    if content is not None:
        return content
    if action is not None and action.content is not None:
        return action.content
    return ""


def _assistant_action(value: dict[str, object]) -> AssistantAction:
    """Convert cookbook's OpenAI-shaped assistant result without inventing malformed tools."""
    content_value = value.get("content")
    content = content_value if isinstance(content_value, str) else None
    tool_values = value.get("tool_calls")
    if tool_values is None:
        tool_calls: tuple[ToolCall, ...] = ()
    elif isinstance(tool_values, list):
        tool_calls = tuple(_tool_call(item, index) for index, item in enumerate(tool_values))
    else:
        raise TinkerSamplingError("Tinker renderer returned tool_calls outside an array")
    try:
        return AssistantAction(content=content, tool_calls=tool_calls)
    except ValueError as exc:
        raise TinkerSamplingError(
            "Tinker renderer returned neither assistant text nor a complete tool call"
        ) from exc


def _tool_call(value: object, index: int) -> ToolCall:
    """Validate one renderer tool call through the shared OpenAI-wire parser."""
    try:
        return parse_openai_wire_tool_call(value, index)
    except ProviderResponseError as exc:
        raise TinkerSamplingError(f"Tinker tool call {index} is invalid: {exc}") from exc

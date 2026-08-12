"""Concrete managed Tinker cross-entropy adapter for the offline SFT runner.

The adapter accepts a prebuilt ``tinker.ServiceClient``.  It never constructs a service client,
reads an environment variable, or resolves a credential.  Application composition owns that
authorization boundary, while tests inject the runner's in-memory ``TrainerBackend`` instead.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

from wmo.common.core.artifacts import canonical_json_bytes
from wmo.common.models import AssistantAction, ToolCall
from wmo.optimize.model.sft.contracts import (
    AssistantActionEvent,
    SFTExample,
    SFTMessage,
    ToolEvent,
)
from wmo.optimize.model.sft.training_contracts import (
    TinkerSFTError,
    TinkerSFTSpec,
    TrainerBatchResult,
    TrainerDatum,
    TrainerSession,
)

if TYPE_CHECKING:
    import tinker
    from tinker_cookbook.renderers import Message, Renderer

_MISSING_TINKER_EXTRA = (
    "Tinker SFT requires the optional distill dependencies; run `uv sync --extra distill` "
    "or install `world-model-optimizer[distill]`"
)


class TinkerSFTDependencyError(TinkerSFTError):
    """The concrete Tinker backend was requested without its optional local SDK dependencies."""


class TinkerFunctionCall(TypedDict):
    """The cookbook-compatible function payload of one complete assistant tool call."""

    name: str
    arguments: str


class TinkerToolCall(TypedDict):
    """The cookbook-compatible representation of one WMO assistant tool call."""

    type: str
    id: str
    function: TinkerFunctionCall


class TinkerConversationMessage(TypedDict):
    """One cookbook-compatible text, assistant-tool, or tool-result message."""

    role: str
    content: str
    tool_calls: NotRequired[list[TinkerToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


@dataclass(frozen=True)
class TinkerSFTDatum:
    """One immutable W12 example rendered as a pinned Tinker cross-entropy wire datum."""

    example_id: str
    supervised_token_count: int
    datum: tinker.Datum


class TinkerTrainerBackend:
    """Create one concrete Tinker LoRA session from a caller-owned service client.

    The supplied service client is intentionally already constructed.  This class has no API-key
    parameter and no fallback constructor, which keeps environment and spend authorization outside
    optimization code.
    """

    def __init__(self, service: tinker.ServiceClient) -> None:
        """Bind a caller-owned service client without creating or configuring it.

        Args:
            service: Existing authorized Tinker service client.  No request occurs until ``open``.
        """
        self._service = service

    def conservative_step_cost(self, spec: TinkerSFTSpec, *, batch_example_count: int) -> None:
        """Return no cost bound because the pinned SDK exposes no supported estimator.

        Args:
            spec: Frozen training settings, unused because no SDK estimate exists.
            batch_example_count: Planned row count, also insufficient for an SDK-backed bound.

        Returns:
            None. A run with ``maximum_cost_usd`` therefore fails before opening this backend.
        """
        return None

    def open(self, spec: TinkerSFTSpec, resume_state_path: str | None) -> TrainerSession:
        """Create a managed LoRA client and restore a durable state before rendering any datum.

        Args:
            spec: Frozen LoRA and cross-entropy settings.
            resume_state_path: Last recorded training-state handle, when continuing a prior run.

        Returns:
            A concrete session that renders W12 examples and performs managed cross-entropy steps.

        Raises:
            TinkerSFTDependencyError: The optional local SDK or cookbook package is unavailable.
        """
        _require_tinker_dependencies()
        from tinker_cookbook.model_info import get_recommended_renderer_name
        from tinker_cookbook.renderers import get_renderer

        client = self._service.create_lora_training_client(
            spec.base_model,
            rank=spec.lora_rank,
            seed=spec.seed,
        )
        if resume_state_path is not None:
            # Tinker requires restore before a weight-affecting call.  In particular, renderer
            # creation waits until this returns, so a resumed optimizer state is never replaced.
            client.load_state_with_optimizer(resume_state_path).result()
        renderer_name = get_recommended_renderer_name(spec.base_model)
        renderer = get_renderer(
            renderer_name,
            client.get_tokenizer(),
            model_name=spec.base_model,
        )
        return TinkerTrainerSession(client=client, renderer=renderer, spec=spec)


class TinkerTrainerSession:
    """One concrete Tinker training client using cookbook rendering and cross-entropy only."""

    def __init__(
        self,
        *,
        client: tinker.TrainingClient,
        renderer: Renderer,
        spec: TinkerSFTSpec,
    ) -> None:
        """Bind a managed client, its authoritative base-model renderer, and frozen settings.

        Args:
            client: Tinker client already created or restored by ``TinkerTrainerBackend``.
            renderer: Cookbook renderer selected for the exact configured base model.
            spec: Frozen datum length and optimizer settings.
        """
        self._client = client
        self._renderer = renderer
        self._spec = spec

    def render_examples(self, examples: Sequence[SFTExample]) -> tuple[TrainerDatum, ...]:
        """Render every complete target action into a cross-entropy datum in W12 row order.

        Only the final assistant message is trainable because W12 expands each accepted assistant
        action into a standalone row.  Previous assistant actions and tool events remain context.

        Args:
            examples: Ordered frozen train examples, never held-out rows.

        Returns:
            One labeled Tinker datum per input example, in exactly the same order.

        Raises:
            TinkerSFTError: Rendering omitted cross-entropy inputs, removed the complete target,
                or left the target with no supervised tokens.
        """
        _require_tinker_dependencies()
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised.data import conversation_to_datum

        rendered: list[TrainerDatum] = []
        for example in examples:
            datum = conversation_to_datum(
                cast("list[Message]", tinker_messages_from_example(example)),
                self._renderer,
                max_length=None,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
                reduction="none",
            )
            datum = _truncate_context_only(
                datum,
                maximum_datum_tokens=self._spec.maximum_datum_tokens,
                example_id=example.example_id,
            )
            weights = datum.loss_fn_inputs.get("weights")
            targets = datum.loss_fn_inputs.get("target_tokens")
            if weights is None or targets is None:
                raise TinkerSFTError("Tinker cookbook did not produce cross-entropy datum inputs")
            if set(datum.loss_fn_inputs) != {"target_tokens", "weights"}:
                raise TinkerSFTError(
                    "Tinker SFT cross-entropy datum must contain only target_tokens and weights"
                )
            supervised_token_count = sum(weight != 0.0 for weight in weights.data)
            if supervised_token_count <= 0:
                raise TinkerSFTError(
                    f"Tinker renderer left no supervised target tokens for {example.example_id}"
                )
            rendered.append(
                TinkerSFTDatum(
                    example_id=example.example_id,
                    supervised_token_count=supervised_token_count,
                    datum=datum,
                )
            )
        return tuple(rendered)

    def train_batch(
        self, datums: Sequence[TrainerDatum], *, learning_rate: float
    ) -> TrainerBatchResult:
        """Run exactly one managed cross-entropy forward-backward and Adam update.

        Args:
            datums: Datums produced by this session's immutable W12 renderer.
            learning_rate: Frozen Adam learning rate for the current batch.

        Returns:
            Only backend-reported loss and gradient metrics.  Tinker exposes no per-step monetary
            meter on this API, so cost remains absent rather than estimated.

        Raises:
            TinkerSFTError: A datum did not originate from this concrete renderer.
        """
        _require_tinker_dependencies()
        import tinker

        tinker_datums: list[tinker.Datum] = []
        for datum in datums:
            if not isinstance(datum, TinkerSFTDatum):
                raise TinkerSFTError("Tinker trainer received a datum from another backend")
            tinker_datums.append(datum.datum)
        forward_backward = self._client.forward_backward(tinker_datums, "cross_entropy").result()
        optimizer = self._client.optim_step(tinker.AdamParams(learning_rate=learning_rate)).result()
        return TrainerBatchResult(
            loss=_named_metric(forward_backward.metrics, ("loss", "total_loss")),
            gradient_norm=_named_metric(optimizer.metrics, ("grad_norm", "gradient_norm")),
        )

    def save_state(self, checkpoint_name: str) -> str:
        """Persist one non-overwriting managed optimizer state.

        Args:
            checkpoint_name: Unique immutable checkpoint name for the current run step.

        Returns:
            Tinker's opaque resume resource identifier.
        """
        response = self._client.save_state(checkpoint_name).result()
        return response.path

    def save_sampling_handle(self, model_name: str) -> str:
        """Persist one completed sampler handle without deploying or serving it.

        Args:
            model_name: Unique immutable resource name for the completed run.

        Returns:
            Tinker's opaque sampling resource identifier.
        """
        response = self._client.save_weights_for_sampler(model_name).result()
        return response.path


def tinker_messages_from_example(example: SFTExample) -> list[TinkerConversationMessage]:
    """Convert one immutable W12 example into a complete cookbook supervised conversation.

    The task is explicit initial context.  W12 observation events have no native cookbook role, so
    they remain text-identical context under ``user``.  Tool call IDs, names, canonical arguments,
    prior assistant actions, tool results, and the final complete target action are preserved.

    Args:
        example: One frozen W12 context and complete assistant-action target.

    Returns:
        Ordered messages whose final assistant message is the only supervised target.

    Raises:
        TinkerSFTError: The frozen context contains an unsupported event type.
    """
    messages: list[TinkerConversationMessage] = [
        {"role": "system", "content": f"Task:\n{example.task}"}
    ]
    for event in example.history:
        if isinstance(event, SFTMessage):
            role = "user" if event.role == "observation" else event.role
            messages.append({"role": role, "content": event.content})
        elif isinstance(event, AssistantActionEvent):
            messages.append(_assistant_message(event.action))
        elif isinstance(event, ToolEvent):
            tool_message: TinkerConversationMessage = {
                "role": "tool",
                "content": event.content,
                "tool_call_id": event.tool_call_id,
            }
            if event.tool_name is not None:
                tool_message["name"] = event.tool_name
            messages.append(tool_message)
        else:
            raise TinkerSFTError("W12 training context contains an unsupported event")
    messages.append(_assistant_message(example.target))
    return messages


def _assistant_message(action: AssistantAction) -> TinkerConversationMessage:
    """Convert a complete typed assistant action without dropping text or ordered tool calls."""
    message: TinkerConversationMessage = {"role": "assistant", "content": action.content or ""}
    if action.tool_calls:
        message["tool_calls"] = [_tool_call(call) for call in action.tool_calls]
    return message


def _tool_call(call: ToolCall) -> TinkerToolCall:
    """Render one typed tool call through canonical JSON so argument bytes are deterministic."""
    return {
        "type": "function",
        "id": call.call_id,
        "function": {
            "name": call.name,
            "arguments": canonical_json_bytes(call.arguments).decode("utf-8"),
        },
    }


def _named_metric(metrics: Mapping[str, float] | None, names: Sequence[str]) -> float | None:
    """Extract one finite backend-reported metric by stable base name without inventing a value."""
    if metrics is None:
        return None
    for key, value in metrics.items():
        if key.split(":", 1)[0] in names:
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise TinkerSFTError(f"Tinker reported a non-finite metric for {key}")
            return numeric_value
    return None


def _truncate_context_only(
    datum: tinker.Datum,
    *,
    maximum_datum_tokens: int | None,
    example_id: str,
) -> tinker.Datum:
    """Trim only unsupervised prompt tokens while retaining the complete target suffix."""
    if maximum_datum_tokens is None:
        return datum
    import tinker

    model_tokens = datum.model_input.to_ints()
    maximum_model_tokens = maximum_datum_tokens - 1
    if len(model_tokens) <= maximum_model_tokens:
        return datum
    weights = datum.loss_fn_inputs["weights"].data
    targets = datum.loss_fn_inputs["target_tokens"].data
    weight_tensor = datum.loss_fn_inputs["weights"]
    target_tensor = datum.loss_fn_inputs["target_tokens"]
    first_supervised = next(
        (index for index, weight in enumerate(weights) if weight != 0.0),
        len(weights),
    )
    trim_count = len(model_tokens) - maximum_model_tokens
    if trim_count > first_supervised:
        raise TinkerSFTError(
            f"maximum_datum_tokens cannot retain the complete supervised target for {example_id}"
        )
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(model_tokens[trim_count:]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=targets[trim_count:],
                dtype=target_tensor.dtype,
                shape=[len(targets) - trim_count],
            ),
            "weights": tinker.TensorData(
                data=weights[trim_count:],
                dtype=weight_tensor.dtype,
                shape=[len(weights) - trim_count],
            ),
        },
    )


def _require_tinker_dependencies() -> None:
    """Fail clearly when the concrete backend is invoked without its optional local dependencies."""
    try:
        import tinker  # noqa: F401
        import tinker_cookbook  # noqa: F401
    except ImportError as exc:
        raise TinkerSFTDependencyError(_MISSING_TINKER_EXTRA) from exc

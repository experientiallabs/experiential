"""Local Tinker datum and adapter tests that never construct a service client."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import AssistantAction, ToolCall
from exp.optimize.model.sft.contracts import (
    AssistantActionEvent,
    SFTExample,
    SFTMessage,
    ToolEvent,
    TraceExampleSource,
)
from exp.optimize.model.sft.tinker import (
    TinkerConversationMessage,
    TinkerSFTDatum,
    TinkerTrainerBackend,
    TinkerTrainerSession,
    tinker_messages_from_example,
)
from exp.optimize.model.sft.training import TinkerSFTError, TinkerSFTSpec

if TYPE_CHECKING:
    import tinker
    import torch
    from tinker_cookbook.renderers import Renderer, TrainOnWhat

_DIGEST = "a" * 64


def _spec() -> TinkerSFTSpec:
    """Build one compact concrete Tinker specification for local adapter tests."""
    return TinkerSFTSpec(
        base_model="Qwen/Qwen3-8B",
        lora_rank=8,
        learning_rate=0.01,
        batch_size=1,
        epochs=1,
        checkpoint_every_steps=1,
        maximum_datum_tokens=32,
    )


def _example() -> SFTExample:
    """Build one complete W12 target with prior and target tool calls for conversion checks."""
    return SFTExample(
        example_id="example-tinker",
        leakage_group_id="lineage-tinker",
        task="Resolve an account request.",
        history=(
            SFTMessage(role="system", content="Follow the support policy."),
            SFTMessage(role="user", content="Please inspect my account."),
            SFTMessage(role="observation", content="The account is active."),
            AssistantActionEvent(
                action=AssistantAction(
                    content="I will inspect it.",
                    tool_calls=(
                        ToolCall(
                            call_id="call-history",
                            name="lookup_account",
                            arguments={"account_id": "A-1"},
                        ),
                    ),
                )
            ),
            ToolEvent(
                tool_call_id="call-history",
                tool_name="lookup_account",
                content="Account A-1 is active.",
            ),
        ),
        target=AssistantAction(
            content="Your account is active.",
            tool_calls=(
                ToolCall(
                    call_id="call-target",
                    name="send_confirmation",
                    arguments={"channel": "email", "priority": 1},
                ),
            ),
        ),
        source=TraceExampleSource(
            trace_id="trace-tinker",
            acceptance_evidence=ArtifactInput(artifact_id="acceptance-evidence", sha256=_DIGEST),
        ),
        source_step_index=2,
    )


def test_messages_preserve_complete_target_action_and_tool_context() -> None:
    """The cookbook conversation retains every target and historical tool field in order."""
    messages = tinker_messages_from_example(_example())

    assert messages == [
        {"role": "system", "content": "Task:\nResolve an account request."},
        {"role": "system", "content": "Follow the support policy."},
        {"role": "user", "content": "Please inspect my account."},
        {"role": "user", "content": "The account is active."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-history",
                    "function": {
                        "name": "lookup_account",
                        "arguments": '{"account_id":"A-1"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "Account A-1 is active.",
            "tool_call_id": "call-history",
            "name": "lookup_account",
        },
        {
            "role": "assistant",
            "content": "Your account is active.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-target",
                    "function": {
                        "name": "send_confirmation",
                        "arguments": '{"channel":"email","priority":1}',
                    },
                }
            ],
        },
    ]


def test_messages_canonicalize_provider_tool_argument_formatting() -> None:
    """Training inputs use semantic JSON even when runtime retained provider wire bytes."""
    example = _example()
    target = example.target.model_copy(
        update={
            "tool_calls": (
                ToolCall(
                    call_id="call-target",
                    name="send_confirmation",
                    arguments={"channel": "email", "priority": 1},
                    raw_arguments='{ "priority": 1, "channel": "email" }',
                ),
            )
        }
    )

    messages = tinker_messages_from_example(example.model_copy(update={"target": target}))

    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == (
        '{"channel":"email","priority":1}'
    )


class _DatumRenderer:
    """Return fixed local token and weight tensors without fetching a tokenizer or model."""

    def __init__(self) -> None:
        """Initialize empty call journals for deterministic renderer assertions."""
        self.conversations: list[list[TinkerConversationMessage]] = []
        self.train_on_what: list[TrainOnWhat] = []

    def build_supervised_example(
        self,
        conversation: list[TinkerConversationMessage],
        train_on_what: TrainOnWhat,
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """Build a deterministic input whose shifted CE layout is easy to inspect exactly."""
        import tinker
        import torch

        self.conversations.append(conversation)
        self.train_on_what.append(train_on_what)
        return (
            tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=[10, 11, 12, 13])]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
        )


class _Future[ValueT]:
    """A synchronous local stand-in for the small Tinker future surface used by the adapter."""

    def __init__(self, value: ValueT) -> None:
        """Store one already-computed local response."""
        self._value = value

    def result(self) -> ValueT:
        """Return the prebuilt local SDK-shaped response."""
        return self._value


class _TrainingClient:
    """Record local concrete training calls without creating a Tinker service session."""

    def __init__(self) -> None:
        """Initialize empty forward, backward, and optimizer call journals."""
        self.forward_backward_calls: list[tuple[Sequence[tinker.Datum], str]] = []
        self.optim_steps: list[tinker.AdamParams] = []

    def forward_backward(
        self, data: Sequence[tinker.Datum], loss_fn: str
    ) -> _Future[tinker.ForwardBackwardOutput]:
        """Return fixed locally constructed loss metrics."""
        import tinker

        self.forward_backward_calls.append((data, loss_fn))
        return _Future(
            tinker.ForwardBackwardOutput(
                loss_fn_output_type="cross_entropy",
                loss_fn_outputs=[],
                metrics={"total_loss:mean": 1.25},
            )
        )

    def optim_step(self, params: tinker.AdamParams) -> _Future[tinker.OptimStepResponse]:
        """Return a fixed local gradient metric."""
        import tinker

        self.optim_steps.append(params)
        return _Future(tinker.OptimStepResponse(metrics={"grad_norm:mean": 0.75}))


def test_datum_fixture_is_shifted_cross_entropy_with_only_target_tokens_and_weights() -> None:
    """The frozen local datum fixture pins cookbook conversion to the current CE wire contract."""
    pytest.importorskip("tinker")
    renderer = _DatumRenderer()
    client = _TrainingClient()
    session = TinkerTrainerSession(
        client=cast("tinker.TrainingClient", client),
        renderer=cast("Renderer", renderer),
        spec=_spec(),
    )

    (datum,) = session.render_examples((_example(),))

    assert isinstance(datum, TinkerSFTDatum)
    assert datum.example_id == "example-tinker"
    assert datum.supervised_token_count == 2
    assert datum.datum.model_input.to_ints() == [10, 11, 12]
    assert datum.datum.loss_fn_inputs["target_tokens"].data == [11, 12, 13]
    assert datum.datum.loss_fn_inputs["weights"].data == [0.0, 1.0, 1.0]
    assert set(datum.datum.loss_fn_inputs) == {"target_tokens", "weights"}
    assert renderer.train_on_what[0].value == "last_assistant_message"

    result = session.train_batch((datum,), learning_rate=0.02)

    assert result.loss == 1.25
    assert result.gradient_norm == 0.75
    assert client.forward_backward_calls[0][1] == "cross_entropy"
    assert client.optim_steps[0].learning_rate == 0.02


@dataclass
class _ResumeClient:
    """Record ordering around the concrete backend's restore and renderer setup calls."""

    calls: list[str]

    def load_state_with_optimizer(self, path: str) -> _Future[None]:
        """Record a local restore with no provider communication."""
        self.calls.append(f"load:{path}")
        return _Future(None)

    def get_tokenizer(self) -> str:
        """Return a marker token source after recording its position in the call sequence."""
        self.calls.append("tokenizer")
        return "local-tokenizer"


@dataclass
class _Service:
    """Expose just the prebuilt service-client creation call used by the concrete backend."""

    client: _ResumeClient
    calls: list[str]

    def create_lora_training_client(
        self, base_model: str, rank: int, seed: int | None = None
    ) -> _ResumeClient:
        """Return the injected local client without constructing a Tinker service."""
        self.calls.append(f"create:{base_model}:{rank}:{seed}")
        return self.client


def test_backend_restores_before_renderer_setup_with_an_injected_local_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backend restores state before tokenizer or renderer access, with no real client."""
    pytest.importorskip("tinker_cookbook")
    import tinker_cookbook.model_info
    import tinker_cookbook.renderers

    calls: list[str] = []
    client = _ResumeClient(calls)
    service = _Service(client, calls)
    renderer = _DatumRenderer()
    monkeypatch.setattr(
        tinker_cookbook.model_info,
        "get_recommended_renderer_name",
        lambda model: "local-renderer",
    )
    monkeypatch.setattr(
        tinker_cookbook.renderers,
        "get_renderer",
        lambda name, tokenizer, model_name: renderer,
    )

    session = TinkerTrainerBackend(cast("tinker.ServiceClient", service)).open(
        _spec(), "fake://state/checkpoint"
    )

    assert isinstance(session, TinkerTrainerSession)
    assert calls == [
        "create:Qwen/Qwen3-8B:8:0",
        "load:fake://state/checkpoint",
        "tokenizer",
    ]


def test_context_truncation_retains_the_complete_two_token_target() -> None:
    """A tight datum limit removes prompt context without dropping either target token."""
    pytest.importorskip("tinker")
    renderer = _DatumRenderer()
    session = TinkerTrainerSession(
        client=cast("tinker.TrainingClient", _TrainingClient()),
        renderer=cast("Renderer", renderer),
        spec=_spec().model_copy(update={"maximum_datum_tokens": 3}),
    )

    (datum,) = session.render_examples((_example(),))

    assert isinstance(datum, TinkerSFTDatum)
    assert datum.supervised_token_count == 2
    assert datum.datum.model_input.to_ints() == [11, 12]
    assert datum.datum.loss_fn_inputs["target_tokens"].data == [12, 13]
    assert datum.datum.loss_fn_inputs["weights"].data == [1.0, 1.0]


def test_backend_cost_bound_uses_explicit_price_and_maximum_tokens() -> None:
    """Calculate a model-specific full-datum bound without accessing the service client."""
    backend = TinkerTrainerBackend(cast("tinker.ServiceClient", object()))
    spec = _spec().model_copy(update={"training_usd_per_million_tokens": 250.0})

    cost = backend.conservative_step_cost(spec, batch_example_count=3)

    assert cost is not None
    assert cost.value == 250.0 * 32 * 3 / 1_000_000
    assert cost.provenance == "estimated"


def test_datum_limit_rejects_instead_of_truncating_the_two_token_target() -> None:
    """A limit too small for the complete target fails before any training dispatch."""
    pytest.importorskip("tinker")
    session = TinkerTrainerSession(
        client=cast("tinker.TrainingClient", _TrainingClient()),
        renderer=cast("Renderer", _DatumRenderer()),
        spec=_spec().model_copy(update={"maximum_datum_tokens": 2}),
    )

    with pytest.raises(TinkerSFTError, match="complete supervised target"):
        session.render_examples((_example(),))

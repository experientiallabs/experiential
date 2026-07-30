from llm_waterfall import ChatRequest, ChatResponse
from router_real_routerbench import _outcome

from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry


class _StructuredOnlyProvider:
    def __init__(self) -> None:
        self.request: ChatRequest | None = None

    def complete(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("reasoning-effort Anthropic arms must not use text completion")

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        self.request = request
        return ChatResponse.model_validate(
            {
                "choices": [{"message": {"role": "assistant", "content": "B"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
        )


def test_anthropic_reasoning_effort_uses_structured_completion() -> None:
    provider = _StructuredOnlyProvider()
    entry = PoolEntry(
        name="fable-5",
        kind=ProviderKind.ANTHROPIC,
        model="claude-fable-5",
        reasoning_effort="max",
        input_per_mtok=10.0,
        output_per_mtok=50.0,
    )

    outcome = _outcome(
        task={"task_id": "task-1", "prompt": "A or B?", "answer": "B"},
        entry=entry,
        attempt=1,
        provider=provider,  # type: ignore[arg-type]
    )

    assert provider.request is not None
    assert provider.request.max_completion_tokens == 1024
    assert outcome.reward == 1.0
    assert outcome.replies == ["B"]
    assert outcome.usage.input_tokens == 10
    assert outcome.usage.output_tokens == 2
    assert outcome.cost_usd == 0.0002

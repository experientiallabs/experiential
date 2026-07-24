"""Tests for the Tinker provider: span recording, response shape, lazy imports.

Everything runs against the deterministic fakes in `wmh.distill.fake_tinker`
plus a minimal char-level `ChatRendering`; the real tinker SDK is never
touched (several tests pin that by poisoning `sys.modules`).
"""

from __future__ import annotations

import ast
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

import pytest
from llm_waterfall.types import (
    ChatFunctionCall,
    ChatFunctionDefinition,
    ChatMessage,
    ChatRequest,
    ChatTool,
    ChatToolCall,
)
from pydantic import JsonValue, ValidationError

import wmh.distill.rendering as rendering_module
import wmh.providers.tinker as tinker_module
from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    RolloutConfig,
    StudentConfig,
    TeacherConfig,
    TrainConfig,
)
from wmh.distill.data import build_datums
from wmh.distill.deadlines import TinkerDeadlineError
from wmh.distill.fake_tinker import FakeSampledSequence, FakeSamplingClient, FakeTokenizer
from wmh.distill.rendering import ParsedAssistantMessage
from wmh.distill.tokens import TrialRecord
from wmh.providers.base import (
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
)
from wmh.providers.registry import get_provider
from wmh.providers.tinker import (
    SdkSampler,
    TinkerChatProvider,
    TinkerSampler,
    TokenRecorder,
    TokenSpan,
)

if TYPE_CHECKING:
    import tinker


class _MiniRendering:
    """Minimal char-level ChatRendering that drives the provider without the cookbook."""

    def __init__(self) -> None:
        self._tok = FakeTokenizer()

    @property
    def stop_sequences(self) -> list[str] | list[int]:
        # Newline: outside the fake sampler's printable-ASCII token range, so
        # deterministic samples always run to max_tokens.
        return [ord("\n")]

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        lines: list[str] = []
        if tools:
            lines.append("tools: " + ",".join(tool.function.name for tool in tools))
        for message in messages:
            content = message.content if isinstance(message.content, str) else ""
            lines.append(f"{message.role}: {content}")
        lines.append("assistant:")
        return self._tok.encode("\n".join(lines))

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        del tools, previous_sampled_ids
        lines = [""]
        for message in messages[delta_start:]:
            content = message.content if isinstance(message.content, str) else ""
            lines.append(f"{message.role}: {content}")
        lines.append("assistant:")
        return self._tok.encode("\n".join(lines))

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        return ParsedAssistantMessage(
            text=self._tok.decode(sampled_ids), tool_calls=[], stopped=False
        )


class _ToolCallRendering(_MiniRendering):
    """Parses every sample into one fixed tool call (tool-call shape tests)."""

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        del sampled_ids
        call = ChatToolCall(
            id="call_0", function=ChatFunctionCall(name="bash", arguments='{"cmd": "ls"}')
        )
        return ParsedAssistantMessage(text="", tool_calls=[call], stopped=True)


class _FlakySampler:
    """Raises on the first sample() calls, then delegates to a fake sampler."""

    def __init__(self, inner: FakeSamplingClient, failures: int = 1) -> None:
        self._inner = inner
        self._failures = failures

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> FakeSampledSequence:
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("simulated sampler outage")
        return self._inner.sample(
            prompt_token_ids, max_tokens=max_tokens, temperature=temperature, stop=stop
        )


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.TINKER,
        model_type="Qwen/Qwen3-8B",
        model="tinker://run/weights/0",
    )


def _request(max_tokens: int = 16) -> ChatRequest:
    return ChatRequest(
        messages=[
            ChatMessage(role="system", content="be terse"),
            ChatMessage(role="user", content="hi"),
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )


def _as_dict(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _as_list(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)
    return value


def _span(call_index: int = 0) -> TokenSpan:
    return TokenSpan(
        call_index=call_index,
        prompt_token_ids=[1, 2],
        sampled_token_ids=[65, 66],
        sampled_logprobs=[-0.5, -1.5],
    )


def test_token_span_requires_aligned_logprobs() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        TokenSpan(
            call_index=0,
            prompt_token_ids=[1],
            sampled_token_ids=[2, 3],
            sampled_logprobs=[-0.1],
        )


def test_recorder_snapshot_is_a_copy() -> None:
    recorder = TokenRecorder()
    recorder.record(_span())
    snapshot = recorder.spans()
    snapshot.clear()
    assert len(recorder) == 1
    assert recorder.spans()[0].call_index == 0


def test_recorder_jsonl_sink_written_incrementally(tmp_path: Path) -> None:
    sink = tmp_path / "spans.jsonl"
    recorder = TokenRecorder(jsonl_path=sink)
    recorder.record(_span(0))
    assert len(sink.read_text(encoding="utf-8").splitlines()) == 1
    recorder.record(_span(1))
    lines = sink.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert [item["call_index"] for item in parsed] == [0, 1]
    assert parsed[0]["sampled_token_ids"] == [65, 66]
    assert parsed[0]["sampled_logprobs"] == [-0.5, -1.5]


def test_complete_chat_shape_and_span_matches_issued_sample() -> None:
    fake = FakeSamplingClient(seed="student-v0")
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(), sampling_client=fake, renderer=_MiniRendering(), recorder=recorder
    )

    response = provider.complete_chat(_request(max_tokens=16))

    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.content
    assert choice.message.tool_calls is None
    # _MiniRendering never reports a stop signal, so truncation reads as length.
    assert choice.finish_reason == "length"
    expected_prompt = _MiniRendering().build_generation_prompt(_request().messages)
    assert response.usage is not None
    assert response.usage.prompt_tokens == len(expected_prompt)
    assert response.usage.completion_tokens == 16
    assert response.model == "tinker://run/weights/0"

    # TITO: the recorded span is byte-identical to what the sampler issued.
    assert len(recorder) == 1
    span = recorder.spans()[0]
    issued = fake.issued[0]
    assert span.call_index == 0
    assert span.prompt_token_ids == list(issued.prompt_ids) == expected_prompt
    assert span.sampled_token_ids == list(issued.sampled_ids)
    assert span.sampled_logprobs == list(issued.logprobs)

    wire = response.wire_payload()
    wire_message = _as_dict(_as_dict(_as_list(wire["choices"])[0])["message"])
    assert wire_message["role"] == "assistant"
    assert _as_dict(wire["usage"])["completion_tokens"] == 16


def test_complete_chat_tool_calls_in_openai_format() -> None:
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_ToolCallRendering()
    )
    response = provider.complete_chat(_request())
    choice = response.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls is not None
    assert choice.message.tool_calls[0].function.name == "bash"

    wire = response.wire_payload()
    wire_message = _as_dict(_as_dict(_as_list(wire["choices"])[0])["message"])
    wire_call = _as_list(wire_message["tool_calls"])[0]
    assert wire_call == {
        "id": "call_0",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"cmd": "ls"}'},
    }
    # Empty text serializes as an absent content key, like OpenAI's null content.
    assert "content" not in wire_message


def test_tool_choice_none_renders_without_tool_schemas() -> None:
    recorder = TokenRecorder()
    rendering = _MiniRendering()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=rendering,
        recorder=recorder,
    )
    request = _request()
    request.tools = [
        ChatTool(
            function=ChatFunctionDefinition(
                name="bash", description="run bash", parameters={"type": "object"}
            )
        )
    ]
    request.tool_choice = "none"
    provider.complete_chat(request)
    prompt_text = rendering.decode(recorder.spans()[0].prompt_token_ids)
    assert "tools:" not in prompt_text


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "bash"}}])
def test_unsupported_tool_choice_raises_actionable_error(choice: JsonValue) -> None:
    provider = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    request = _request()
    request.tool_choice = choice
    with pytest.raises(ValueError, match="tool_choice"):
        provider.complete_chat(request)


def test_span_recorded_once_per_success_and_not_on_failure() -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=_FlakySampler(FakeSamplingClient(seed="s"), failures=1),
        renderer=_MiniRendering(),
        recorder=recorder,
    )
    # First attempt fails mid-sampling: an outer retry wrapper would re-invoke
    # complete_chat, and the failed attempt must not leave a span behind.
    with pytest.raises(RuntimeError, match="outage"):
        provider.complete_chat(_request())
    assert len(recorder) == 0

    provider.complete_chat(_request())
    assert [span.call_index for span in recorder.spans()] == [0]

    provider.complete_chat(_request())
    assert [span.call_index for span in recorder.spans()] == [0, 1]


class _FramedRendering(_MiniRendering):
    """Scripted renderer with explicit per-message framing and suffix rendering.

    Each message renders as `<role>content|calls=name:args</>`, the generation
    header is `<assistant>`, and `</>` is the end-of-turn framing. Tool-call
    arguments render VERBATIM, so a caller that echoes an assistant turn with
    reformatted JSON spacing changes the re-rendered tokens, exactly the live
    defect the incremental prompt construction exists to absorb.
    """

    def _segment(self, message: ChatMessage) -> str:
        content = message.content if isinstance(message.content, str) else ""
        calls = ""
        if message.tool_calls:
            calls = "|calls=" + ";".join(
                f"{call.function.name}:{call.function.arguments}" for call in message.tool_calls
            )
        return f"<{message.role}>{content}{calls}</>"

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        prefix = ""
        if tools:
            prefix = "<tools>" + ",".join(tool.function.name for tool in tools) + "</>"
        body = "".join(self._segment(message) for message in messages)
        return self._tok.encode(prefix + body + "<assistant>")

    def render_suffix(
        self,
        messages: list[ChatMessage],
        delta_start: int,
        tools: list[ChatTool] | None = None,
        *,
        previous_sampled_ids: list[int],
    ) -> list[int]:
        del tools
        end_of_turn = self._tok.encode("</>")
        tokens: list[int] = []
        if previous_sampled_ids[-len(end_of_turn) :] != end_of_turn:
            tokens.extend(end_of_turn)
        for message in messages[delta_start:]:
            tokens.extend(self._tok.encode(self._segment(message)))
        tokens.extend(self._tok.encode("<assistant>"))
        return tokens

    def parse_response(self, sampled_ids: list[int]) -> ParsedAssistantMessage:
        text = self._tok.decode(sampled_ids)
        stopped = text.endswith("</>")
        return ParsedAssistantMessage(text=text.removesuffix("</>"), tool_calls=[], stopped=stopped)


class _ScriptedSampler:
    """Returns canned token sequences in order, recording every prompt."""

    def __init__(self, texts: list[str]) -> None:
        self._tok = FakeTokenizer()
        self._outputs = [self._tok.encode(text) for text in texts]
        self.prompts: list[list[int]] = []

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> FakeSampledSequence:
        del max_tokens, temperature, stop
        self.prompts.append(list(prompt_token_ids))
        tokens = self._outputs.pop(0)
        return FakeSampledSequence(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason="stop")


def _distill_cfg() -> DistillConfig:
    return DistillConfig(
        student=StudentConfig(base_model="Qwen/Qwen3-8B"),
        teacher=TeacherConfig(model="Qwen/Qwen3-32B"),
        harbor=HarborConfig(job_template="job.yaml"),
        rollout=RolloutConfig(),
        train=TrainConfig(),
    )


def _trial(recorder: TokenRecorder) -> TrialRecord:
    return TrialRecord(
        task_id="task-a",
        attempt=1,
        trial_name="task-a__x1",
        reward=1.0,
        passed=True,
        spans=recorder.spans(),
        stop_reason="submitted",
        artifact_dir="/tmp/jobs/task-a__x1",
    )


def _framed_provider(texts: list[str]) -> tuple[TinkerChatProvider, TokenRecorder]:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=_ScriptedSampler(texts),
        renderer=_FramedRendering(),
        recorder=recorder,
    )
    return provider, recorder


def _chat(provider: TinkerChatProvider, messages: list[ChatMessage]) -> ChatMessage:
    response = provider.complete_chat(
        ChatRequest(messages=messages, temperature=0.0, max_tokens=64)
    )
    return response.choices[0].message


_HISTORY = [
    ChatMessage(role="system", content="be terse"),
    ChatMessage(role="user", content="list files"),
]


def _echo(arguments: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content="ok",
        tool_calls=[
            ChatToolCall(id="call_0", function=ChatFunctionCall(name="bash", arguments=arguments))
        ],
    )


def _tool_result(content: str) -> ChatMessage:
    return ChatMessage.model_validate(
        {"role": "tool", "content": content, "tool_call_id": "call_0"}
    )


def test_multi_turn_verbatim_echo_prompts_extend_and_merge() -> None:
    # Regression: when the caller echoes the assistant turn verbatim, prompts
    # are prefix-extending AND identical to a full re-render, validating that
    # the suffix composition matches full-render framing.
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd": "ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    assert second.prompt_token_ids[: len(episode)] == episode
    assert second.prompt_token_ids == rendering.build_generation_prompt(extended)
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0
    assert recorder.fallback_count == 0


def test_reformatted_assistant_echo_still_extends_and_merges() -> None:
    # The live defect: the agent re-serializes the assistant turn (different
    # JSON spacing in tool_calls), so a full re-render would NOT extend the
    # sampled tokens; the incremental prompt must still extend and merge.
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(['ok|calls=bash:{"cmd": "ls"}</>', "done</>"])
    _chat(provider, _HISTORY)
    extended = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt b.txt")]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    # The defect is real in this scripted world: a full re-render diverges.
    assert rendering.build_generation_prompt(extended)[: len(episode)] != episode
    # The fix: the incrementally built prompt still extends the token history.
    assert second.prompt_token_ids[: len(episode)] == episode
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0
    assert stats.fragmentation_rate == 0.0
    assert recorder.fallback_count == 0


def test_genuine_history_edit_falls_back_and_fragments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rendering = _FramedRendering()
    provider, recorder = _framed_provider(["a</>", "b</>", "c</>"])
    _chat(provider, _HISTORY)
    second_messages = [*_HISTORY, _echo('{"cmd":"ls"}'), _tool_result("a.txt")]
    _chat(provider, second_messages)
    # A changed tool-result message mid-history is a genuine edit: fall back.
    edited = [
        *_HISTORY,
        _echo('{"cmd":"ls"}'),
        _tool_result("EDITED"),
        ChatMessage(role="assistant", content="b"),
        _tool_result("more"),
    ]
    with caplog.at_level("INFO", logger="wmh.providers.tinker"):
        _chat(provider, edited)

    assert recorder.fallback_count == 1
    assert any("incoming message 3" in record.message for record in caplog.records)
    spans = recorder.spans()
    # The fallback prompt is a correct full render of the edited history.
    assert spans[2].prompt_token_ids == rendering.build_generation_prompt(edited)
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 2
    assert stats.fragments == 1


def test_max_tokens_truncation_gets_end_of_turn_framing() -> None:
    tok = FakeTokenizer()
    provider, recorder = _framed_provider(["par", "done</>"])
    first_response = _chat(provider, _HISTORY)
    assert first_response.content == "par"
    extended = [
        *_HISTORY,
        ChatMessage(role="assistant", content="par"),
        ChatMessage(role="user", content="continue"),
    ]
    _chat(provider, extended)

    first, second = recorder.spans()
    episode = first.prompt_token_ids + first.sampled_token_ids
    assert second.prompt_token_ids[: len(episode)] == episode
    # The suffix supplies the missing end-of-turn framing before the new message.
    expected_suffix = tok.encode("</>" + "<user>continue</>" + "<assistant>")
    assert second.prompt_token_ids[len(episode) :] == expected_suffix
    datums, stats = build_datums([_trial(recorder)], _distill_cfg())
    assert len(datums) == 1
    assert stats.fragments == 0


def test_re_asked_identical_history_reuses_the_exact_prompt() -> None:
    provider, recorder = _framed_provider(["a</>", "b</>"])
    _chat(provider, _HISTORY)
    _chat(provider, _HISTORY)
    first, second = recorder.spans()
    assert second.prompt_token_ids == first.prompt_token_ids
    assert recorder.fallback_count == 0


def test_tool_schema_change_mid_episode_falls_back() -> None:
    provider, recorder = _framed_provider(["a</>", "b</>"])
    tool = ChatTool(
        function=ChatFunctionDefinition(
            name="bash", description="run bash", parameters={"type": "object"}
        )
    )
    request = ChatRequest(messages=list(_HISTORY), temperature=0.0, max_tokens=64)
    request.tools = [tool]
    provider.complete_chat(request)
    follow_up = ChatRequest(
        messages=[*_HISTORY, ChatMessage(role="assistant", content="a"), _tool_result("out")],
        temperature=0.0,
        max_tokens=64,
    )
    provider.complete_chat(follow_up)
    assert recorder.fallback_count == 1


def test_complete_plain_text_uses_same_machinery() -> None:
    recorder = TokenRecorder()
    rendering = _MiniRendering()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=rendering,
        recorder=recorder,
    )
    completion = provider.complete(
        "sys prompt", [Message(role="user", content="do it")], temperature=0.5, max_tokens=8
    )
    span = recorder.spans()[0]
    assert completion.text == rendering.decode(span.sampled_token_ids)
    assert completion.usage.input_tokens == len(span.prompt_token_ids)
    assert completion.usage.output_tokens == 8
    # The system prompt travels as a leading system message.
    assert "system: sys prompt" in rendering.decode(span.prompt_token_ids)


def test_embed_raises_actionable_error() -> None:
    provider = TinkerChatProvider(_config(), sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="embedder"):
        provider.embed(["text"])


def test_verify_ok_via_fakes_and_never_records() -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(
        _config(),
        sampling_client=FakeSamplingClient(seed="s"),
        renderer=_MiniRendering(),
        recorder=recorder,
    )
    result = provider.verify()
    assert result.ok is True
    assert result.kind is ProviderKind.TINKER
    assert result.model == "tinker://run/weights/0"
    assert len(recorder) == 0


def test_registry_constructs_provider_without_touching_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Poison the SDK modules: construction and fake-backed completions must
    # never import tinker or tinker_cookbook.
    monkeypatch.setitem(sys.modules, "tinker", None)
    monkeypatch.setitem(sys.modules, "tinker_cookbook", None)
    provider = get_provider(_config())
    assert isinstance(provider, TinkerChatProvider)
    assert isinstance(provider, Provider)
    assert isinstance(provider, ToolCallingProvider)

    injected = TinkerChatProvider(
        _config(), sampling_client=FakeSamplingClient(seed="s"), renderer=_MiniRendering()
    )
    response = injected.complete_chat(_request())
    assert response.choices[0].message.role == "assistant"


def test_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tinker", None)
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())
    with pytest.raises(ImportError, match="uv sync --extra distill"):
        provider.complete_chat(_request())


def test_missing_api_key_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering())
    # The message must state the problem (not set) and the remedy (set it).
    with pytest.raises(RuntimeError, match="TINKER_API_KEY is not set"):
        provider.complete_chat(_request())


def test_tinker_path_without_model_type_is_actionable() -> None:
    config = ProviderConfig(kind=ProviderKind.TINKER, model="tinker://run/weights/0")
    provider = TinkerChatProvider(config, sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="model_type is unset"):
        provider.complete_chat(_request())


def test_tinker_path_in_model_type_names_the_swapped_field() -> None:
    # The swapped-fields mistake (weights path in model_type) must not claim that
    # model_type is unset; the message points at the field that actually holds the path.
    config = ProviderConfig(
        kind=ProviderKind.TINKER,
        model="Qwen/Qwen3-8B",
        model_type="tinker://run/weights/0",
    )
    provider = TinkerChatProvider(config, sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(ValueError, match="weights paths belong in config.model"):
        provider.complete_chat(_request())


def test_injected_sampler_without_tokenizer_requires_renderer() -> None:
    provider = TinkerChatProvider(_config(), sampling_client=FakeSamplingClient(seed="s"))
    with pytest.raises(RuntimeError, match="renderer="):
        provider.complete_chat(_request())


def _module_scope_import_roots(path: Path) -> set[str]:
    """Top-level import roots of a module (TYPE_CHECKING blocks excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".")[0])
    return roots


def test_module_scope_never_imports_the_distill_extra() -> None:
    # The tinker/tinker-cookbook SDKs are optional; module import must stay
    # lazy so the provider modules load without the distill extra installed.
    for module in (tinker_module, rendering_module):
        assert module.__file__ is not None
        roots = _module_scope_import_roots(Path(module.__file__))
        assert not roots & {"tinker", "tinker_cookbook"}, module.__name__


# --- deadlines: wedged sessions become retryable errors with fresh clients ----------------------


class _WedgedSampler:
    """A sampler whose every call reports a deadline expiry (a wedged session)."""

    def __init__(self) -> None:
        self.calls = 0

    def sample(
        self,
        prompt_token_ids: list[int],
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | list[int] | None = None,
    ) -> NoReturn:
        del prompt_token_ids, max_tokens, temperature, stop
        self.calls += 1
        raise TinkerDeadlineError("sample", elapsed_s=0.05, deadline_s=0.05)


def test_sampling_deadline_drops_and_rebuilds_the_lazy_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = TokenRecorder()
    provider = TinkerChatProvider(_config(), renderer=_MiniRendering(), recorder=recorder)
    builds: list[TinkerSampler] = []

    def build_sampler() -> TinkerSampler:
        sampler: TinkerSampler = _WedgedSampler() if not builds else FakeSamplingClient(seed="s")
        builds.append(sampler)
        return sampler

    monkeypatch.setattr(provider, "_build_sdk_sampler", build_sampler)

    with pytest.raises(TinkerDeadlineError, match="timed out"):
        provider.complete_chat(_request())
    # The timed-out call recorded no span, and the retry wrapper's next
    # attempt (simulated by calling again) builds a fresh client and succeeds.
    assert len(recorder) == 0
    response = provider.complete_chat(_request())
    assert response.choices[0].message.role == "assistant"
    assert len(builds) == 2
    assert [span.call_index for span in recorder.spans()] == [0]


def test_injected_sampling_client_is_never_dropped_on_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An injected client cannot be rebuilt; poison the SDK so any accidental
    # rebuild attempt would fail loudly instead of hitting the network.
    monkeypatch.setitem(sys.modules, "tinker", None)
    sampler = _WedgedSampler()
    provider = TinkerChatProvider(_config(), sampling_client=sampler, renderer=_MiniRendering())
    for _ in range(2):
        with pytest.raises(TinkerDeadlineError):
            provider.complete_chat(_request())
    assert sampler.calls == 2


class _NeverResolvingFuture:
    """Mimics the SDK future of a wedged session: result(timeout) honors the timeout."""

    def __init__(self) -> None:
        self._never = threading.Event()

    def result(self, timeout: float | None = None) -> NoReturn:
        self._never.wait(timeout)
        raise TimeoutError(f"fake future gave up after {timeout}s")


def test_sdk_sampler_bounds_the_sample_future(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tinker")
    monkeypatch.setenv("WMH_TINKER_DEADLINE_SAMPLE", "0.05")

    class _WedgedClient:
        def sample(
            self, prompt: object, num_samples: int, sampling_params: object
        ) -> _NeverResolvingFuture:
            del prompt, num_samples, sampling_params
            return _NeverResolvingFuture()

    sampler = SdkSampler(cast("tinker.SamplingClient", _WedgedClient()))
    with pytest.raises(TinkerDeadlineError, match="tinker sample timed out"):
        sampler.sample([1, 2, 3], max_tokens=4, temperature=1.0)

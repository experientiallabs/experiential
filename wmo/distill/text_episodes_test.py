"""Tests for the teacher-text to cross-entropy bridge."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm_waterfall.types import ChatFunctionCall, ChatMessage, ChatTool, ChatToolCall
from llm_waterfall.types import ChatFunctionDefinition as FunctionDef

from wmo.distill.config import DistillConfig
from wmo.distill.data import attach_advantages, build_datums
from wmo.distill.store import DistillRunStore
from wmo.distill.text_episodes import (
    TeacherEpisode,
    episode_spans,
    episodes_to_trial_records,
    text_warmup_manifest,
)


class _StubRendering:
    """A deterministic stand-in: one token per character, headers as sentinels.

    Mirrors the shape that matters (a prompt ending in a generation header, a
    turn rendering to its own output tokens) without a real tokenizer.
    """

    def build_generation_prompt(
        self, messages: list[ChatMessage], tools: list[ChatTool] | None = None
    ) -> list[int]:
        text = "|".join(f"{m.role}:{m.content or ''}" for m in messages)
        prefix = "T|" if tools else ""
        return [ord(c) for c in f"{prefix}{text}|assistant>"]

    def render_assistant_turn(
        self, messages: list[ChatMessage], index: int, tools: list[ChatTool] | None = None
    ) -> list[int]:
        message = messages[index]
        if not (message.content or message.tool_calls):
            return []
        content = message.content
        rendered = content if isinstance(content, str) else ""
        for call in message.tool_calls or []:
            rendered = f"{rendered}<{call.function.name}>"
        return [ord(c) for c in rendered]


def _cfg() -> DistillConfig:
    return DistillConfig.model_validate(
        {
            "student": {"base_model": "Qwen/Qwen3.5-9B"},
            "teacher": {"model": "Qwen/Qwen3.6-27B"},
            "tau2": {"tau2_bin": "/x/tau2", "data_dir": "/x/data"},
            "train": {"steps": 0},
            "warmup": {"steps": 1},
        }
    )


def _episode(**overrides: object) -> TeacherEpisode:
    base = {
        "task_id": "airline/7",
        "messages": [
            ChatMessage(role="system", content="policy"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="hi there"),
            ChatMessage(role="user", content="do it"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ChatToolCall(
                        id="c1", function=ChatFunctionCall(name="book", arguments='{"id": "7"}')
                    )
                ],
            ),
            ChatMessage(role="tool", content="ok", tool_call_id="c1"),
            ChatMessage(role="assistant", content="done"),
        ],
        "reward": 1.0,
        "passed": True,
        "teacher_model": "kimi-k3",
        "source": "fireworks",
    }
    return TeacherEpisode.model_validate(base | overrides)


class TestEpisodeValidation:
    def test_transcript_without_an_assistant_turn_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no assistant turn"):
            _episode(
                messages=[
                    ChatMessage(role="system", content="p"),
                    ChatMessage(role="user", content="u"),
                ]
            )

    def test_transcript_opening_on_an_assistant_turn_is_rejected(self) -> None:
        # Real tau2 transcripts DO open on the agent's greeting; the reader must
        # supply the system prompt, and this is the check that forces it to.
        with pytest.raises(ValueError, match="opens on an assistant turn"):
            _episode(messages=[ChatMessage(role="assistant", content="hi")])

    def test_trial_name_is_filesystem_safe(self) -> None:
        assert _episode().trial_name == "airline-7-a01"


class TestSpans:
    def test_one_span_per_assistant_turn_with_placeholder_logprobs(self) -> None:
        spans = episode_spans(_episode(), _StubRendering())
        assert len(spans) == 3
        assert [s.call_index for s in spans] == [0, 1, 2]
        for span in spans:
            assert span.logprobs_are_placeholders is True
            assert len(span.sampled_logprobs) == len(span.sampled_token_ids)
            assert set(span.sampled_logprobs) == {0.0}

    def test_tool_call_turns_are_trained_on(self) -> None:
        spans = episode_spans(_episode(), _StubRendering())
        rendered = "".join(chr(c) for c in spans[1].sampled_token_ids)
        assert rendered == "<book>", "a turn whose whole output is a tool call must train"

    def test_empty_assistant_turns_are_skipped_and_indices_stay_contiguous(self) -> None:
        episode = _episode(
            messages=[
                ChatMessage(role="system", content="p"),
                ChatMessage(role="user", content="u"),
                ChatMessage(role="assistant", content=""),
                ChatMessage(role="user", content="u2"),
                ChatMessage(role="assistant", content="real"),
            ]
        )
        spans = episode_spans(episode, _StubRendering())
        assert [s.call_index for s in spans] == [0]


class TestRecordsAndDatums:
    def test_records_carry_outcome_and_need_no_disk(self) -> None:
        [record] = episodes_to_trial_records([_episode()], _StubRendering())
        assert record.task_id == "airline/7"
        assert record.passed is True
        assert record.infra_failed is False
        assert record.artifact_dir == ""
        assert len(record.spans) == 3

    def test_datums_are_hard_target_only_and_train_the_turns(self) -> None:
        records = episodes_to_trial_records([_episode()], _StubRendering())
        datums, _stats = build_datums(records, _cfg())
        assert datums, "the episode must yield trainable datums"
        assert all(d.hard_targets_only for d in datums)
        assert all(sum(d.loss_mask) > 0 for d in datums), "every datum needs loss tokens"

    def test_the_advantage_path_refuses_text_derived_datums(self) -> None:
        # The guard that makes the whole bridge safe: placeholder logprobs must
        # never become a reverse-KL baseline.
        records = episodes_to_trial_records([_episode()], _StubRendering())
        datums, _ = build_datums(records, _cfg())
        teacher_logprobs = [[None] * len(d.model_input_tokens) for d in datums]
        with pytest.raises(ValueError, match="placeholder logprobs"):
            attach_advantages(datums, teacher_logprobs, _cfg())


class TestManifest:
    def test_manifest_records_the_teacher_and_every_episode(self) -> None:
        episodes = [_episode(), _episode(attempt=2)]
        manifest = text_warmup_manifest(episodes, _StubRendering(), teacher_model="kimi-k3")
        assert manifest.teacher_model == "kimi-k3"
        assert [r.trial_name for r in manifest.records] == ["airline-7-a01", "airline-7-a02"]

    def test_a_mixed_teacher_corpus_is_rejected(self) -> None:
        episodes = [_episode(), _episode(attempt=2, teacher_model="Qwen/Qwen3.6-27B")]
        with pytest.raises(ValueError, match="mixing teachers"):
            text_warmup_manifest(episodes, _StubRendering(), teacher_model="kimi-k3")

    def test_an_empty_corpus_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="zero teacher episodes"):
            text_warmup_manifest([], _StubRendering(), teacher_model="kimi-k3")

    def test_manifest_round_trips_through_the_run_store(self, tmp_path: Path) -> None:
        store = DistillRunStore(tmp_path)
        manifest = text_warmup_manifest([_episode()], _StubRendering(), teacher_model="kimi-k3")
        store.write_warmup_trials(manifest)
        loaded = store.read_warmup_trials()
        assert loaded is not None
        assert loaded.teacher_model == "kimi-k3"
        assert loaded.records[0].spans[0].logprobs_are_placeholders is True


class TestRealRenderer:
    """The property that decides whether this bridge produces trainable data."""

    def test_real_qwen_rendering_produces_loss_bearing_turns(self) -> None:
        pytest.importorskip("tinker_cookbook")
        from wmo.distill.rendering import build_offline_rendering

        try:
            rendering = build_offline_rendering("Qwen/Qwen3.5-9B")
        except OSError:
            pytest.skip("the Qwen3.5 tokenizer is not cached locally")
        episode = _episode(
            tools=[
                ChatTool(
                    function=FunctionDef(
                        name="book",
                        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
                    )
                )
            ]
        )
        spans = episode_spans(episode, rendering)
        assert len(spans) == 3
        for span in spans:
            assert span.sampled_token_ids, "every kept turn must render to tokens"
            assert span.prompt_token_ids
        # The prompt of a later turn extends the earlier prompt (the shared
        # system/user prefix), even though the primed thinking block means it
        # does not extend prompt+sampled (see the module docstring).
        assert (
            spans[1].prompt_token_ids[: len(spans[0].prompt_token_ids) - 4]
            == (spans[0].prompt_token_ids[: len(spans[0].prompt_token_ids) - 4])
        )
        datums, _ = build_datums(episodes_to_trial_records([episode], rendering), _cfg())
        assert datums
        assert all(d.hard_targets_only for d in datums)

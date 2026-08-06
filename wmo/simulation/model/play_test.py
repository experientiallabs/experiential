"""Tests for the interactive play engine: line parsing and a human-driven turn (no network)."""

from __future__ import annotations

import pytest

from wmo.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmo.simulation.model.play import parse_action, play_turn
from wmo.simulation.model.world_model import WorldModel
from wmo.simulation.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(
            text='{"output": "user u1 found", "is_error": false, "state_note": "looked up u1"}'
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_parse_action_tool_call_with_json_args() -> None:
    action = parse_action('get_user {"id": "u1"}')
    assert action.kind == ActionKind.TOOL_CALL
    assert action.name == "get_user"
    assert action.arguments == {"id": "u1"}


def test_parse_action_bare_tool_name_has_no_args() -> None:
    action = parse_action("list_flights")
    assert action.kind == ActionKind.TOOL_CALL
    assert action.name == "list_flights"
    assert action.arguments == {}


def test_parse_action_say_prefix_forces_message() -> None:
    action = parse_action("say hello there")
    assert action.kind == ActionKind.MESSAGE
    assert action.content == "hello there"


def test_parse_action_prose_is_a_message() -> None:
    action = parse_action("what is the weather?")
    assert action.kind == ActionKind.MESSAGE
    assert action.content == "what is the weather?"


def test_parse_action_rejects_empty_and_bad_json() -> None:
    with pytest.raises(ValueError, match="empty action"):
        parse_action("   ")
    with pytest.raises(ValueError, match="JSON object"):
        parse_action('get_user ["not", "an", "object"]')


def test_parse_action_non_ascii_first_word_is_a_message() -> None:
    # Tool names are ASCII identifiers; a non-ASCII first word is prose, not a tool call
    # (str.isalpha() would otherwise accept Unicode letters and misread it).
    action = parse_action("café {}")
    assert action.kind == ActionKind.MESSAGE
    action2 = parse_action("日本 hello")
    assert action2.kind == ActionKind.MESSAGE


def test_play_turn_steps_and_evolves_scratchpad() -> None:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    retriever.index(
        [
            Trace(
                trace_id="t",
                steps=[
                    Step(
                        action=Action(
                            kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u0"}
                        ),
                        observation=Observation(content="found u0"),
                    )
                ],
            )
        ]
    )
    wm = WorldModel(FakeProvider(), retriever, top_k=3)
    session = wm.new_session(task="look up users")

    turn = play_turn(wm, session.id, parse_action('get_user {"id": "u1"}'))
    assert turn.observation.content == "user u1 found"
    assert "get_user" in turn.env_prompt
    # The state note folds into the session scratchpad so later turns stay consistent.
    assert "looked up u1" in wm.get_session(session.id).state.scratchpad
    assert len(wm.get_session(session.id).history) == 1

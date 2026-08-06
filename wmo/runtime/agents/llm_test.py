"""Tests for the baseline LLM agent's reply parsing and turn rendering."""

from __future__ import annotations

from wmo.common.core.types import Action, ActionKind, EnvState, Observation, Step
from wmo.common.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmo.runtime.agents.llm import LLMAgent
from wmo.runtime.episode import DONE_SIGNAL


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._reply = reply
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class ScriptedProvider(FakeProvider):
    """Returns a fixed sequence of replies (last one repeats) and counts the calls it served."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies[0])
        self._replies = replies
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_user = messages[0].content
        index = min(self.calls, len(self._replies) - 1)
        self.calls += 1
        return Completion(text=self._replies[index])


def test_agent_parses_tool_call() -> None:
    agent = LLMAgent(FakeProvider('{"tool": "search", "arguments": {"q": "x"}}'))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"
    assert action.arguments == {"q": "x"}


def test_agent_parses_done() -> None:
    agent = LLMAgent(FakeProvider('{"done": true, "summary": "finished"}'))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == DONE_SIGNAL


def test_agent_surfaces_garbage_as_message() -> None:
    agent = LLMAgent(FakeProvider("I think I should search first"))
    action = agent.act("find x", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == "I think I should search first"


def test_agent_prompt_includes_task_state_and_history() -> None:
    provider = FakeProvider('{"done": true}')
    history = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="search", arguments={"q": "x"}),
            observation=Observation(content="found it", is_error=False),
        )
    ]
    LLMAgent(provider).act("find x", EnvState(scratchpad="db is empty"), history)
    prompt = provider.last_user or ""
    assert "TASK: find x" in prompt
    assert "db is empty" in prompt
    assert "search" in prompt and "found it" in prompt


def test_tools_hint_lands_in_the_system_prompt() -> None:
    class _CaptureProvider:
        def __init__(self) -> None:
            self.system = ""
            self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            self.system = system
            return Completion(text='{"done": true, "summary": "x"}')

        def embed(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def verify(self) -> VerifyResult:
            raise NotImplementedError

    provider = _CaptureProvider()
    agent = LLMAgent(provider, tools_hint="TOOLS: run_sql(query), list_tables()")
    agent.act("do it", EnvState(), [])
    assert "run_sql(query)" in provider.system
    bare = LLMAgent(_CaptureProvider())
    assert bare is not None  # no hint -> unchanged system (covered by existing tests)


def test_prose_tool_call_executes_instead_of_stalling() -> None:
    # glm-5.2 sometimes writes `get_user_details({"user_id": "x"})` instead of the envelope.
    # Reading that as done ended 34% of glm's wm episodes unexecuted at step 1; falling
    # through to a message action merely wasted the turn. Run the call the model meant.
    agent = LLMAgent(FakeProvider('get_user_details({"user_id": "chen_silva_2201"})'))
    action = agent.act("task", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "get_user_details"
    assert action.arguments == {"user_id": "chen_silva_2201"}


def test_prose_tool_call_fused_with_prose_is_recovered() -> None:
    reply = 'Let me look them up first. get_user_details({"user_id": "x"}) and then decide.'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "get_user_details"


def test_first_of_several_prose_calls_wins() -> None:
    reply = 'search({"q": "a"}) then update_order({"id": "b"})'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.name == "search"


def test_envelope_wins_over_a_later_prose_call() -> None:
    reply = '{"tool": "search", "arguments": {"q": "x"}} (not get_user_details({"id": "y"}))'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.name == "search"


def test_a_rejected_hypothesis_does_not_outrank_the_chosen_envelope() -> None:
    # A reasoning model deliberates before it decides. An earlier draft let the mentioned call
    # suppress the envelope, so this reply issued a DB write the model had just ruled out.
    reply = (
        'Thinking: the schema is update_order({"order_id": "A1"}). Not applicable here.\n'
        '```json\n{"done": true, "summary": "nothing to do"}\n```'
    )
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == DONE_SIGNAL


def test_envelope_is_found_after_a_leading_non_envelope_object() -> None:
    reply = 'Let me think about search({"q": 1}) as an option.\n{"tool": "list_orders"}'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "list_orders"


def test_a_done_envelope_wrapped_in_call_syntax_still_terminates() -> None:
    # `finish({"done": true})` must not become a call to a tool named "finish": that stopped
    # terminating the episode and burned the rest of the step budget.
    action = LLMAgent(FakeProvider('finish({"done": true, "summary": "all set"})')).act(
        "task", EnvState(), []
    )
    assert action.content == DONE_SIGNAL


def test_prose_naming_a_tool_before_an_envelope_does_not_become_the_tool() -> None:
    # The space before "(" is what tells prose apart from call syntax. Without that, this ran
    # a tool literally named "tool" with the whole envelope as its arguments.
    reply = 'I will use the search tool ({"tool": "search", "arguments": {"q": "x"}})'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"
    assert action.arguments == {"q": "x"}


def test_a_hyphenated_call_name_is_not_truncated_to_its_last_segment() -> None:
    # Matching only "details" invented a tool nobody offers; the whole name at least produces
    # an honest unknown-tool error from the env.
    action = LLMAgent(FakeProvider('get-user-details({"id": "x"})')).act("task", EnvState(), [])
    assert action.name == "get-user-details"


def test_unclosed_prose_call_still_falls_back_to_a_message() -> None:
    reply = 'get_user_details({"user_id": "x"'
    action = LLMAgent(FakeProvider(reply)).act("task", EnvState(), [])
    assert action.kind is ActionKind.MESSAGE
    assert action.content == reply


def test_explicit_done_still_terminates() -> None:
    agent = LLMAgent(FakeProvider('{"done": true}'))
    action = agent.act("task", EnvState(), [])
    assert action.content == DONE_SIGNAL


def test_empty_completion_is_retried() -> None:
    provider = ScriptedProvider(["", "   ", '{"tool": "search", "arguments": {}}'])
    action = LLMAgent(provider).act("task", EnvState(), [])
    assert provider.calls == 3
    assert action.kind is ActionKind.TOOL_CALL
    assert action.name == "search"


def test_empty_completion_retries_are_bounded() -> None:
    provider = ScriptedProvider([""])
    action = LLMAgent(provider).act("task", EnvState(), [])
    assert provider.calls == 3  # the first call plus two retries, then give up
    assert action.kind is ActionKind.MESSAGE
    assert action.content == ""


def test_a_good_first_completion_is_not_retried() -> None:
    provider = ScriptedProvider(['{"done": true}'])
    LLMAgent(provider).act("task", EnvState(), [])
    assert provider.calls == 1


def test_history_observations_survive_a_tau_sized_payload() -> None:
    # The 500-char cap truncated `get_user_details` payloads before their useful fields, so
    # the agent re-fetched the same record verbatim until the step budget died.
    payload = "x" * 900 + "MEMBERSHIP=gold"
    provider = FakeProvider('{"done": true}')
    history = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get_user_details", arguments={}),
            observation=Observation(content=payload),
        )
    ]
    LLMAgent(provider).act("task", EnvState(), history)
    assert "MEMBERSHIP=gold" in (provider.last_user or "")


def test_history_chars_bounds_the_rendered_observation() -> None:
    provider = FakeProvider('{"done": true}')
    history = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get_user_details", arguments={}),
            observation=Observation(content="y" * 50 + "TAIL"),
        )
    ]
    LLMAgent(provider, history_chars=10).act("task", EnvState(), history)
    assert "TAIL" not in (provider.last_user or "")


def test_history_chars_bounds_the_message_fallback() -> None:
    action = LLMAgent(FakeProvider("z" * 40), history_chars=10).act("task", EnvState(), [])
    assert action.content == "z" * 10


def test_empty_retries_can_be_disabled_for_latency_measurement() -> None:
    provider = ScriptedProvider([""])
    action = LLMAgent(provider, empty_retries=0).act("task", EnvState(), [])
    assert provider.calls == 1
    assert action.kind is ActionKind.MESSAGE

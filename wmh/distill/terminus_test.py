"""Tests for the clean context stop on distillation rollouts.

These run harbor's REAL `Terminus2.run`, its real `Chat`, and (in
`test_the_stop_reaches_the_verifier_where_the_raise_did_not`) harbor's real
`SingleStepTrial._run_agent`, with only the per-episode loop body and the LLM
replaced. The whole point of the change is what harbor does with the exception,
so a test that stubbed harbor out would prove nothing.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import pytest
from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import BaseLLM, ContextLengthExceededError, LLMResponse
from harbor.llms.chat import Chat
from harbor.models.agent.context import AgentContext
from harbor.models.metric.usage_info import UsageInfo
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from harbor.utils.import_path import import_class

from wmh.distill.rollouts import TERMINUS_2_AGENT_IMPORT_PATH, rollout_stats
from wmh.distill.terminus import CleanStopTerminus2
from wmh.distill.tokens import (
    TERMINUS_STOP_REASON_METADATA_KEY,
    assemble_harbor_trial_records,
    read_terminus_stop_reason,
)
from wmh.harness.runtime import SCAFFOLD_LOSS_STOP_REASONS, StopReason
from wmh.harness.scoring import ScoreCell

_MAX_TURNS = 100
_ANSWERED_TURNS = 3
"""Turns the fake student answers before the next prompt no longer fits."""


class _OverflowingLLM(BaseLLM):
    """A student that answers `turns` times, then cannot be prompted again.

    Emits the same three per-turn lists `harbor.llms.tinker.TinkerLLM` emits under
    `collect_rollout_details=True`, so harbor's `Chat` records real spans, and then
    raises the error TinkerLLM raises when the rendered prompt outgrows the context
    limit (`harbor/llms/tinker.py:206`), BEFORE it samples anything.

    Each turn's prompt EXTENDS the previous turn's prompt plus its completion plus a
    rendered user delta, which is the verbatim prefix chain a live episode records and
    the one `load_trial_rollout_spans` checks before it trusts the stored chat history.
    A fixture that skipped it would leave the whole canonical-delta half of the
    tokens-in-tokens-out contract unexercised.
    """

    def __init__(self, turns: int) -> None:
        super().__init__()
        self.calls = 0
        self._turns = turns
        self._prompt_ids = [1, 2]

    async def call(self, prompt: str, **kwargs: object) -> LLMResponse:
        if self.calls >= self._turns:
            raise ContextLengthExceededError(
                "Prompt length (65100 tokens) exceeds model context limit (65024 tokens)"
            )
        index = self.calls
        self.calls += 1
        prompt_ids = self._prompt_ids
        completion_ids = [900 + index, 901 + index]
        self._prompt_ids = [*prompt_ids, *completion_ids, 500 + index]
        return LLMResponse(
            content=f"reply {index}",
            usage=UsageInfo(prompt_tokens=10, completion_tokens=2, cache_tokens=0, cost_usd=0.0),
            prompt_token_ids=prompt_ids,
            completion_token_ids=completion_ids,
            logprobs=[-0.5, -1.5],
        )

    def get_model_context_limit(self) -> int:
        return 65024

    def get_model_output_limit(self) -> int | None:
        return 16384


class _FakeSession:
    """The two tmux behaviors `Terminus2.run` needs before it enters its loop."""

    async def get_incremental_output(self) -> str:
        return "root@host:/app#"

    async def is_session_alive(self) -> bool:
        return True


_NO_ENVIRONMENT = cast("BaseEnvironment", None)
"""Terminus-2 touches `environment` only through `_build_skills_section`, which returns
before it at an unset `skills_dir` (`terminus_2.py:443`), so there is nothing to fake."""


async def _echo_loop(self: Terminus2, initial_prompt: str, chat: Chat, **_: object) -> None:
    """Terminus-2's episode loop with the parsing and terminal work removed.

    Keeps the two things that matter to this change: `_n_episodes` is bumped BEFORE
    the LLM call (as at `terminus_2.py:1255`, which is why a stop on the last allowed
    turn would otherwise read as the turn cap), and the call itself goes through the
    real `Chat`, so rollout details and `all_messages` accumulate exactly as in a
    live trial.
    """
    prompt = initial_prompt
    for episode in range(self._max_episodes):
        self._n_episodes = episode + 1
        response = await chat.chat(prompt)
        prompt = f"observation after {response.content}"


def _agent(
    tmp_path: Path,
    *,
    agent_class: type[Terminus2] = CleanStopTerminus2,
    answered_turns: int = _ANSWERED_TURNS,
    max_turns: int = _MAX_TURNS,
) -> tuple[Terminus2, _OverflowingLLM]:
    """A real agent wired the way `terminus_2_agent_kwargs` wires the live one."""
    logs_dir = tmp_path / "trial" / "agent"
    logs_dir.mkdir(parents=True, exist_ok=True)
    agent = agent_class(
        logs_dir=logs_dir,
        model_name="openai/gpt-4o",
        max_turns=max_turns,
        collect_rollout_details=True,
        enable_summarize=False,
        store_all_messages=True,
        suppress_max_turns_warning=True,
    )
    llm = _OverflowingLLM(answered_turns)
    agent._llm = llm
    agent._session = cast("TmuxSession", _FakeSession())
    return agent, llm


def _run_episode(agent: Terminus2) -> AgentContext:
    context = AgentContext()
    asyncio.run(agent.run("solve the task", _NO_ENVIRONMENT, context))
    return context


@pytest.fixture(autouse=True)
def _stub_episode_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Terminus2, "_run_agent_loop", _echo_loop)


def test_the_import_path_resolves_to_the_clean_stop_subclass() -> None:
    """Harbor's agent factory imports this by string, so the string has to resolve."""
    resolved = import_class(TERMINUS_2_AGENT_IMPORT_PATH, label="agent")

    assert resolved is CleanStopTerminus2
    assert issubclass(CleanStopTerminus2, Terminus2)
    # Same agent identity to harbor: the trial dirs and the ATIF trajectory keep their names.
    assert CleanStopTerminus2.name() == Terminus2.name()


def test_a_context_stop_keeps_every_span_sampled_before_it(tmp_path: Path) -> None:
    """The turns that DID complete are valid training data and must all survive.

    Harbor's `Chat` appends to `rollout_details` and to `messages` only after a call
    returns, so the overflowing turn contributes nothing to either and cannot
    desynchronize them. What this asserts is that the stop does not cost the episode
    anything else either: the exact sampled ids, their logprobs, and the 2:1
    message alignment the cross-tokenizer teacher joins on are all still there.
    """
    agent, llm = _agent(tmp_path)

    context = _run_episode(agent)

    assert llm.calls == _ANSWERED_TURNS
    [detail] = context.rollout_details or []
    prompts = detail["prompt_token_ids"]
    completions = detail["completion_token_ids"]
    assert completions == [[900, 901], [901, 902], [902, 903]]
    assert detail["logprobs"] == [[-0.5, -1.5]] * _ANSWERED_TURNS
    # The verbatim prefix property, turn by turn: nothing was rewritten on the way out.
    for index in range(1, _ANSWERED_TURNS):
        carried = [*prompts[index - 1], *completions[index - 1]]
        assert prompts[index][: len(carried)] == carried
    assert context.metadata is not None
    # One [user, assistant] pair per recorded turn: the canonical view stays aligned.
    assert len(context.metadata["all_messages"]) == 2 * _ANSWERED_TURNS
    # The failed turn is counted as an episode entered, not as a turn recorded.
    assert context.metadata["n_episodes"] == _ANSWERED_TURNS + 1
    assert json.loads((agent.logs_dir / "trajectory.json").read_text(encoding="utf-8"))


def test_a_context_stop_records_its_own_reason(tmp_path: Path) -> None:
    """Not `submitted` (nothing was claimed) and not `max_turns` (the cap was never hit)."""
    agent, _ = _agent(tmp_path)

    context = _run_episode(agent)

    assert context.metadata is not None
    assert context.metadata[TERMINUS_STOP_REASON_METADATA_KEY] == "context_exhausted"
    assert StopReason.CONTEXT_EXHAUSTED in SCAFFOLD_LOSS_STOP_REASONS


def test_the_stop_outranks_the_turn_cap_it_coincides_with(tmp_path: Path) -> None:
    """An overflow on the last allowed turn is still the overflow, not the cap.

    `_n_episodes` is bumped before the call that fails, so an episode whose context
    runs out on turn `max_turns` records exactly `max_turns` episodes and would read
    as a cap stop from the episode count alone. The agent's own account is read first
    for this reason: a reader chasing `max_turns` would raise a cap that never bound.
    """
    agent, _ = _agent(tmp_path, answered_turns=4, max_turns=5)
    trial_dir = tmp_path / "trial"

    context = _run_episode(agent)
    _write_result(trial_dir, {"agent_result": context.model_dump(mode="json")})

    assert context.metadata is not None
    assert context.metadata["n_episodes"] == 5
    assert read_terminus_stop_reason(trial_dir, max_turns=5) == "context_exhausted"


def test_other_llm_failures_still_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the context overflow is converted; a real failure must still fail the trial."""
    agent, _ = _agent(tmp_path)

    async def _die(self: Terminus2, initial_prompt: str, chat: Chat, **_: object) -> None:
        raise AgentTimeoutError("the agent phase ran out of wall clock")

    monkeypatch.setattr(Terminus2, "_run_agent_loop", _die)

    with pytest.raises(AgentTimeoutError):
        _run_episode(agent)


def test_the_stop_reaches_the_verifier_where_the_raise_did_not(tmp_path: Path) -> None:
    """The mechanism itself: harbor's own agent phase, run against both agents.

    `SingleStepTrial._run_agent` catches `AgentTimeoutError` and
    `NonZeroAgentExitCodeError` and nothing else, so a propagating
    `ContextLengthExceededError` escapes it, escapes `_run()`, and is recorded by
    `Trial.run` as a trial-level exception. `_run_verifier` is the statement AFTER
    the one that raised, so it never executes and the trial ends with no reward for
    the scorer's key. This drives that real harbor method (bound to a stand-in trial,
    since a real one needs an environment) with the stock agent and with ours, and
    asserts the difference is exactly whether the exception escapes.
    """

    async def phase(agent: Terminus2) -> None:
        await agent.run("solve the task", _NO_ENVIRONMENT, AgentContext())

    stock, _ = _agent(tmp_path / "stock", agent_class=Terminus2)
    with pytest.raises(ContextLengthExceededError):
        asyncio.run(_harbor_agent_phase(stock, phase))

    clean, _ = _agent(tmp_path / "clean")
    # Returns, so harbor goes on to collect artifacts and run the verifier.
    asyncio.run(_harbor_agent_phase(clean, phase))


def test_the_stopped_trial_is_graded_and_keeps_its_spans(tmp_path: Path) -> None:
    """End to end over the collector's own vocabulary: graded, counted, not infra-failed.

    Two trials, one per outcome of the SAME overflow. The propagating one is what the
    run produces today: harbor records the exception, never verifies, and the scorer
    has no reward for its key, so the cell is `infra_failed` and leaves the solve-rate
    denominator. The stopped one is verified like any other trial, so it scores its
    real 0.0, keeps every span it sampled, and reports `context_exhausted`.
    """
    stopped, _ = _agent(tmp_path / "stopped")
    stopped_context = _run_episode(stopped)
    stopped_dir = tmp_path / "stopped" / "trial"
    _write_result(stopped_dir, {"agent_result": stopped_context.model_dump(mode="json")})

    died_dir = tmp_path / "died" / "trial"
    _write_result(
        died_dir,
        {
            "exception_info": {"exception_type": "ContextLengthExceededError"},
            "agent_result": {"metadata": {"n_episodes": _ANSWERED_TURNS + 1}},
        },
    )

    records = assemble_harbor_trial_records(
        [
            ScoreCell(
                task_id="task-a",
                attempt=1,
                reward=0.0,
                passed=False,
                artifact_dir=str(stopped_dir),
            ),
            ScoreCell(
                task_id="task-b",
                attempt=1,
                reward=0.0,
                passed=False,
                artifact_dir=str(died_dir),
                infra_failed=True,
                note="infra-failure: ContextLengthExceededError; no verifier evidence",
            ),
        ],
        max_turns=_MAX_TURNS,
    )
    stats = rollout_stats(records, max_tokens=16384)

    graded, ungraded = records
    assert graded.stop_reason == "context_exhausted"
    assert graded.infra_failed is False
    # Every completed turn is still training data, carrying the exact sampled ids.
    assert [span.sampled_token_ids for span in graded.spans] == [
        [900, 901],
        [901, 902],
        [902, 903],
    ]
    # And still SCORABLE by a cross-tokenizer teacher: the canonical message delta of
    # every surviving turn came through, which only holds while the prefix chain does.
    assert [span.delta_start for span in graded.spans] == [0, 2, 4]
    assert all(span.delta_messages is not None for span in graded.spans)
    assert ungraded.stop_reason == "provider_error"
    assert ungraded.infra_failed is True

    # The stopped trial is IN the solve-rate denominator; the dead one is not.
    assert (stats.trials, stats.executed_trials, stats.infra_failed_trials) == (2, 1, 1)
    assert stats.stop_reason_counts == {"context_exhausted": 1, "provider_error": 1}
    assert (stats.trials_with_spans, stats.trials_without_delta) == (1, 0)
    # Both are scaffold losses: neither model declared itself done.
    assert stats.scaffold_loss_rate == 1.0


def _write_result(trial_dir: Path, payload: dict[str, object]) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")


async def _harbor_agent_phase(
    agent: Terminus2, phase: Callable[[Terminus2], Awaitable[None]]
) -> None:
    """Run harbor's real `SingleStepTrial._run_agent` around one agent phase.

    Bound to a stand-in `self` rather than a real trial: constructing one needs a
    task, a config and a live environment, none of which this is about. Everything
    the method itself does (which exceptions it catches, what it lets escape) is
    harbor's own pinned code.
    """

    task_config = type("_TaskConfig", (), {"agent": type("_AgentConfig", (), {"user": None})()})()

    class _StandInTrial:
        result = object()
        task = type("_Task", (), {"instruction": "solve the task", "config": task_config})()
        _agent_timeout_sec = 60.0

        async def _run_agent_phase(self, **_: object) -> None:
            await phase(agent)

        def _record_exception(self, exc: BaseException) -> None:
            raise AssertionError(f"harbor recorded {exc!r} as a trial-level failure")

        async def _sync_agent_output(self, _: object) -> None:
            return None

    await SingleStepTrial._run_agent(cast("SingleStepTrial", _StandInTrial()))

"""Terminus-2 with a clean context stop, so an overflowed episode is still graded.

Harbor's terminus-2 has exactly two answers when the next prompt no longer fits
the model's context, and distillation can use neither:

- **Summarize.** `_query_llm` catches `ContextLengthExceededError` and falls back
  to `_unwind_messages_to_free_tokens` plus a summary handoff
  (`harbor/agents/terminus_2/terminus_2.py:1014-1067`). That rewrites the chat
  history mid-episode, which destroys the verbatim token prefix the
  cross-tokenizer teacher joins on, so `wmh.distill.rollouts` pins
  `enable_summarize=False` and this path is unavailable by construction.
- **Die.** With summarization off the same handler re-raises. For a SINGLE-STEP
  task, which every TerminalBench-2 task is, the exception escapes
  `SingleStepTrial._run_agent` (`harbor/trial/single_step.py:74-84` catches only
  `AgentTimeoutError` and `NonZeroAgentExitCodeError`), so `_run()` never reaches
  `_run_verifier` and the trial finishes with an `exception_info` and NO verifier
  result. The scorer then has no reward for its key, marks the cell `infra_failed`
  and EXCLUDES it from the solve rate.

That exclusion is a measurement failure, not a scoring one: 6 of 16 holdout
trials vanished this way in one run, 8 of 17 and 5 of 17 in two earlier ones, so
the comparison rested on whatever short tasks survived. An episode stopped at the
TURN CAP has the opposite fate: `run()` returns, the verifier grades whatever the
agent left behind, and the trial stays in the denominator scoring what it earned.

`CleanStopTerminus2` converts the first outcome into the second, and does it in a
wmh subclass rather than a patch because harbor is a pinned wheel (`harbor==0.20.0`)
whose `.venv` copy any `uv sync` restores. `wmh.distill.rollouts` points harbor's
agent factory here instead of at `Terminus2` directly; everything else about the
agent, including every rollout-detail invariant, is inherited untouched.

Reactive (catch the error) rather than proactive (project the next prompt against
the ceiling before sending it), because the reactive catch is already exact and the
proactive version would not be:

- Nothing is sampled on the overflowing turn either way. Harbor's `TinkerLLM`
  measures the rendered prompt and raises BEFORE it calls `sample_async`
  (`harbor/llms/tinker.py:200-210`), and a server-side rejection returns no tokens
  either. `Chat.chat` appends to `rollout_details` and to `messages` only after a
  successful call (`harbor/llms/chat.py:83-121`), so the failed turn leaves no
  half-recorded span and no unpaired message. Catching one frame further out costs
  nothing that a pre-check would have saved.
- Harbor's own check IS the projection, at the budget wmh sets: `context_limit`
  comes from `rollout.context_budget_tokens`, which the TB2 run pins to 65,024
  precisely to sit under the 65,530 the sampler actually allows. A second
  projection would have to re-render the prompt to count it (duplicating harbor
  internals that the pinned wheel is free to change) and reach through
  `TinkerLLM`'s private `_service_client` for `get_server_capabilities()`, to
  re-derive a ceiling the run config already sits below.
- Reactive also covers the case a projection cannot: a rejection that only the
  SERVER can detect, which harbor maps back to the same exception type
  (`harbor/llms/tinker.py:272-284`).

The stop is recorded as `StopReason.CONTEXT_EXHAUSTED` in the agent's own metadata,
never as `submitted` (the agent claimed nothing) and never as `max_turns` (that
would hide the cause and send a reader to raise a cap that was never reached).
"""

from __future__ import annotations

import logging
from typing import override

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext

from wmh.distill.tokens import TERMINUS_STOP_REASON_METADATA_KEY
from wmh.harness.runtime import StopReason

logger = logging.getLogger(__name__)


def _completed_turns(context: AgentContext) -> int:
    """How many turns harbor recorded token ids for on the main agent chat.

    The overflowing turn is not among them: `Chat.chat` appends to
    `rollout_details` only after the LLM call returns, so this is exactly the
    training evidence the stop preserves.
    """
    details = context.rollout_details
    if not details:
        return 0
    return len(details[0].get("completion_token_ids", []))


class CleanStopTerminus2(Terminus2):
    """Harbor's terminus-2, ending a context-exhausted episode instead of failing it."""

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the episode, returning normally when the context runs out.

        Everything harbor records about a finished episode is already recorded by
        the time the overflow gets here: `Terminus2.run` populates
        `context.rollout_details`, the token/cost totals and
        `context.metadata` (`n_episodes`, `all_messages`) inside a `finally`, and
        dumps the ATIF trajectory there too, all of which runs BEFORE the exception
        leaves it (`harbor/agents/terminus_2/terminus_2.py:1606-1643`). So this
        override adds one metadata key and swallows the error; it neither replays
        nor rewrites anything, and every turn that completed keeps the exact ids
        the sampler issued.

        Args:
            instruction: The task instruction, passed through.
            environment: The task environment, passed through.
            context: Harbor's per-run output record; carries the stop marker back
                into the trial's `result.json` under `agent_result.metadata`.
        """
        try:
            await super().run(instruction, environment, context)
        except ContextLengthExceededError as error:
            metadata = context.metadata if context.metadata is not None else {}
            metadata[TERMINUS_STOP_REASON_METADATA_KEY] = StopReason.CONTEXT_EXHAUSTED.value
            context.metadata = metadata
            logger.warning(
                "terminus-2 episode ran out of context after %s completed turn(s) and was "
                "STOPPED rather than failed, so the trial is still verified and still counts "
                "toward the solve rate (stop reason %s): %s",
                _completed_turns(context),
                StopReason.CONTEXT_EXHAUSTED.value,
                error,
            )

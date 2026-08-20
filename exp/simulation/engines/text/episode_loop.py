"""World-terminal-driven text episode orchestration over the customer agent contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from exp.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from exp.common.models import AssistantAction
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode, AgentRuntime, execute_agent_episode
from exp.simulation.engines.text.environment import TextOnlyEnvironmentRuntime
from exp.simulation.engines.text.recording import RecordingCandidateClient


@dataclass(frozen=True)
class TextEpisodeOutcome:
    """Terminal result of looping a customer agent until the text world model ends the task."""

    episodes: tuple[AgentEpisode, ...]
    stop_reason: StopReason
    failure: StructuredFailure | None
    final_output: AssistantAction | None


def execute_text_episode_loop(
    *,
    agent_factory: Callable[[], AgentRuntime],
    task: TaskCase,
    recorder: RecordingCandidateClient,
) -> TextEpisodeOutcome:
    """Drive candidate and world turns until an explicit world terminal or pinned limit.

    A customer ``AgentRuntime`` owns its own internal loop and may return ``COMPLETED`` after one
    candidate call. That state alone never ends text simulation: a nonterminal world response is
    appended to the visible transcript and the same agent is invoked again. This preserves the
    normal customer runtime seam while making the simulator, rather than a one-turn adapter,
    authoritative for scenario termination.

    Exhausting the pinned candidate turn ceiling is a judgeable episode outcome, not an
    infrastructure failure: the recorded transcript is complete evidence that the candidate did
    not finish the task within the pinned budget, so the cell stops with ``MAXIMUM_STEPS`` and
    no structured failure.

    Args:
        agent_factory: Creates one isolated customer runtime for the simulation cell.
        task: Canonical text-only representative task.
        recorder: Candidate client that records each candidate and world-model transition.

    Returns:
        Complete internal agent evidence and the world-terminal-driven cell outcome.
    """
    agent = agent_factory()
    episodes: list[AgentEpisode] = []
    while True:
        prior_turn_count = recorder.candidate_turn_count
        episode = execute_agent_episode(
            agent,
            TextOnlyEnvironmentRuntime(),
            task,
            recorder,
        )
        episodes.append(episode)
        if recorder.world_model_terminal:
            return TextEpisodeOutcome(
                episodes=tuple(episodes),
                stop_reason=StopReason.COMPLETED,
                failure=None,
                final_output=episode.final_action or recorder.last_candidate_action,
            )
        text_error = recorder.terminal_error
        if text_error is not None:
            judgeable = text_error.stop_reason in (
                StopReason.COMPLETED,
                StopReason.MAXIMUM_STEPS,
            )
            return TextEpisodeOutcome(
                episodes=tuple(episodes),
                stop_reason=text_error.stop_reason,
                failure=None if judgeable else text_error.failure,
                final_output=episode.final_action or recorder.last_candidate_action,
            )
        if recorder.turn_limit_reached:
            return TextEpisodeOutcome(
                episodes=tuple(episodes),
                stop_reason=StopReason.MAXIMUM_STEPS,
                failure=None,
                final_output=episode.final_action or recorder.last_candidate_action,
            )
        if episode.stop_reason == StopReason.COMPLETED:
            if recorder.candidate_turn_count == prior_turn_count:
                failure = StructuredFailure(
                    code=FailureCode.INTERNAL,
                    message="customer agent completed without requesting a candidate turn",
                    attribution=FailureAttribution.AGENT,
                    details={"phase": "agent_no_progress"},
                )
                return TextEpisodeOutcome(
                    episodes=tuple(episodes),
                    stop_reason=StopReason.FAILURE,
                    failure=failure,
                    final_output=episode.final_action or recorder.last_candidate_action,
                )
            continue
        return TextEpisodeOutcome(
            episodes=tuple(episodes),
            stop_reason=episode.stop_reason,
            failure=episode.failure,
            final_output=episode.final_action or recorder.last_candidate_action,
        )

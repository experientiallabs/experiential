"""World-terminal-driven text episode orchestration over the customer agent contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from wmo.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure
from wmo.common.models import AssistantAction
from wmo.common.rollouts import StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.agents import AgentEpisode, AgentRuntime, execute_agent_episode
from wmo.simulation.engines.text.environment import TextOnlyEnvironmentRuntime
from wmo.simulation.engines.text.recording import RecordingCandidateClient


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
            return TextEpisodeOutcome(
                episodes=tuple(episodes),
                stop_reason=text_error.stop_reason,
                failure=(
                    None if text_error.stop_reason == StopReason.COMPLETED else text_error.failure
                ),
                final_output=episode.final_action or recorder.last_candidate_action,
            )
        limit_error = recorder.terminal_limit_error()
        if limit_error is not None:
            return TextEpisodeOutcome(
                episodes=tuple(episodes),
                stop_reason=limit_error.stop_reason,
                failure=limit_error.failure,
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

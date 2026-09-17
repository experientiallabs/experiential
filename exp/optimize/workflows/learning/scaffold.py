"""Collect an existing customer-agent episode without implementing another agent loop."""

from dataclasses import dataclass

from exp.common.tasks import TaskCase
from exp.runtime.agents.interface import AgentEpisode, AgentRuntime
from exp.runtime.agents.lifecycle import execute_agent_episode
from exp.runtime.claas.client import LearningClient
from exp.runtime.environments.interface import EnvironmentRuntime


@dataclass(frozen=True)
class LearningEpisode:
    """Customer-agent evidence and the exact model calls available for explicit feedback."""

    episode: AgentEpisode
    response_ids: tuple[str, ...]


def run_learning_episode(
    *, client: LearningClient, agent: AgentRuntime, environment: EnvironmentRuntime, task: TaskCase
) -> LearningEpisode:
    """Run the supplied scaffold with its environment, retaining IDs even when the episode fails.

    The caller supplies scoring and chooses which response receives each signal. A world-model
    environment is one possible implementation of the existing executable environment contract.
    No feedback is inferred from an episode ending or propagated to earlier actions implicitly.
    """
    model = client.model_client()
    episode = execute_agent_episode(agent, environment, task, model)
    return LearningEpisode(episode=episode, response_ids=model.response_ids)

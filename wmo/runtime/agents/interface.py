"""Customer-agent contracts for one complete model-injected episode."""

from __future__ import annotations

import inspect
from typing import Protocol, runtime_checkable

from pydantic import model_validator

from wmo.common.core.artifacts import ContractModel, StructuredFailure
from wmo.common.models import AssistantAction, Usage
from wmo.common.rollouts import RolloutSpan, StopReason
from wmo.common.tasks import TaskCase
from wmo.runtime.environments import EnvironmentSession

# Mandatory W3 restack after c44569df is available in this worktree:
# replace the temporary object annotations in AgentRuntime.run, lifecycle.py, and pi.py with
# wmo.common.models.ModelClient. Update every temporary model fixture in interface_test.py,
# lifecycle_test.py, and pi_test.py with a conforming fake. Do not define a local protocol.


class AgentEpisode(ContractModel):
    """In-memory events and terminal state emitted by one customer-agent run."""

    events: tuple[RolloutSpan, ...] = ()
    final_action: AssistantAction | None = None
    stop_reason: StopReason
    usage: Usage | None = None
    failure: StructuredFailure | None = None

    @model_validator(mode="after")
    def _require_consistent_terminal_state(self) -> AgentEpisode:
        span_ids = tuple(event.span_id for event in self.events)
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("agent episode event IDs must be unique")
        for previous, current in zip(self.events, self.events[1:], strict=False):
            if current.started_at < previous.ended_at:
                raise ValueError("agent episode events must be ordered by completion time")
        if self.stop_reason == StopReason.FAILURE and self.failure is None:
            raise ValueError("failed agent episodes require a structured failure")
        if self.stop_reason != StopReason.FAILURE and self.failure is not None:
            raise ValueError("only failed agent episodes may contain a structured failure")
        return self


@runtime_checkable
class AgentRuntime(Protocol):
    """Runs one whole customer-agent episode with dependencies supplied by WMO."""

    def run(
        self,
        task: TaskCase,
        *,
        model: object,  # W3 restack: use the canonical common ModelClient.
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Run the agent's own loop with an injected model and execute-only environment.

        Args:
            task: The task and allowed tool schemas for this episode.
            model: The candidate model selected by WMO for this episode.
            environment: The simulator-owned session that permits only tool execution.

        Returns:
            Ordered agent events, terminal output, usage, and stop reason.
        """


class AgentAdapterPreflightError(ValueError):
    """A customer agent cannot expose the required model injection seam."""


def preflight_agent_runtime(agent: AgentRuntime) -> None:
    """Validate that a customer adapter accepts WMO's injected dependencies.

    Args:
        agent: Customer adapter expected to implement the whole-episode contract.

    Raises:
        AgentAdapterPreflightError: The adapter does not name keyword-addressable model and
            environment parameters.
    """
    try:
        parameters = inspect.signature(agent.run).parameters
    except (TypeError, ValueError) as exc:
        raise AgentAdapterPreflightError(
            "Customer agent adapters must expose "
            "run(task, *, model: ModelClient, environment: EnvironmentSession) -> AgentEpisode. "
            "Expose that method so WMO can inject each candidate model."
        ) from exc
    missing = tuple(
        name
        for name in ("model", "environment")
        if name not in parameters or parameters[name].kind is inspect.Parameter.POSITIONAL_ONLY
    )
    if missing:
        names = ", ".join(missing)
        raise AgentAdapterPreflightError(
            "Customer agent adapters must expose "
            "run(task, *, model: ModelClient, environment: EnvironmentSession) -> AgentEpisode. "
            f"Add keyword-addressable {names} parameter(s) so WMO can inject each candidate model."
        )

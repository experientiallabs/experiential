"""Project agent-factory resolution tests."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from exp.common.models import ModelClient
from exp.common.project import AgentConfiguration
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode, ChatAgentRuntime
from exp.runtime.agents.factory import (
    AgentFactoryError,
    preflight_agent_factory,
    resolve_agent_factory,
)
from exp.runtime.environments import EnvironmentSession


def test_absent_project_factory_uses_bounded_chat_runtime() -> None:
    """The standard happy path needs no customer factory module."""
    factory = resolve_agent_factory(None, maximum_model_calls=3)

    agent = factory()

    assert isinstance(agent, ChatAgentRuntime)
    preflight_agent_factory(factory)


def test_explicit_project_factory_remains_supported() -> None:
    """An explicit import reference creates a fresh validated runtime per call."""
    module_name = "exp_test_custom_agent_factory"
    module = ModuleType(module_name)
    module.__dict__["create_agent"] = _CustomAgent
    sys.modules[module_name] = module
    try:
        factory = resolve_agent_factory(
            AgentConfiguration(factory=f"{module_name}:create_agent"),
            maximum_model_calls=3,
        )

        first = factory()
        second = factory()

        assert isinstance(first, _CustomAgent)
        assert isinstance(second, _CustomAgent)
        assert first is not second
    finally:
        sys.modules.pop(module_name, None)


def test_invalid_custom_factory_fails_during_preflight() -> None:
    """Import and constructor failures are actionable before simulation dispatch."""
    with pytest.raises(AgentFactoryError, match="cannot import"):
        resolve_agent_factory(
            AgentConfiguration(factory="missing_exp_agent:create"),
            maximum_model_calls=3,
        )


class _CustomAgent:
    """Minimal compatible customer agent fixture."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        """Return a completed episode through the required injected signature.

        Args:
            task: Task supplied by the simulator.
            model: Candidate model supplied by EXP.
            environment: Execute-only environment supplied by the simulator.

        Returns:
            Completed fixture episode.
        """
        del task, model, environment
        return AgentEpisode(stop_reason=StopReason.COMPLETED)

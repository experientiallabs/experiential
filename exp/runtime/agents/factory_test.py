"""Project agent-factory resolution tests."""

from __future__ import annotations

import sys
from functools import partial
from types import ModuleType

import pytest

from exp.common.models import ModelClient
from exp.common.project import AgentConfiguration
from exp.common.rollouts import StopReason
from exp.common.tasks import TaskCase
from exp.runtime.agents import AgentEpisode, ChatAgentRuntime
from exp.runtime.agents.factory import (
    AgentFactoryError,
    is_builtin_chat_factory,
    preflight_agent_factory,
    resolve_agent_factory,
)
from exp.runtime.environments import EnvironmentSession


def test_absent_project_factory_uses_bounded_chat_runtime() -> None:
    """The standard happy path needs no customer factory module."""
    factory = resolve_agent_factory(None, maximum_model_calls=3)

    agent = factory()

    assert isinstance(agent, ChatAgentRuntime)
    assert is_builtin_chat_factory(factory)
    preflight_agent_factory(factory)


def test_builtin_identity_check_does_not_invoke_custom_code() -> None:
    """Factory identity admits exact built-ins without constructing custom wrappers."""
    calls = 0

    def custom() -> ChatAgentRuntime:
        """Track accidental initialization despite returning a built-in instance."""
        nonlocal calls
        calls += 1
        return ChatAgentRuntime()

    assert is_builtin_chat_factory(ChatAgentRuntime)
    assert is_builtin_chat_factory(partial(ChatAgentRuntime, maximum_model_calls=1000))
    assert not is_builtin_chat_factory(custom)
    assert not is_builtin_chat_factory(partial(custom))
    assert calls == 0


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

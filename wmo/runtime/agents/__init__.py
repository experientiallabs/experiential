"""Whole-episode customer-agent runtime contracts and the built-in Pi adapter."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wmo.runtime.agents.pi import PiAgentRuntime as PiAgentRuntime
    from wmo.runtime.agents.pi import PiInvocationTimeoutError as PiInvocationTimeoutError
    from wmo.runtime.agents.pi import PiRuntimePreflightError as PiRuntimePreflightError
    from wmo.runtime.agents.pi import PiTranscriptError as PiTranscriptError

from wmo.runtime.agents.chat import ChatAgentRuntime
from wmo.runtime.agents.factory import (
    AgentFactory,
    AgentFactoryError,
    agent_factory_sha256,
    preflight_agent_factory,
    resolve_agent_factory,
)
from wmo.runtime.agents.interface import (
    AgentAdapterPreflightError,
    AgentEpisode,
    AgentRuntime,
    preflight_agent_runtime,
)
from wmo.runtime.agents.lifecycle import execute_agent_episode

_PI_EXPORTS = frozenset(
    {
        "PiAgentRuntime",
        "PiInvocationTimeoutError",
        "PiRuntimePreflightError",
        "PiTranscriptError",
    }
)

__all__ = [
    "AgentAdapterPreflightError",
    "AgentEpisode",
    "AgentFactory",
    "AgentFactoryError",
    "AgentRuntime",
    "agent_factory_sha256",
    "ChatAgentRuntime",
    "PiAgentRuntime",
    "PiInvocationTimeoutError",
    "PiRuntimePreflightError",
    "PiTranscriptError",
    "execute_agent_episode",
    "preflight_agent_factory",
    "preflight_agent_runtime",
    "resolve_agent_factory",
]


def __getattr__(name: str) -> object:
    """Resolve one Pi export without loading the router HTTP stack at package import.

    The laziness is load-bearing, not a convenience: `wmo/cli/tests/startup_test.py` pins the CLI
    import to stay free of heavy third parties (fastapi arrives with the router endpoint that
    the Pi adapter reuses), so the adapter loads on first attribute access instead.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported public Pi adapter object loaded from its owning module.

    Raises:
        AttributeError: The name is not a lazy Pi export.
    """
    if name not in _PI_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("wmo.runtime.agents.pi"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy Pi exports."""
    return sorted(set(globals()) | _PI_EXPORTS)

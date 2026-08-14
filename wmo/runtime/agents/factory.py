"""Project agent-factory resolution with a built-in standard chat fallback."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

from wmo.common.core.artifacts import Sha256, sha256_json
from wmo.common.project import AgentConfiguration
from wmo.runtime.agents.chat import ChatAgentRuntime
from wmo.runtime.agents.interface import AgentRuntime, preflight_agent_runtime

AgentFactory = Callable[[], AgentRuntime]


class AgentFactoryError(ValueError):
    """A configured customer agent factory cannot provide the runtime contract."""


def agent_factory_sha256(
    configuration: AgentConfiguration | None,
    *,
    maximum_model_calls: int,
) -> Sha256:
    """Bind the effective built-in or custom agent configuration to one digest.

    Args:
        configuration: Optional project-owned custom factory reference.
        maximum_model_calls: Request ceiling applied when the built-in chat agent is selected.

    Returns:
        Stable semantic identity for simulation and replay checks.

    Raises:
        ValueError: The built-in request ceiling is not positive.
    """
    if maximum_model_calls <= 0:
        raise ValueError("maximum_model_calls must be positive")
    if configuration is not None and configuration.code_revision is None:
        raise ValueError(
            "custom agent configuration requires an immutable code_revision for exact replay"
        )
    if configuration is not None:
        module_name, separator, attribute_name = configuration.factory.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError("agent factory must use the form 'module:attribute'")
    binding = (
        {
            "version": "agent-factory-v1",
            "kind": "custom",
            "configuration": configuration.model_dump(mode="json"),
        }
        if configuration is not None
        else {
            "version": "agent-factory-v1",
            "kind": "built-in-chat",
            "maximum_model_calls": maximum_model_calls,
        }
    )
    return sha256_json(binding)


def resolve_agent_factory(
    configuration: AgentConfiguration | None,
    *,
    maximum_model_calls: int,
) -> AgentFactory:
    """Resolve a configured custom factory or the built-in bounded chat agent.

    Args:
        configuration: Optional project-owned ``module:attribute`` factory reference.
        maximum_model_calls: Hard request ceiling applied to the built-in agent.

    Returns:
        Fresh-runtime factory suitable for concurrent simulation cells.

    Raises:
        AgentFactoryError: The custom reference is malformed, unavailable, or not callable.
    """
    if configuration is None:
        return lambda: ChatAgentRuntime(maximum_model_calls=maximum_model_calls)
    module_name, separator, attribute_name = configuration.factory.partition(":")
    if not separator or not module_name or not attribute_name:
        raise AgentFactoryError("agent factory must use the form 'module:attribute'")
    try:
        factory = getattr(import_module(module_name), attribute_name)
    except (AttributeError, ImportError) as exc:
        raise AgentFactoryError(
            f"cannot import configured agent factory {configuration.factory!r}"
        ) from exc
    if not callable(factory):
        raise AgentFactoryError(
            f"configured agent factory {configuration.factory!r} is not callable"
        )

    def create() -> AgentRuntime:
        """Create and validate one isolated custom runtime.

        Returns:
            Customer runtime implementing the injected episode contract.

        Raises:
            AgentFactoryError: Construction fails or returns an incompatible object.
        """
        try:
            agent = factory()
            preflight_agent_runtime(cast(AgentRuntime, agent))
        except Exception as exc:
            raise AgentFactoryError(
                f"configured agent factory {configuration.factory!r} failed preflight: {exc}"
            ) from exc
        return cast(AgentRuntime, agent)

    return create


def preflight_agent_factory(factory: AgentFactory) -> None:
    """Construct one runtime before simulation so factory errors precede provider calls.

    Args:
        factory: Resolved built-in or project-provided runtime factory.

    Raises:
        AgentFactoryError: Construction or runtime-contract validation fails.
    """
    try:
        agent = factory()
        preflight_agent_runtime(agent)
    except AgentFactoryError:
        raise
    except Exception as exc:
        raise AgentFactoryError(f"agent factory failed preflight: {exc}") from exc

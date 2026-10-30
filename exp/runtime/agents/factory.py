"""Project agent-factory resolution with a built-in standard chat fallback."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, Sha256, sha256_bytes, sha256_json
from exp.common.project import AgentConfiguration
from exp.runtime.agents.chat import ChatAgentRuntime, normalize_chat_system_prompt
from exp.runtime.agents.interface import AgentRuntime, preflight_agent_runtime
from exp.runtime.agents.pi import PiAgentRuntime

AgentFactory = Callable[[], AgentRuntime]


class ActorHarness(ContractModel):
    """Select a maintained actor adapter without changing the injected model or tool owner.

    Attributes:
        kind: Chat or installed Pi; neither grants new environment capabilities.
        executable: Absolute Pi entrypoint, checked before any new execution.
        executable_sha256: Digest of the resolved entrypoint bytes, not a containment claim.
        version: Operator-declared Pi release label, recorded separately from content identity.
        dependencies: Additional exact installed files, such as the interpreter and lockfile.
            The inventory is declared, not proof that every transitive dependency is listed.
        timeout_seconds: Positive per-process bound; provider deadlines remain authoritative.
    """

    kind: Literal["chat", "pi"] = "chat"
    executable: Path | None = None
    executable_sha256: Sha256 | None = None
    version: str | None = Field(default=None, min_length=1)
    dependencies: dict[Path, Sha256] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        """Require concrete Pi identity and reject silently ignored chat configuration."""
        if self.kind == "pi":
            if self.executable is None or self.executable_sha256 is None or self.version is None:
                raise ValueError("Pi requires executable, executable_sha256 and version")
            if not self.executable.is_absolute() or any(
                not path.is_absolute() for path in self.dependencies
            ):
                raise ValueError("Pi executable and dependency pins must use absolute paths")
            if self.executable in self.dependencies:
                raise ValueError("Pi executable has one digest; omit it from dependencies")
        elif (
            self.executable is not None
            or self.executable_sha256 is not None
            or self.version is not None
            or self.dependencies
            or self.timeout_seconds != 300
        ):
            raise ValueError("chat harness cannot accept ignored Pi settings")
        return self

    def preflight(self) -> None:
        """Verify the declared installed files without subprocesses or provider credentials."""
        if self.kind == "chat":
            return
        assert self.executable is not None and self.executable_sha256 is not None
        for path, expected in {
            self.executable: self.executable_sha256,
            **self.dependencies,
        }.items():
            if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
                raise ValueError(
                    f"Pi runtime file changed or unavailable: {path}; restore its pinned bytes "
                    "or register a fresh evaluation with the new installation"
                )
        PiAgentRuntime(executable=str(self.executable)).preflight()


def create_actor_runtime(
    profile: ActorHarness, *, maximum_model_calls: int, maximum_output_tokens: int = 16384
) -> AgentRuntime:
    """Create a fresh bounded maintained actor; models and tools are still injected per cell."""
    profile.preflight()
    if profile.kind == "chat":
        return ChatAgentRuntime(maximum_model_calls=maximum_model_calls)
    assert profile.executable is not None
    return PiAgentRuntime(
        executable=str(profile.executable),
        timeout_seconds=profile.timeout_seconds,
        maximum_model_calls=maximum_model_calls,
        maximum_output_tokens=maximum_output_tokens,
    )


class AgentFactoryError(ValueError):
    """A configured customer agent factory cannot provide the runtime contract."""


def is_builtin_chat_factory(factory: AgentFactory) -> bool:
    """Recognize explicit built-in construction without calling arbitrary customer code.

    Args:
        factory: Constructor selected for each agent episode.

    Returns:
        True only for the built-in class or a partial applying arguments to that exact class.
        Custom wrappers and subclasses cannot assert built-in continuation semantics.
    """
    return factory is ChatAgentRuntime or (
        type(factory) is partial and factory.func is ChatAgentRuntime
    )


def agent_factory_sha256(
    configuration: AgentConfiguration | None,
    *,
    maximum_model_calls: int,
    system_prompt: str | None = None,
) -> Sha256:
    """Bind the effective built-in or custom agent configuration to one digest.

    Args:
        configuration: Optional project-owned custom factory reference.
        maximum_model_calls: Request ceiling applied when the built-in chat agent is selected.
        system_prompt: Optional built-in system instruction applied to every model request.

    Returns:
        Stable semantic identity for simulation and replay checks.

    Raises:
        ValueError: The built-in request ceiling or system prompt is invalid.
    """
    if maximum_model_calls < 1:
        raise ValueError("maximum_model_calls must be positive")
    if configuration is not None and configuration.code_revision is None:
        raise ValueError(
            "custom agent configuration requires an immutable code_revision for exact replay"
        )
    if configuration is not None:
        module_name, separator, attribute_name = configuration.factory.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError("agent factory must use the form 'module:attribute'")
    normalized_prompt = (
        None if configuration is not None else normalize_chat_system_prompt(system_prompt)
    )
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
            "system_prompt": normalized_prompt,
            "maximum_model_calls": maximum_model_calls,
        }
    )
    return sha256_json(binding)


def resolve_agent_factory(
    configuration: AgentConfiguration | None,
    *,
    maximum_model_calls: int,
    system_prompt: str | None = None,
) -> AgentFactory:
    """Resolve a configured custom factory or the built-in bounded chat agent.

    Args:
        configuration: Optional project-owned ``module:attribute`` factory reference.
        maximum_model_calls: Hard request ceiling applied to the built-in agent.
        system_prompt: Optional built-in instruction prepended to candidate requests.

    Returns:
        Fresh-runtime factory suitable for concurrent simulation cells.

    Raises:
        AgentFactoryError: The custom reference is malformed, unavailable, or not callable.
    """
    if configuration is None:
        prompt = normalize_chat_system_prompt(system_prompt)
        return partial(
            ChatAgentRuntime,
            maximum_model_calls=maximum_model_calls,
            system_prompt=prompt,
        )
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

"""Harbor and E2B executable-environment adapters behind the canonical runtime seam.

The optional Harbor and E2B dependencies stay outside this module's import path. A caller gives
this adapter a narrow synchronous session factory, normally backed by Harbor's E2B lifecycle, so
the simulator still owns one context exit, partial transcript, and cleanup ledger.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import TracebackType
from typing import Literal, Protocol, runtime_checkable

from wmo.common.core.artifacts import ContractModel, JsonObject
from wmo.common.models import OperationEconomics, ToolCall
from wmo.common.tasks import TaskCase
from wmo.runtime.environments.interface import EnvironmentSession, Observation
from wmo.runtime.harness.e2b_ledger import SandboxLedger

E2B_TEMPLATE_POLICY_VERSION = "1"
"""Version pinned into E2B template resource identities."""

E2B_DEFAULT_CPU_COUNT = 2
E2B_DEFAULT_MEMORY_MB = 1024
DEFAULT_COMMAND_TIMEOUT_SECONDS = 240
DEFAULT_RETRY_DELAYS_SECONDS = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_MAXIMUM_OBSERVATION_CHARACTERS = 20_000


class HarborRetryableCommandError(RuntimeError):
    """A safe read-only Harbor command may be retried against the same environment."""


class HarborTemplateStatusError(RuntimeError):
    """A retryable template-status request did not succeed within its configured attempts."""


class HarborCleanupUnprovenError(RuntimeError):
    """An injected Harbor context returned without proving that its sandbox was released."""


@dataclass(frozen=True)
class E2BTemplateResources:
    """Resource-qualified E2B template values used in the persistent template identity."""

    cpu_count: int
    memory_mb: int


class HarborCommandResult(ContractModel):
    """One normalized output returned by an injected Harbor executable session."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int
    timed_out: bool = False
    economics: OperationEconomics = OperationEconomics()


class HarborTranscriptEntry(ContractModel):
    """One tool attempt retained when an episode ends before its final artifact is written."""

    action: ToolCall
    observation: Observation


@runtime_checkable
class HarborExecutableSession(Protocol):
    """Narrow synchronous slice of an already-open Harbor or E2B task environment."""

    sandbox_id: str
    template_id: str

    def execute_command(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None,
        timeout_seconds: int,
    ) -> HarborCommandResult:
        """Execute one environment command and return normalized process evidence.

        Args:
            command: Shell command to execute in the task environment.
            environment: Optional explicit environment passed to the command.
            timeout_seconds: Finite command execution timeout.

        Returns:
            Normalized exit, output, and economics evidence.
        """

    def cleanup_verified(self) -> bool:
        """Return whether the backing service proved the exact sandbox is no longer live.

        Returns:
            Whether cleanup conclusively released this session's sandbox.
        """


@runtime_checkable
class HarborSessionFactory(Protocol):
    """Opens an optional Harbor or E2B environment for exactly one canonical task."""

    def open(
        self,
        task: TaskCase,
        *,
        template_name: str,
    ) -> AbstractContextManager[HarborExecutableSession]:
        """Create an externally-managed session whose context exit proves cleanup.

        Args:
            task: Canonical task to execute in the environment.
            template_name: Resource-qualified template selected for the task.

        Returns:
            Context manager that yields one executable environment session.
        """


class HarborEnvironmentRuntime:
    """Adapts an injected Harbor or E2B task session to ``EnvironmentRuntime``.

    This class neither downloads benchmark tasks nor scores them. Its only job is environment
    lifecycle and the three canonical file or shell tools needed by a customer agent.
    """

    def __init__(
        self,
        factory: HarborSessionFactory,
        *,
        environment_id: str,
        template_name: str,
        ledger: SandboxLedger | None = None,
        command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        retry_delays_seconds: Sequence[float] = DEFAULT_RETRY_DELAYS_SECONDS,
        maximum_observation_characters: int = DEFAULT_MAXIMUM_OBSERVATION_CHARACTERS,
        workspace_root: str = "/workspace",
    ) -> None:
        """Configure one optional Harbor backend without creating a cloud sandbox.

        Args:
            factory: Injected optional Harbor or E2B implementation.
            environment_id: Stable customer environment identity recorded in rollout provenance.
            template_name: Resource-qualified E2B template alias selected by the caller.
            ledger: Durable sandbox ledger. The default preserves legacy orphan-reaping behavior.
            command_timeout_seconds: Finite limit forwarded to every environment command.
            retry_delays_seconds: Retry delays used only for read-only transport failures.
            maximum_observation_characters: Bound for tool text retained in a rollout transcript.
            workspace_root: Absolute task root used to confine canonical file tools.
        """
        if not environment_id:
            raise ValueError("Harbor environment_id must be nonempty")
        if not template_name:
            raise ValueError("Harbor template_name must be nonempty")
        if isinstance(command_timeout_seconds, bool) or command_timeout_seconds < 1:
            raise ValueError("Harbor command_timeout_seconds must be a positive integer")
        if isinstance(maximum_observation_characters, bool) or maximum_observation_characters < 1:
            raise ValueError("Harbor maximum_observation_characters must be a positive integer")
        normalized_root = PurePosixPath(workspace_root)
        if not normalized_root.is_absolute() or ".." in normalized_root.parts:
            raise ValueError("Harbor workspace_root must be an absolute normalized path")
        normalized_delays = tuple(float(delay) for delay in retry_delays_seconds)
        if any(delay < 0 for delay in normalized_delays):
            raise ValueError("Harbor retry delays must be nonnegative")
        self._factory = factory
        self.environment_id = environment_id
        self.template_name = template_name
        self._ledger = ledger or SandboxLedger()
        self._command_timeout_seconds = command_timeout_seconds
        self._retry_delays_seconds = normalized_delays
        self._maximum_observation_characters = maximum_observation_characters
        self._workspace_root = normalized_root.as_posix()

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        """Return the simulator-owned cleanup context for one task environment.

        Args:
            task: Canonical task passed to the injected Harbor or E2B session factory.

        Returns:
            An execute-only environment context that records proven cleanup in the ledger.
        """
        return _HarborSessionContext(self, task)


class _HarborSessionContext(AbstractContextManager[EnvironmentSession]):
    """Records a created sandbox before use and releases it only after a successful context exit."""

    def __init__(self, runtime: HarborEnvironmentRuntime, task: TaskCase) -> None:
        self._runtime = runtime
        self._task = task
        self._context: AbstractContextManager[HarborExecutableSession] | None = None
        self._session: HarborExecutableSession | None = None

    def __enter__(self) -> EnvironmentSession:
        """Open, ledger-record, and adapt one task-local Harbor session."""
        context = self._runtime._factory.open(self._task, template_name=self._runtime.template_name)
        self._context = context
        try:
            session = context.__enter__()
        except Exception:
            self._context = None
            raise
        if not session.sandbox_id or not session.template_id:
            try:
                context.__exit__(None, None, None)
            finally:
                self._context = None
            raise ValueError(
                "Harbor executable sessions must expose nonempty sandbox and template IDs"
            )
        self._session = session
        self._runtime._ledger.record_created(
            sandbox_id=session.sandbox_id,
            template_id=session.template_id,
        )
        return _HarborEnvironmentSession(
            session,
            command_timeout_seconds=self._runtime._command_timeout_seconds,
            retry_delays_seconds=self._runtime._retry_delays_seconds,
            maximum_observation_characters=self._runtime._maximum_observation_characters,
            workspace_root=self._runtime._workspace_root,
        )

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Release a ledger entry only when the injected session cleanup actually returns."""
        context = self._context
        session = self._session
        if context is None or session is None:
            return False
        try:
            suppressed = context.__exit__(exception_type, exception, traceback)
        except Exception:
            raise
        if not session.cleanup_verified():
            raise HarborCleanupUnprovenError(
                "Harbor cleanup returned without proof that the sandbox was released"
            )
        self._runtime._ledger.record_released(session.sandbox_id)
        return bool(suppressed)


class _HarborEnvironmentSession:
    """Maps canonical tool calls to safe Harbor task-environment commands."""

    def __init__(
        self,
        session: HarborExecutableSession,
        *,
        command_timeout_seconds: int,
        retry_delays_seconds: tuple[float, ...],
        maximum_observation_characters: int,
        workspace_root: str,
    ) -> None:
        self._session = session
        self._command_timeout_seconds = command_timeout_seconds
        self._retry_delays_seconds = retry_delays_seconds
        self._maximum_observation_characters = maximum_observation_characters
        self._workspace_root = workspace_root
        self._partial_transcript: list[HarborTranscriptEntry] = []

    @property
    def partial_transcript(self) -> tuple[HarborTranscriptEntry, ...]:
        """Return every tool observation observed before a complete rollout is materialized."""
        return tuple(self._partial_transcript)

    def execute(self, action: ToolCall) -> Observation:
        """Execute one canonical action and retain its result even when the environment fails."""
        try:
            observation = self._execute(action)
        except Exception as error:  # noqa: BLE001 - customer environment failures are evidence
            observation = Observation(
                content=f"environment command failed: {type(error).__name__}",
                is_error=True,
                metadata={
                    "exception_type": type(error).__name__,
                    "retryable": isinstance(error, HarborRetryableCommandError),
                },
            )
        self._partial_transcript.append(
            HarborTranscriptEntry(action=action, observation=observation)
        )
        return observation

    def _execute(self, action: ToolCall) -> Observation:
        """Validate and dispatch only the supported canonical executable tools."""
        if action.name == "bash":
            command = _string_argument(action.arguments, "command")
            if command is None:
                return _invalid_arguments("bash", "command must be a string")
            return _command_observation(self._run(command, retryable=False))
        if action.name == "read_file":
            raw_path = _string_argument(action.arguments, "path", nonempty=True)
            path = _safe_relative_path(raw_path) if raw_path is not None else None
            if path is None:
                return _invalid_arguments("read_file", "path must be a nonempty string")
            return _command_observation(
                self._run(
                    _guarded_read_command(),
                    environment=self._file_environment(path),
                    retryable=True,
                ),
                maximum_characters=self._maximum_observation_characters,
            )
        if action.name == "write_file":
            raw_path = _string_argument(action.arguments, "path", nonempty=True)
            path = _safe_relative_path(raw_path) if raw_path is not None else None
            content = _string_argument(action.arguments, "content")
            if path is None or content is None:
                return _invalid_arguments(
                    "write_file",
                    "path must be nonempty and content must be a string",
                )
            encoded = base64.b64encode(content.encode()).decode()
            result = self._run(
                _guarded_write_command(),
                environment={
                    **self._file_environment(path),
                    "WMO_FILE_CONTENT_B64": encoded,
                },
                retryable=False,
            )
            observation = _command_observation(
                result,
                maximum_characters=self._maximum_observation_characters,
            )
            if observation.is_error:
                return observation
            return Observation(content=f"wrote {path}", metadata=observation.metadata)
        return Observation(content=f"tool {action.name!r} not available", is_error=True)

    def _file_environment(self, path: str) -> dict[str, str]:
        """Return non-secret variables consumed by the path-confinement shell fragment."""
        return {"WMO_FILE_PATH": path, "WMO_SANDBOX_ROOT": self._workspace_root}

    def _run(
        self,
        command: str,
        *,
        environment: Mapping[str, str] | None = None,
        retryable: bool,
    ) -> HarborCommandResult:
        """Run one command, retrying only a read-only transport failure."""
        attempts = len(self._retry_delays_seconds) + 1 if retryable else 1
        for attempt in range(attempts):
            try:
                return self._session.execute_command(
                    command,
                    environment=environment,
                    timeout_seconds=self._command_timeout_seconds,
                )
            except HarborRetryableCommandError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(self._retry_delays_seconds[attempt])
        raise AssertionError("Harbor command retry loop exhausted without a result")


def resolve_e2b_template_resources(
    *,
    cpu_count: int | None,
    memory_mb: int | None,
) -> E2BTemplateResources:
    """Resolve absent template resource settings to the frozen Harbor E2B defaults.

    Args:
        cpu_count: Requested CPU count, or ``None`` for the policy default.
        memory_mb: Requested memory in MiB, or ``None`` for the policy default.

    Returns:
        Validated concrete resource settings.

    Raises:
        ValueError: A supplied value is not an integer or is below the policy minimum.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (cpu_count, memory_mb)
        if value is not None
    ):
        raise ValueError("E2B template CPU and memory values must be integers")
    resolved_cpu = E2B_DEFAULT_CPU_COUNT if cpu_count is None else cpu_count
    resolved_memory = E2B_DEFAULT_MEMORY_MB if memory_mb is None else memory_mb
    if resolved_cpu < 1 or resolved_memory < 128:
        raise ValueError("E2B template CPU must be positive and memory must be at least 128 MiB")
    return E2BTemplateResources(cpu_count=resolved_cpu, memory_mb=resolved_memory)


def e2b_template_resource_payload(
    *,
    environment_id: str,
    build_source_kind: Literal["docker_image", "dockerfile"],
    build_source_reference: str,
    resources: E2BTemplateResources,
    harbor_version: str,
    e2b_sdk_version: str,
) -> dict[str, int | str]:
    """Return the frozen resource-complete payload used to identify one E2B template.

    Args:
        environment_id: Canonical Harbor environment identity.
        build_source_kind: Whether the template is built from an image or Dockerfile.
        build_source_reference: Immutable build source reference.
        resources: Concrete CPU and memory allocation.
        harbor_version: Harbor dependency version used for the build.
        e2b_sdk_version: E2B SDK version used for the build.

    Returns:
        Canonical JSON-compatible identity payload.

    Raises:
        ValueError: A required identity or version value is empty.
    """
    if not environment_id:
        raise ValueError("Harbor environment_id must be nonempty")
    if not build_source_reference:
        raise ValueError("E2B template build source must be nonempty")
    if not harbor_version or not e2b_sdk_version:
        raise ValueError("Harbor and E2B versions must be nonempty")
    return {
        "schema_version": E2B_TEMPLATE_POLICY_VERSION,
        "harbor_environment_id": environment_id,
        "build_source_kind": build_source_kind,
        "build_source_reference": build_source_reference,
        "cpu_count": resources.cpu_count,
        "memory_mb": resources.memory_mb,
        "harbor_version": harbor_version,
        "e2b_sdk_version": e2b_sdk_version,
    }


def e2b_template_resource_digest(
    *,
    environment_id: str,
    build_source_kind: Literal["docker_image", "dockerfile"],
    build_source_reference: str,
    resources: E2BTemplateResources,
    harbor_version: str,
    e2b_sdk_version: str,
) -> str:
    """Hash one canonical Harbor content, resources, and dependency-version identity.

    Args:
        environment_id: Canonical Harbor environment identity.
        build_source_kind: Whether the template is built from an image or Dockerfile.
        build_source_reference: Immutable build source reference.
        resources: Concrete CPU and memory allocation.
        harbor_version: Harbor dependency version used for the build.
        e2b_sdk_version: E2B SDK version used for the build.

    Returns:
        Stable SHA-256 digest of the canonical identity payload.
    """
    payload = e2b_template_resource_payload(
        environment_id=environment_id,
        build_source_kind=build_source_kind,
        build_source_reference=build_source_reference,
        resources=resources,
        harbor_version=harbor_version,
        e2b_sdk_version=e2b_sdk_version,
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def qualify_harbor_e2b_template_name(
    base_name: str,
    *,
    environment_id: str,
    build_source_kind: Literal["docker_image", "dockerfile"],
    build_source_reference: str,
    resources: E2BTemplateResources,
    harbor_version: str,
    e2b_sdk_version: str,
    snapshot_hash_length: int,
) -> str:
    """Derive the resource-qualified E2B alias without importing optional SDKs.

    ``snapshot_hash_length`` comes from the optional Harbor integration, preserving its exact
    native content-name validation without making ordinary local runs import Harbor.

    Args:
        base_name: Harbor's content-derived base template name.
        environment_id: Canonical Harbor environment identity.
        build_source_kind: Whether the template is built from an image or Dockerfile.
        build_source_reference: Immutable build source reference.
        resources: Concrete CPU and memory allocation.
        harbor_version: Harbor dependency version used for the build.
        e2b_sdk_version: E2B SDK version used for the build.
        snapshot_hash_length: Harbor's validated environment-ID prefix length.

    Returns:
        Resource-qualified immutable E2B template alias.

    Raises:
        ValueError: The environment identity, prefix length, or base alias is invalid.
    """
    if len(environment_id) != 32 or any(
        character not in "0123456789abcdef" for character in environment_id
    ):
        raise ValueError("Harbor environment_id must be 32 lowercase hexadecimal characters")
    if snapshot_hash_length < 1:
        raise ValueError("Harbor snapshot_hash_length must be positive")
    if not base_name.endswith(environment_id[:snapshot_hash_length]):
        raise ValueError("Harbor E2B template name does not match its environment_id")
    digest = e2b_template_resource_digest(
        environment_id=environment_id,
        build_source_kind=build_source_kind,
        build_source_reference=build_source_reference,
        resources=resources,
        harbor_version=harbor_version,
        e2b_sdk_version=e2b_sdk_version,
    )
    return f"wmo-hb-v1-{digest}"


def retry_template_status[T](
    read_status: Callable[[], T],
    *,
    retry_delays_seconds: Sequence[float] = DEFAULT_RETRY_DELAYS_SECONDS,
) -> T:
    """Retry an idempotent template-status read without replaying a template submission.

    Args:
        read_status: Idempotent status read to retry after transient status errors.
        retry_delays_seconds: Delays before each permitted retry.

    Returns:
        The first successful status result.

    Raises:
        HarborTemplateStatusError: The final permitted status read still fails.
    """
    delays = tuple(retry_delays_seconds)
    for attempt in range(len(delays) + 1):
        try:
            return read_status()
        except HarborTemplateStatusError:
            if attempt == len(delays):
                raise
            time.sleep(delays[attempt])
    raise AssertionError("Harbor template status retry loop exhausted without a result")


def _string_argument(
    arguments: JsonObject,
    name: str,
    *,
    nonempty: bool = False,
) -> str | None:
    """Return one string tool argument when it meets the requested nonempty constraint."""
    value = arguments.get(name)
    if not isinstance(value, str) or (nonempty and not value):
        return None
    return value


def _invalid_arguments(tool: str, message: str) -> Observation:
    """Build one agent-visible tool-input error without executing a customer environment command."""
    return Observation(content=f"invalid {tool} arguments: {message}", is_error=True)


def _safe_relative_path(value: str) -> str | None:
    """Accept one normalized relative POSIX path with no traversal component."""
    path = PurePosixPath(value)
    if path.is_absolute() or value in {"", "."} or any(part in {"", ".."} for part in path.parts):
        return None
    normalized = path.as_posix()
    if normalized != value:
        return None
    return normalized


def _guarded_read_command() -> str:
    """Return a shell fragment that rejects lexical and symlink escapes before reading."""
    return (
        'set -eu; root=$(cd "$WMO_SANDBOX_ROOT" && pwd -P); '
        "rel=$WMO_FILE_PATH; parent=${rel%/*}; name=${rel##*/}; "
        '[ "$parent" = "$rel" ] && parent=.; '
        'target_parent=$(cd "$root/$parent" && pwd -P); '
        'case "$target_parent/" in "$root/"*) ;; *) exit 73 ;; esac; '
        '[ ! -L "$target_parent/$name" ] || exit 73; cat "$target_parent/$name"'
    )


def _guarded_write_command() -> str:
    """Return a shell fragment that confines creation and replacement to the task root."""
    return (
        'set -eu; root=$(cd "$WMO_SANDBOX_ROOT" && pwd -P); '
        "rel=$WMO_FILE_PATH; parent=${rel%/*}; name=${rel##*/}; "
        '[ "$parent" = "$rel" ] && parent=.; '
        'target_parent=$(cd "$root/$parent" && pwd -P); '
        'case "$target_parent/" in "$root/"*) ;; *) exit 73 ;; esac; '
        '[ ! -L "$target_parent/$name" ] || exit 73; '
        'printf \'%s\' "$WMO_FILE_CONTENT_B64" | base64 -d > "$target_parent/$name"'
    )


def _command_observation(
    result: HarborCommandResult,
    *,
    maximum_characters: int = DEFAULT_MAXIMUM_OBSERVATION_CHARACTERS,
) -> Observation:
    """Convert process evidence into a bounded canonical tool observation."""
    content = result.stdout if result.stdout else result.stderr
    if len(content) > maximum_characters:
        omitted = len(content) - maximum_characters
        content = f"{content[:maximum_characters]}\n[truncated {omitted} characters]"
    metadata: JsonObject = {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "economics": result.economics.model_dump(mode="json"),
    }
    if result.timed_out:
        return Observation(
            content=content or "environment command timed out",
            is_error=True,
            metadata=metadata,
        )
    if result.exit_code != 0:
        return Observation(
            content=content or f"command exited {result.exit_code}",
            is_error=True,
            metadata=metadata,
        )
    return Observation(content=content, metadata=metadata)

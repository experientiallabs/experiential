"""Static credential-boundary checks for task-authored Harbor inputs.

Harbor deliberately resolves task environment templates from the host and passes the host
environment to Docker Compose. A benchmark evaluator that keeps model credentials on the host
must therefore reject task sources that reference credential-like host variables before Harbor
constructs an environment.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Final

import yaml
from harbor.models.task.task import Task
from pydantic import JsonValue, TypeAdapter, ValidationError

_ENVIRONMENT_NAME_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_COMPOSE_DOCUMENT_ADAPTER: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(
    dict[str, JsonValue]
)
_PROTECTED_NAME_SEGMENTS: Final = frozenset(
    {
        "AUTH",
        "AUTHORIZATION",
        "CRED",
        "CREDENTIAL",
        "CREDENTIALS",
        "KEY",
        "PASSWORD",
        "SECRET",
        "TOKEN",
    }
)
_KNOWN_PROTECTED_HOST_ENVIRONMENT_NAMES: Final = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AZURE_CLIENT_SECRET",
        "AZURE_FEDERATED_TOKEN_FILE",
        "AZURE_OPENAI_API_KEY",
        "E2B_API_KEY",
        "OPENAI_API_KEY",
        "WMH_ENDPOINT_API_KEY",
    }
)


@dataclass(frozen=True, order=True, slots=True)
class ProtectedHostEnvironmentReference:
    """One protected host variable named by a task-controlled source."""

    source: str
    variable: str


@dataclass(frozen=True, order=True, slots=True)
class TaskCredentialAuditFailure:
    """One reason a task-controlled Compose dependency could not be audited safely."""

    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class _TaskCredentialAudit:
    references: tuple[ProtectedHostEnvironmentReference, ...]
    failures: tuple[TaskCredentialAuditFailure, ...]


class TaskCredentialBoundaryError(ValueError):
    """A task source imports credentials or cannot be audited without resolving host state."""

    def __init__(
        self,
        references: tuple[ProtectedHostEnvironmentReference, ...] = (),
        failures: tuple[TaskCredentialAuditFailure, ...] = (),
    ) -> None:
        self.references = references
        self.failures = failures
        details: list[str] = []
        if references:
            reference_details = "; ".join(
                f"{reference.source}: {reference.variable}" for reference in references
            )
            details.append(
                f"task source references protected host environment variables ({reference_details})"
            )
        if failures:
            failure_details = "; ".join(
                f"{failure.source}: {failure.reason}" for failure in failures
            )
            details.append(f"task Compose credential audit failed closed ({failure_details})")
        super().__init__("; ".join(details) + "; credential values were not read")


class _ComposeDocumentError(ValueError):
    """A Compose source is not a plain string-keyed YAML mapping."""


def is_protected_host_environment_name(name: str) -> bool:
    """Return whether a host variable name is reserved for credential-like data.

    Known evaluator/provider names are protected explicitly. Future uppercase credential names
    are covered by tokenized markers without treating unrelated names such as ``MONKEY`` or
    route selectors such as ``AWS_REGION`` as credentials.
    """
    if name in _KNOWN_PROTECTED_HOST_ENVIRONMENT_NAMES:
        return True
    if name != name.upper() or _ENVIRONMENT_NAME_RE.fullmatch(name) is None:
        return False
    return bool(_PROTECTED_NAME_SEGMENTS.intersection(name.split("_")))


def find_protected_host_environment_references(
    task: Task,
) -> tuple[ProtectedHostEnvironmentReference, ...]:
    """Find protected host variables referenced by one fully resolved Harbor task.

    The audit covers every task-authored environment map used by WMH trials and the complete
    local ``include`` and ``extends`` closure of task Compose files. It records variable names
    and source locations only.

    Args:
        task: Resolved Harbor task to inspect before environment construction.

    Returns:
        Sorted, de-duplicated protected references. Secret values are never resolved or read.

    Raises:
        TaskCredentialBoundaryError: If a Compose dependency cannot be audited safely.
    """
    audit = _audit_task_credential_boundary(task)
    if audit.failures:
        raise TaskCredentialBoundaryError(audit.references, audit.failures)
    return audit.references


def validate_task_credential_boundary(task: Task) -> None:
    """Reject a task that could import credential-like values from the host.

    Args:
        task: Resolved Harbor task to audit.

    Raises:
        TaskCredentialBoundaryError: If a task names a protected host variable or a Compose
            dependency cannot be audited safely.
    """
    audit = _audit_task_credential_boundary(task)
    if audit.references or audit.failures:
        raise TaskCredentialBoundaryError(audit.references, audit.failures)


def _audit_task_credential_boundary(task: Task) -> _TaskCredentialAudit:
    references: set[ProtectedHostEnvironmentReference] = set()
    failures: set[TaskCredentialAuditFailure] = set()
    for source, env in _task_environment_maps(task):
        for value in env.values():
            for name in _compose_interpolation_references(value):
                _record_protected_reference(source, name, references)

    task_dir = task.task_dir.resolve()
    try:
        environment_dir = task.paths.environment_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        failures.add(
            TaskCredentialAuditFailure(
                source="environment",
                reason="task environment directory cannot be resolved locally",
            )
        )
        return _TaskCredentialAudit(
            references=tuple(sorted(references)),
            failures=tuple(sorted(failures)),
        )
    if not environment_dir.is_relative_to(task_dir):
        failures.add(
            TaskCredentialAuditFailure(
                source="environment",
                reason="task environment directory escapes the task directory",
            )
        )
        return _TaskCredentialAudit(
            references=tuple(sorted(references)),
            failures=tuple(sorted(failures)),
        )
    visited: set[tuple[Path, Path]] = set()
    for candidate in _task_compose_root_paths(task.paths.environment_dir):
        path = _resolve_existing_task_path(
            candidate,
            environment_dir=environment_dir,
            source=_task_source(candidate, task_dir),
            relation="Compose source",
            failures=failures,
        )
        if path is None:
            continue
        _audit_compose_file(
            path,
            project_dir=path.parent,
            environment_dir=environment_dir,
            task_dir=task_dir,
            references=references,
            failures=failures,
            visited=visited,
            active=(),
        )

    for candidate in sorted(task.paths.environment_dir.rglob("*.env")):
        path = _resolve_existing_task_path(
            candidate,
            environment_dir=environment_dir,
            source=_task_source(candidate, task_dir),
            relation="environment source",
            failures=failures,
        )
        if path is not None:
            _audit_interpolation_file(
                path,
                source=_task_source(path, task_dir),
                references=references,
                failures=failures,
            )

    return _TaskCredentialAudit(
        references=tuple(sorted(references)),
        failures=tuple(sorted(failures)),
    )


def _task_environment_maps(task: Task) -> Iterator[tuple[str, Mapping[str, str]]]:
    yield "task.toml [environment].env", task.config.environment.env
    yield "task.toml [verifier].env", task.config.verifier.env
    if task.config.verifier.environment is not None:
        yield "task.toml [verifier.environment].env", task.config.verifier.environment.env
    for index, step in enumerate(task.config.steps or []):
        yield f"task.toml steps[{index}].verifier.env", step.verifier.env
        if step.verifier.environment is not None:
            yield (
                f"task.toml steps[{index}].verifier.environment.env",
                step.verifier.environment.env,
            )


def _task_compose_root_paths(environment_dir: Path) -> tuple[Path, ...]:
    candidates = {
        environment_dir / "docker-compose.yaml",
        environment_dir / "docker-compose.yml",
    }
    for pattern in ("*compose*.yaml", "*compose*.yml"):
        candidates.update(environment_dir.rglob(pattern))
    return tuple(sorted(path for path in candidates if path.exists() or path.is_symlink()))


def _audit_compose_file(
    path: Path,
    *,
    project_dir: Path,
    environment_dir: Path,
    task_dir: Path,
    references: set[ProtectedHostEnvironmentReference],
    failures: set[TaskCredentialAuditFailure],
    visited: set[tuple[Path, Path]],
    active: tuple[Path, ...],
) -> None:
    source = _task_source(path, task_dir)
    if path in active:
        failures.add(
            TaskCredentialAuditFailure(source=source, reason="cyclic Compose file reference")
        )
        return
    visit_key = (path, project_dir)
    if visit_key in visited:
        return
    visited.add(visit_key)

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason="Compose source cannot be read as UTF-8",
            )
        )
        return
    for name in _compose_interpolation_references(text):
        _record_protected_reference(source, name, references)
    try:
        document = _parse_compose_document(text)
    except _ComposeDocumentError:
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason="Compose source is not a valid string-keyed YAML mapping",
            )
        )
        return
    for name in _compose_inherited_environment_names(document):
        _record_protected_reference(source, name, references)
    _audit_compose_service_env_files(
        document,
        project_dir=project_dir,
        compose_path=path,
        environment_dir=environment_dir,
        task_dir=task_dir,
        references=references,
        failures=failures,
    )

    next_active = (*active, path)
    _audit_compose_includes(
        document,
        compose_path=path,
        environment_dir=environment_dir,
        task_dir=task_dir,
        references=references,
        failures=failures,
        visited=visited,
        active=next_active,
    )
    _audit_compose_extends(
        document,
        project_dir=project_dir,
        source=source,
        environment_dir=environment_dir,
        task_dir=task_dir,
        references=references,
        failures=failures,
        visited=visited,
        active=next_active,
    )


def _audit_compose_service_env_files(
    document: dict[str, JsonValue],
    *,
    project_dir: Path,
    compose_path: Path,
    environment_dir: Path,
    task_dir: Path,
    references: set[ProtectedHostEnvironmentReference],
    failures: set[TaskCredentialAuditFailure],
) -> None:
    """Audit every service env file that Compose can read from the host."""
    services = document.get("services")
    if not isinstance(services, dict):
        return
    compose_source = _task_source(compose_path, task_dir)
    for service_name, service in services.items():
        if not isinstance(service, dict) or (env_file := service.get("env_file")) is None:
            continue
        declaration_source = f"{compose_source} [services.{service_name}.env_file]"
        if isinstance(env_file, str):
            entries = [env_file]
        elif isinstance(env_file, list):
            entries = env_file
        else:
            failures.add(
                TaskCredentialAuditFailure(
                    source=declaration_source,
                    reason="unsupported Compose service env_file declaration",
                )
            )
            continue

        for index, entry in enumerate(entries):
            entry_source = f"{declaration_source}[{index}]"
            if isinstance(entry, str):
                path_value = entry
            elif isinstance(entry, dict) and isinstance(entry.get("path"), str):
                path_value = entry["path"]
            else:
                failures.add(
                    TaskCredentialAuditFailure(
                        source=entry_source,
                        reason="unsupported Compose service env_file declaration",
                    )
                )
                continue
            env_path = _resolve_compose_reference(
                path_value,
                base_dir=project_dir,
                environment_dir=environment_dir,
                source=entry_source,
                relation="Compose service env_file",
                failures=failures,
                expect_directory=False,
            )
            if env_path is not None:
                _audit_interpolation_file(
                    env_path,
                    source=_task_source(env_path, task_dir),
                    references=references,
                    failures=failures,
                )


def _audit_compose_includes(
    document: dict[str, JsonValue],
    *,
    compose_path: Path,
    environment_dir: Path,
    task_dir: Path,
    references: set[ProtectedHostEnvironmentReference],
    failures: set[TaskCredentialAuditFailure],
    visited: set[tuple[Path, Path]],
    active: tuple[Path, ...],
) -> None:
    include_value = document.get("include")
    if include_value is None:
        return
    if isinstance(include_value, list):
        entries = include_value
    elif isinstance(include_value, (str, dict)):
        entries = [include_value]
    else:
        failures.add(
            TaskCredentialAuditFailure(
                source=f"{_task_source(compose_path, task_dir)} [include]",
                reason="unsupported Compose include declaration",
            )
        )
        return

    for index, entry in enumerate(entries):
        source = f"{_task_source(compose_path, task_dir)} [include[{index}]]"
        path_values, project_value, env_values = _compose_include_entry(
            entry,
            source=source,
            failures=failures,
        )
        resolved_paths = tuple(
            path
            for value in path_values
            if (
                path := _resolve_compose_reference(
                    value,
                    base_dir=compose_path.parent,
                    environment_dir=environment_dir,
                    source=source,
                    relation="Compose include",
                    failures=failures,
                    expect_directory=False,
                )
            )
            is not None
        )
        if project_value is None:
            child_project_dir = resolved_paths[0].parent if resolved_paths else None
        else:
            child_project_dir = _resolve_compose_reference(
                project_value,
                base_dir=compose_path.parent,
                environment_dir=environment_dir,
                source=source,
                relation="Compose include project_directory",
                failures=failures,
                expect_directory=True,
            )

        env_base_dir = child_project_dir or compose_path.parent
        for env_value in env_values:
            env_path = _resolve_compose_reference(
                env_value,
                base_dir=env_base_dir,
                environment_dir=environment_dir,
                source=source,
                relation="Compose include env_file",
                failures=failures,
                expect_directory=False,
            )
            if env_path is not None:
                _audit_interpolation_file(
                    env_path,
                    source=_task_source(env_path, task_dir),
                    references=references,
                    failures=failures,
                )

        if child_project_dir is None:
            continue
        default_env = child_project_dir / ".env"
        if default_env.exists() or default_env.is_symlink():
            default_env_path = _resolve_existing_task_path(
                default_env,
                environment_dir=environment_dir,
                source=source,
                relation="Compose include default environment source",
                failures=failures,
            )
            if default_env_path is not None:
                _audit_interpolation_file(
                    default_env_path,
                    source=_task_source(default_env_path, task_dir),
                    references=references,
                    failures=failures,
                )
        for path in resolved_paths:
            _audit_compose_file(
                path,
                project_dir=child_project_dir,
                environment_dir=environment_dir,
                task_dir=task_dir,
                references=references,
                failures=failures,
                visited=visited,
                active=active,
            )


def _compose_include_entry(
    entry: JsonValue,
    *,
    source: str,
    failures: set[TaskCredentialAuditFailure],
) -> tuple[tuple[str, ...], str | None, tuple[str, ...]]:
    if isinstance(entry, str):
        return (entry,), None, ()
    if not isinstance(entry, dict):
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason="unsupported Compose include entry",
            )
        )
        return (), None, ()
    paths = _compose_string_or_list(
        entry.get("path"),
        source=source,
        field="path",
        required=True,
        failures=failures,
    )
    project_values = _compose_string_or_list(
        entry.get("project_directory"),
        source=source,
        field="project_directory",
        required=False,
        failures=failures,
    )
    if len(project_values) > 1:
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason="unsupported Compose include project_directory declaration",
            )
        )
    env_values = _compose_string_or_list(
        entry.get("env_file"),
        source=source,
        field="env_file",
        required=False,
        failures=failures,
    )
    return paths, project_values[0] if len(project_values) == 1 else None, env_values


def _compose_string_or_list(
    value: JsonValue | None,
    *,
    source: str,
    field: str,
    required: bool,
    failures: set[TaskCredentialAuditFailure],
) -> tuple[str, ...]:
    if value is None:
        if required:
            failures.add(
                TaskCredentialAuditFailure(
                    source=source,
                    reason=f"Compose include {field} is required",
                )
            )
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    failures.add(
        TaskCredentialAuditFailure(
            source=source,
            reason=f"unsupported Compose include {field} declaration",
        )
    )
    return ()


def _audit_compose_extends(
    document: dict[str, JsonValue],
    *,
    project_dir: Path,
    source: str,
    environment_dir: Path,
    task_dir: Path,
    references: set[ProtectedHostEnvironmentReference],
    failures: set[TaskCredentialAuditFailure],
    visited: set[tuple[Path, Path]],
    active: tuple[Path, ...],
) -> None:
    services = document.get("services")
    if not isinstance(services, dict):
        return
    for service_name, service in services.items():
        if not isinstance(service, dict) or (extends := service.get("extends")) is None:
            continue
        extends_source = f"{source} [services.{service_name}.extends]"
        if not isinstance(extends, dict):
            failures.add(
                TaskCredentialAuditFailure(
                    source=extends_source,
                    reason="unsupported Compose extends declaration",
                )
            )
            continue
        file_value = extends.get("file")
        if file_value is None:
            continue
        if not isinstance(file_value, str):
            failures.add(
                TaskCredentialAuditFailure(
                    source=extends_source,
                    reason="unsupported Compose extends file declaration",
                )
            )
            continue
        path = _resolve_compose_reference(
            file_value,
            base_dir=project_dir,
            environment_dir=environment_dir,
            source=extends_source,
            relation="Compose extends file",
            failures=failures,
            expect_directory=False,
        )
        if path is not None:
            _audit_compose_file(
                path,
                project_dir=project_dir,
                environment_dir=environment_dir,
                task_dir=task_dir,
                references=references,
                failures=failures,
                visited=visited,
                active=active,
            )


def _resolve_compose_reference(
    value: str,
    *,
    base_dir: Path,
    environment_dir: Path,
    source: str,
    relation: str,
    failures: set[TaskCredentialAuditFailure],
    expect_directory: bool,
) -> Path | None:
    if not value or "$" in value:
        failures.add(TaskCredentialAuditFailure(source=source, reason=f"dynamic {relation} path"))
        return None
    path_value = Path(value)
    if path_value.is_absolute() or PureWindowsPath(value).is_absolute():
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason=f"{relation} path escapes the task environment directory",
            )
        )
        return None
    return _resolve_existing_task_path(
        base_dir / path_value,
        environment_dir=environment_dir,
        source=source,
        relation=relation,
        failures=failures,
        expect_directory=expect_directory,
    )


def _resolve_existing_task_path(
    candidate: Path,
    *,
    environment_dir: Path,
    source: str,
    relation: str,
    failures: set[TaskCredentialAuditFailure],
    expect_directory: bool = False,
) -> Path | None:
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason=f"{relation} cannot be resolved to a local file",
            )
        )
        return None
    if not path.is_relative_to(environment_dir):
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason=f"{relation} path escapes the task environment directory",
            )
        )
        return None
    expected_kind = path.is_dir() if expect_directory else path.is_file()
    if not expected_kind:
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason=f"{relation} cannot be resolved to a local file",
            )
        )
        return None
    return path


def _audit_interpolation_file(
    path: Path,
    *,
    source: str,
    references: set[ProtectedHostEnvironmentReference],
    failures: set[TaskCredentialAuditFailure],
) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        failures.add(
            TaskCredentialAuditFailure(
                source=source,
                reason="environment source cannot be read as UTF-8",
            )
        )
        return
    for name in _compose_interpolation_references(text):
        _record_protected_reference(source, name, references)


def _record_protected_reference(
    source: str,
    name: str,
    references: set[ProtectedHostEnvironmentReference],
) -> None:
    if is_protected_host_environment_name(name):
        references.add(ProtectedHostEnvironmentReference(source=source, variable=name))


def _task_source(path: Path, task_dir: Path) -> str:
    try:
        return path.relative_to(task_dir.resolve()).as_posix()
    except ValueError:
        return "environment/[outside task boundary]"


def _parse_compose_document(text: str) -> dict[str, JsonValue]:
    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise _ComposeDocumentError from exc
    if loaded is None:
        return {}
    try:
        return _COMPOSE_DOCUMENT_ADAPTER.validate_python(loaded, strict=True)
    except (ValidationError, RecursionError) as exc:
        raise _ComposeDocumentError from exc


def _compose_host_environment_references(text: str) -> frozenset[str]:
    """Extract host variable names from supported Docker Compose reference forms."""
    references = set(_compose_interpolation_references(text))
    try:
        document = _parse_compose_document(text)
    except _ComposeDocumentError:
        return frozenset(references)
    references.update(_compose_inherited_environment_names(document))
    return frozenset(references)


def _compose_inherited_environment_names(document: dict[str, JsonValue]) -> frozenset[str]:
    references: set[str] = set()
    services = document.get("services")
    if isinstance(services, dict):
        for service in services.values():
            if not isinstance(service, dict):
                continue
            references.update(_bare_environment_names(service.get("environment")))
            build = service.get("build")
            if isinstance(build, dict):
                references.update(_bare_environment_names(build.get("args")))
    for resource_key in ("secrets", "configs"):
        resources = document.get(resource_key)
        if not isinstance(resources, dict):
            continue
        for resource in resources.values():
            if not isinstance(resource, dict):
                continue
            environment = resource.get("environment")
            if isinstance(environment, str):
                references.add(environment)
    return frozenset(references)


def _bare_environment_names(value: JsonValue | None) -> frozenset[str]:
    if isinstance(value, dict):
        return frozenset(name for name, item in value.items() if item is None)
    if isinstance(value, list):
        return frozenset(
            item
            for item in value
            if isinstance(item, str) and "=" not in item and _ENVIRONMENT_NAME_RE.fullmatch(item)
        )
    return frozenset()


def _compose_interpolation_references(text: str) -> frozenset[str]:
    references: set[str] = set()
    index = 0
    while index < len(text):
        if text[index] != "$":
            index += 1
            continue
        if index + 1 >= len(text):
            break
        next_character = text[index + 1]
        if next_character == "$":
            index += 2
            continue
        name_start = index + 2 if next_character == "{" else index + 1
        match = _ENVIRONMENT_NAME_RE.match(text, name_start)
        if match is None:
            index += 1
            continue
        references.add(match.group())
        index = match.end()
    return frozenset(references)

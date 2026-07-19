"""Tests for Harbor task credential-boundary auditing."""

from pathlib import Path

import pytest
from harbor.models.task.task import Task

from wmh.evals.harbor.task_security import (
    ProtectedHostEnvironmentReference,
    TaskCredentialBoundaryError,
    _compose_host_environment_references,
    find_protected_host_environment_references,
    is_protected_host_environment_name,
    validate_task_credential_boundary,
)


def _write_task(tmp_path: Path, task_toml: str = "") -> Task:
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    (task_dir / "instruction.md").write_text("Solve it.\n", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    return Task(task_dir)


@pytest.mark.parametrize(
    "name",
    [
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_OPENAI_API_KEY",
        "E2B_API_KEY",
        "EXAMPLE_REFRESH_TOKEN",
        "WMH_ENDPOINT_API_KEY",
    ],
)
def test_protected_name_contract_covers_known_and_future_credentials(name: str) -> None:
    assert is_protected_host_environment_name(name) is True


@pytest.mark.parametrize("name", ["AWS_REGION", "AZURE_OPENAI_ENDPOINT", "MONKEY", "token"])
def test_protected_name_contract_allows_non_credentials(name: str) -> None:
    assert is_protected_host_environment_name(name) is False


def test_compose_extraction_handles_templates_bare_imports_and_escapes() -> None:
    source = """
services:
  main:
    environment:
      - AWS_SESSION_TOKEN
      - SAFE=$SAFE_VALUE
      - NESTED=${SAFE_VALUE:-${AWS_SECRET_ACCESS_KEY}}
      - ESCAPED=$$E2B_API_KEY
    build:
      args: [AZURE_OPENAI_API_KEY]
secrets:
  endpoint-auth:
    environment: WMH_ENDPOINT_API_KEY
"""

    assert _compose_host_environment_references(source) == frozenset(
        {
            "AWS_SESSION_TOKEN",
            "SAFE_VALUE",
            "AWS_SECRET_ACCESS_KEY",
            "AZURE_OPENAI_API_KEY",
            "WMH_ENDPOINT_API_KEY",
        }
    )


def test_compose_extraction_handles_flow_mappings_and_explicit_values() -> None:
    source = """
services:
  main:
    environment: {AWS_SECRET_ACCESS_KEY: null, E2B_API_KEY: "task-fixture"}
    build: {args: {AZURE_OPENAI_API_KEY: null, OPENAI_API_KEY: "task-fixture"}}
"""

    assert _compose_host_environment_references(source) == frozenset(
        {"AWS_SECRET_ACCESS_KEY", "AZURE_OPENAI_API_KEY"}
    )


def test_task_audit_reports_config_and_compose_sources_without_reading_values(
    tmp_path: Path,
) -> None:
    task = _write_task(
        tmp_path,
        """
[environment.env]
DATASET_AUTH = "${AZURE_OPENAI_API_KEY}"

[verifier.env]
JUDGE_AUTH = "${AWS_SESSION_TOKEN:-fallback-value}"
""",
    )
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    environment:\n      - E2B_API_KEY\n",
        encoding="utf-8",
    )

    references = find_protected_host_environment_references(task)

    assert references == (
        ProtectedHostEnvironmentReference(
            source="environment/docker-compose.yaml", variable="E2B_API_KEY"
        ),
        ProtectedHostEnvironmentReference(
            source="task.toml [environment].env", variable="AZURE_OPENAI_API_KEY"
        ),
        ProtectedHostEnvironmentReference(
            source="task.toml [verifier].env", variable="AWS_SESSION_TOKEN"
        ),
    )
    with pytest.raises(TaskCredentialBoundaryError) as caught:
        validate_task_credential_boundary(task)
    message = str(caught.value)
    assert "fallback-value" not in message
    assert "AZURE_OPENAI_API_KEY" in message
    assert "credential values were not read" in message


def test_task_audit_follows_compose_include_and_extends_closure(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    environment_dir = task.paths.environment_dir
    (environment_dir / "docker-compose.yaml").write_text(
        "include:\n  - compose-parts/feature.arbitrary.yaml\n",
        encoding="utf-8",
    )
    (environment_dir / "compose-parts").mkdir()
    (environment_dir / "compose-parts" / "feature.arbitrary.yaml").write_text(
        """
services:
  worker:
    extends: {file: ../fragments/base.fragment, service: base}
    environment: {AWS_SESSION_TOKEN: null}
""",
        encoding="utf-8",
    )
    (environment_dir / "fragments").mkdir()
    (environment_dir / "fragments" / "base.fragment").write_text(
        "services: {base: {build: {args: {AZURE_OPENAI_API_KEY: null}}}}\n",
        encoding="utf-8",
    )

    assert find_protected_host_environment_references(task) == (
        ProtectedHostEnvironmentReference(
            source="environment/compose-parts/feature.arbitrary.yaml",
            variable="AWS_SESSION_TOKEN",
        ),
        ProtectedHostEnvironmentReference(
            source="environment/fragments/base.fragment",
            variable="AZURE_OPENAI_API_KEY",
        ),
    )


def test_task_audit_rejects_compose_reference_cycles(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        "include: [nested.fragment]\n",
        encoding="utf-8",
    )
    (task.paths.environment_dir / "nested.fragment").write_text(
        "include: [docker-compose.yaml]\n",
        encoding="utf-8",
    )

    with pytest.raises(TaskCredentialBoundaryError, match="cyclic Compose file reference"):
        validate_task_credential_boundary(task)


@pytest.mark.parametrize(
    ("compose_source", "expected_reason"),
    [
        ("include: [../../outside.yaml]\n", "escapes the task environment directory"),
        ("include: [missing.yaml]\n", "cannot be resolved to a local file"),
        ("include: ['${COMPOSE_FRAGMENT}']\n", "dynamic Compose include path"),
    ],
)
def test_task_audit_rejects_unresolvable_compose_references(
    tmp_path: Path,
    compose_source: str,
    expected_reason: str,
) -> None:
    task = _write_task(tmp_path)
    (tmp_path / "outside.yaml").write_text("services: {}\n", encoding="utf-8")
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        compose_source,
        encoding="utf-8",
    )

    with pytest.raises(TaskCredentialBoundaryError, match=expected_reason):
        validate_task_credential_boundary(task)


def test_task_audit_rejects_compose_symlink_escape(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    outside = tmp_path / "outside.yaml"
    outside.write_text("services: {}\n", encoding="utf-8")
    (task.paths.environment_dir / "linked.yaml").symlink_to(outside)
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        "include: [linked.yaml]\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TaskCredentialBoundaryError,
        match="escapes the task environment directory",
    ):
        validate_task_credential_boundary(task)


def test_task_audit_rejects_symlinked_environment_directory_escape(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "task.toml").write_text("", encoding="utf-8")
    (task_dir / "instruction.md").write_text("Solve it.\n", encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    outside_environment = tmp_path / "outside-environment"
    outside_environment.mkdir()
    (outside_environment / "Dockerfile").write_text("FROM alpine:3.20\n", encoding="utf-8")
    (task_dir / "environment").symlink_to(outside_environment, target_is_directory=True)
    task = Task(task_dir)

    with pytest.raises(
        TaskCredentialBoundaryError,
        match="task environment directory escapes the task directory",
    ):
        validate_task_credential_boundary(task)


def test_task_audit_accepts_noncredential_host_parameters(tmp_path: Path) -> None:
    task = _write_task(
        tmp_path,
        """
[environment.env]
REGION = "${AWS_REGION:-us-east-1}"
""",
    )
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    environment:\n      - AZURE_OPENAI_ENDPOINT\n",
        encoding="utf-8",
    )

    validate_task_credential_boundary(task)
    assert find_protected_host_environment_references(task) == ()


def test_task_audit_accepts_explicit_compose_credential_values(tmp_path: Path) -> None:
    task = _write_task(tmp_path)
    (task.paths.environment_dir / "docker-compose.yaml").write_text(
        """
services:
  main:
    environment: {AWS_SECRET_ACCESS_KEY: "task-fixture", E2B_API_KEY: ""}
    build: {args: {AZURE_OPENAI_API_KEY: "task-fixture"}}
""",
        encoding="utf-8",
    )

    validate_task_credential_boundary(task)
    assert find_protected_host_environment_references(task) == ()

"""Tests for resource-complete Harbor E2B template identity."""

from __future__ import annotations

import hashlib
from typing import cast

import pytest
from harbor.environments.definition import SNAPSHOT_HASH_LEN

import wmh.evals.harbor.e2b_template_policy as policy


def test_qualified_name_replaces_harbor_name_with_fixed_safe_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.8.1")
    environment_id = hashlib.sha256(b"environment").hexdigest()[:32]
    base_name = f"task__{environment_id[:SNAPSHOT_HASH_LEN]}"
    resources = policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=2048)

    qualified = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=resources,
    )

    assert qualified.startswith("wmh-hb-v1-")
    assert qualified != base_name
    assert len(qualified) == 74
    assert qualified.replace("-", "").isalnum()


def test_qualified_name_changes_with_resources_and_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_id = hashlib.sha256(b"environment").hexdigest()[:32]
    base_name = f"task__{environment_id[:SNAPSHOT_HASH_LEN]}"
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.8.1")
    first = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
    )
    more_memory = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=2048),
    )
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.9.0")
    newer_sdk = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
    )

    assert len({first, more_memory, newer_sdk}) == 3


def test_qualified_name_changes_with_harbor_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_id = hashlib.sha256(b"environment").hexdigest()[:32]
    base_name = f"task__{environment_id[:SNAPSHOT_HASH_LEN]}"
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.31.0")
    monkeypatch.setattr(policy, "harbor_version", lambda: "0.20.0")
    first = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
    )
    monkeypatch.setattr(policy, "harbor_version", lambda: "0.21.0")
    upgraded = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
    )

    assert first != upgraded


def test_qualified_name_rejects_unrelated_harbor_name() -> None:
    with pytest.raises(ValueError, match="does not match"):
        policy.qualify_harbor_e2b_template_name(
            "task__not-the-hash",
            environment_id="a" * 32,
            build_source_kind="dockerfile",
            build_source_reference="a" * 32,
            resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
        )


@pytest.mark.parametrize("environment_id", ["a" * 31, "a" * 33, "G" * 32])
def test_qualified_name_rejects_noncanonical_environment_id(environment_id: str) -> None:
    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        policy.qualify_harbor_e2b_template_name(
            f"task__{environment_id[:SNAPSHOT_HASH_LEN]}",
            environment_id=environment_id,
            build_source_kind="dockerfile",
            build_source_reference="a" * 32,
            resources=policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024),
        )


def test_omitted_resources_resolve_to_explicit_pinned_defaults() -> None:
    resources = policy.resolve_e2b_template_resources(cpu_count=None, memory_mb=None)

    assert resources == policy.E2BTemplateResources(
        cpu_count=2,
        memory_mb=1024,
        cpu_source="provider_default",
        memory_source="provider_default",
    )


def test_equal_effective_resources_share_identity_regardless_of_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.31.0")
    monkeypatch.setattr(policy, "harbor_version", lambda: "0.20.0")
    environment_id = hashlib.sha256(b"environment").hexdigest()[:32]
    base_name = f"task__{environment_id[:SNAPSHOT_HASH_LEN]}"
    explicit = policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024)
    defaults = policy.resolve_e2b_template_resources(cpu_count=None, memory_mb=None)

    explicit_name = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=explicit,
    )
    default_name = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="dockerfile",
        build_source_reference=environment_id,
        resources=defaults,
    )

    assert explicit_name == default_name


@pytest.mark.parametrize(
    ("cpu_count", "memory_mb"),
    [(True, 1024), (2, False), (1.5, 1024), (2, 1024.0)],
)
def test_resource_resolution_rejects_non_integer_values(
    cpu_count: object,
    memory_mb: object,
) -> None:
    with pytest.raises(ValueError, match="integers"):
        policy.resolve_e2b_template_resources(
            cpu_count=cast("int | None", cpu_count),
            memory_mb=cast("int | None", memory_mb),
        )


def test_docker_image_reference_changes_identity_even_with_same_environment_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy, "e2b_sdk_version", lambda: "2.8.1")
    environment_id = hashlib.sha256(b"environment").hexdigest()[:32]
    base_name = f"task__{environment_id[:SNAPSHOT_HASH_LEN]}"
    resources = policy.resolve_e2b_template_resources(cpu_count=2, memory_mb=1024)

    first = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="docker_image",
        build_source_reference="ubuntu:22.04",
        resources=resources,
    )
    second = policy.qualify_harbor_e2b_template_name(
        base_name,
        environment_id=environment_id,
        build_source_kind="docker_image",
        build_source_reference="ubuntu:24.04",
        resources=resources,
    )

    assert first != second

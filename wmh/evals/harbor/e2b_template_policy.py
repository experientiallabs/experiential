"""Stable identity and policy for prepared Harbor E2B task templates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import harbor
from harbor.environments.definition import SNAPSHOT_HASH_LEN

E2B_TEMPLATE_POLICY_VERSION = "1"
E2B_TEMPLATE_BUILD_CONCURRENCY = 10
E2B_TEMPLATE_BUILD_CONCURRENCY_LIMIT = 20
E2B_DEFAULT_CPU_COUNT = 2
E2B_DEFAULT_MEMORY_MB = 1024
WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH = "wmh.evals.harbor.e2b_environment:WmhE2BEnvironment"
_HARBOR_ENVIRONMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class E2BTemplateResources:
    """Normalized resources passed explicitly to E2B template builds."""

    cpu_count: int
    memory_mb: int
    cpu_source: Literal["runtime_override", "task", "provider_default"]
    memory_source: Literal["runtime_override", "task", "provider_default"]


def resolve_e2b_template_resources(
    *,
    cpu_count: int | None,
    memory_mb: int | None,
    override_cpu_count: int | None = None,
    override_memory_mb: int | None = None,
) -> E2BTemplateResources:
    """Resolve omitted Harbor values to the pinned E2B numeric defaults."""
    values = (cpu_count, memory_mb, override_cpu_count, override_memory_mb)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
        if value is not None
    ):
        raise ValueError("E2B template CPU and memory values must be integers")
    resolved_cpu = (
        override_cpu_count
        if override_cpu_count is not None
        else E2B_DEFAULT_CPU_COUNT
        if cpu_count is None
        else cpu_count
    )
    resolved_memory = (
        override_memory_mb
        if override_memory_mb is not None
        else E2B_DEFAULT_MEMORY_MB
        if memory_mb is None
        else memory_mb
    )
    if resolved_cpu < 1 or resolved_memory < 128:
        raise ValueError("E2B template CPU must be positive and memory must be at least 128 MiB")
    return E2BTemplateResources(
        cpu_count=resolved_cpu,
        memory_mb=resolved_memory,
        cpu_source=(
            "runtime_override"
            if override_cpu_count is not None
            else "provider_default"
            if cpu_count is None
            else "task"
        ),
        memory_source=(
            "runtime_override"
            if override_memory_mb is not None
            else "provider_default"
            if memory_mb is None
            else "task"
        ),
    )


def e2b_sdk_version() -> str:
    """Return the installed E2B SDK version required by the E2B runtime path."""
    try:
        return version("e2b")
    except PackageNotFoundError as error:
        raise RuntimeError("Harbor E2B readiness requires the wmh e2b extra") from error


def harbor_version() -> str:
    """Return the Harbor version whose build semantics define template identity."""
    return harbor.__version__


def e2b_template_resource_payload(
    *,
    environment_id: str,
    build_source_kind: Literal["docker_image", "dockerfile"],
    build_source_reference: str,
    resources: E2BTemplateResources,
) -> dict[str, int | str]:
    """Return the canonical resource-complete cache identity used by E2B."""
    if not environment_id:
        raise ValueError("Harbor environment_id must be nonempty")
    if not build_source_reference:
        raise ValueError("E2B template build source must be nonempty")
    return {
        "schema_version": E2B_TEMPLATE_POLICY_VERSION,
        "harbor_environment_id": environment_id,
        "build_source_kind": build_source_kind,
        "build_source_reference": build_source_reference,
        "cpu_count": resources.cpu_count,
        "memory_mb": resources.memory_mb,
        "harbor_version": harbor_version(),
        "e2b_sdk_version": e2b_sdk_version(),
    }


def e2b_template_resource_digest(
    *,
    environment_id: str,
    build_source_kind: Literal["docker_image", "dockerfile"],
    build_source_reference: str,
    resources: E2BTemplateResources,
) -> str:
    """Hash one canonical Harbor-content and E2B-resource identity."""
    payload = e2b_template_resource_payload(
        environment_id=environment_id,
        build_source_kind=build_source_kind,
        build_source_reference=build_source_reference,
        resources=resources,
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
) -> str:
    """Replace Harbor's content-only name with a fixed-safe complete identity."""
    if _HARBOR_ENVIRONMENT_ID_PATTERN.fullmatch(environment_id) is None:
        raise ValueError("Harbor environment_id must be 32 lowercase hexadecimal characters")
    inherited_suffix = environment_id[:SNAPSHOT_HASH_LEN]
    if not base_name.endswith(inherited_suffix):
        raise ValueError("Harbor E2B template name does not match its environment_id")
    digest = e2b_template_resource_digest(
        environment_id=environment_id,
        build_source_kind=build_source_kind,
        build_source_reference=build_source_reference,
        resources=resources,
    )
    return f"wmh-hb-v1-{digest}"


def e2b_template_readiness_policy_payload() -> dict[str, int | str | bool]:
    """Return the frozen readiness policy included in scorer identity."""
    return {
        "schema_version": E2B_TEMPLATE_POLICY_VERSION,
        "harbor_version": harbor_version(),
        "e2b_sdk_version": e2b_sdk_version(),
        "default_cpu_count": E2B_DEFAULT_CPU_COUNT,
        "default_memory_mb": E2B_DEFAULT_MEMORY_MB,
        "build_concurrency": E2B_TEMPLATE_BUILD_CONCURRENCY,
        "build_concurrency_limit": E2B_TEMPLATE_BUILD_CONCURRENCY_LIMIT,
        "resource_qualified_alias": True,
        "force_build": False,
        "storage_policy": "reject_requested",
    }

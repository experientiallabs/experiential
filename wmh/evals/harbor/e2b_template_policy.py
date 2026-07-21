"""Stable identity for WMH-built Harbor E2B task templates.

The template alias replaces harbor's content-only name with a resource-complete identity:
harbor's native naming has no resource qualification, so two tasks sharing environment content
but differing in cpu/memory would collide on one alias. The digest input is FROZEN: a fleet of
templates already exists on the E2B account under this exact derivation, and byte-identical
naming is what lets a new run reuse them instead of re-paying every build.

Everything here imports harbor but never the e2b SDK at module scope, so the scorer can import
the environment routing constant on a docker-only install (`wmh[harbor]` without `e2b`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import harbor
from harbor.environments.definition import SNAPSHOT_HASH_LEN

# Version marker inside the frozen digest payload. Bumping it renames every derived alias and
# orphans all existing prebuilt templates; do not change it without that intent.
E2B_TEMPLATE_POLICY_VERSION = "1"
# At most this many template builds run concurrently (the E2B account limit is 20; half leaves
# headroom for sandbox operations and other processes).
E2B_TEMPLATE_BUILD_CONCURRENCY = 10
E2B_TEMPLATE_BUILD_STATUS_POLL_INTERVAL_MS = 1_000
# Fixed backoff for retrying the idempotent build-status GET only (never the submission).
E2B_TEMPLATE_BUILD_STATUS_RETRY_DELAYS_MS = (250, 500, 1_000, 2_000, 4_000)
E2B_DEFAULT_CPU_COUNT = 2
E2B_DEFAULT_MEMORY_MB = 1024
WMH_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH = "wmh.evals.harbor.e2b_environment:WmhE2BEnvironment"
_HARBOR_ENVIRONMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class E2BTemplateResources:
    """Normalized resources passed explicitly to E2B template builds."""

    cpu_count: int
    memory_mb: int


def resolve_e2b_template_resources(
    *,
    cpu_count: int | None,
    memory_mb: int | None,
) -> E2BTemplateResources:
    """Resolve omitted Harbor values to the pinned E2B numeric defaults."""
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


def e2b_sdk_version() -> str:
    """Return the installed E2B SDK version embedded in template identity."""
    try:
        return version("e2b")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "the harbor E2B task backend needs the e2b extra; run `uv sync --extra e2b`"
        ) from error


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
    """Return the canonical resource-complete cache identity used by E2B (frozen bytes)."""
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

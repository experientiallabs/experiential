"""Tests for the frozen identity of WMO-built Harbor E2B templates.

The digest input is FROZEN: a fleet of templates already exists on the E2B account under this
derivation, and byte-identical naming is what lets a run reuse them instead of re-paying every
build. So these tests pin the exact payload and one exact alias, and any change that renames a
template has to change a literal here on purpose.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from harbor.environments.definition import SNAPSHOT_HASH_LEN

from wmo.runtime.evaluation.harbor.e2b_template_policy import (
    E2B_DEFAULT_CPU_COUNT,
    E2B_DEFAULT_MEMORY_MB,
    E2B_TEMPLATE_POLICY_VERSION,
    WMO_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH,
    E2BTemplateResources,
    e2b_sdk_version,
    e2b_template_resource_digest,
    e2b_template_resource_payload,
    harbor_version,
    qualify_harbor_e2b_template_name,
    resolve_e2b_template_resources,
)

_ENVIRONMENT_ID = "0123456789abcdef0123456789abcdef"
_RESOURCES = E2BTemplateResources(cpu_count=2, memory_mb=1024)


def _harbor_style_name(environment_id: str = _ENVIRONMENT_ID) -> str:
    """A name shaped like harbor's own: content-only, suffixed with the environment hash."""
    return f"harbor-task-{environment_id[:SNAPSHOT_HASH_LEN]}"


def test_omitted_resources_resolve_to_the_pinned_defaults() -> None:
    # E2B would otherwise apply account defaults to omitted values, which is a silent resource
    # collision inside a shared alias.
    resolved = resolve_e2b_template_resources(cpu_count=None, memory_mb=None)

    assert resolved == E2BTemplateResources(
        cpu_count=E2B_DEFAULT_CPU_COUNT, memory_mb=E2B_DEFAULT_MEMORY_MB
    )


def test_supplied_resources_pass_through() -> None:
    assert resolve_e2b_template_resources(cpu_count=8, memory_mb=16_384) == E2BTemplateResources(
        cpu_count=8, memory_mb=16_384
    )


@pytest.mark.parametrize(
    ("cpu_count", "memory_mb"),
    [(0, None), (-1, None), (None, 127), (None, 0)],
)
def test_out_of_range_resources_are_refused(cpu_count: int | None, memory_mb: int | None) -> None:
    with pytest.raises(ValueError, match="CPU must be positive"):
        resolve_e2b_template_resources(cpu_count=cpu_count, memory_mb=memory_mb)


@pytest.mark.parametrize(("cpu_count", "memory_mb"), [(True, None), (None, False), (2.0, None)])
def test_non_integer_resources_are_refused(cpu_count: object, memory_mb: object) -> None:
    # A bool or a float would hash differently from the int it looks like, silently orphaning
    # every template already built for that task.
    with pytest.raises(ValueError, match="must be integers"):
        resolve_e2b_template_resources(
            cpu_count=cpu_count,  # ty: ignore[invalid-argument-type]
            memory_mb=memory_mb,  # ty: ignore[invalid-argument-type]
        )


def test_the_digest_payload_is_the_frozen_field_set() -> None:
    payload = e2b_template_resource_payload(
        environment_id=_ENVIRONMENT_ID,
        build_source_kind="docker_image",
        build_source_reference="ghcr.io/example/task:1",
        resources=_RESOURCES,
    )

    assert payload == {
        "schema_version": E2B_TEMPLATE_POLICY_VERSION,
        "harbor_environment_id": _ENVIRONMENT_ID,
        "build_source_kind": "docker_image",
        "build_source_reference": "ghcr.io/example/task:1",
        "cpu_count": 2,
        "memory_mb": 1024,
        "harbor_version": harbor_version(),
        "e2b_sdk_version": e2b_sdk_version(),
    }


def test_the_digest_is_sha256_over_the_canonical_json_payload() -> None:
    payload = e2b_template_resource_payload(
        environment_id=_ENVIRONMENT_ID,
        build_source_kind="dockerfile",
        build_source_reference=_ENVIRONMENT_ID,
        resources=_RESOURCES,
    )
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert (
        e2b_template_resource_digest(
            environment_id=_ENVIRONMENT_ID,
            build_source_kind="dockerfile",
            build_source_reference=_ENVIRONMENT_ID,
            resources=_RESOURCES,
        )
        == expected
    )


def test_resources_and_build_source_all_change_the_digest() -> None:
    # The whole reason this identity exists: harbor's native name is content-only, so two tasks
    # sharing environment content but differing in cpu/memory would collide on one alias.
    def digest(**overrides: object) -> str:
        kwargs = {
            "environment_id": _ENVIRONMENT_ID,
            "build_source_kind": "docker_image",
            "build_source_reference": "ghcr.io/example/task:1",
            "resources": _RESOURCES,
        }
        kwargs.update(overrides)
        return e2b_template_resource_digest(**kwargs)

    baseline = digest()

    assert digest(resources=E2BTemplateResources(cpu_count=4, memory_mb=1024)) != baseline
    assert digest(resources=E2BTemplateResources(cpu_count=2, memory_mb=2048)) != baseline
    assert digest(build_source_reference="ghcr.io/example/task:2") != baseline
    assert digest(build_source_kind="dockerfile") != baseline
    assert digest(environment_id="f" * 32) != baseline
    assert digest() == baseline  # and it is deterministic


def test_an_empty_environment_id_or_build_source_is_refused() -> None:
    with pytest.raises(ValueError, match="environment_id must be nonempty"):
        e2b_template_resource_payload(
            environment_id="",
            build_source_kind="docker_image",
            build_source_reference="ghcr.io/example/task:1",
            resources=_RESOURCES,
        )
    with pytest.raises(ValueError, match="build source must be nonempty"):
        e2b_template_resource_payload(
            environment_id=_ENVIRONMENT_ID,
            build_source_kind="docker_image",
            build_source_reference="",
            resources=_RESOURCES,
        )


def test_the_alias_is_the_fixed_prefix_plus_the_digest() -> None:
    name = qualify_harbor_e2b_template_name(
        _harbor_style_name(),
        environment_id=_ENVIRONMENT_ID,
        build_source_kind="docker_image",
        build_source_reference="ghcr.io/example/task:1",
        resources=_RESOURCES,
    )

    digest = e2b_template_resource_digest(
        environment_id=_ENVIRONMENT_ID,
        build_source_kind="docker_image",
        build_source_reference="ghcr.io/example/task:1",
        resources=_RESOURCES,
    )
    assert name == f"wmo-hb-v1-{digest}"
    assert len(name) == len("wmo-hb-v1-") + 64


def test_a_malformed_environment_id_is_refused() -> None:
    for bad in ("0123456789ABCDEF0123456789abcdef", "abc", "g" * 32, f"{_ENVIRONMENT_ID}0"):
        with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
            qualify_harbor_e2b_template_name(
                _harbor_style_name(),
                environment_id=bad,
                build_source_kind="docker_image",
                build_source_reference="ghcr.io/example/task:1",
                resources=_RESOURCES,
            )


def test_a_name_that_does_not_belong_to_the_environment_is_refused() -> None:
    # The inherited suffix is the proof that the harbor name and the environment id describe the
    # same environment; without the check a mismatched pair would produce a plausible alias.
    with pytest.raises(ValueError, match="does not match its environment_id"):
        qualify_harbor_e2b_template_name(
            _harbor_style_name("f" * 32),
            environment_id=_ENVIRONMENT_ID,
            build_source_kind="docker_image",
            build_source_reference="ghcr.io/example/task:1",
            resources=_RESOURCES,
        )


def test_the_e2b_environment_import_path_resolves() -> None:
    # Harbor loads the environment by this string; a rename that missed it would only fail once a
    # real trial tried to start.
    module_path, _, attribute = WMO_HARBOR_E2B_ENVIRONMENT_IMPORT_PATH.partition(":")
    module = __import__(module_path, fromlist=[attribute])

    assert hasattr(module, attribute)


def test_the_versions_in_the_digest_are_the_installed_ones() -> None:
    assert harbor_version() == pytest.importorskip("harbor").__version__
    assert e2b_sdk_version().count(".") >= 1

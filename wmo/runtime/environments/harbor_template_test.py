"""Tests for deterministic Harbor template identity and status reads."""

from wmo.runtime.environments.harbor import (
    HarborTemplateStatusError,
    e2b_template_resource_digest,
    qualify_harbor_e2b_template_name,
    resolve_e2b_template_resources,
    retry_template_status,
)


def test_template_identity_includes_resources_and_dependency_versions() -> None:
    """Changing resources or an SDK version cannot silently reuse another template."""
    resources = resolve_e2b_template_resources(cpu_count=None, memory_mb=None)
    first = e2b_template_resource_digest(
        environment_id="a" * 32,
        build_source_kind="docker_image",
        build_source_reference="fixture:image",
        resources=resources,
        harbor_version="0.20.0",
        e2b_sdk_version="2.31.0",
    )
    second = e2b_template_resource_digest(
        environment_id="a" * 32,
        build_source_kind="docker_image",
        build_source_reference="fixture:image",
        resources=resolve_e2b_template_resources(cpu_count=4, memory_mb=None),
        harbor_version="0.20.0",
        e2b_sdk_version="2.31.0",
    )

    assert first != second
    assert (
        qualify_harbor_e2b_template_name(
            "harbor-template-aaaaaaaa",
            environment_id="a" * 32,
            build_source_kind="docker_image",
            build_source_reference="fixture:image",
            resources=resources,
            harbor_version="0.20.0",
            e2b_sdk_version="2.31.0",
            snapshot_hash_length=8,
        )
        == f"wmo-hb-v1-{first}"
    )


def test_template_status_retry_retries_only_the_idempotent_read() -> None:
    """The retry helper is usable for status reads without making submission replayable."""
    attempts = 0

    def read_status() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise HarborTemplateStatusError("temporary status transport failure")
        return "ready"

    assert retry_template_status(read_status, retry_delays_seconds=(0.0,)) == "ready"
    assert attempts == 2

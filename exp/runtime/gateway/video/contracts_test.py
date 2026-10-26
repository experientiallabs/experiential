"""Tests for the canonical video contracts: sizes, ceilings, rates, and rendering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from openai.types import Video, VideoDeleteResponse
from pydantic import ValidationError

from exp.common.models import (
    BilledUnitKind,
    BillingSource,
    ExactModelDeployment,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
    GatewayUnitPrices,
)
from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.gateway.ledger_valuation import NanoUsdOverflowError
from exp.runtime.gateway.video.contracts import (
    VIDEO_SIZES,
    VideoJob,
    VideoRequest,
    VideoStatus,
    render_deleted_object,
    render_video_object,
    video_ceiling_nano_usd,
    video_unit_rate,
)


def test_a_request_defaults_to_openais_documented_shape() -> None:
    """Omitted seconds and size default to OpenAI's 4 seconds at 720x1280."""
    request = VideoRequest(prompt="a cat")
    assert request.surface is GatewayApiSurface.VIDEO
    assert (request.seconds, request.size) == (4, "720x1280")
    assert (request.aspect_ratio, request.resolution) == ("9:16", "720p")


@pytest.mark.parametrize(
    ("size", "aspect", "resolution"),
    [("1280x720", "16:9", "720p"), ("1080x1920", "9:16", "1080p"), ("3840x2160", "16:9", "4k")],
)
def test_every_size_names_its_aspect_and_priced_resolution(
    size: str, aspect: str, resolution: str
) -> None:
    """The size table is the single source of the aspect ratio and the price variant."""
    request = VideoRequest(prompt="p", size=size)
    assert (request.aspect_ratio, request.resolution) == (aspect, resolution)


def test_sora_only_sizes_have_no_provider_equivalent() -> None:
    """OpenAI's Sora-only portrait and landscape sizes are not in the table."""
    assert "1024x1792" not in VIDEO_SIZES
    assert "1792x1024" not in VIDEO_SIZES


def test_prompt_and_seconds_are_bounded() -> None:
    """Prompts past 4,000 characters and durations past 20 seconds are invalid."""
    with pytest.raises(ValidationError):
        VideoRequest(prompt="x" * 4_001)
    with pytest.raises(ValidationError):
        VideoRequest(prompt="p", seconds=21)


def test_the_ceiling_is_requested_seconds_at_the_rate() -> None:
    """The reservation is exactly seconds times the resolution's rate."""
    request = VideoRequest(prompt="p", seconds=8, size="1280x720")
    assert video_ceiling_nano_usd(request, 100_000_000) == 800_000_000
    assert video_ceiling_nano_usd(request, None) is None
    with pytest.raises(NanoUsdOverflowError):
        video_ceiling_nano_usd(request, 2**62)


def test_the_unit_rate_reads_the_resolution_variant_then_the_flat_rate() -> None:
    """A per-resolution card prices its variants; a flat card prices every resolution."""
    per_resolution = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"720p": 10, "1080p": 12})
    )
    assert video_unit_rate(per_resolution, "1080p") == 12
    assert video_unit_rate(per_resolution, "4k") is None
    flat = GatewayTokenPrices(
        units=GatewayUnitPrices(kind=BilledUnitKind.VIDEO_SECOND, rates={"": 5})
    )
    assert video_unit_rate(flat, "4k") == 5
    other = GatewayTokenPrices(units=GatewayUnitPrices(kind=BilledUnitKind.IMAGE, rates={"": 5}))
    assert video_unit_rate(other, "720p") is None
    assert video_unit_rate(GatewayTokenPrices(), "720p") is None


def job_fixture(**update: object) -> VideoJob:
    """Build one 8-second 720p Veo job record with test defaults.

    Args:
        **update: Fields to override.

    Returns:
        The job.
    """
    created = datetime(2026, 10, 18, 12, tzinfo=UTC)
    deployment = ExactModelDeployment(
        deployment_id="veo-deployment",
        source_alias="veo",
        exact_model_id="veo-exact",
        connection="gemini-main",
        provider="gemini",
        provider_model="veo-3.1-fast-generate-preview",
        billing_source=BillingSource.HOST_MANAGED,
        connection_sha256="1" * 64,
        capabilities_sha256="2" * 64,
        gateway=GatewayDeploymentMetadata(
            prices=GatewayTokenPrices(
                units=GatewayUnitPrices(
                    kind=BilledUnitKind.VIDEO_SECOND, rates={"720p": 100_000_000}
                )
            ),
        ),
    )
    job = VideoJob(
        video_id="video_fixture",
        organization_id="org",
        identity_id="owner",
        key_id="key",
        alias="veo",
        request_id="request-1",
        attempt_id="attempt-1",
        deployment=deployment,
        alias_revision_id="revision-veo",
        catalog_sha256="0" * 64,
        dialect="veo",
        provider_job_id="models/veo/operations/op-1",
        seconds=8,
        size="1280x720",
        variant="720p",
        rate_nano_usd=100_000_000,
        reserved_nano_usd=800_000_000,
        created_at=created,
        settle_by=created + timedelta(hours=2),
    )
    return job.model_copy(update=update)


_job = job_fixture


def test_a_completed_job_renders_the_openai_video_object() -> None:
    """The rendered object matches the SDK's ``Video`` model, with the alias as model."""
    done = datetime(2026, 10, 18, 12, 5, tzinfo=UTC)
    job = _job(
        status=VideoStatus.COMPLETED,
        progress=40,
        completed_at=done,
        expires_at=done + timedelta(days=2),
    )
    rendered = render_video_object(job)
    assert rendered["object"] == "video"
    assert rendered["model"] == job.alias
    assert rendered["progress"] == 100
    assert rendered["seconds"] == "8"
    assert rendered["completed_at"] == int(done.timestamp())
    assert rendered["error"] is None
    Video.model_validate(rendered)


def test_a_failed_job_renders_its_error() -> None:
    """A failed job carries the OpenAI error object with code and message."""
    rendered = render_video_object(
        _job(status=VideoStatus.FAILED, error_code="content_filtered", error_message="blocked")
    )
    assert rendered["error"] == {"code": "content_filtered", "message": "blocked"}


def test_the_deleted_object_matches_the_sdk() -> None:
    """The deletion confirmation is the SDK's ``VideoDeleteResponse``."""
    VideoDeleteResponse.model_validate(render_deleted_object("video_x"))

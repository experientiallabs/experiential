"""Canonical asynchronous video-generation contracts: request, job, and public object.

The public surface mirrors OpenAI's Videos API (``POST /v1/videos``, then poll
``GET /v1/videos/{id}`` and download ``/content``) so the official SDK's
``client.videos`` methods work unchanged, while the gateway serves it from
whichever provider rung the alias names. A request names its output by
``size`` (``"1280x720"``) and ``seconds``; the gateway maps the size to an
aspect ratio and a resolution, and the resolution is the priced variant on the
lane's ``video_second`` unit card.

A job is content-free once accepted: it records who owns it, which frozen
deployment and ledger attempt carry its reservation, and the provider's job
handle, never the prompt or the reference image.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256
from exp.common.models.catalog_prices import BilledUnitKind, GatewayTokenPrices
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.ledger_valuation import require_representable_nano_usd
from exp.runtime.gateway.surfaces import GatewayApiSurface

MAXIMUM_VIDEO_PROMPT_CHARACTERS = 4_000
"""Prompt cap: Veo reads at most 1,024 prompt tokens, about 4,000 characters."""

MAXIMUM_VIDEO_SECONDS = 20
"""Canonical duration cap; each provider narrows it to the durations it renders."""

MAXIMUM_REFERENCE_IMAGE_BYTES = 10 * 1024 * 1024
"""First-frame image cap. It travels base64-inline to the provider, well under
the 20 MB Gemini request limit after encoding."""

REFERENCE_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp")

DEFAULT_VIDEO_SECONDS = 4
"""OpenAI's documented default clip length."""

DEFAULT_VIDEO_SIZE = "720x1280"
"""OpenAI's documented default size (portrait 720p)."""

VideoResolution = Literal["480p", "720p", "1080p", "4k"]
VideoAspectRatio = Literal["16:9", "9:16"]

VIDEO_SIZES: dict[str, tuple[VideoAspectRatio, VideoResolution]] = {
    "854x480": ("16:9", "480p"),
    "480x854": ("9:16", "480p"),
    "1280x720": ("16:9", "720p"),
    "720x1280": ("9:16", "720p"),
    "1920x1080": ("16:9", "1080p"),
    "1080x1920": ("9:16", "1080p"),
    "3840x2160": ("16:9", "4k"),
    "2160x3840": ("9:16", "4k"),
}
"""Every size the gateway can express to some provider, with its aspect ratio and
priced resolution. OpenAI's Sora-only sizes (``1024x1792``, ``1792x1024``)
have no provider equivalent and are refused by name."""

VIDEO_JOB_SETTLEMENT_SECONDS = 2 * 60 * 60
"""How long an accepted job may stay non-terminal before the gateway finalizes it.

Provider renders take seconds to a few minutes (Veo documents 11 s to 6 min), and
the gateway's own poller drives every job to a terminal state without the
caller polling. The bound finalizes an unconfirmed job as failed and releases its
customer reservation without declaring its provider usage zero. Late observed
usage can still reconcile provider cost, preserving the customer waiver."""

VIDEO_LEDGER_DEADLINE_SLACK_SECONDS = 10 * 60
"""Extra time on the ledger request deadline past the settlement bound, so the
startup crash reconciler never claims an attempt the poller still owns."""


class VideoReferenceImage(ContractModel):
    """The facts of one first-frame reference image, never its bytes.

    Attributes:
        content_type: The image media type.
        bytes: Size of the decoded image.
        sha256: Lowercase hex digest of the decoded image.
    """

    content_type: Literal["image/jpeg", "image/png", "image/webp"]
    bytes: int = Field(ge=1, le=MAXIMUM_REFERENCE_IMAGE_BYTES)
    sha256: Sha256


class VideoRequest(ContractModel):
    """Canonical, provider-neutral video-generation request.

    Attributes:
        surface: Always the video surface.
        prompt: The text describing the video.
        seconds: Requested clip length in whole seconds.
        size: Output size as ``WIDTHxHEIGHT``; a key of :data:`VIDEO_SIZES`.
        input_reference: The optional first-frame image's facts.
    """

    surface: Literal[GatewayApiSurface.VIDEO] = GatewayApiSurface.VIDEO
    prompt: str = Field(min_length=1, max_length=MAXIMUM_VIDEO_PROMPT_CHARACTERS)
    seconds: int = Field(default=DEFAULT_VIDEO_SECONDS, ge=1, le=MAXIMUM_VIDEO_SECONDS)
    size: str = Field(default=DEFAULT_VIDEO_SIZE, min_length=3, max_length=16)
    input_reference: VideoReferenceImage | None = None

    @property
    def aspect_ratio(self) -> VideoAspectRatio:
        """The aspect ratio the size names."""
        return VIDEO_SIZES[self.size][0]

    @property
    def resolution(self) -> VideoResolution:
        """The resolution the size names, which is also the priced variant."""
        return VIDEO_SIZES[self.size][1]


def video_ceiling_nano_usd(request: VideoRequest, rate: int | None) -> int | None:
    """Return the reservation for one video: requested seconds at the variant's rate.

    The provider renders exactly the requested length, and settlement bills at
    most the requested seconds, so this is both the reservation and the
    settlement bound.

    Args:
        request: The canonical request.
        rate: Integer nano-USD per second for the request's resolution, or None.

    Returns:
        The ceiling, or None when the lane cannot price this resolution.

    Raises:
        NanoUsdOverflowError: The ceiling does not fit the ledger column.
    """
    if rate is None:
        return None
    return require_representable_nano_usd(request.seconds * rate, what="video reservation")


class VideoStatus(StrEnum):
    """The OpenAI lifecycle states one video job reports."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_VIDEO_STATUSES = frozenset({VideoStatus.COMPLETED, VideoStatus.FAILED})


class VideoJob(ContractModel):
    """One durable gateway video job: ownership, money, and provider handle.

    Attributes:
        video_id: The gateway-issued public id (``video_<hex>``).
        organization_id: Owning tenant.
        identity_id: Owning identity; every read is scoped to it.
        key_id: The virtual key that created the job.
        alias: The public model alias the caller named.
        request_id: The ledger request carrying the reservation.
        attempt_id: The ledger attempt dispatched to the accepting rung.
        deployment: The frozen deployment the job was submitted to.
        alias_revision_id: The authorized alias revision, for catalog lookup.
        catalog_sha256: The authorized runtime catalog digest.
        dialect: The provider video dialect serving the job.
        provider_job_id: The provider's job handle; None until accepted.
        status: Current public status.
        progress: Approximate completion percentage.
        seconds: Requested clip length.
        size: Requested size.
        variant: The priced resolution variant.
        rate_nano_usd: Frozen nano-USD per second for the variant.
        reserved_nano_usd: The reservation the attempt holds.
        billed_seconds_milli: Settled seconds in thousandths; None until settled.
        error_code: Public failure code when the job failed.
        error_message: Public failure message when the job failed.
        content_url: The provider's download locator once completed; None for
            an ``output_stored`` job, whose video lives in host storage.
        output_stored: Whether the provider uploads the rendered video to the
            host's :class:`~exp.runtime.gateway.video.output.VideoOutputStore`
            (fixed at dispatch; content is then served from that store).
        created_at: When the job was accepted.
        completed_at: When the job reached a terminal state.
        expires_at: When the provider deletes the rendered assets.
        settle_by: The settlement bound; past it a live job finalizes as failed.
        last_polled_at: The latest provider poll, for poll throttling.
        settled: Whether the ledger attempt is terminal.
        deleted: Whether the owner deleted the job from listings and reads.
    """

    video_id: str = Field(min_length=7, max_length=64)
    organization_id: str
    identity_id: str
    key_id: str | None = None
    alias: str
    request_id: str
    attempt_id: str
    deployment: ExactModelDeployment
    alias_revision_id: str
    catalog_sha256: Sha256
    dialect: str
    provider_job_id: str | None = None
    status: VideoStatus = VideoStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    seconds: int = Field(ge=1, le=MAXIMUM_VIDEO_SECONDS)
    size: str
    variant: str
    rate_nano_usd: int = Field(ge=0)
    reserved_nano_usd: int = Field(ge=0)
    billed_seconds_milli: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_message: str | None = Field(default=None, max_length=2_048)
    content_url: str | None = None
    output_stored: bool = False
    created_at: AwareDatetime
    completed_at: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    settle_by: AwareDatetime
    last_polled_at: AwareDatetime | None = None
    settled: bool = False
    deleted: bool = False

    @property
    def terminal(self) -> bool:
        """Whether the job reached a terminal public status."""
        return self.status in TERMINAL_VIDEO_STATUSES


def _unix(moment: datetime | None) -> int | None:
    """Render one optional aware datetime as integer Unix seconds."""
    return None if moment is None else int(moment.timestamp())


def render_video_object(job: VideoJob) -> JsonObject:
    """Render one job as the OpenAI ``video`` object the SDK parses.

    Args:
        job: The durable job.

    Returns:
        The public object; ``model`` is the caller's alias and the prompt is
        never echoed (the job does not retain it).
    """
    rendered: JsonObject = {
        "id": job.video_id,
        "object": "video",
        "model": job.alias,
        "status": job.status.value,
        "progress": 100 if job.status is VideoStatus.COMPLETED else job.progress,
        "created_at": _unix(job.created_at),
        "completed_at": _unix(job.completed_at),
        "expires_at": _unix(job.expires_at),
        "seconds": str(job.seconds),
        "size": job.size,
        "error": None,
        "remixed_from_video_id": None,
    }
    if job.status is VideoStatus.FAILED:
        rendered["error"] = {
            "code": job.error_code or "video_generation_failed",
            "message": job.error_message or "The video could not be generated.",
        }
    return rendered


def render_deleted_object(video_id: str) -> JsonObject:
    """Render the OpenAI ``video.deleted`` confirmation."""
    return {"id": video_id, "object": "video.deleted", "deleted": True}


def video_unit_rate(prices: GatewayTokenPrices, resolution: str) -> int | None:
    """The lane's nano-USD per second for one resolution, or None when unpriced.

    Video lanes bill one ``video_second`` card whose variants are resolutions
    (``"720p"``); a flat ``""`` rate prices every resolution the lane renders.

    Args:
        prices: The deployment's price card.
        resolution: The request's resolution variant.

    Returns:
        The rate, or None when the card is absent, of another kind, or has no
        rate for the resolution.
    """
    units = prices.units
    if units is None or units.kind != BilledUnitKind.VIDEO_SECOND:
        return None
    return units.rate_for(resolution)

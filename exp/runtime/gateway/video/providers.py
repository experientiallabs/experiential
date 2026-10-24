"""Provider clients for asynchronous video jobs: Google Veo and xAI.

Each client speaks one provider's job protocol over the connection's resolved
wire profile (endpoint plus authenticated headers) and normalizes it into the
gateway's lifecycle. A submit outcome is classified for failover: a
:class:`VideoSubmitRejected` means the provider answered and created nothing,
so the next rung may serve; a :class:`VideoSubmitAmbiguous` means the provider
may have accepted the job, so the gateway must not dispatch it again.

Verified contracts (2026-10-18):

- Veo: ``POST {root}/models/{model}:predictLongRunning`` with ``instances`` and
  ``parameters`` (``aspectRatio`` 16:9 or 9:16, ``resolution`` 720p, 1080p or
  4k, ``durationSeconds`` 4, 6 or 8, and 8 only at 1080p and 4k; Lite renders
  no 4k), answering an operation ``name`` polled with ``GET {root}/{name}``
  until ``done``. Samples download from ``video.uri`` with the API key and are
  kept for 2 days. A blocked render is not charged.
- xAI: ``POST {root}/videos/generations`` with ``duration`` 1 to 15,
  ``aspect_ratio`` and ``resolution`` (480p, 720p, 1080p), answering a
  ``request_id`` polled with ``GET {root}/videos/{request_id}`` until
  ``status`` is ``done`` (``video.url``, ``video.duration``) or ``failed``.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeGuard
from urllib.parse import urlsplit

import httpx

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.stream_contracts import GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.video.contracts import VideoRequest, VideoStatus
from exp.runtime.models.providers.base import GatewayWireProfile

VEO_DIALECT = "veo"
XAI_VIDEOS_DIALECT = "xai_videos"

VEO_ASSET_LIFETIME_SECONDS = 2 * 24 * 60 * 60
"""Veo keeps generated videos for 2 days, then removes them."""

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_MAXIMUM_DETAIL_CHARACTERS = 240


@dataclass(frozen=True)
class VideoSubmission:
    """Everything one provider submit carries for an admitted request.

    Attributes:
        request: The canonical request.
        model_id: The exact provider model identifier.
        reference_image: The first-frame image bytes, when the request has one.
        upload_url: The presigned ``PUT`` destination the provider uploads the
            rendered video to, when the host stores outputs and the dialect
            supports it (see :mod:`exp.runtime.gateway.video.output`).
    """

    request: VideoRequest
    model_id: str
    reference_image: bytes | None = field(default=None, repr=False)
    upload_url: str | None = field(default=None, repr=False)


class VideoProviderSnapshot(ContractModel):
    """One normalized provider observation of a job.

    Attributes:
        status: The public lifecycle status.
        progress: Approximate completion percentage.
        duration_milli: Rendered length in thousandths of a second, when reported.
        content_url: The asset locator once completed.
        error_code: Public failure code for a failed job.
        error_message: Bounded provider explanation for a failed job.
        asset_lifetime_seconds: How long the provider keeps the asset, if known.
    """

    status: VideoStatus
    progress: int = 0
    duration_milli: int | None = None
    content_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    asset_lifetime_seconds: int | None = None


@dataclass(frozen=True)
class VideoContentFetch:
    """How the data plane downloads one asset.

    The data plane sends ``headers`` only to ``url`` itself and drops them on
    any redirect, so a provider credential never reaches a storage host.

    Attributes:
        url: The asset URL.
        headers: Request headers for that URL only.
        content_type: The media type to answer with.
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    content_type: str = "video/mp4"


class VideoSubmitRejected(Exception):
    """The provider answered a submit without creating a job."""

    def __init__(self, failure: GatewayFailure) -> None:
        """Retain the classified failure.

        Args:
            failure: Sanitized failure; ``failover_eligible`` says whether the
                next rung may serve.
        """
        super().__init__(failure.safe_message)
        self.failure = failure


class VideoSubmitAmbiguous(Exception):
    """The provider may have accepted the submit; it must not be dispatched again."""


class VideoPollUnavailable(Exception):
    """One poll failed transiently; the job stays open for a later poll."""


HttpFactory = Callable[[], httpx.Client]


def default_http_factory() -> httpx.Client:
    """Build one bounded synchronous HTTP client for provider job calls."""
    return httpx.Client(timeout=_TIMEOUT, follow_redirects=False)


def _bounded(text: object) -> str | None:
    """Collapse one provider explanation to a bounded single line, or None."""
    if not isinstance(text, str):
        return None
    collapsed = " ".join(text.split())
    return collapsed[:_MAXIMUM_DETAIL_CHARACTERS] or None


def _provider_message(response: httpx.Response) -> str | None:
    """Read the provider's error sentence from a JSON error body, if any."""
    try:
        body = response.json()
    except ValueError:
        return None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return _bounded(error.get("message"))
        if isinstance(error, str):
            return _bounded(error)
        return _bounded(body.get("message"))
    return None


def rejection_failure(response: httpx.Response) -> GatewayFailure:
    """Classify one non-success submit answer for failover.

    A client error about the request itself (400, 413, 422) is the caller's
    to fix and never fails over; account, credential, routing, throttle, and
    server failures are the rung's and fail over to the next rung.

    Args:
        response: The provider's non-2xx answer.

    Returns:
        The sanitized failure.
    """
    status = response.status_code
    detail = _provider_message(response)
    if status in {400, 413, 422}:
        return GatewayFailure(
            failure_class=GatewayFailureClass.INVALID_REQUEST,
            safe_message="the video provider rejected the request",
            provider_detail=detail,
        )
    classes = {
        401: GatewayFailureClass.PROVIDER_AUTHENTICATION,
        402: GatewayFailureClass.PROVIDER_QUOTA,
        403: GatewayFailureClass.PROVIDER_AUTHENTICATION,
        404: GatewayFailureClass.PROVIDER_NOT_FOUND,
        429: GatewayFailureClass.THROTTLED,
    }
    return GatewayFailure(
        failure_class=classes.get(status, GatewayFailureClass.PROVIDER_INTERNAL),
        safe_message=f"the video provider answered HTTP {status}",
        failover_eligible=True,
    )


def _post_submit(
    http: HttpFactory, url: str, headers: dict[str, str], body: JsonObject
) -> JsonObject:
    """POST one job submit and return the parsed acceptance body.

    Raises:
        VideoSubmitRejected: The provider answered without creating a job, or
            the connection never opened (nothing was sent).
        VideoSubmitAmbiguous: The request may have reached the provider but no
            readable acceptance came back.
    """
    try:
        with http() as client:
            response = client.post(url, headers=headers, content=json.dumps(body))
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        raise VideoSubmitRejected(
            GatewayFailure(
                failure_class=GatewayFailureClass.TRANSPORT,
                safe_message="the video provider could not be reached",
                failover_eligible=True,
            )
        ) from exc
    except httpx.HTTPError as exc:
        raise VideoSubmitAmbiguous("the video provider did not answer the submit") from exc
    if not response.is_success:
        raise VideoSubmitRejected(rejection_failure(response))
    try:
        parsed = response.json()
    except ValueError as exc:
        raise VideoSubmitAmbiguous("the video provider's acceptance was unreadable") from exc
    if not isinstance(parsed, dict):
        raise VideoSubmitAmbiguous("the video provider's acceptance was not an object")
    return parsed


def _get_poll(http: HttpFactory, url: str, headers: dict[str, str]) -> httpx.Response:
    """GET one job status, mapping transient failures to a retryable error.

    Raises:
        VideoPollUnavailable: Transport failure, throttle, or server error.
    """
    try:
        with http() as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise VideoPollUnavailable("the video provider could not be polled") from exc
    if response.status_code == 429 or response.status_code >= 500:
        raise VideoPollUnavailable(f"the video provider answered HTTP {response.status_code}")
    return response


def _json_object(response: httpx.Response) -> JsonObject:
    """Parse one poll body, treating an unreadable body as transient.

    Raises:
        VideoPollUnavailable: The body is not a JSON object.
    """
    try:
        parsed = response.json()
    except ValueError as exc:
        raise VideoPollUnavailable("the video provider's status was unreadable") from exc
    if not isinstance(parsed, dict):
        raise VideoPollUnavailable("the video provider's status was not an object")
    return parsed


def _vanished(response: httpx.Response) -> VideoProviderSnapshot | None:
    """Map a poll for a job the provider no longer knows to a failed snapshot."""
    if response.status_code in {404, 410}:
        return VideoProviderSnapshot(
            status=VideoStatus.FAILED,
            error_code="video_not_found",
            error_message="the provider no longer has this video job",
        )
    if not response.is_success:
        raise VideoPollUnavailable(f"the video provider answered HTTP {response.status_code}")
    return None


def _asset_url(value: object) -> TypeGuard[str]:
    """Whether one provider-reported asset locator is an absolute HTTP(S) URL.

    The scheme and destination are enforced where the bytes are fetched: the
    data plane's upstream address policy admits only public HTTPS hosts on a
    hosted gateway.
    """
    return isinstance(value, str) and value.startswith(("https://", "http://"))


class VideoProviderClient(Protocol):
    """One provider's video job protocol over a resolved wire profile."""

    dialect: str
    stores_output: bool

    def unsupported(self, request: VideoRequest, model_id: str) -> tuple[str, str] | None:
        """Return ``(param, message)`` when this model cannot render the request."""
        ...

    def submit(self, profile: GatewayWireProfile, submission: VideoSubmission) -> str:
        """Submit one job and return the provider's job handle."""
        ...

    def poll(
        self, profile: GatewayWireProfile, provider_job_id: str, *, output_stored: bool = False
    ) -> VideoProviderSnapshot:
        """Observe one job; ``output_stored`` says it uploaded to host storage."""
        ...

    def content(self, profile: GatewayWireProfile, content_url: str) -> VideoContentFetch:
        """Describe how to download one completed job's video."""
        ...


def _duration_milli(value: object) -> int | None:
    """A provider-reported positive duration in seconds as thousandths, else None.

    xAI reports ``video.duration`` as a JSON number, which may be fractional.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    return round(value * 1000)


def _api_root(videos_url: str) -> str:
    """The Gemini API root (``.../v1beta``) in front of a ``/models/...`` endpoint."""
    return videos_url.split("/models/", 1)[0]


class VeoVideoClient:
    """Google Veo over the Gemini API's long-running prediction operations."""

    dialect = VEO_DIALECT
    stores_output = False

    def __init__(self, *, http: HttpFactory = default_http_factory) -> None:
        """Bind the HTTP client factory.

        Args:
            http: Builds one bounded client per call; tests inject a transport.
        """
        self._http = http

    def unsupported(self, request: VideoRequest, model_id: str) -> tuple[str, str] | None:
        """Name the request field this Veo model cannot render, if any.

        Args:
            request: The canonical request.
            model_id: The exact Veo model identifier.

        Returns:
            ``(param, message)`` for the first unsupported field, or None.
        """
        resolution = request.resolution
        lite = "lite" in model_id
        renders = ("720p", "1080p") if lite else ("720p", "1080p", "4k")
        if resolution not in renders:
            sizes = "1280x720, 720x1280, 1920x1080, or 1080x1920"
            if not lite:
                sizes += ", 3840x2160, or 2160x3840"
            return "size", f"This model renders size {sizes}; {request.size} is unavailable."
        if request.seconds not in {4, 6, 8}:
            return "seconds", "This model renders 4, 6, or 8 seconds."
        if resolution != "720p" and request.seconds != 8:
            return "seconds", "This model renders 1080p and 4k video at 8 seconds only."
        return None

    def submit(self, profile: GatewayWireProfile, submission: VideoSubmission) -> str:
        """Start one prediction operation and return its operation name.

        Raises:
            VideoSubmitRejected: No operation was created.
            VideoSubmitAmbiguous: An operation may exist without a readable name.
        """
        url = profile.videos_url
        if url is None:  # pragma: no cover - the engine filters rungs without one.
            raise VideoSubmitAmbiguous("the rung has no video endpoint")
        request = submission.request
        instance: JsonObject = {"prompt": request.prompt}
        if submission.reference_image is not None and request.input_reference is not None:
            instance["image"] = {
                "inlineData": {
                    "mimeType": request.input_reference.content_type,
                    "data": base64.b64encode(submission.reference_image).decode("ascii"),
                }
            }
        body: JsonObject = {
            "instances": [instance],
            "parameters": {
                "aspectRatio": request.aspect_ratio,
                "resolution": request.resolution,
                "durationSeconds": request.seconds,
            },
        }
        accepted = _post_submit(self._http, url, dict(profile.headers), body)
        name = accepted.get("name")
        if not isinstance(name, str) or not name or "/operations/" not in name:
            raise VideoSubmitAmbiguous("the Veo acceptance named no operation")
        return name

    def poll(
        self, profile: GatewayWireProfile, provider_job_id: str, *, output_stored: bool = False
    ) -> VideoProviderSnapshot:
        """Read one operation and normalize it (Veo never uploads to host storage).

        Raises:
            VideoPollUnavailable: The poll failed transiently.
        """
        del output_stored
        url = profile.videos_url
        if url is None:  # pragma: no cover - jobs are only created on rungs with one.
            raise VideoPollUnavailable("the rung has no video endpoint")
        response = _get_poll(
            self._http, f"{_api_root(url)}/{provider_job_id}", dict(profile.headers)
        )
        vanished = _vanished(response)
        if vanished is not None:
            return vanished
        operation = _json_object(response)
        if operation.get("done") is not True:
            return VideoProviderSnapshot(status=VideoStatus.IN_PROGRESS)
        error = operation.get("error")
        if isinstance(error, dict):
            return VideoProviderSnapshot(
                status=VideoStatus.FAILED,
                error_code="video_generation_failed",
                error_message=_bounded(error.get("message")),
            )
        result = operation.get("response")
        generated = result.get("generateVideoResponse") if isinstance(result, dict) else None
        samples = generated.get("generatedSamples") if isinstance(generated, dict) else None
        video = samples[0].get("video") if isinstance(samples, list) and samples else None
        uri = video.get("uri") if isinstance(video, dict) else None
        if _asset_url(uri):
            return VideoProviderSnapshot(
                status=VideoStatus.COMPLETED,
                progress=100,
                content_url=uri,
                asset_lifetime_seconds=VEO_ASSET_LIFETIME_SECONDS,
            )
        reasons = generated.get("raiMediaFilteredReasons") if isinstance(generated, dict) else None
        return VideoProviderSnapshot(
            status=VideoStatus.FAILED,
            error_code="content_filtered",
            error_message=(
                _bounded(reasons[0])
                if isinstance(reasons, list) and reasons
                else "the provider returned no video for this request"
            ),
        )

    def content(self, profile: GatewayWireProfile, content_url: str) -> VideoContentFetch:
        """Download from the sample URI, sending the key only to the API host."""
        api_host = urlsplit(profile.videos_url or "").hostname
        same_host = urlsplit(content_url).hostname == api_host
        return VideoContentFetch(
            url=content_url, headers=dict(profile.headers) if same_host else {}
        )


class XaiVideoClient:
    """xAI Grok Imagine video over its asynchronous generation API."""

    dialect = XAI_VIDEOS_DIALECT
    stores_output = True

    def __init__(self, *, http: HttpFactory = default_http_factory) -> None:
        """Bind the HTTP client factory.

        Args:
            http: Builds one bounded client per call; tests inject a transport.
        """
        self._http = http

    def unsupported(self, request: VideoRequest, model_id: str) -> tuple[str, str] | None:
        """Name the request field xAI cannot render, if any.

        A first-frame image is refused: xAI takes an image URL whose data-URL
        support is unverified, so the rung is narrowed past instead.

        Args:
            request: The canonical request.
            model_id: The exact xAI model identifier (unused; the family shares limits).

        Returns:
            ``(param, message)`` for the first unsupported field, or None.
        """
        del model_id
        if request.resolution == "4k":
            return "size", "This model renders at most 1080p; choose 1920x1080 or smaller."
        if not 1 <= request.seconds <= 15:
            return "seconds", "This model renders 1 to 15 seconds."
        if request.input_reference is not None:
            return "input_reference", "This model does not accept input_reference."
        return None

    def submit(self, profile: GatewayWireProfile, submission: VideoSubmission) -> str:
        """Start one generation and return its request id.

        Raises:
            VideoSubmitRejected: No generation was created.
            VideoSubmitAmbiguous: A generation may exist without a readable id.
        """
        url = profile.videos_url
        if url is None:  # pragma: no cover - the engine filters rungs without one.
            raise VideoSubmitAmbiguous("the rung has no video endpoint")
        request = submission.request
        body: JsonObject = {
            "model": submission.model_id,
            "prompt": request.prompt,
            "duration": request.seconds,
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
        }
        if submission.upload_url is not None:
            # xAI PUTs the raw MP4 to this URL and keeps no copy (required for
            # Zero Data Retention teams, honored for every team).
            body["output"] = {"upload_url": submission.upload_url}
        accepted = _post_submit(self._http, url, dict(profile.headers), body)
        request_id = accepted.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise VideoSubmitAmbiguous("the xAI acceptance named no request id")
        return request_id

    def poll(
        self, profile: GatewayWireProfile, provider_job_id: str, *, output_stored: bool = False
    ) -> VideoProviderSnapshot:
        """Read one generation and normalize it.

        A ``done`` generation that uploaded to host storage completes without
        a provider URL; the content route then serves the stored object.

        Raises:
            VideoPollUnavailable: The poll failed transiently.
        """
        url = profile.videos_url
        if url is None:  # pragma: no cover - jobs are only created on rungs with one.
            raise VideoPollUnavailable("the rung has no video endpoint")
        root = url.removesuffix("/videos/generations")
        response = _get_poll(self._http, f"{root}/videos/{provider_job_id}", dict(profile.headers))
        vanished = _vanished(response)
        if vanished is not None:
            return vanished
        body = _json_object(response)
        status = body.get("status")
        if status == "pending":
            return VideoProviderSnapshot(status=VideoStatus.IN_PROGRESS)
        if status != "done":
            return VideoProviderSnapshot(
                status=VideoStatus.FAILED,
                error_code="video_generation_failed",
                error_message=_bounded(body.get("error")) or f"the provider reported {status}",
            )
        video = body.get("video")
        video_url = video.get("url") if isinstance(video, dict) else None
        duration = video.get("duration") if isinstance(video, dict) else None
        duration_milli = _duration_milli(duration)
        if output_stored:
            return VideoProviderSnapshot(
                status=VideoStatus.COMPLETED, progress=100, duration_milli=duration_milli
            )
        if not _asset_url(video_url):
            return VideoProviderSnapshot(
                status=VideoStatus.FAILED,
                error_code="content_filtered",
                error_message="the provider returned no video for this request",
            )
        return VideoProviderSnapshot(
            status=VideoStatus.COMPLETED,
            progress=100,
            content_url=video_url,
            duration_milli=duration_milli,
        )

    def content(self, profile: GatewayWireProfile, content_url: str) -> VideoContentFetch:
        """Download from the presigned asset URL with no credentials."""
        del profile
        return VideoContentFetch(url=content_url)


def video_dialect(profile: GatewayWireProfile) -> str | None:
    """The video job dialect a resolved profile speaks, or None.

    Args:
        profile: A rung's resolved wire profile.

    Returns:
        ``veo`` on a Gemini connection with a video endpoint, ``xai_videos`` on
        an xAI-origin OpenAI-compatible connection, otherwise None.
    """
    if profile.videos_url is None:
        return None
    if profile.dialect == "gemini_generate_content":
        return VEO_DIALECT
    if profile.dialect == "openai_compatible":
        return XAI_VIDEOS_DIALECT
    return None


def default_video_clients(
    http: HttpFactory = default_http_factory,
) -> dict[str, VideoProviderClient]:
    """Build the client registry keyed by dialect.

    Args:
        http: The HTTP client factory every client shares.

    Returns:
        One client per supported dialect.
    """
    return {VEO_DIALECT: VeoVideoClient(http=http), XAI_VIDEOS_DIALECT: XaiVideoClient(http=http)}

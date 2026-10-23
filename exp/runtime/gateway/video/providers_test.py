"""Tests for the Veo and xAI video clients over scripted HTTP transports."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable

import httpx
import pytest

from exp.runtime.gateway.stream_contracts import GatewayFailureClass
from exp.runtime.gateway.video.contracts import VideoReferenceImage, VideoRequest, VideoStatus
from exp.runtime.gateway.video.providers import (
    VEO_ASSET_LIFETIME_SECONDS,
    VeoVideoClient,
    VideoPollUnavailable,
    VideoSubmission,
    VideoSubmitAmbiguous,
    VideoSubmitRejected,
    XaiVideoClient,
    video_dialect,
)
from exp.runtime.models.providers.base import GatewayWireProfile

_VEO = GatewayWireProfile(
    dialect="gemini_generate_content",
    url="https://generativelanguage.googleapis.com/v1beta/models/veo:streamGenerateContent",
    videos_url="https://generativelanguage.googleapis.com/v1beta/models/veo:predictLongRunning",
    headers={"x-goog-api-key": "secret"},
    model_id="veo-3.1-fast-generate-preview",
)
_XAI = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://api.x.ai/v1/chat/completions",
    videos_url="https://api.x.ai/v1/videos/generations",
    headers={"authorization": "Bearer secret"},
    model_id="grok-imagine-video",
)

Handler = Callable[[httpx.Request], httpx.Response]


def _http(handler: Handler) -> Callable[[], httpx.Client]:
    """Build an HTTP factory over one scripted handler."""
    return lambda: httpx.Client(transport=httpx.MockTransport(handler))


def _answer(status: int, payload: object) -> Handler:
    """A handler answering every request with one JSON response."""
    return lambda request: httpx.Response(status, json=payload)


def test_veo_submit_speaks_predict_long_running_with_an_inline_frame() -> None:
    """The submit carries the prompt, inline image, and Veo's parameter names."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record and accept."""
        seen.append(request)
        return httpx.Response(200, json={"name": "models/veo/operations/op-1"})

    image = b"\x89PNGframe"
    request = VideoRequest(
        prompt="a boat",
        seconds=6,
        size="720x1280",
        input_reference=VideoReferenceImage(
            content_type="image/png", bytes=len(image), sha256="a" * 64
        ),
    )
    client = VeoVideoClient(http=_http(handler))
    handle = client.submit(
        _VEO, VideoSubmission(request=request, model_id="veo", reference_image=image)
    )
    assert handle == "models/veo/operations/op-1"
    (sent,) = seen
    assert str(sent.url) == _VEO.videos_url
    assert sent.headers["x-goog-api-key"] == "secret"
    assert json.loads(sent.content) == {
        "instances": [
            {
                "prompt": "a boat",
                "image": {
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": base64.b64encode(image).decode(),
                    }
                },
            }
        ],
        "parameters": {"aspectRatio": "9:16", "resolution": "720p", "durationSeconds": 6},
    }


@pytest.mark.parametrize(
    ("status", "failure_class", "failover"),
    [
        (400, GatewayFailureClass.INVALID_REQUEST, False),
        (422, GatewayFailureClass.INVALID_REQUEST, False),
        (401, GatewayFailureClass.PROVIDER_AUTHENTICATION, True),
        (402, GatewayFailureClass.PROVIDER_QUOTA, True),
        (404, GatewayFailureClass.PROVIDER_NOT_FOUND, True),
        (429, GatewayFailureClass.THROTTLED, True),
        (503, GatewayFailureClass.PROVIDER_INTERNAL, True),
    ],
)
def test_submit_rejections_are_classified_for_failover(
    status: int, failure_class: GatewayFailureClass, failover: bool
) -> None:
    """Request errors stay with the caller; rung errors fail over."""
    client = VeoVideoClient(http=_http(_answer(status, {"error": {"message": "nope"}})))
    with pytest.raises(VideoSubmitRejected) as rejected:
        client.submit(_VEO, VideoSubmission(request=VideoRequest(prompt="p"), model_id="veo"))
    assert rejected.value.failure.failure_class is failure_class
    assert rejected.value.failure.failover_eligible is failover
    if not failover:
        assert rejected.value.failure.provider_detail == "nope"


def test_an_unconnected_submit_is_a_definitive_rejection() -> None:
    """A connection that never opened sent nothing, so it may fail over."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Fail before any bytes are sent."""
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(VideoSubmitRejected) as rejected:
        VeoVideoClient(http=_http(handler)).submit(
            _VEO, VideoSubmission(request=VideoRequest(prompt="p"), model_id="veo")
        )
    assert rejected.value.failure.failover_eligible


@pytest.mark.parametrize(
    "handler",
    [
        _answer(200, {"done": False}),
        lambda request: httpx.Response(200, content=b"not json"),
    ],
)
def test_an_unreadable_acceptance_is_ambiguous(handler: Handler) -> None:
    """A 2xx without a usable operation name may still have created a job."""
    with pytest.raises(VideoSubmitAmbiguous):
        VeoVideoClient(http=_http(handler)).submit(
            _VEO, VideoSubmission(request=VideoRequest(prompt="p"), model_id="veo")
        )


def test_veo_polls_normalize_every_terminal_shape() -> None:
    """Running, completed, failed, filtered, and vanished operations map to statuses."""
    root = "https://generativelanguage.googleapis.com/v1beta"
    cases: list[tuple[int, object, VideoStatus, str | None]] = [
        (200, {"done": False}, VideoStatus.IN_PROGRESS, None),
        (
            200,
            {"done": True, "error": {"message": "boom"}},
            VideoStatus.FAILED,
            "video_generation_failed",
        ),
        (
            200,
            {
                "done": True,
                "response": {"generateVideoResponse": {"raiMediaFilteredReasons": ["x"]}},
            },
            VideoStatus.FAILED,
            "content_filtered",
        ),
        (404, {"error": {"message": "gone"}}, VideoStatus.FAILED, "video_not_found"),
    ]
    for status, payload, expected, code in cases:
        seen: list[str] = []

        def handler(
            request: httpx.Request,
            status: int = status,
            payload: object = payload,
            seen: list[str] = seen,
        ) -> httpx.Response:
            """Record the poll URL and answer the case."""
            seen.append(str(request.url))
            return httpx.Response(status, json=payload)

        snapshot = VeoVideoClient(http=_http(handler)).poll(_VEO, "models/veo/operations/op-1")
        assert seen == [f"{root}/models/veo/operations/op-1"]
        assert (snapshot.status, snapshot.error_code) == (expected, code)
    sample = {"video": {"uri": f"{root}/files/abc:download?alt=media"}}
    done = {"done": True, "response": {"generateVideoResponse": {"generatedSamples": [sample]}}}
    snapshot = VeoVideoClient(http=_http(_answer(200, done))).poll(
        _VEO, "models/veo/operations/op-1"
    )
    assert snapshot.status is VideoStatus.COMPLETED
    assert snapshot.content_url == sample["video"]["uri"]
    assert snapshot.asset_lifetime_seconds == VEO_ASSET_LIFETIME_SECONDS


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_poll_failures_raise_for_a_retry(status: int) -> None:
    """Throttles and server errors leave the job open."""
    with pytest.raises(VideoPollUnavailable):
        VeoVideoClient(http=_http(_answer(status, {}))).poll(_VEO, "models/veo/operations/op")


def test_veo_sends_its_key_only_to_its_own_api_host() -> None:
    """A sample on the API host carries the key; any other host gets no credential."""
    client = VeoVideoClient()
    own = client.content(_VEO, "https://generativelanguage.googleapis.com/v1beta/files/x")
    assert own.headers == {"x-goog-api-key": "secret"}
    assert client.content(_VEO, "https://storage.example.com/x.mp4").headers == {}


@pytest.mark.parametrize(
    ("request_fields", "param"),
    [
        ({"size": "854x480"}, "size"),
        ({"seconds": 5}, "seconds"),
        ({"seconds": 4, "size": "1920x1080"}, "seconds"),
    ],
)
def test_veo_refuses_what_it_cannot_render(request_fields: dict[str, object], param: str) -> None:
    """Resolutions and durations outside Veo's grid name the field."""
    request = VideoRequest.model_validate({"prompt": "p", **request_fields})
    reason = VeoVideoClient().unsupported(request, "veo-3.1")
    assert reason is not None and reason[0] == param


def test_veo_lite_renders_no_4k() -> None:
    """The Lite model tops out at 1080p."""
    request = VideoRequest(prompt="p", seconds=8, size="3840x2160")
    assert VeoVideoClient().unsupported(request, "veo-3.1-generate-preview") is None
    reason = VeoVideoClient().unsupported(request, "veo-3.1-lite-generate-preview")
    assert reason is not None and reason[0] == "size"


def test_xai_submit_poll_and_content() -> None:
    """xAI's generation wire, its done shape with a duration, and keyless downloads."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Accept the submit, then report the finished generation."""
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"request_id": "req-1"})
        return httpx.Response(
            200,
            json={"status": "done", "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 5}},
        )

    client = XaiVideoClient(http=_http(handler))
    request = VideoRequest(prompt="p", seconds=6, size="1280x720")
    assert (
        client.submit(_XAI, VideoSubmission(request=request, model_id="grok-imagine-video"))
        == "req-1"
    )
    assert json.loads(seen[0].content) == {
        "model": "grok-imagine-video",
        "prompt": "p",
        "duration": 6,
        "aspect_ratio": "16:9",
        "resolution": "720p",
    }
    snapshot = client.poll(_XAI, "req-1")
    assert str(seen[1].url) == "https://api.x.ai/v1/videos/req-1"
    assert (snapshot.status, snapshot.duration_milli) == (VideoStatus.COMPLETED, 5_000)
    assert client.content(_XAI, "https://vidgen.x.ai/a.mp4").headers == {}


@pytest.mark.parametrize(
    ("payload", "status", "code"),
    [
        ({"status": "pending"}, VideoStatus.IN_PROGRESS, None),
        ({"status": "failed", "error": "moderated"}, VideoStatus.FAILED, "video_generation_failed"),
        ({"status": "expired"}, VideoStatus.FAILED, "video_generation_failed"),
        ({"status": "done", "video": {"url": None}}, VideoStatus.FAILED, "content_filtered"),
    ],
)
def test_xai_poll_shapes(payload: object, status: VideoStatus, code: str | None) -> None:
    """Pending, failed, expired, and URL-less generations normalize."""
    snapshot = XaiVideoClient(http=_http(_answer(200, payload))).poll(_XAI, "req")
    assert (snapshot.status, snapshot.error_code) == (status, code)


def test_veo_sends_duration_seconds_as_a_json_number() -> None:
    """Google rejects a string ``durationSeconds``; the wire carries an integer."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the submit and accept it."""
        seen.append(request)
        return httpx.Response(200, json={"name": "models/veo/operations/op-1"})

    client = VeoVideoClient(http=_http(handler))
    request = VideoRequest(prompt="p", seconds=4, size="1280x720")
    client.submit(_VEO, VideoSubmission(request=request, model_id="veo-3.1-lite-generate-preview"))
    parameters = json.loads(seen[0].content)["parameters"]
    assert parameters["durationSeconds"] == 4
    assert type(parameters["durationSeconds"]) is int


def test_xai_submit_names_the_host_upload_url_only_when_given() -> None:
    """A host upload destination rides ``output.upload_url``; none adds no ``output``."""
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record each submit body and accept it."""
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"request_id": f"req-{len(bodies)}"})

    client = XaiVideoClient(http=_http(handler))
    request = VideoRequest(prompt="p", seconds=4, size="854x480")
    upload = "https://bucket.example/put/video_1.mp4?sig=abc"
    client.submit(
        _XAI, VideoSubmission(request=request, model_id="grok-imagine-video", upload_url=upload)
    )
    client.submit(_XAI, VideoSubmission(request=request, model_id="grok-imagine-video"))
    assert bodies[0]["output"] == {"upload_url": upload}
    assert "output" not in bodies[1]
    assert type(bodies[0]["duration"]) is int


def test_xai_reports_a_fractional_duration_in_thousandths() -> None:
    """xAI's ``video.duration`` is a JSON number; 4.04 s bills 4,040 milliseconds."""
    payload = {"status": "done", "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": 4.04}}
    snapshot = XaiVideoClient(http=_http(_answer(200, payload))).poll(_XAI, "req")
    assert (snapshot.status, snapshot.duration_milli) == (VideoStatus.COMPLETED, 4_040)


@pytest.mark.parametrize("duration", [0, -1, True, "5", None])
def test_xai_ignores_an_unusable_duration(duration: object) -> None:
    """A missing, non-numeric, or non-positive duration bills the requested seconds."""
    payload = {
        "status": "done",
        "video": {"url": "https://vidgen.x.ai/a.mp4", "duration": duration},
    }
    snapshot = XaiVideoClient(http=_http(_answer(200, payload))).poll(_XAI, "req")
    assert snapshot.duration_milli is None


def test_an_uploaded_xai_generation_completes_without_a_provider_url() -> None:
    """With the video in host storage, ``done`` completes even with no ``video.url``."""
    payload = {"status": "done", "video": {"duration": 4}}
    client = XaiVideoClient(http=_http(_answer(200, payload)))
    stored = client.poll(_XAI, "req", output_stored=True)
    assert (stored.status, stored.content_url, stored.duration_milli) == (
        VideoStatus.COMPLETED,
        None,
        4_000,
    )
    assert client.poll(_XAI, "req").status is VideoStatus.FAILED
    failed = client.poll(_XAI, "req", output_stored=False)
    assert failed.error_code == "content_filtered"
    rejected = XaiVideoClient(http=_http(_answer(200, {"status": "failed", "error": "x"})))
    assert rejected.poll(_XAI, "req", output_stored=True).status is VideoStatus.FAILED


def test_only_xai_stores_output() -> None:
    """Veo never uploads to a caller URL; xAI does."""
    assert (VeoVideoClient.stores_output, XaiVideoClient.stores_output) == (False, True)


def test_xai_refuses_a_reference_image_and_4k() -> None:
    """Unverified image input and out-of-range output are narrowed past."""
    client = XaiVideoClient()
    image = VideoReferenceImage(content_type="image/png", bytes=1, sha256="a" * 64)
    reason = client.unsupported(VideoRequest(prompt="p", input_reference=image), "grok")
    assert reason is not None and reason[0] == "input_reference"
    reason = client.unsupported(VideoRequest(prompt="p", size="3840x2160"), "grok")
    assert reason is not None and reason[0] == "size"
    assert client.unsupported(VideoRequest(prompt="p", seconds=16), "grok") is not None


def test_video_dialect_reads_the_profile() -> None:
    """Gemini and xAI-origin profiles speak a video dialect; others do not."""
    assert video_dialect(_VEO) == "veo"
    assert video_dialect(_XAI) == "xai_videos"
    no_video = GatewayWireProfile(dialect="openai_compatible", url="https://x/v1/chat/completions")
    assert video_dialect(no_video) is None

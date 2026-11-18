"""Traffic capture of a video job, from admission to its settled render.

A video request settles when its render does, long after the create call
answers. Its capture entry therefore lives until the job's settlement deadline,
holds the reference image as an input media part, and records the job object as
its response: once at creation and again at the terminal state, just before
the settlement that releases the record.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.native_capture import (
    CaptureController,
    begin_capture,
    discard_capture,
)
from exp.runtime.gateway.video.contracts import (
    VIDEO_JOB_SETTLEMENT_SECONDS,
    VIDEO_LEDGER_DEADLINE_SLACK_SECONDS,
    VideoJob,
    VideoRequest,
    render_video_object,
)

_LOGGER = logging.getLogger(__name__)

CAPTURE_TTL_SECONDS = VIDEO_JOB_SETTLEMENT_SECONDS + VIDEO_LEDGER_DEADLINE_SLACK_SECONDS
"""How long a video capture waits for its job's settlement."""


class VideoCapture:
    """One engine's optional capture; a no-op unless the controller captures video."""

    def __init__(self, controller: CaptureController | None) -> None:
        """Bind the host's capture controller, or None to capture nothing."""
        self.controller = (
            controller if controller is not None and "video" in controller.protocols else None
        )

    def begin(
        self,
        authorization: AuthorizationSnapshot,
        request: VideoRequest,
        reference_image: bytes | None,
        session_id: str | None,
    ) -> bool:
        """Register an accepted job's capture; False when capture is unavailable."""
        if self.controller is None:
            return True
        if not begin_capture(
            self.controller,
            authorization,
            request,
            session_id=session_id,
            ttl_seconds=CAPTURE_TTL_SECONDS,
        ):
            return False
        if reference_image is not None and request.input_reference is not None:
            try:
                self.controller.native.attach_media(
                    authorization.request_id,
                    "input",
                    "image",
                    request.input_reference.content_type,
                    reference_image,
                )
            except Exception:  # noqa: BLE001 - capture is observational only.
                _LOGGER.warning("capture.media_dropped request_id=%s", authorization.request_id)
        return True

    @contextmanager
    def releasing(self, request_id: str) -> Iterator[None]:
        """Release the capture of a create that fails before a job exists."""
        try:
            yield
        except BaseException:
            discard_capture(self.controller, request_id)
            raise

    def record(self, job: VideoJob) -> VideoJob:
        """Record the job's current object as its captured response and return the job."""
        if self.controller is None:
            return job
        try:
            self.controller.native.finish_job(
                job.request_id,
                json.dumps(render_video_object(job), separators=(",", ":")),
                job.provider_job_id or job.video_id,
                job.settle_by.timestamp(),
                job.deployment.deployment_id,
            )
        except Exception:  # noqa: BLE001 - capture is observational only.
            _LOGGER.warning("capture.job_dropped request_id=%s", job.request_id)
        return job

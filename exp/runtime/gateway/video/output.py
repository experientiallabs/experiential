"""Host-owned storage for rendered videos (the provider uploads, the gateway serves).

xAI refuses video generation for Zero Data Retention teams unless the request
names ``output.upload_url``: xAI then ``PUT``s the rendered MP4's raw bytes to
that URL and keeps nothing itself. A host that installs a
:class:`VideoOutputStore` (``components.video_outputs``) makes every xAI job
take that path, whatever the team's retention setting: the store mints a
presigned upload URL per gateway ``video_id`` before submit, and
``/v1/videos/{id}/content`` streams from the store instead of the provider.
Without a store, xAI jobs are submitted without ``output`` (a ZDR team's
provider 400 then settles the attempt at $0 like any submit refusal).

Veo never uploads to a caller URL; its jobs ignore the store.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol

from exp.runtime.gateway.video.providers import VideoContentFetch


@dataclass(frozen=True)
class VideoUploadTarget:
    """One presigned destination the provider writes a rendered video to.

    Attributes:
        url: An absolute ``https`` URL accepting one HTTP ``PUT`` of the raw
            MP4 bytes with no extra headers. It must stay valid until
            ``expires_at``, which must reach past the job's settlement bound
            (two hours after creation).
        expires_at: When the URL stops accepting the upload.
    """

    url: str = field(repr=False)
    expires_at: datetime


class VideoOutputStore(Protocol):
    """Storage the provider uploads rendered videos into, keyed by gateway video id.

    Every method is called from synchronous worker threads and must be safe
    to call concurrently from every gateway worker sharing the store.
    """

    def upload_target(self, *, video_id: str, organization_id: str) -> VideoUploadTarget:
        """Mint the presigned ``PUT`` destination for one new job's video.

        Raises:
            Exception: Any failure; the rung is then settled unsent at $0 and
                the walk fails over.
        """
        ...

    def download(self, *, video_id: str, organization_id: str) -> VideoContentFetch | None:
        """Describe how the data plane fetches one stored video, or None when absent.

        The returned URL is fetched with ``GET`` (headers sent to that URL
        only) and streamed to the owner.
        """
        ...

    def delete(self, *, video_id: str, organization_id: str) -> None:
        """Remove one stored video; a missing object is not an error."""
        ...


class InMemoryVideoOutputStore:
    """A process-local :class:`VideoOutputStore` for tests and local launches.

    Upload URLs point at ``base_url``; :meth:`put` stands in for the provider's
    upload. Downloads answer a ``base_url`` locator the test resolves itself.
    """

    def __init__(self, *, base_url: str = "https://videos.invalid") -> None:
        """Bind the locator prefix.

        Args:
            base_url: Prefix of every minted URL.
        """
        self._base = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._objects: dict[tuple[str, str], bytes] = {}
        self.minted: list[str] = []

    def upload_target(self, *, video_id: str, organization_id: str) -> VideoUploadTarget:
        """Mint a deterministic upload URL valid for one day."""
        url = f"{self._base}/upload/{organization_id}/{video_id}.mp4"
        with self._lock:
            self.minted.append(video_id)
        return VideoUploadTarget(url=url, expires_at=datetime.now(UTC) + timedelta(days=1))

    def put(self, *, video_id: str, organization_id: str, data: bytes) -> None:
        """Record one provider upload."""
        with self._lock:
            self._objects[(organization_id, video_id)] = data

    def download(self, *, video_id: str, organization_id: str) -> VideoContentFetch | None:
        """Answer the stored object's locator, or None when nothing was uploaded."""
        with self._lock:
            present = (organization_id, video_id) in self._objects
        if not present:
            return None
        return VideoContentFetch(url=f"{self._base}/download/{organization_id}/{video_id}.mp4")

    def delete(self, *, video_id: str, organization_id: str) -> None:
        """Drop one stored object."""
        with self._lock:
            self._objects.pop((organization_id, video_id), None)

    def stored(self, *, video_id: str, organization_id: str) -> bytes | None:
        """Read one stored object (test helper)."""
        with self._lock:
            return self._objects.get((organization_id, video_id))

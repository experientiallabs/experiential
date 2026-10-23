"""Tests for the in-memory reference :class:`VideoOutputStore`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from exp.runtime.gateway.video.output import InMemoryVideoOutputStore, VideoOutputStore


def test_the_reference_store_round_trips_one_upload() -> None:
    """Mint, upload, download, and delete follow the protocol's contract."""
    store: VideoOutputStore = InMemoryVideoOutputStore(base_url="https://store.test/")
    target = store.upload_target(video_id="video_1", organization_id="org")
    assert target.url == "https://store.test/upload/org/video_1.mp4"
    assert target.expires_at > datetime.now(UTC) + timedelta(hours=2)
    assert store.download(video_id="video_1", organization_id="org") is None
    assert isinstance(store, InMemoryVideoOutputStore)
    store.put(video_id="video_1", organization_id="org", data=b"mp4")
    fetch = store.download(video_id="video_1", organization_id="org")
    assert fetch is not None and fetch.url == "https://store.test/download/org/video_1.mp4"
    assert fetch.headers == {}
    store.delete(video_id="video_1", organization_id="org")
    store.delete(video_id="video_1", organization_id="org")
    assert store.download(video_id="video_1", organization_id="org") is None


def test_objects_are_scoped_to_their_organization() -> None:
    """One organization's upload is invisible under another organization."""
    store = InMemoryVideoOutputStore()
    store.put(video_id="video_1", organization_id="org-a", data=b"mp4")
    assert store.download(video_id="video_1", organization_id="org-b") is None
    target = store.upload_target(video_id="video_1", organization_id="org-a")
    assert target.url not in repr(target)  # A presigned URL is a credential; never logged.

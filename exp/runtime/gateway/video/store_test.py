"""Tests for the SQLite video job store: ownership, compare-and-set, paging, and open jobs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from exp.runtime.gateway.sqlite.migrations import initialize_database
from exp.runtime.gateway.video.contracts import VideoStatus
from exp.runtime.gateway.video.contracts_test import job_fixture
from exp.runtime.gateway.video.store import SQLiteVideoJobStore


@pytest.fixture(name="store")
def _store(tmp_path: Path) -> SQLiteVideoJobStore:
    """A store over a freshly migrated gateway database."""
    database = tmp_path / "gateway.db"
    initialize_database(database)
    return SQLiteVideoJobStore(database)


def test_reads_are_scoped_to_the_owning_organization_and_identity(
    store: SQLiteVideoJobStore,
) -> None:
    """Another identity or tenant reads an existing id exactly like an absent one."""
    job = job_fixture()
    store.create(job=job)
    assert store.load(video_id=job.video_id, organization_id="org", identity_id="owner") == job
    assert store.load(video_id=job.video_id, organization_id="org", identity_id="other") is None
    assert store.load(video_id=job.video_id, organization_id="else", identity_id="owner") is None
    assert store.load_any(video_id=job.video_id) == job


def test_a_settled_job_cannot_be_regressed(store: SQLiteVideoJobStore) -> None:
    """Writes are compare-and-set on settled = 0: a slower writer cannot undo settlement."""
    job = job_fixture()
    store.create(job=job)
    settled = job.model_copy(update={"status": VideoStatus.COMPLETED, "settled": True})
    assert store.save_open(job=settled)
    stale = job.model_copy(update={"status": VideoStatus.IN_PROGRESS})
    assert not store.save_open(job=stale)
    assert store.load_any(video_id=job.video_id) == settled


def test_only_settled_owned_jobs_can_be_deleted(store: SQLiteVideoJobStore) -> None:
    """Deleting an open or foreign job is refused; a settled owned job disappears."""
    job = job_fixture()
    store.create(job=job)
    owner = {"video_id": job.video_id, "organization_id": "org", "identity_id": "owner"}
    assert not store.mark_deleted(**owner)
    assert store.save_open(job=job.model_copy(update={"settled": True}))
    assert not store.mark_deleted(video_id=job.video_id, organization_id="org", identity_id="x")
    assert store.mark_deleted(**owner)
    assert store.load(**owner) is None
    assert not store.mark_deleted(**owner)


def test_listing_pages_newest_first_and_honors_the_cursor(store: SQLiteVideoJobStore) -> None:
    """Pages walk creation order in either direction and skip deleted jobs."""
    base = job_fixture()
    ids = []
    for index in range(4):
        job = base.model_copy(
            update={
                "video_id": f"video_{index}",
                "created_at": base.created_at + timedelta(minutes=index),
            }
        )
        store.create(job=job)
        ids.append(job.video_id)
    store.create(job=base.model_copy(update={"video_id": "video_foreign", "identity_id": "x"}))
    owner = {"organization_id": "org", "identity_id": "owner"}
    first = store.list_jobs(**owner, limit=2, after=None, order="desc")
    assert [job.video_id for job in first] == ["video_3", "video_2"]
    second = store.list_jobs(**owner, limit=2, after="video_2", order="desc")
    assert [job.video_id for job in second] == ["video_1", "video_0"]
    ascending = store.list_jobs(**owner, limit=10, after="video_1", order="asc")
    assert [job.video_id for job in ascending] == ["video_2", "video_3"]
    foreign_cursor = store.list_jobs(**owner, limit=1, after="video_foreign", order="desc")
    assert [job.video_id for job in foreign_cursor] == ["video_3"]


def test_open_jobs_are_the_unsettled_ones_oldest_first(store: SQLiteVideoJobStore) -> None:
    """The poller sees every unsettled job and no settled one."""
    base = job_fixture()
    store.create(job=base.model_copy(update={"video_id": "video_new"}))
    older = base.model_copy(
        update={"video_id": "video_old", "created_at": base.created_at - timedelta(hours=1)}
    )
    store.create(job=older)
    store.create(job=base.model_copy(update={"video_id": "video_done", "settled": True}))
    assert [job.video_id for job in store.open_jobs(limit=10)] == ["video_old", "video_new"]
    assert [job.video_id for job in store.open_jobs(limit=1)] == ["video_old"]

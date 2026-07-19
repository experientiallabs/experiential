from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import wmh.evals.study_journal as mod
from wmh.evals.study_journal import (
    STUDY_PHASES,
    ExternalPublicationReceipt,
    StudyJournalStore,
    StudyPhase,
    StudyPhaseCommitment,
    append_study_phase,
    load_study_journal,
)


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


class _Publisher:
    configuration_digest = _digest("test-publisher-config")

    def __init__(self) -> None:
        self.published: list[str] = []
        self.verified: list[str] = []
        self.receipts: dict[str, ExternalPublicationReceipt] = {}

    def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        self.published.append(commitment.digest)
        receipt = self.receipts.get(commitment.digest)
        if receipt is None:
            receipt = ExternalPublicationReceipt(
                commitment_digest=commitment.digest,
                publisher="test-transparency-log",
                publication_id=f"entry-{len(self.receipts):04d}",
                immutable_locator=f"test://study-log/{commitment.digest}",
                published_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
                evidence={"entry_digest": _digest(commitment.digest)},
            )
            self.receipts[commitment.digest] = receipt
        return receipt

    def verify(
        self,
        commitment: StudyPhaseCommitment,
        receipt: ExternalPublicationReceipt,
    ) -> None:
        self.verified.append(commitment.digest)
        if self.receipts.get(commitment.digest) != receipt:
            raise ValueError("publication receipt is not present in the external log")


def _store(tmp_path: Path) -> StudyJournalStore:
    return StudyJournalStore.create(
        tmp_path / "journal",
        study_id="study-1",
        publisher_configuration_digest=_Publisher.configuration_digest,
    )


def test_journal_requires_exact_phase_order_and_chains_external_commitments(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()

    records = []
    for index, phase in enumerate(STUDY_PHASES):
        record = append_study_phase(
            store,
            phase=phase,
            payload_digest=_digest(f"payload-{index}"),
            publisher=publisher,
        )
        records.append(record)

    assert tuple(record.commitment.phase for record in records) == STUDY_PHASES
    assert records[0].commitment.previous_record_digest is None
    assert tuple(record.commitment.sequence for record in records) == tuple(range(len(records)))
    for previous, current in zip(records, records[1:], strict=False):
        assert current.commitment.previous_record_digest == previous.digest
    assert publisher.published == [record.commitment.digest for record in records]
    assert publisher.verified == [record.commitment.digest for record in records]

    restarted = StudyJournalStore(
        store.directory,
        study_id="study-1",
        publisher_configuration_digest=_Publisher.configuration_digest,
    )
    loaded = load_study_journal(restarted, publisher=publisher)
    assert loaded == tuple(records)
    assert publisher.verified[-len(records) :] == [record.commitment.digest for record in records]


def test_append_is_idempotent_without_republishing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    first = append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )

    repeated = append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )

    assert repeated == first
    assert publisher.published == [first.commitment.digest]
    assert publisher.verified == [first.commitment.digest, first.commitment.digest]


def test_append_rejects_skip_reorder_and_payload_drift(tmp_path: Path) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )

    with pytest.raises(ValueError, match="next study phase"):
        append_study_phase(
            store,
            phase=StudyPhase.PROTOCOL_PUBLISHED,
            payload_digest=_digest("protocol"),
            publisher=publisher,
        )
    with pytest.raises(ValueError, match="already committed with a different payload"):
        append_study_phase(
            store,
            phase=StudyPhase.PREPARATION_PLANNED,
            payload_digest=_digest("different"),
            publisher=publisher,
        )


def test_bad_publication_receipt_never_creates_a_local_record(tmp_path: Path) -> None:
    store = _store(tmp_path)

    class _BadPublisher(_Publisher):
        def publish(self, commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
            return ExternalPublicationReceipt(
                commitment_digest=_digest("wrong"),
                publisher="test-transparency-log",
                publication_id="bad-entry",
                immutable_locator="test://study-log/bad-entry",
                published_at=datetime(2026, 7, 19, 12, 0, tzinfo=UTC),
                evidence={},
            )

    with pytest.raises(ValueError, match="different commitment"):
        append_study_phase(
            store,
            phase=StudyPhase.PREPARATION_PLANNED,
            payload_digest=_digest("plan"),
            publisher=_BadPublisher(),
        )
    assert load_study_journal(store) == ()


def test_load_rejects_local_record_tampering(tmp_path: Path) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    record = append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )
    record_path = store.directory / "000-preparation_planned.json"
    raw = json.loads(record_path.read_text(encoding="utf-8"))
    raw["commitment"]["payload_digest"] = _digest("tampered")
    record_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="record digest"):
        load_study_journal(store)
    assert record.commitment.payload_digest != raw["commitment"]["payload_digest"]


def test_store_rejects_wrong_study_and_symlinked_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="different study"):
        StudyJournalStore(
            store.directory,
            study_id="study-2",
            publisher_configuration_digest=_Publisher.configuration_digest,
        )

    with pytest.raises(ValueError, match="different publisher"):
        StudyJournalStore(
            store.directory,
            study_id="study-1",
            publisher_configuration_digest=_digest("other-publisher"),
        )

    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    (store.directory / "000-preparation_planned.json").symlink_to(target)
    with pytest.raises(OSError, match="regular file"):
        load_study_journal(store)


def test_load_rejects_backward_publication_time_and_public_record_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    first = append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )

    original_publish = publisher.publish

    def publish_backdated(commitment: StudyPhaseCommitment) -> ExternalPublicationReceipt:
        receipt = original_publish(commitment).model_copy(
            update={"published_at": first.publication.published_at - timedelta(seconds=1)}
        )
        publisher.receipts[commitment.digest] = receipt
        return receipt

    monkeypatch.setattr(publisher, "publish", publish_backdated)
    with pytest.raises(ValueError, match="timestamps move backwards"):
        append_study_phase(
            store,
            phase=StudyPhase.ROSTER_QUALIFIED,
            payload_digest=_digest("roster"),
            publisher=publisher,
        )
    assert not (store.directory / "001-roster_qualified.json").exists()

    first_path = store.directory / "000-preparation_planned.json"
    os.chmod(first_path, 0o644)
    with pytest.raises(OSError, match="group or other users"):
        load_study_journal(store)


def test_append_never_follows_a_replaced_journal_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    original_directory = tmp_path / "original-journal"
    original_publish = publisher.publish

    def publish_after_replacement(
        commitment: StudyPhaseCommitment,
    ) -> ExternalPublicationReceipt:
        receipt = original_publish(commitment)
        store.directory.rename(original_directory)
        store.directory.mkdir(mode=0o700)
        return receipt

    monkeypatch.setattr(publisher, "publish", publish_after_replacement)

    with pytest.raises(OSError, match="directory was replaced"):
        append_study_phase(
            store,
            phase=StudyPhase.PREPARATION_PLANNED,
            payload_digest=_digest("plan"),
            publisher=publisher,
        )

    assert not (store.directory / "000-preparation_planned.json").exists()
    assert (original_directory / "000-preparation_planned.json").is_file()


def test_create_fsyncs_the_parent_after_creating_the_journal_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_identity = (tmp_path.stat().st_dev, tmp_path.stat().st_ino)
    fsynced_identities: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        fsynced_identities.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record_fsync)

    _store(tmp_path)

    assert parent_identity in fsynced_identities


def test_create_requires_an_existing_parent_directory(tmp_path: Path) -> None:
    journal = tmp_path / "missing" / "journal"

    with pytest.raises(OSError, match="parent directory does not exist"):
        StudyJournalStore.create(
            journal,
            study_id="study-1",
            publisher_configuration_digest=_Publisher.configuration_digest,
        )

    assert not journal.parent.exists()


def test_create_rejects_a_copied_genesis_directory_swap_before_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal = tmp_path / "journal"
    original_directory = tmp_path / "original-journal"
    real_open = mod._open_private_directory
    swapped = False

    def swap_then_open(
        path: Path,
        *,
        expected_identity: tuple[int, int] | None = None,
    ) -> int:
        nonlocal swapped
        if path == journal and not swapped:
            swapped = True
            journal.rename(original_directory)
            journal.mkdir(mode=0o700)
            replacement_genesis = journal / "study-journal.json"
            replacement_genesis.write_bytes(
                (original_directory / "study-journal.json").read_bytes()
            )
            replacement_genesis.chmod(0o600)
        return real_open(path, expected_identity=expected_identity)

    monkeypatch.setattr(mod, "_open_private_directory", swap_then_open)

    with pytest.raises(OSError, match="replaced during create"):
        StudyJournalStore.create(
            journal,
            study_id="study-1",
            publisher_configuration_digest=_Publisher.configuration_digest,
        )

    assert swapped is True
    assert (original_directory / "study-journal.json").is_file()


def test_directory_inode_lease_cannot_be_bypassed_by_lock_path_replacement(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    mutable_lock_path = store.directory / ".study-journal.lock"
    mutable_lock_path.write_text("old", encoding="utf-8")
    mutable_lock_path.chmod(0o600)

    with store.locked():
        mutable_lock_path.unlink()
        mutable_lock_path.write_text("replacement", encoding="utf-8")
        mutable_lock_path.chmod(0o600)
        with pytest.raises(RuntimeError, match="already locked"):
            with store.locked():
                pytest.fail("a second directory lease must not enter")


def test_temporary_cleanup_failure_never_masks_the_primary_publish_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    real_unlink = os.unlink

    def fail_record_write(_descriptor: int, _payload: bytes) -> int:
        raise OSError("forced record write failure")

    def fail_temporary_unlink(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if str(path).startswith(".tmp-"):
            raise OSError("forced temporary unlink failure")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "write", fail_record_write)
    monkeypatch.setattr(os, "unlink", fail_temporary_unlink)

    with pytest.raises(OSError, match="forced record write failure") as captured:
        append_study_phase(
            store,
            phase=StudyPhase.PREPARATION_PLANNED,
            payload_digest=_digest("plan"),
            publisher=publisher,
        )

    notes = getattr(captured.value, "__notes__", [])
    assert any("forced temporary unlink failure" in note for note in notes)
    assert not (store.directory / "000-preparation_planned.json").exists()


def test_descriptor_close_failure_never_masks_the_primary_journal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)
    real_close = os.close

    def fail_close(candidate: int) -> None:
        if candidate == descriptor:
            raise OSError("forced journal descriptor close failure")
        real_close(candidate)

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(os, "close", fail_close)
            with pytest.raises(ValueError, match="primary journal failure") as captured:
                with mod._managed_descriptor(descriptor):
                    raise ValueError("primary journal failure")
    finally:
        real_close(descriptor)

    notes = getattr(captured.value, "__notes__", [])
    assert any("forced journal descriptor close failure" in note for note in notes)


def test_temporary_unlink_is_followed_by_a_directory_fsync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    publisher = _Publisher()
    directory_identity = (store.directory.stat().st_dev, store.directory.stat().st_ino)
    events: list[str] = []
    real_fsync = os.fsync
    real_unlink = os.unlink

    def record_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == directory_identity:
            events.append("directory-fsync")
        real_fsync(descriptor)

    def record_unlink(
        path: str | bytes,
        *,
        dir_fd: int | None = None,
    ) -> None:
        if str(path).startswith(".tmp-"):
            events.append("temporary-unlink")
        real_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "unlink", record_unlink)

    append_study_phase(
        store,
        phase=StudyPhase.PREPARATION_PLANNED,
        payload_digest=_digest("plan"),
        publisher=publisher,
    )

    unlink_index = events.index("temporary-unlink")
    assert "directory-fsync" in events[unlink_index + 1 :]

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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

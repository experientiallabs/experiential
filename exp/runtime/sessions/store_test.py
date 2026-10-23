"""Tests for the durable SQLite session store, including crash, replay, and concurrency paths."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.runtime.sessions.store import (
    ClaimLostError,
    RequestPreflightReceipt,
    SessionBinding,
    SessionDrainingError,
    SessionExistsError,
    SessionNotFoundError,
    SessionStore,
    SessionStoreError,
    StepAmbiguousError,
    StepConflictError,
    StepInFlightError,
    StepRejectedError,
    VersionConflictError,
)

BINDING = SessionBinding(
    deadline=datetime(2099, 1, 1, tzinfo=UTC),
    target_id="retail@v1",
    initialization_receipt_sha256=sha256_json({"receipt": 1}),
    knowledge_sha256=sha256_json({"knowledge": 1}),
    corpus_sha256=sha256_json({"corpus": 1}),
    representation_sha256=sha256_json({"representation": 1}),
    execution_policy_sha256=sha256_json({"policy": "offline-store-test"}),
)


def _store(tmp_path: Path) -> SessionStore:
    """Create a fresh store in a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        A store with one session named ``s1`` at version 0.
    """
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.create_session("s1", BINDING, {"orders": {"o1": {"status": "open"}}})
    return store


def _digest(label: str) -> str:
    """Return a stable request digest for test requests.

    Args:
        label: Human-readable request label.

    Returns:
        A SHA-256 hex digest.
    """
    return sha256_json({"request": label})


@pytest.mark.parametrize("accepted", [True, False])
def test_preflight_receipt_is_durable_idempotent_and_does_not_change_state(
    tmp_path: Path, accepted: bool
) -> None:
    """Claimed-step audit receipts survive restart without consuming provider-start budget."""
    store = _store(tmp_path)
    before = store.read_session("s1")
    claim = store.begin_step("s1", "one", _digest("one"), 0, BINDING).claim
    assert claim is not None
    receipt = RequestPreflightReceipt(
        request_sha256="a" * 64,
        capacity_sha256="b" * 64,
        input_token_upper_bound=200,
        output_tokens=100,
        accepted=accepted,
    )
    store.record_request_preflight(claim, receipt)
    restarted = SessionStore(tmp_path / "sessions.sqlite")
    restarted.record_request_preflight(claim, receipt)
    assert restarted.read_session("s1") == before
    with sqlite3.connect(tmp_path / "sessions.sqlite") as connection:
        rows = connection.execute(
            "SELECT event, detail_sha256 FROM audit WHERE step_id = 'one' ORDER BY seq"
        ).fetchall()
    assert [event for event, _ in rows] == ["step_claimed", "request_preflight"]
    assert rows[-1][1] == sha256_json(receipt.model_dump(mode="json"))
    with pytest.raises(StepConflictError, match="receipt differs"):
        restarted.record_request_preflight(
            claim, receipt.model_copy(update={"request_sha256": "c" * 64})
        )
    restarted.release_step(claim)
    with pytest.raises(ClaimLostError):
        restarted.record_request_preflight(claim, receipt)


def test_preflight_receipt_cannot_be_added_after_dispatch_or_recovery(tmp_path: Path) -> None:
    """A stale or started claim cannot acquire a retroactive pre-dispatch qualification."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "one", _digest("one"), 0, BINDING).claim
    assert claim is not None
    receipt = RequestPreflightReceipt(
        request_sha256="a" * 64,
        capacity_sha256="b" * 64,
        input_token_upper_bound=200,
        output_tokens=100,
        accepted=True,
    )
    store.mark_provider_started(claim, 1)
    with pytest.raises(ClaimLostError):
        store.record_request_preflight(claim, receipt)
    store.recover()
    with pytest.raises(ClaimLostError):
        store.record_request_preflight(claim, receipt)


def test_complete_step_commits_state_response_and_version_together(tmp_path: Path) -> None:
    """A completed step advances the version and replays the stored response verbatim."""
    store = _store(tmp_path)
    claim = store.begin_step(
        "s1", "step-1", _digest("a"), expected_version=0, binding=BINDING
    ).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )

    record = store.complete_step(
        claim,
        next_state={"orders": {"o1": {"status": "cancelled"}}},
        response={"status": "supported", "reply_raw": '{"ok": true}  \u00e9'},
    )

    assert record.version == 1
    assert record.state == {"orders": {"o1": {"status": "cancelled"}}}
    replay = store.begin_step("s1", "step-1", _digest("a"), expected_version=0, binding=BINDING)
    assert replay.kind == "completed"
    assert replay.completed is not None
    assert replay.completed.response == {
        "status": "supported",
        "reply_raw": '{"ok": true}  \u00e9',
    }
    assert replay.completed.resulting_version == 1


def test_replay_after_lost_response_never_reaches_provider_twice(tmp_path: Path) -> None:
    """A completed step replays from the store even when the caller lost its response."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(claim, {"orders": {}}, {"reply_raw": "x"})

    second = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING)

    assert second.kind == "completed"
    assert store.read_session("s1").version == 1


def test_crash_before_provider_start_is_released_and_retried_under_new_step(
    tmp_path: Path,
) -> None:
    """A claim that never reached the provider is released, not marked ambiguous."""
    store = _store(tmp_path)
    store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING)

    report = SessionStore(tmp_path / "sessions.sqlite").recover()

    assert report.released == (("s1", "step-1"),)
    assert report.ambiguous == ()
    assert store.read_session("s1").version == 0
    fresh = store.begin_step("s1", "step-2", _digest("a"), 0, binding=BINDING)
    assert fresh.kind == "claimed"


def test_crash_after_provider_start_becomes_ambiguous_and_blocks_retry(tmp_path: Path) -> None:
    """A claim that reached the provider without a response is ambiguous and never retried."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)

    restarted = SessionStore(tmp_path / "sessions.sqlite")
    report = restarted.recover()

    assert report.ambiguous == (("s1", "step-1"),)
    assert restarted.open_steps("s1") == ("step-1",)
    with pytest.raises(StepAmbiguousError):
        restarted.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING)
    with pytest.raises(StepInFlightError):
        restarted.begin_step("s1", "step-2", _digest("b"), 0, binding=BINDING)
    assert restarted.read_session("s1").version == 0


def test_stale_expected_version_is_refused_without_claim(tmp_path: Path) -> None:
    """A caller that computed its request against an older version gets no claim."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(claim, {"orders": {}}, {"reply_raw": "x"})

    with pytest.raises(VersionConflictError):
        store.begin_step("s1", "step-2", _digest("b"), expected_version=0, binding=BINDING)
    assert store.open_steps("s1") == ()


def test_concurrent_claims_on_one_session_allow_exactly_one(tmp_path: Path) -> None:
    """Eight threads race for the same session version; exactly one wins the claim."""
    store = _store(tmp_path)
    outcomes: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def attempt(index: int) -> None:
        """Try one claim on the shared session and record its outcome.

        Args:
            index: Distinct per-thread step suffix.
        """
        barrier.wait()
        try:
            store.begin_step(
                "s1", f"step-{index}", _digest(str(index)), expected_version=0, binding=BINDING
            )
            result = "claimed"
        except (StepInFlightError, VersionConflictError):
            result = "refused"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count("claimed") == 1
    assert outcomes.count("refused") == 7
    assert len(store.open_steps("s1")) == 1


def test_step_id_reuse_with_different_request_is_refused(tmp_path: Path) -> None:
    """A step ID cannot silently stand for a different request."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(claim, {"orders": {}}, {"reply_raw": "x"})

    with pytest.raises(StepConflictError):
        store.begin_step("s1", "step-1", _digest("different"), 0, binding=BINDING)


def test_claim_token_must_match_open_claim(tmp_path: Path) -> None:
    """A stale or forged claim cannot complete or release a step."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    forged = claim.model_copy(update={"claim_token": "forged"})

    with pytest.raises(ClaimLostError):
        store.complete_step(forged, {"orders": {}}, {"reply_raw": "x"})
    with pytest.raises(ClaimLostError):
        store.release_step(forged)
    store.release_step(claim)
    assert store.read_session("s1").version == 0


def test_provider_started_claim_cannot_be_released(tmp_path: Path) -> None:
    """Once the provider may have run, a claim cannot be released back to retryable."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)

    with pytest.raises(ClaimLostError):
        store.release_step(claim)


def test_draining_refuses_new_steps_but_not_existing_completion(tmp_path: Path) -> None:
    """Draining blocks new work while a completed step still replays for its caller."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(claim, {"orders": {}}, {"reply_raw": "x"})

    store.set_draining("s1", True)

    assert store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).kind == "completed"
    with pytest.raises(SessionDrainingError):
        store.begin_step("s1", "step-2", _digest("b"), 1, binding=BINDING)


def test_schema_mismatch_fails_closed(tmp_path: Path) -> None:
    """A database from another schema version is refused rather than migrated."""
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()

    with pytest.raises(SessionStoreError):
        SessionStore(path)


def test_unknown_session_and_duplicate_create_are_refused(tmp_path: Path) -> None:
    """Missing sessions and duplicate identifiers raise typed errors."""
    store = _store(tmp_path)

    with pytest.raises(SessionNotFoundError):
        store.read_session("missing")
    with pytest.raises(SessionExistsError):
        store.create_session("s1", BINDING, {})


def test_binding_is_stored_and_immutable(tmp_path: Path) -> None:
    """The binding round-trips on read and cannot be rewritten in place."""
    store = _store(tmp_path)
    assert store.read_session("s1").binding == BINDING

    connection = sqlite3.connect(tmp_path / "sessions.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE sessions SET binding_json = '{}' WHERE session_id = 's1'")
    connection.close()


def test_replay_matches_original_request_after_version_advances(tmp_path: Path) -> None:
    """A lost response replays after later steps advance the version."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(claim, {"orders": {}}, {"reply_raw": "first"})
    second = store.begin_step("s1", "step-2", _digest("b"), 1, binding=BINDING).claim
    assert second is not None
    store.mark_provider_started(second, 100)
    store.record_proposal(
        second,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    store.complete_step(second, {"orders": {"o2": {}}}, {"reply_raw": "second"})

    replay = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING)

    assert replay.completed is not None
    assert replay.completed.response == {"reply_raw": "first"}
    assert replay.completed.resulting_version == 1
    assert store.read_session("s1").version == 2


def test_rejected_step_stays_closed_without_state_change(tmp_path: Path) -> None:
    """A rejected step closes with its reason and never advances the version."""
    store = _store(tmp_path)
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.reject_step(claim, reason="ReplyRejectedError")

    with pytest.raises(StepRejectedError, match="ReplyRejectedError"):
        store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING)
    assert store.read_session("s1").version == 0
    assert store.open_steps("s1") == ()


def test_binding_refuses_missing_execution_policy_digest() -> None:
    """A binding without an execution-policy digest is refused, not defaulted."""
    fields = BINDING.model_dump(mode="json")
    del fields["execution_policy_sha256"]
    with pytest.raises(ValueError):
        SessionBinding.model_validate(fields)


def test_binding_refuses_unversioned_target() -> None:
    """A target name without a deployed profile version is refused."""
    with pytest.raises(ValueError):
        SessionBinding.model_validate({**BINDING.model_dump(mode="json"), "target_id": "retail"})


ORDERED_OPTIONS: JsonObject = {"color": "red", "band material": "steel", "display": "lcd"}


def _options(state: JsonObject) -> JsonObject:
    """Return the nested options object of a test state, narrowing its JSON type.

    Args:
        state: A state built by these tests.

    Returns:
        The nested ``product.options`` object.

    Raises:
        AssertionError: If the structure is not the expected nested object.
    """
    product = state["product"]
    assert isinstance(product, dict)
    options = product["options"]
    assert isinstance(options, dict)
    return options


def test_nested_state_keeps_declared_key_order_across_reload(tmp_path: Path) -> None:
    """A committed nested state reloads with its declared key order, not sorted order."""
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.create_session("s1", BINDING, {"product": {"options": ORDERED_OPTIONS}})
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    store.record_proposal(
        claim,
        {"status": "supported", "reply": {"reply_raw": "x"}, "state_delta": [], "reason": None},
    )
    record = store.complete_step(
        claim, {"product": {"options": ORDERED_OPTIONS}}, {"reply_raw": "x"}
    )

    reloaded = SessionStore(tmp_path / "sessions.sqlite").read_session("s1")

    assert list(_options(record.state)) == list(ORDERED_OPTIONS)
    assert list(_options(reloaded.state)) == list(ORDERED_OPTIONS)


def test_proposal_and_response_keep_declared_key_order(tmp_path: Path) -> None:
    """A stored proposal and response keep the order the provider declared."""
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.create_session("s1", BINDING, {})
    claim = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).claim
    assert claim is not None
    store.mark_provider_started(claim, 100)
    proposal: JsonObject = {
        "status": "supported",
        "reply": {"reply_raw": "x"},
        "state_delta": [],
        "reason": None,
        "zeta": 1,
        "alpha": 2,
    }
    store.record_proposal(claim, proposal)
    store.complete_step(claim, {}, {"zeta": 1, "alpha": 2, "reply_raw": "x"})

    stored_proposal = store.proposal("s1", "step-1")
    stored_response = store.begin_step("s1", "step-1", _digest("a"), 0, binding=BINDING).completed

    assert stored_proposal is not None and list(stored_proposal) == list(proposal)
    assert stored_response is not None and list(stored_response.response) == [
        "zeta",
        "alpha",
        "reply_raw",
    ]


def test_binding_digest_stays_canonical_and_order_independent() -> None:
    """Identity digests ignore key order: the binding digest is the canonical one."""
    from exp.common.core.artifacts import sha256_json

    reordered = {
        key: BINDING.model_dump(mode="json")[key]
        for key in reversed(list(BINDING.model_dump(mode="json")))
    }
    assert sha256_json(SessionBinding.model_validate(reordered)) == sha256_json(BINDING)


def test_schema_one_database_is_refused_without_migration(tmp_path: Path) -> None:
    """A database stamped with schema 1 is refused and never rewritten."""
    import sqlite3

    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    connection.execute("PRAGMA user_version = 1")
    connection.commit()
    connection.close()

    with pytest.raises(SessionStoreError, match="schema 1 is not 3"):
        SessionStore(path)

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA user_version").fetchone()[0] == 1
        assert [r[0] for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")] == [
            "sessions"
        ]
    finally:
        raw.close()


def test_late_result_is_retained_and_terminal_without_mutation(tmp_path: Path) -> None:
    """Commit authorization is sampled under the transaction after proposal validation."""
    from exp.runtime.sessions.store import StepDeadlineExceededError

    before = datetime(2030, 1, 1, tzinfo=UTC)
    deadline = datetime(2030, 1, 2, tzinfo=UTC)
    clock = [before]
    binding = BINDING.model_copy(update={"deadline": deadline})
    store = SessionStore(tmp_path / "late.sqlite", clock=lambda: clock[0])
    store.create_session("s1", binding, {"items": []})
    claim = store.begin_step("s1", "op", _digest("late"), 0, binding).claim
    assert claim is not None
    store.mark_provider_started(claim, 1)
    proposal: JsonObject = {"result": "late", "items": ["new"]}
    store.record_proposal(claim, proposal)
    clock[0] = deadline
    with pytest.raises(StepDeadlineExceededError):
        store.complete_step(claim, {"items": ["new"]}, {"reply": "ok"})
    reopened = SessionStore(tmp_path / "late.sqlite")
    assert reopened.read_session("s1").version == 0
    assert reopened.read_session("s1").state == {"items": []}
    assert reopened.proposal("s1", "op") == proposal
    with pytest.raises(StepRejectedError):
        reopened.begin_step("s1", "op", _digest("late"), 0, binding)

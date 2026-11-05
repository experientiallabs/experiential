"""The real shared physical ledger keeps legacy and parent scope ceilings distinct."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject
from exp.common.project import ProjectConfig, ProjectStore, artifact_input
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.scoped_spending import (
    EnvelopeCompletion,
    ParentSpendPolicy,
    ParentSpendRef,
    ScopeBudgetRefused,
    SpendEnvelopePlan,
    SpendEnvelopeRef,
    associate_reservation,
    open_spend_envelope,
    parent_scope_summary,
    register_parent_policy,
    scoped_spend_ceiling,
    settle_spend_envelope,
    spend_envelope_scope,
)
from exp.runtime.models.spend_ceiling import (
    AliasTariff,
    OperationSpendCeiling,
    UnpricedDispatchRefused,
    current_spend_ceiling,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.runtime.models.spend_ceiling_test import _transport

URL = "https://fixture.invalid/v1/chat/completions"
PAYLOAD: JsonObject = {"messages": [], "max_tokens": 1}
REF = ArtifactInput(artifact_id="fixture-proof", sha256="a" * 64)


def setup_policy(
    tmp_path: Path,
    *,
    legacy: int = 2,
    parent: int = 5,
    retained: int = 0,
    known_usd: float = 0,
    tariffs: dict[str, AliasTariff] | None = None,
) -> tuple[OperationSpendCeiling, ParentSpendRef]:
    """Create native original authority and an explicitly consented synthetic parent policy."""
    project = ProjectStore(tmp_path, "scope")
    project.initialize(ProjectConfig(project_id="scope"))
    proof = artifact_input(
        project.artifacts.write_json(
            artifact_id="scope-proof",
            artifact_type="fixture-scope-proof",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="fixture",
            ),
            files={"proof.json": {"fixture": True}},
        )
    )
    ceiling = OperationSpendCeiling(tmp_path / "spend.sqlite", tariffs=tariffs or {})
    ceiling.authorize_policy(ceiling_usd=known_usd, unpriced_attempts=legacy)
    for index in range(retained):
        claim = ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
        ceiling.mark_dispatched(claim, f"original-attempt-{index}")
        if index == 0:
            ceiling.retain_unknown(claim)
        else:
            ceiling.settle(claim, JsonHttpResponse(status_code=200, body={}))
    policy = ParentSpendPolicy(
        operation="original-operation",
        legacy_attempt_limit=legacy,
        parent_attempt_limit=parent,
        maximum_known_usd=known_usd,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        aliases=("actor",),
        producer_revision="fixture",
        authorization=proof,
        preparation=proof,
        execution=proof,
    )
    registered = register_parent_policy(ceiling, policy)
    return ceiling, registered


def plan(policy: ParentSpendRef, identity: str, attempts: int) -> SpendEnvelopePlan:
    """Bind a synthetic native inventory and immutable logical requirements to an envelope."""
    return SpendEnvelopePlan(
        policy=policy,
        scope=identity,
        inventory=REF,
        maximum_attempts=attempts,
        maximum_known_usd=0,
        remaining_minimum_attempts=0,
        requirements_sha256="b" * 64,
    )


def test_scoped_requests_do_not_expand_original_preparation_allowance(tmp_path: Path) -> None:
    """The new parent permits council scope while the original unscoped path keeps its cap."""
    ceiling, policy = setup_policy(tmp_path)
    history = ceiling.summary().authorizations
    for _ in range(2):
        ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
    envelope = open_spend_envelope(ceiling, plan(policy, "council", 3))
    scoped = scoped_spend_ceiling(ceiling, envelope)
    assert isinstance(scoped, OperationSpendCeiling) and scoped.path == ceiling.path
    for _ in range(3):
        scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)
    assert sum(row.attempts for row in ceiling.summary().roles) == 5
    assert ceiling.summary().authorizations == history
    assert ceiling.summary().unpriced_attempt_ceiling == 2
    with pytest.raises(UnpricedDispatchRefused):
        ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
    with pytest.raises((ScopeBudgetRefused, UnpricedDispatchRefused)):
        scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)


def test_legacy_calls_cannot_take_outstanding_council_headroom(tmp_path: Path) -> None:
    """Unscoped dispatch honors durable booked capacity, even before scoped calls start."""
    ceiling, policy = setup_policy(tmp_path, legacy=4, parent=5)
    open_spend_envelope(ceiling, plan(policy, "council", 4))
    ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
    with pytest.raises(ScopeBudgetRefused, match="parent"):
        ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)


def test_parallel_envelopes_cannot_overbook_same_parent(tmp_path: Path) -> None:
    """SQLite admission admits one full native wave, rather than racing two stale summaries."""
    ceiling, policy = setup_policy(tmp_path)

    def reserve(identity: str) -> bool:
        """One independent native planner attempts to encumber three physical requests."""
        try:
            open_spend_envelope(ceiling, plan(policy, identity, 3))
            return True
        except ScopeBudgetRefused:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = tuple(pool.map(reserve, ("a", "b")))
    assert sum(outcomes) == 1
    assert not ceiling.summary().roles


def completion(
    root: Path,
    envelope: SpendEnvelopeRef,
    inventory: ArtifactInput,
    *,
    producer: str = "fixture",
) -> tuple[ProjectStore, ArtifactInput]:
    """Save an explicit synthetic domain certificate, never task/runtime QA evidence."""
    project = ProjectStore(root, "scope")
    proof = artifact_input(project.artifacts.read("scope-proof").manifest)
    owner = artifact_input(
        project.artifacts.write_json(
            artifact_id="owner-" + envelope.identity[:20],
            artifact_type="fixture-native-owner",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="fixture",
            ),
            files={"owner.json": {"fixture_complete": True}},
        )
    )
    record = EnvelopeCompletion(
        envelope=envelope,
        inventory=inventory,
        owners=(owner,),
        requirements_sha256="b" * 64,
    )
    certificate = artifact_input(
        project.artifacts.write_json(
            artifact_id="completion-" + producer + "-" + envelope.identity[:20],
            artifact_type="native-spend-envelope-completion",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=producer,
                inputs=tuple(sorted({inventory, proof, owner}, key=lambda ref: ref.artifact_id)),
            ),
            files={"completion.json": record},
        )
    )
    return project, certificate


def test_native_admitted_transport_uses_scoped_view_without_extra_claims(tmp_path: Path) -> None:
    """The unchanged real admitted transport reserves and settles against the exact envelope."""
    ceiling, policy = setup_policy(tmp_path)
    envelope = open_spend_envelope(ceiling, plan(policy, "one", 1))
    scoped = scoped_spend_ceiling(ceiling, envelope)
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})])
    transport, project, dispatch = _transport(tmp_path / "wire", inner, scoped)
    assert transport.spend_ceiling is scoped
    transport.post(URL, headers={}, payload=PAYLOAD, timeout_seconds=10)
    artifacts = set(project.artifacts.list_ids())
    with pytest.raises(ScopeBudgetRefused, match="envelope"):
        transport.post(URL, headers={}, payload=PAYLOAD, timeout_seconds=10)
    assert len(inner.requests) == dispatch.used() == 1
    assert set(project.artifacts.list_ids()) == artifacts
    facts = parent_scope_summary(ceiling)
    assert facts is not None and facts.parent_attempts == 1 and facts.legacy_attempts == 0


def test_floor_survives_settlement_and_unknown_claims_keep_envelope(tmp_path: Path) -> None:
    """Native settlement releases only unused wave headroom; the future study floor remains."""
    ceiling, policy = setup_policy(tmp_path, legacy=4, parent=5)
    proposal = plan(policy, "wave", 3).model_copy(update={"remaining_minimum_attempts": 1})
    envelope = open_spend_envelope(ceiling, proposal)
    scoped = scoped_spend_ceiling(ceiling, envelope)
    claim = scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)
    project, certificate = completion(tmp_path, envelope, proposal.inventory)
    with pytest.raises(ScopeBudgetRefused, match="unsettled"):
        settle_spend_envelope(project, ceiling, envelope, certificate)
    scoped.mark_dispatched(claim, "fixture-attempt")
    scoped.settle(claim, JsonHttpResponse(status_code=200, body={}))
    settle_spend_envelope(project, ceiling, envelope, certificate)
    settle_spend_envelope(project, ceiling, envelope, certificate)
    facts = parent_scope_summary(ceiling)
    assert facts is not None and facts.parent_attempts == 1 and facts.encumbered_attempts == 1
    with pytest.raises(ScopeBudgetRefused, match="settled"):
        scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)

    next_wave = open_spend_envelope(ceiling, plan(policy, "unknown", 3))
    unknown = scoped_spend_ceiling(ceiling, next_wave)
    liability = unknown.reserve(alias="actor", url=URL, payload=PAYLOAD)
    unknown.mark_dispatched(liability, "unknown-attempt")
    unknown.retain_unknown(liability)
    _, unknown_certificate = completion(tmp_path, next_wave, REF)
    with pytest.raises(ScopeBudgetRefused, match="unknown"):
        settle_spend_envelope(project, ceiling, next_wave, unknown_certificate)
    facts = parent_scope_summary(ceiling)
    assert facts is not None and facts.parent_attempts == 2 and facts.encumbered_attempts == 2


def test_scope_rekey_wrong_alias_wrong_producer_and_foreign_ledger_refuse(tmp_path: Path) -> None:
    """Neither inventory edits nor other sources/routes can reuse or replenish a grant."""
    ceiling, policy = setup_policy(tmp_path)
    proposal = plan(policy, "stable-owner", 2)
    envelope = open_spend_envelope(ceiling, proposal)
    assert open_spend_envelope(ceiling, proposal) == envelope
    with pytest.raises(ValueError, match="another immutable"):
        open_spend_envelope(ceiling, proposal.model_copy(update={"maximum_attempts": 1}))
    scoped = scoped_spend_ceiling(ceiling, envelope)
    with pytest.raises(ScopeBudgetRefused, match="alias"):
        scoped.reserve(alias="other-model", url=URL, payload=PAYLOAD)
    project, certificate = completion(tmp_path, envelope, proposal.inventory, producer="unreviewed")
    with pytest.raises(ValueError, match="producer"):
        settle_spend_envelope(project, ceiling, envelope, certificate)
    foreign = OperationSpendCeiling(tmp_path / "foreign.sqlite", tariffs={})
    with pytest.raises(ValueError, match="original operation"):
        scoped_spend_ceiling(foreign, envelope)
    assert not ceiling.summary().roles


def test_preconstructed_native_client_inherits_only_its_own_isolated_child_scope(
    tmp_path: Path,
) -> None:
    """Shared clients retain routes while parallel dispatch owns distinct native leases."""
    ceiling, policy = setup_policy(tmp_path)
    a = open_spend_envelope(ceiling, plan(policy, "a", 1))
    b = open_spend_envelope(ceiling, plan(policy, "b", 1))
    inner = ScriptedJsonTransport([JsonHttpResponse(status_code=200, body={})] * 2)
    transport, _, ledger = _transport(tmp_path / "wire", inner, ceiling)
    barrier = Barrier(2)

    def run(envelope: SpendEnvelopeRef) -> SpendEnvelopeRef | None:
        """Construct the native isolated scope before the existing shared client dispatches."""
        with spend_envelope_scope(ceiling, envelope) as scoped:
            barrier.wait(timeout=5)
            assert current_spend_ceiling() is scoped and transport.spend_ceiling is scoped
            transport.post(URL, headers={}, payload=PAYLOAD, timeout_seconds=10)
            return scoped.envelope

    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        with pytest.raises(ValueError, match="isolated nested"):
            install_spend_ceiling(scoped_spend_ceiling(ceiling, a))
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(copy_context().run, run, envelope) for envelope in (a, b)]
            actual = tuple(future.result() for future in futures)
        assert current_spend_ceiling() is ceiling and transport.spend_ceiling is ceiling
    assert actual == (a, b) and ledger.used() == 2
    facts = parent_scope_summary(ceiling)
    assert facts is not None and facts.legacy_attempts == 0 and facts.parent_attempts == 2


def test_original_unknown_claims_and_grants_are_kept_and_cannot_be_reassociated(
    tmp_path: Path,
) -> None:
    """The registration binds the real retained prefix and never copies or relabels old claims."""
    ceiling, policy = setup_policy(tmp_path, retained=2)
    original_grants = ceiling.summary().authorizations
    with ceiling._transaction() as connection:
        original = connection.execute("SELECT * FROM reservations ORDER BY seq").fetchall()
    envelope = open_spend_envelope(ceiling, plan(policy, "new-study", 3))
    with ceiling._transaction() as connection:
        with pytest.raises(ValueError, match="never be relabeled"):
            associate_reservation(connection, original[0][0], envelope)
        unchanged = connection.execute("SELECT * FROM reservations ORDER BY seq").fetchall()
        assert unchanged == original
        assert connection.execute("SELECT COUNT(*) FROM scoped_claims").fetchone()[0] == 0
    scoped = scoped_spend_ceiling(ceiling, envelope)
    for _ in range(3):
        scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)
    assert ceiling.summary().authorizations == original_grants
    facts = parent_scope_summary(ceiling)
    assert facts is not None and facts.legacy_attempts == 2 and facts.parent_attempts == 5
    with ceiling._transaction() as connection:
        assert (
            connection.execute("SELECT * FROM reservations WHERE seq <= 2 ORDER BY seq").fetchall()
            == original
        )


def test_known_price_bounds_cannot_be_hidden_by_parent_request_allowance(tmp_path: Path) -> None:
    """Known-price ceilings and encumbered dollars remain binding for both native scopes."""
    tariff = AliasTariff(input_usd_per_million=0, output_usd_per_million=1)
    ceiling, policy = setup_policy(
        tmp_path,
        legacy=4,
        parent=5,
        known_usd=0.01,
        tariffs={"actor": tariff},
    )
    proposal = plan(policy, "priced", 2).model_copy(update={"maximum_known_usd": 0.006})
    envelope = open_spend_envelope(ceiling, proposal)
    request: JsonObject = {"messages": [], "max_tokens": 5000}
    with pytest.raises(ScopeBudgetRefused, match="dollars"):
        ceiling.reserve(alias="actor", url=URL, payload=request)
    scoped = scoped_spend_ceiling(ceiling, envelope)
    scoped.reserve(alias="actor", url=URL, payload=request)
    with pytest.raises(ScopeBudgetRefused, match="known-price"):
        scoped.reserve(alias="actor", url=URL, payload=request)
    assert ceiling.summary().accounted_usd == pytest.approx(0.005)
    assert sum(role.attempts for role in ceiling.summary().roles) == 1


def test_mandatory_future_floor_is_physically_reserved_between_waves(tmp_path: Path) -> None:
    """Legacy traffic cannot steal the remaining study minimum after an earlier wave settles."""
    ceiling, policy = setup_policy(tmp_path, legacy=4, parent=5)
    proposal = plan(policy, "first", 1).model_copy(update={"remaining_minimum_attempts": 3})
    envelope = open_spend_envelope(ceiling, proposal)
    scoped = scoped_spend_ceiling(ceiling, envelope)
    claim = scoped.reserve(alias="actor", url=URL, payload=PAYLOAD)
    scoped.mark_dispatched(claim, "first-attempt")
    scoped.settle(claim, JsonHttpResponse(status_code=200, body={}))
    project, certificate = completion(tmp_path, envelope, proposal.inventory)
    settle_spend_envelope(project, ceiling, envelope, certificate)
    ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
    with pytest.raises(ScopeBudgetRefused, match="parent"):
        ceiling.reserve(alias="actor", url=URL, payload=PAYLOAD)
    final = open_spend_envelope(ceiling, plan(policy, "rest", 3))
    assert final != envelope


def test_scoped_context_cannot_redirect_a_client_of_another_original_operation(
    tmp_path: Path,
) -> None:
    """A mismatched operation refuses before either ledger claim, intent or model request."""
    first, _ = setup_policy(tmp_path / "first")
    other, policy = setup_policy(tmp_path / "other")
    envelope = open_spend_envelope(other, plan(policy, "other", 1))
    inner = ScriptedJsonTransport([])
    transport, project, ledger = _transport(tmp_path / "wire", inner, first)
    artifacts = set(project.artifacts.list_ids())
    with spend_envelope_scope(other, envelope):
        with pytest.raises(ValueError, match="different original operation"):
            transport.post(URL, headers={}, payload=PAYLOAD, timeout_seconds=10)
    assert not inner.requests and ledger.used() == 0
    assert set(project.artifacts.list_ids()) == artifacts
    assert not first.summary().roles and not other.summary().roles

"""A policy declares every dimension once and is fixed before the results it judges."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.equivalence.policy import (
    DIMENSIONS,
    AdmissionPolicy,
    Dimension,
    DimensionRule,
    Gate,
    load_admission_policy,
    policy_identity,
    register_admission_policy,
    require_policy_precedes,
)

GATES: dict[Dimension, Gate] = {
    "goal_success": "required",
    "policy_compliance": "when_applicable",
    "forbidden_effects": "when_applicable",
    "state_fidelity": "required",
    "tool_validity": "required",
    "recovery": "when_applicable",
}


def fixture_policy(
    *,
    margin: float = 0.7,
    minimum_units: int = 2,
    families: dict[str, str] | None = None,
    gates: dict[Dimension, Gate] | None = None,
    false_pass_margin: float | None = None,
) -> AdmissionPolicy:
    """A deployment-equivalence policy with loose margins that small fixtures can reach.

    Args:
        margin: Margin of every dimension.
        minimum_units: Fewest independent units of every dimension.
        families: Declared lineage families.
        gates: Gate overrides by dimension.
        false_pass_margin: False-pass margin of every dimension.

    Returns:
        A complete policy over every dimension.
    """
    chosen = {**GATES, **(gates or {})}
    return AdmissionPolicy(
        description="Fixture deployment-equivalence policy; margins are not recommendations.",
        dimensions=tuple(
            DimensionRule(
                dimension=dimension,
                gate=chosen[dimension],
                margin=margin,
                false_pass_margin=false_pass_margin,
                minimum_units=minimum_units,
            )
            for dimension in DIMENSIONS
        ),
        families=families or {},
    )


def registered_policy(
    store: ArtifactStore, policy: AdmissionPolicy | None = None, *, at: datetime | None = None
) -> ArtifactInput:
    """Register a fixture policy, by default now, before any result it judges."""
    return register_admission_policy(
        store,
        policy or fixture_policy(),
        created_at=at or datetime.now(UTC),
        code_revision="test",
    )


def test_every_dimension_is_declared_exactly_once_in_order() -> None:
    """A missing, repeated or reordered dimension refuses before registration."""
    policy = fixture_policy()
    for dimensions in (
        policy.dimensions[:-1],
        (*policy.dimensions[:-1], policy.dimensions[0]),
        tuple(reversed(policy.dimensions)),
    ):
        with pytest.raises(ValueError, match="every dimension once"):
            AdmissionPolicy(description="x", dimensions=dimensions)


def test_state_fidelity_cannot_trust_the_simulator_itself() -> None:
    """Simulated state fidelity is only ever known from calibration."""
    with pytest.raises(ValueError, match="only known from calibration"):
        DimensionRule(dimension="state_fidelity", gate="required", simulated="observed", margin=0.1)
    with pytest.raises(ValueError):
        DimensionRule(dimension="goal_success", gate="required", margin=0)
    with pytest.raises(ValueError):
        DimensionRule(dimension="goal_success", gate="required", margin=0.1, minimum_units=1)


def test_units_merge_declared_family_lineages_and_never_split_them() -> None:
    """Lineages share a declared family; undeclared lineages are their own family."""
    policy = fixture_policy(families={"lineage-a": "orders", "lineage-b": "orders"})
    assert policy.unit("goal_success", "lineage-a") == policy.unit("goal_success", "lineage-b")
    assert policy.unit("goal_success", "lineage-a") == "family:orders"
    assert policy.unit("goal_success", "lineage-c") == "family:lineage-c"
    by_lineage = policy.model_copy(
        update={
            "dimensions": tuple(
                rule.model_copy(update={"unit": "lineage"}) for rule in policy.dimensions
            )
        }
    )
    assert by_lineage.unit("recovery", "lineage-a") == "lineage:lineage-a"
    with pytest.raises(ValueError, match="non-empty"):
        fixture_policy(families={"lineage-a": " "})


def test_registration_is_content_addressed_and_replays_its_original_time(tmp_path: Path) -> None:
    """Re-registering identical content keeps the earliest registration time."""
    store = ProjectStore(tmp_path, "case").artifacts
    early = datetime(2026, 1, 1, tzinfo=UTC)
    first = registered_policy(store, at=early)
    again = registered_policy(store, at=early + timedelta(days=3))
    assert first == again and first.artifact_id == policy_identity(fixture_policy())
    assert store.read(first.artifact_id).manifest.created_at == early
    assert load_admission_policy(store, first) == fixture_policy()
    other = registered_policy(store, fixture_policy(margin=0.2))
    assert other != first


def results_registration(
    store: ArtifactStore, name: str, *, at: datetime, cites: tuple[ArtifactInput, ...] = ()
) -> str:
    """Save a results registration at a claimed time, citing ``cites``, and return its ID."""
    return store.write_json(
        artifact_id=name,
        artifact_type="continuation-study",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=at, code_revision="test", inputs=cites
        ),
        files={"record.json": {}},
    ).artifact_id


def test_a_policy_the_results_do_not_cite_is_refused_even_with_an_earlier_clock(
    tmp_path: Path,
) -> None:
    """Margins chosen after seeing results cannot admit them, whatever the policy clock claims."""
    store = ProjectStore(tmp_path, "case").artifacts
    now = datetime.now(UTC)
    results = results_registration(store, "study-registration", at=now)
    # Registered after the results exist, on a host whose clock runs a day behind.
    skewed = registered_policy(store, fixture_policy(margin=0.3), at=now - timedelta(days=1))
    with pytest.raises(ValueError, match="not cited by study study-registration"):
        require_policy_precedes(store, skewed, results, subject="study")


def test_a_cited_policy_precedes_its_results_even_with_a_later_clock(tmp_path: Path) -> None:
    """A registration citing the saved policy proves the order, so a fast policy clock is fine."""
    store = ProjectStore(tmp_path, "case").artifacts
    now = datetime.now(UTC)
    # Registered first, on a host whose clock runs a day ahead of the results' host.
    policy = registered_policy(store, at=now + timedelta(days=1))
    results = results_registration(store, "study-registration", at=now, cites=(policy,))
    require_policy_precedes(store, policy, results, subject="study")
    other = registered_policy(store, fixture_policy(margin=0.3), at=now - timedelta(days=1))
    with pytest.raises(ValueError, match="register it first"):
        require_policy_precedes(store, other, results, subject="study")


def test_loading_refuses_other_artifacts_and_changed_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the exact saved policy under the current protocol loads."""
    store = ProjectStore(tmp_path, "case").artifacts
    reference = registered_policy(store)
    other = store.write_json(
        artifact_id="not-a-policy",
        artifact_type="continuation-study",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"record.json": {}},
    )
    with pytest.raises(ValueError, match="is not the saved admission policy"):
        load_admission_policy(store, artifact_input(other))
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.policy.ADMISSION_POLICY_PROTOCOL",
        "admission-policy-v0",
    )
    with pytest.raises(ValueError, match="registered under another admission protocol"):
        load_admission_policy(store, reference)

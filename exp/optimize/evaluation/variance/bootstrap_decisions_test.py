"""The actual native preparation launch composes and accounts its required System One client."""

import socket
import sqlite3
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings
from exp.optimize.evaluation.rules.rule_tags import RuleTaggingSettings
from exp.optimize.evaluation.rules.system_one import DecisionBoundary
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import bootstrap_plan as owner
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapDependencies,
    BootstrapPlan,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    BootstrapResumeBinding,
    load_bootstrap_resume,
)
from exp.optimize.evaluation.variance.bootstrap_resume_test import _origin, _register
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.optimize.evaluation.variance.build_test import DecisionWire
from exp.runtime.gateway.decisions_contracts import DecisionRequest, NoulQuestion
from exp.runtime.models.spend_ceiling import current_spend_ceiling
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.environment_review import CompatibilityReviewer


def launch_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision_alias: str = "jev",
    profile: str = "jev_memory",
) -> tuple[ProjectStore, BootstrapPreparationSpec, ArtifactInput]:
    """Freeze real native consent for a preparation with required rule-memory decisions."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("DECISION_TEST_KEY", "fixture-only")
    assert spec.bootstrap.rules is not None
    rules = spec.bootstrap.rules.model_copy(
        update={
            "profile": profile,
            "memory": (
                MemorySetSettings(decision_alias=decision_alias, maximum_topics=11)
                if profile == "jev_memory"
                else None
            ),
            "tagging": (
                RuleTaggingSettings(decision_alias=decision_alias, maximum_rules=1024)
                if profile == "trajectory_tags"
                else None
            ),
        }
    )
    spec = spec.model_copy(update={"bootstrap": spec.bootstrap.model_copy(update={"rules": rules})})
    paths = _origin(project, spec, tmp_path)
    reference = _register(project, paths)
    return project, spec, reference


def no_network(_socket: socket.socket, _address: str | tuple[str, int]) -> None:
    """All smoke dispatches must stay on the injected offline physical wire."""
    raise AssertionError("offline launch smoke must not open a socket")


@pytest.mark.parametrize("profile", ["jev_memory", "trajectory_tags"])
def test_actual_launch_composes_system_one_and_retains_original_claims_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    """An omitted driver collaborator is filled natively and buys through the original grant."""
    project, spec, reference = launch_fixture(tmp_path, monkeypatch, profile=profile)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    historical = ceiling.reserve(alias="unpriced", url="https://fixture.invalid/v1", payload={})
    ceiling.mark_dispatched(historical, "earlier-physical-attempt")
    ceiling.retain_unknown(historical)
    before = ceiling.summary()
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_decisions.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(socket.socket, "connect", no_network)

    class SmokeComplete(Exception):
        """Stop after the native collaborator and physical accounting are proved."""

    def inspect_launch(
        actual: ProjectStore,
        execution: BootstrapPreparationSpec,
        plan: BootstrapPlan,
        collaborators: BootstrapDependencies,
        reviewer: CompatibilityReviewer,
        revision: str,
        bound: BootstrapResumeBinding | None,
    ) -> BootstrapPreparationReport:
        """Exercise the actual composed client after all original-operation admission gates."""
        assert actual is project and execution == spec and bound == binding
        assert revision == "corrected" and plan.identity == binding.original_operation
        active = current_spend_ceiling()
        assert active is not None and active.path == ceiling.path
        client = collaborators.decisions
        assert isinstance(client, DecisionsTraceTagger) and client.cache is not None
        assert client.model_identity["model_id"] == "jev-fixture"
        request = DecisionRequest(
            state="launch-wiring-smoke",
            questions={
                "supported": NoulQuestion(
                    instructions="Does this region support the requested rule?"
                )
            },
        )
        boundary = DecisionBoundary.from_tagger(client)
        response, bought = boundary.buy(request)
        replay, bought_again = boundary.buy(request)
        assert response == replay and bought and not bought_again
        assert wire.states == ["launch-wiring-smoke"]
        assert ceiling.summary().unpriced_attempts == before.unpriced_attempts + 1
        assert ceiling.summary().authorizations == before.authorizations
        path = project.artifacts.project_directory / f"{plan.identity}-rules-decisions.sqlite"
        with sqlite3.connect(path) as db:
            registered_cap = db.execute("SELECT cap FROM registration").fetchone()[0]
            slots = db.execute("SELECT COUNT(*) FROM invocation_slots").fetchone()[0]
        assert registered_cap == sum(row.attempts for row in plan.costs if row.name == "jev")
        assert slots == 1
        raise SmokeComplete

    monkeypatch.setattr(owner, "_run", inspect_launch)
    with pytest.raises(SmokeComplete):
        run_bootstrap_preparation(
            project, spec, code_revision="corrected", resume_binding=reference
        )
    assert ceiling.summary().authorizations == before.authorizations
    assert ceiling.summary().unpriced_attempt_ceiling == before.unpriced_attempt_ceiling


def test_missing_native_system_one_configuration_fails_before_any_paid_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat-only alias cannot reach either the rules author or a decision POST."""
    project, spec, reference = launch_fixture(tmp_path, monkeypatch, decision_alias="world")
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    before = ceiling.summary()
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_decisions.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(socket.socket, "connect", no_network)

    def no_roles(*args: object, **kwargs: object) -> None:
        """A missing native client must fail before entering the first authoring stage."""
        raise AssertionError("roles must not run without a native System One client")

    monkeypatch.setattr(owner, "_run", no_roles)
    with pytest.raises(ValueError, match="native decision client|typed|System One|tag"):
        run_bootstrap_preparation(
            project, spec, code_revision="corrected", resume_binding=reference
        )
    assert not wire.states and ceiling.summary() == before


def test_actual_resume_retains_the_registered_cap_when_pending_build_inventory_shrinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving a real build between launches cannot replace the original decision cap or claims."""
    from dataclasses import replace

    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.variance.bootstrap_selection_test import compatible
    from exp.optimize.evaluation.variance.build import run_variance_build
    from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
    from exp.simulation.mining.environment_families_test import reviewer as scripted_reviewer

    project, spec, reference = launch_fixture(tmp_path, monkeypatch)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    before = ceiling.summary()
    wire = DecisionWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.bootstrap_decisions.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(socket.socket, "connect", no_network)
    _, reviewer = scripted_reviewer(compatible)
    reviewer = replace(reviewer, configuration_sha256=sha256_json(spec.build.reviewer))
    registered: list[int] = []
    pending: list[int] = []

    class Interrupted(Exception):
        """Stop before any bootstrap roles while retaining the real saved build."""

    def checkpoint(
        actual: ProjectStore,
        execution: BootstrapPreparationSpec,
        plan: BootstrapPlan,
        collaborators: BootstrapDependencies,
        actual_reviewer: CompatibilityReviewer,
        revision: str,
        bound: BootstrapResumeBinding | None,
    ) -> BootstrapPreparationReport:
        """Use the actual collaborator, checkpoint real build owners, then resume it."""
        assert actual is project and bound == binding and revision == "corrected"
        client = collaborators.decisions
        assert isinstance(client, DecisionsTraceTagger)
        request = DecisionRequest(
            state="same-original-rules-objective",
            questions={"support": NoulQuestion(instructions="Does this region support the rule?")},
        )
        _, bought = DecisionBoundary.from_tagger(client).buy(request)
        assert bought == (not pending)
        pending.append(sum(row.attempts for row in plan.costs if row.name == "jev"))
        path = project.artifacts.project_directory / f"{plan.identity}-rules-decisions.sqlite"
        with sqlite3.connect(path) as db:
            registered.append(db.execute("SELECT cap FROM registration").fetchone()[0])
            assert db.execute("SELECT COUNT(*) FROM invocation_slots").fetchone()[0] == 1
        if len(pending) == 1:
            run_variance_build(
                project,
                spec.build.spec,
                code_revision="original",
                reviewer=reviewer,
                tagger=_tagger(project, FakeDecisions()),
            )
        raise Interrupted

    monkeypatch.setattr(owner, "_run", checkpoint)
    for _ in range(2):
        with pytest.raises(Interrupted):
            run_bootstrap_preparation(
                project,
                spec,
                code_revision="corrected",
                resume_binding=reference,
                dependencies=BootstrapDependencies(reviewer=reviewer),
            )
    assert pending[1] < pending[0] and registered == [pending[0], pending[0]]
    assert wire.states == ["same-original-rules-objective"]
    assert ceiling.summary().authorizations == before.authorizations
    assert ceiling.summary().unpriced_attempts == before.unpriced_attempts + 1

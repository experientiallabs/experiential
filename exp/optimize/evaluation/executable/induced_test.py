"""The built-in induced runtime: exact programs over the built grounded simulator."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from exp.common.models import BillingSource, ModelSnapshot, ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable.grounded_test import _record_spec
from exp.optimize.evaluation.executable.induced import continuation_runtime
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.induction.service import RESULT_TYPE, InductionResult, InductionSpec
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.induction.contracts import (
    InducedCoverage,
    ProgramDomain,
    ProgramProposal,
    TransitionProgram,
    ValidationReport,
)

REVISION = "c" * 40
PROGRAM = TransitionProgram(
    tool="inspect",
    source="def transition(state, args, facts):\n    return state['records']['status']\n",
)


def _saved_induction(project: ProjectStore, manifest: WorldManifest) -> ArtifactInput:
    """Save an induction whose single accepted program reads the records status."""
    spec = InductionSpec(
        manifest=manifest,
        inducer_alias="fixture",
        inducer_model=ModelSnapshot(
            provider="fixture",
            model_id="inducer",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        catalog_sha256="c" * 64,
        maximum_physical_attempts=1,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    identity = spec.identity(REVISION)
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision=REVISION
    )
    proposal = project.artifacts.write_json(
        artifact_id=identity + "-tool-0-attempt-0-proposal",
        artifact_type="induced-transition-proposal",
        envelope=envelope,
        files={
            "proposal.json": ProgramProposal(
                tool="inspect", attempt=0, operation_id="tool-0-attempt-0", program=PROGRAM
            )
        },
    )
    coverage = InducedCoverage(
        manifest=manifest,
        programs=(PROGRAM,),
        domains={"inspect": ProgramDomain(argument_keys=(("key",),), lines=(2,))},
        induced_tools=("inspect",),
        induced_domains=("records",),
    )
    result = InductionResult(
        spec=spec,
        episodes_sha256="d" * 64,
        programs=(PROGRAM,),
        report=ValidationReport(
            manifest_sha256=sha256_json(manifest),
            episodes=1,
            steps=1,
            rounds=1,
            programs=(),
            coverage=coverage,
        ),
    )
    manifest_record = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=RESULT_TYPE,
        envelope=envelope.model_copy(update={"inputs": (artifact_input(proposal),)}),
        files={"induction.json": result},
    )
    return artifact_input(manifest_record)


def test_induced_calls_answer_from_exact_code_without_a_provider_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The induced tool runs as code; the probe reads the same simulated state."""
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    transport = ScriptedJsonTransport()
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    spec = _record_spec(project, fidelity)
    coverage = _saved_induction(project, spec.manifest)
    runtime = continuation_runtime(
        project,
        {"grounded": spec.model_dump(mode="json"), "coverage": coverage.model_dump(mode="json")},
    )
    with runtime.world.open(spec.task) as world:
        reply = world.execute(ToolCall(name="inspect", call_id="read", arguments={"key": "now"}))
        assert (reply.content, reply.metadata["transition"]) == ("pending", "induced")
        assert runtime.probe(world.checkpoint()) == {"status": "pending"}
    assert transport.requests == []


def test_coverage_under_another_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Programs validated for other domains never serve this world."""
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    spec = _record_spec(project, fidelity)
    other = WorldManifest(
        domains=(StateDomain(name="records", owner="simulated"),),
        tools=(ToolEffectPolicy(name="inspect", reads=("records",), duration=2),),
    )
    coverage = _saved_induction(project, other)
    with pytest.raises(ValueError, match="another world manifest"):
        continuation_runtime(
            project,
            {
                "grounded": spec.model_dump(mode="json"),
                "coverage": coverage.model_dump(mode="json"),
            },
        )

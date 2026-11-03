"""Durable induction over a real offline build: saved draws, replay and re-validation."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.common.core.verified_cache import VerifiedCache
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.common.project.causality import cites
from exp.common.tasks import ToolSchema
from exp.common.traces import Trace
from exp.common.traces import store as trace_store
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.audit.panel_test import SimulatedProvider, _tool_reply, build_project
from exp.optimize.evaluation.induction import episodes as induction_episodes
from exp.optimize.evaluation.induction import service
from exp.optimize.evaluation.induction.service import (
    InductionSpec,
    TransitionValidation,
    TransitionValidationSpec,
    induce_transitions,
    load_coverage,
    load_induction,
    preflight_induction,
    validate_transitions,
)
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.induction.proposals import InductionSettings
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings

REVISION = "c" * 40
MANIFEST = WorldManifest(
    domains=(StateDomain(name="orders", owner="simulated"),),
    tools=(
        ToolEffectPolicy(name="find_user_id_by_email"),
        ToolEffectPolicy(name="get_user_details"),
        ToolEffectPolicy(name="get_order_details", reads=("orders",)),
        ToolEffectPolicy(name="cancel_pending_order", reads=("orders",), writes=("orders",)),
    ),
)
PROGRAMS = {
    "cancel_pending_order": "def transition(state, args, facts):\n    raise OutOfDomain('x')\n",
    "find_user_id_by_email": (
        "def transition(state, args, facts):\n    return {'user_id': args['email'].split('@')[0]}\n"
    ),
    "get_order_details": "def transition(state, args, facts):\n    raise OutOfDomain('x')\n",
    "get_user_details": (
        "def transition(state, args, facts):\n"
        "    number = int(args['user_id'][5:])\n"
        "    return {'user_id': args['user_id'], 'orders': ['#W' + str(1000 + number)],\n"
        "            'payment': 'gift_card'}\n"
    ),
}


def induction_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, InductionSpec, ScriptedJsonTransport]:
    """Build the offline retail project and script one program per inducible tool."""
    _, project, _ = build_project(tmp_path, monkeypatch, SimulatedProvider([]))
    catalog = load_model_catalog(project.model_catalog_path)
    snapshot, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("world")
    spec = InductionSpec(
        manifest=MANIFEST,
        inducer_alias="world",
        inducer_model=snapshot,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        settings=InductionSettings(maximum_attempts=1, minimum_support=2),
        maximum_physical_attempts=8,
        deadline=datetime.now(UTC) + timedelta(hours=1),
    )
    transport = ScriptedJsonTransport(
        tuple(
            _tool_reply("world-v1", "submit_program", {"source": PROGRAMS[tool]})
            for tool in sorted(PROGRAMS)
        )
    )
    monkeypatch.setattr(service, "HttpxJsonTransport", lambda: transport)
    return project, spec, transport


def test_induction_saves_every_draw_and_replays_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure tools are owned by induced code; tools on unknown order state stay predicted."""
    project, spec, transport = induction_fixture(tmp_path, monkeypatch)
    episodes = preflight_induction(project, spec)
    assert episodes
    reference = induce_transitions(project, spec, code_revision=REVISION)
    assert len(transport.requests) == len(PROGRAMS)
    result = load_induction(project.artifacts, reference)
    coverage = result.report.coverage
    assert coverage.induced_tools == ("find_user_id_by_email", "get_user_details")
    verdicts = {row.tool: row for row in result.report.programs}
    assert verdicts["get_order_details"].status == "rejected"
    assert verdicts["get_order_details"].unverifiable == len(episodes)
    proposals = [
        row.artifact_id
        for row in project.artifacts.read(reference.artifact_id).manifest.inputs
        if row.artifact_id.endswith("-proposal")
    ]
    assert len(proposals) == len(PROGRAMS)
    build = project.load_project().build
    assert build is not None
    for evidence in (build.trace_dataset, build.task_set):
        assert cites(project.artifacts, reference.artifact_id, evidence), (
            "the induction cites the fit evidence its programs were validated on"
        )
    request = json.loads(
        project.artifacts.read_bytes(
            f"{reference.artifact_id}-tool-1-attempt-0-intent", "request.json"
        )
    )
    context = json.loads(request["messages"][1]["content"])
    assert context["tool"] == "find_user_id_by_email"
    assert {row["episode"] for row in context["episodes"]} <= {row.episode_id for row in episodes}
    again = induce_transitions(project, spec, code_revision=REVISION)
    assert again == reference and len(transport.requests) == len(PROGRAMS)
    assert load_coverage(project.artifacts, reference) == coverage


def test_fit_episodes_exclude_held_out_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every induction episode is a fit trace of the frozen build."""
    project, spec, _ = induction_fixture(tmp_path, monkeypatch)
    episodes = preflight_induction(project, spec)
    build = project.load_project().build
    assert build is not None
    partitions = {
        row.trace_id: row.partition
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    assert {partitions[row.episode_id] for row in episodes} == {"fit"}
    assert "held_out" in partitions.values()


def test_tool_schemas_come_only_from_fit_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A schema a held-out trace declares first never reaches the inducer.

    The adversarial declarations are injected where every trace is decoded from the saved
    dataset, so any reader of the build's traces sees them: held-out traces declare a marker,
    fit traces before the first held-out trace declare nothing, and later fit traces declare
    the expected description.
    """
    project, spec, _ = induction_fixture(tmp_path, monkeypatch)
    build = project.load_project().build
    assert build is not None
    bindings = load_completed_build_rag_lineage_bindings(project.artifacts, build)
    held_out = {row.trace_id for row in bindings if row.partition == "held_out"}
    order = [
        row.trace_id
        for row in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    ]
    first_held_out = min(order.index(trace_id) for trace_id in held_out)
    assert any(trace_id not in held_out for trace_id in order[first_held_out:])
    decode = trace_store._decode_trace_record

    def adversarial(line: str, *, legacy: bool) -> Trace:
        """Decode one saved trace, then give it the declarations this test needs."""
        trace = decode(line, legacy=legacy)
        if trace.trace_id in held_out:
            description = "HELD-OUT-MARKER"
        elif order.index(trace.trace_id) > first_held_out:
            description = "fit"
        else:
            return trace.model_copy(update={"tools": ()})
        tools = tuple(
            ToolSchema(name=tool.name, description=description, input_schema={})
            for tool in spec.manifest.tools
        )
        return trace.model_copy(update={"tools": tools})

    monkeypatch.setattr(trace_store, "_decode_trace_record", adversarial)
    monkeypatch.setattr(trace_store, "_DECODED", VerifiedCache(4))
    # The binding loader re-checks that saved records are canonical, which the injected
    # declarations break; the bindings themselves are the real ones computed above.
    monkeypatch.setattr(
        induction_episodes, "load_completed_build_rag_lineage_bindings", lambda *_: bindings
    )
    schemas = service._tool_schemas(project, spec)
    assert [schema.name for schema in schemas] == sorted(tool.name for tool in spec.manifest.tools)
    assert {schema.description for schema in schemas} == {"fit"}


def test_a_draw_with_an_unknown_outcome_is_never_redrawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider failure after the intent leaves the induction stopped, not retried."""
    project, spec, _ = induction_fixture(tmp_path, monkeypatch)
    failing = ScriptedJsonTransport((TimeoutError("lost"),) * 4)
    monkeypatch.setattr(service, "HttpxJsonTransport", lambda: failing)
    with pytest.raises(Exception):  # noqa: B017 - transport wraps the lost response
        induce_transitions(project, spec, code_revision=REVISION)
    with pytest.raises(ValueError, match="unknown outcome"):
        induce_transitions(project, spec, code_revision=REVISION)


def test_revalidation_is_deterministic_and_reports_unchanged_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved programs replay to the same verdicts and coverage without any provider."""
    project, spec, transport = induction_fixture(tmp_path, monkeypatch)
    reference = induce_transitions(project, spec, code_revision=REVISION)
    calls = len(transport.requests)
    validated = validate_transitions(
        project, TransitionValidationSpec(induction=reference), code_revision=REVISION
    )
    saved = TransitionValidation.model_validate_json(
        project.artifacts.read_bytes(validated.artifact_id, "validation.json")
    )
    original = load_induction(project.artifacts, reference).report
    assert saved.unchanged_evidence and saved.report == original
    assert load_coverage(project.artifacts, validated) == original.coverage
    stricter = validate_transitions(
        project,
        TransitionValidationSpec(induction=reference, minimum_support=10_000),
        code_revision=REVISION,
    )
    strict = TransitionValidation.model_validate_json(
        project.artifacts.read_bytes(stricter.artifact_id, "validation.json")
    )
    assert strict.report.coverage.induced_tools == ()
    assert len(transport.requests) == calls


def test_preflight_refuses_a_manifest_no_fit_trace_uses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest naming no recorded tool fails before any credential or call."""
    project, spec, transport = induction_fixture(tmp_path, monkeypatch)
    other = spec.model_copy(
        update={
            "manifest": WorldManifest(
                domains=(StateDomain(name="x", owner="simulated"),),
                tools=(ToolEffectPolicy(name="never_called", reads=("x",)),),
            )
        }
    )
    with pytest.raises(ValueError, match="no fit trace"):
        preflight_induction(project, other)
    assert transport.requests == []

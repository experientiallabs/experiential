"""The configured mixed factory connects trace-built prediction to actual native workbook state."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_bytes, sha256_json
from exp.common.models import ToolCall, load_model_catalog
from exp.common.project import ProjectStore
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.continuations.capture import CaptureDefinition, capture_initial_world
from exp.optimize.evaluation.continuations.study import restore_continuation_world
from exp.optimize.evaluation.executable import grounded as grounded_runtime
from exp.optimize.evaluation.executable import mixed as mixed_runtime
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.mixed import (
    MixedRuntimeDefinition,
    NativeRuntimeBinding,
    continuation_runtime,
)
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.runtime.environments.mixed import MixedCheckpoint, ToolRoute
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.world_artifacts import persist_world_image
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.environments.world_grounding import NativeReadBridge
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.continuations import load_continuation_checkpoint


def mixed_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, MixedRuntimeDefinition, ArtifactInput, ScriptedJsonTransport]:
    """Configure existing native/frozen-simulator adapters without a custom mixed router."""
    project, fidelity = source_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    native_case = workbook_case()
    with native_case.runtime.open(native_case.task) as native:
        image = persist_world_image(
            project.artifacts,
            native.checkpoint(),
            created_at=datetime.now(UTC),
            code_revision="test",
        )
    task = fidelity.task.model_copy(
        update={
            "origin": "authored",
            "partition": "fit",
            "source_trace_ids": (),
            "tools": (
                *fidelity.task.tools,
                ToolSchema(
                    name="edit_workbook",
                    description="Native row edit",
                    input_schema={"type": "object"},
                ),
            ),
        }
    )
    manifest = WorldManifest(
        domains=(
            StateDomain(name="input.xlsx", owner="native"),
            StateDomain(name="output.xlsx", owner="native"),
            StateDomain(name="records", owner="simulated"),
        ),
        tools=(
            native_case.runtime.manifest.tool("edit_workbook"),
            ToolEffectPolicy(name="inspect", reads=("records",)),
            ToolEffectPolicy(name="inspect_workbook", reads=("output.xlsx",), duration=0),
        ),
        native_reads=(
            NativeReadBridge(
                name="current-workbook",
                target_tool="inspect",
                probe_tool="inspect_workbook",
                arguments={"sheet": "Orders", "row_count": 2},
                argument_keys={"source": "key"},
            ),
        ),
    )
    probe_sha = sha256_json({"native": "row-count-from-real-workbook"})
    native_file = tmp_path / "reviewed_native.py"
    native_file.write_text("""from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.continuations.study_test import probe
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.workbook_edits import WorkbookEditAdapter
from exp.runtime.environments.world_artifacts import load_world_image
from exp.runtime.environments.world_contracts import WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime

def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["image"])
    image = load_world_image(project.artifacts, reference)
    factory = ArtifactWorldFactory(
        WorkbookEditAdapter(), files=image.image.parts, outputs=("output.xlsx",)
    )
    world = ExecutableWorldRuntime(WorldManifest.model_validate(configuration["manifest"]), factory)
    return ContinuationRuntime(world, probe, configuration["probe"])
""")
    spec = MixedRuntimeDefinition(
        native=NativeRuntimeBinding(
            runtime_file=native_file,
            runtime_sha256=sha256_bytes(native_file.read_bytes()),
            configuration={
                "image": image.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json"),
                "probe": probe_sha,
            },
            probe_sha256=probe_sha,
        ),
        simulated=GroundedRuntimeDefinition(
            task=task,
            world_model=fidelity.world_model,
            catalog_sha256=sha256_json(
                load_model_catalog(project.model_catalog_path).model_dump(mode="json")
            ),
            manifest=manifest,
            initial_state={"records": {"status": "pending"}},
            probe_paths={"status": "/records/status"},
            maximum_physical_attempts=1,
            deadline=datetime.now(UTC) + timedelta(minutes=5),
        ),
        routes=(
            ToolRoute(name="edit_workbook", mode="real", effect="write"),
            ToolRoute(name="inspect", mode="simulated", effect="read"),
        ),
    )
    prediction = {
        "observations": [{"call_id": "read", "reply": "predicted reply", "encoding": "text"}],
        "effect": {"kind": "edits", "edits": []},
    }
    transport = ScriptedJsonTransport(
        (
            JsonHttpResponse(
                status_code=200,
                body={
                    "id": "pred",
                    "model": "fixture-v1",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "proposal",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_transition",
                                            "arguments": json.dumps(prediction),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            ),
        )
    )
    monkeypatch.setattr(grounded_runtime, "HttpxJsonTransport", lambda: transport)
    return project, spec, image, transport


def test_configured_factory_captures_restores_grounds_and_records_actual_native_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing capture plus the builtin compositor binds both state owners with no actor calls."""
    project, spec, image, transport = mixed_fixture(tmp_path, monkeypatch)
    runtime_file = Path(mixed_runtime.__file__)
    capture = CaptureDefinition(
        task=spec.simulated.task,
        sources=(spec.simulated.world_model, image),
        runtime_file=runtime_file,
        runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
        configuration=spec.model_dump(mode="json"),
        probe_sha256=spec.probe_sha256,
    )
    reference = capture_initial_world(project, capture, code_revision="test")
    assert not transport.requests
    point = load_continuation_checkpoint(project.artifacts, reference)
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with restore_continuation_world(runtime.world, point.world_task, point.world) as world:
        world.execute(workbook_case().first)
        result = world.execute(
            ToolCall(name="inspect", call_id="read", arguments={"key": "output.xlsx"})
        )
        saved = world.checkpoint()
        assert result.metadata["authority"] == "simulated" and result.content == "predicted reply"
        projection = runtime.probe(saved)
        native_projection = projection["native"]
        assert isinstance(native_projection, dict) and native_projection["rows"] == 6
        assert projection["simulated"] == {"status": "pending"}
    events = project.artifacts.list_ids(artifact_type="mixed-executable-event")
    grounding = [
        json.loads(project.artifacts.read_bytes(key, "event.json"))
        for key in events
        if key.endswith("-grounding")
    ]
    assert len(grounding) == 1
    read = grounding[0]["reads"][0]
    assert json.loads(read["observation"]["content"])["projection"]["output.xlsx"]["max_row"] == 6
    with restore_continuation_world(runtime.world, point.world_task, saved) as restored:
        assert runtime.probe(restored.checkpoint()) == projection
    spec.native.runtime_file.unlink()
    assert capture_initial_world(project, capture, code_revision="test") == reference
    assert len(transport.requests) == 1


def test_nested_runtime_source_drift_refuses_before_native_import_or_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The outer compositor's digest cannot hide a changed nested native implementation."""
    project, spec, _, transport = mixed_fixture(tmp_path, monkeypatch)
    spec.native.runtime_file.write_text("raise AssertionError('must not import')")
    with pytest.raises(ValueError, match="source changed"):
        continuation_runtime(project, spec.model_dump(mode="json"))
    assert not transport.requests


def test_refused_native_read_reaches_actor_without_poisoning_or_predicting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read beyond current rows is a recoverable tool error, not an unknown dispatch."""
    project, spec, _, transport = mixed_fixture(tmp_path, monkeypatch)
    manifest = spec.simulated.manifest
    bridge = manifest.native_reads[0]
    bridge = bridge.model_copy(update={"arguments": {"sheet": "Orders", "row_count": 128}})
    manifest = manifest.model_copy(update={"native_reads": (bridge,)})
    spec = spec.model_copy(
        update={
            "native": spec.native.model_copy(
                update={
                    "configuration": {
                        **spec.native.configuration,
                        "manifest": manifest.model_dump(mode="json"),
                    }
                }
            ),
            "simulated": spec.simulated.model_copy(update={"manifest": manifest}),
        }
    )
    runtime = continuation_runtime(project, spec.model_dump(mode="json"))
    with runtime.world.open(spec.simulated.task) as session:
        session.execute(workbook_case().first)
        before = session.checkpoint()
        refused = session.execute(
            ToolCall(call_id="bounds", name="inspect", arguments={"key": "output.xlsx"})
        )
        assert refused.is_error and "beyond existing cells" in refused.content
        assert refused.metadata["authority"] == "native"
        assert refused.metadata["grounding_refused"] is True
        after = session.checkpoint()
        assert isinstance(before, MixedCheckpoint) and isinstance(after, MixedCheckpoint)
        assert after.native == before.native and after.candidate == before.candidate
        assert not transport.requests, "unavailable facts must never be handed to the predictor"
        # A later valid native call still executes in the same episode.
        session.execute(workbook_case().first.model_copy(update={"call_id": "again"}))
        final = session.checkpoint()
        assert isinstance(final, MixedCheckpoint) and final.sequence == after.sequence + 1


def test_declared_workbook_facts_reach_the_simulated_side_beside_native_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only declared facts and receipts enter the request; native state is never derived."""
    from exp.optimize.evaluation.executable.grounded_test import (
        assert_wire_facts,
        wire_messages,
        with_facts,
    )
    from exp.simulation.world_model.executable_test import workbook_facts

    project, spec, image, transport = mixed_fixture(tmp_path, monkeypatch)
    facts = workbook_facts()
    configuration = {
        **spec.model_dump(mode="json"),
        "simulated": with_facts(spec.simulated, facts),
    }
    bound = MixedRuntimeDefinition.model_validate(configuration)
    assert bound.simulated.facts is not None and spec.simulated.facts is None
    assert "facts" not in json.dumps(spec.model_dump(mode="json")["simulated"])
    runtime_file = Path(mixed_runtime.__file__)
    captures = []
    for definition in (spec.model_dump(mode="json"), configuration):
        capture = CaptureDefinition(
            task=spec.simulated.task,
            sources=(spec.simulated.world_model, image),
            runtime_file=runtime_file,
            runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
            configuration=definition,
            probe_sha256=spec.probe_sha256,
        )
        captures.append(capture_initial_world(project, capture, code_revision="test"))
    assert captures[0] != captures[1] and not transport.requests
    point = load_continuation_checkpoint(project.artifacts, captures[1])
    runtime = continuation_runtime(project, configuration)
    with restore_continuation_world(runtime.world, point.world_task, point.world) as world:
        world.execute(workbook_case().first)
        result = world.execute(
            ToolCall(name="inspect", call_id="read", arguments={"key": "output.xlsx"})
        )
        saved = world.checkpoint()
        assert result.content == "predicted reply"
        assert runtime.probe(saved)["simulated"] == {"status": "pending"}
    assert len(transport.requests) == 1
    expected = facts.conditioning(0)
    assert expected is not None
    payload = transport.requests[0].payload
    assert_wire_facts(payload, expected)
    channel = json.loads(wire_messages(payload)[1])
    rows = channel["instance_facts"]["facts"]
    assert len(rows) == len(facts.facts) + 1 and rows[-1]["key"].startswith("native-read/")
    assert set(channel["environment_state"]["domains"]) == {"records"}


def _released_native(spec: MixedRuntimeDefinition, marker: Path) -> MixedRuntimeDefinition:
    """Rewrite the reviewed native factory so closing it leaves a marker file."""
    source = spec.native.runtime_file.read_text().replace(
        'return ContinuationRuntime(world, probe, configuration["probe"])',
        'return ContinuationRuntime(world, probe, configuration["probe"], '
        'lambda: Path(configuration["marker"]).touch())',
    )
    spec.native.runtime_file.write_text("from pathlib import Path\n" + source)
    native = spec.native.model_copy(
        update={
            "runtime_sha256": sha256_bytes(spec.native.runtime_file.read_bytes()),
            "configuration": {**spec.native.configuration, "marker": str(marker)},
        }
    )
    return spec.model_copy(update={"native": native})


def test_closing_the_composition_closes_its_native_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mixed runtime owns both parts; leaving its block releases the native one."""
    project, spec, _, _ = mixed_fixture(tmp_path, monkeypatch)
    marker = tmp_path / "native-closed"
    spec = _released_native(spec, marker)
    with continuation_runtime(project, spec.model_dump(mode="json")) as runtime:
        assert not marker.exists()
        assert isinstance(runtime.world, mixed_runtime.MixedRealityEnvironmentRuntime)
    assert marker.exists()


def test_a_refused_composition_closes_the_native_runtime_it_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest refusal after the native factory ran still releases what it opened."""
    project, spec, _, transport = mixed_fixture(tmp_path, monkeypatch)
    marker = tmp_path / "native-closed"
    spec = _released_native(spec, marker)
    other = spec.simulated.manifest.model_copy(
        update={
            "domains": (*spec.simulated.manifest.domains, StateDomain(name="extra", owner="native"))
        }
    )
    native = spec.native.model_copy(
        update={
            "configuration": {
                **spec.native.configuration,
                "manifest": other.model_dump(mode="json"),
            }
        }
    )
    with pytest.raises(ValueError, match="manifest differs"):
        continuation_runtime(
            project, spec.model_copy(update={"native": native}).model_dump(mode="json")
        )
    assert marker.exists()
    assert not transport.requests

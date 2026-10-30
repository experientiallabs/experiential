"""Recorded workbook bytes need exact typed provenance and independently replayed outcomes."""

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import Workbook

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    SourceIdentity,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import ToolCall
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.workbook_edits import (
    WorkbookEditAdapter,
    WorkbookEditArguments,
    read_supported_workbook,
    workbook_projection,
)
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
    persist_specification,
    record_label_witness,
    verify_label_witness,
)
from exp.simulation.evaluation.admission_test import saved, specification, witnessed
from exp.simulation.evaluation.record_execution import (
    RecordReplayRuntime,
    execute_record_script,
    load_record_execution,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier, VerifierChallenge

NOW = datetime(2026, 10, 23, tzinfo=UTC)


def workbook_bytes(*, reference: bool = False) -> bytes:
    """Create finite fixture documents independently of the native edit adapter."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["Header"])
    sheet.append(["Keep"])
    if not reference:
        sheet.append(["Remove"])
    stream = BytesIO()
    workbook.save(stream)
    workbook.close()
    return stream.getvalue()


def raw_source(
    project: ProjectStore,
    *,
    variant: str = "valid",
    parents: tuple[ArtifactInput, ...] = (),
    reference: bool = False,
) -> ArtifactInput:
    """Publish fixture file provenance without disguising malformed or generated sources."""
    raw = workbook_bytes(reference=reference)
    member = "spreadsheet/fixture/golden.xlsx" if reference else "spreadsheet/fixture/input.xlsx"
    digest = sha256_bytes(raw)
    provenance: JsonObject = {
        "archive_sha256": "a" * 64,
        "archive_member": member,
        "raw_sha256": digest,
        "role": "reference" if reference else "initial",
        "origin": "Locally authored XLSX test fixture, not a model output",
        "exposure": "Development fixture",
        "authority": "Recorded bytes only; archive hash is a declaration, not authentication",
    }
    source = SourceIdentity(kind="file", source_id=f"fixture.tar.gz!{member}", sha256=digest)
    version = 1
    files = {"workbook.xlsx": raw}
    if variant == "raw_hash":
        provenance["raw_sha256"] = "b" * 64
    elif variant == "source_hash":
        source = source.model_copy(update={"sha256": "b" * 64})
    elif variant == "member_mismatch":
        source = source.model_copy(update={"source_id": "fixture.tar.gz!other.xlsx"})
    elif variant == "source_kind":
        source = source.model_copy(update={"kind": "generated"})
    elif variant == "unsafe_member":
        provenance["archive_member"] = "../input.xlsx"
    elif variant == "windows_member":
        provenance["archive_member"] = "C:/input.xlsx"
    elif variant == "null_member":
        provenance["archive_member"] = "input\x00.xlsx"
    elif variant == "wrong_extension":
        provenance["archive_member"] = "input.csv"
    elif variant == "unknown_role":
        provenance["role"] = "reviewed_label"
    elif variant == "missing_metadata":
        del provenance["archive_sha256"]
    elif variant == "unknown_metadata":
        provenance["human_reviewed"] = True
    elif variant == "envelope_version":
        version = 2
    if variant != "missing_provenance":
        files["provenance.json"] = (
            b"[]" if variant == "malformed" else json.dumps(provenance).encode()
        )
    if variant == "attached_bytes":
        files["workbook.xlsx"] = b"changed bytes"
    if variant == "extra_file":
        files["unclassified.json"] = b"{}"
    return artifact_input(
        project.artifacts.write(
            artifact_id=f"raw-{variant}-{'reference' if reference else 'initial'}",
            artifact_type="unknown-workbook-kind"
            if variant == "unknown_kind"
            else "external-workbook-source",
            envelope=ArtifactEnvelope(
                schema_version=version,
                created_at=NOW,
                code_revision="fixture",
                source=None if variant == "missing_source" else source,
                inputs=parents,
            ),
            files=files,
        )
    )


@pytest.mark.parametrize(
    "variant",
    [
        "raw_hash",
        "source_hash",
        "member_mismatch",
        "source_kind",
        "unsafe_member",
        "wrong_extension",
        "windows_member",
        "null_member",
        "unknown_role",
        "missing_metadata",
        "unknown_metadata",
        "envelope_version",
        "missing_provenance",
        "malformed",
        "attached_bytes",
        "extra_file",
        "unknown_kind",
        "missing_source",
    ],
)
def test_workbook_type_name_does_not_admit_invalid_source(tmp_path: Path, variant: str) -> None:
    """Every malformed file declaration remains unclassified even with a plausible type name."""
    project = ProjectStore(tmp_path, "admission")
    source = raw_source(project, variant=variant)
    challenge = witnessed(project, specification(project, sources=(source,)), "done")
    refusal = verify_label_witness(project.artifacts, challenge, label="pass", compared=())
    assert refusal is not None and refusal.reason == "self_approval"


def test_valid_workbook_cannot_launder_generated_ancestor(tmp_path: Path) -> None:
    """A valid file envelope still walks the entire original provenance closure."""
    project = ProjectStore(tmp_path, "admission")
    generated = saved(project, "generated-parent", "task-proposal")
    source = raw_source(project, parents=(generated,))
    challenge = witnessed(project, specification(project, sources=(source,)), "done")
    refusal = verify_label_witness(project.artifacts, challenge, label="pass", compared=())
    assert refusal is not None and refusal.reason == "self_approval"
    assert "generated-parent" in refusal.detail


def test_recorded_workbooks_admit_only_replayed_native_witness(tmp_path: Path) -> None:
    """Actual native XLSX edits still need the independent oracle and explicit replay runtime."""
    project = ProjectStore(tmp_path, "admission")
    initial = raw_source(project)
    reference = raw_source(project, reference=True)
    raw = project.artifacts.read_bytes(initial.artifact_id, "workbook.xlsx")
    expected = workbook_projection(
        read_supported_workbook(
            project.artifacts.read_bytes(reference.artifact_id, "workbook.xlsx")
        )
    )
    world = ExecutableWorldRuntime(
        WorldManifest(
            domains=(
                StateDomain(name="input.xlsx", owner="native"),
                StateDomain(name="output.xlsx", owner="native"),
            ),
            tools=(
                ToolEffectPolicy(
                    name="edit_workbook",
                    reads=("input.xlsx", "output.xlsx"),
                    writes=("output.xlsx",),
                ),
            ),
        ),
        ArtifactWorldFactory(
            WorkbookEditAdapter(), files={"input.xlsx": raw}, outputs=("output.xlsx",)
        ),
    )
    task = TaskCase(
        task_id="fixture",
        lineage_group_id="fixture",
        origin="authored",
        partition="fit",
        workload_weight=1,
        instruction="Remove row three and preserve the header.",
        tools=(
            ToolSchema(
                name="edit_workbook",
                description="Delete rows",
                input_schema=WorkbookEditArguments.model_json_schema(),
            ),
        ),
    )

    def probe(checkpoint: ExecutableCheckpoint) -> JsonObject:
        """Read actual checkpoint bytes through the native workbook decoder."""
        payload = checkpoint.image.parts.get("output.xlsx")
        return {} if payload is None else workbook_projection(read_supported_workbook(payload))

    replay = RecordReplayRuntime(world, probe, sha256_json({"probe": "fixture-xlsx-v1"}))
    with world.open(task) as opened:
        baseline = opened.checkpoint()
    receipt = execute_record_script(
        project.artifacts,
        replay,
        task=task,
        baseline=baseline,
        actions=(
            ToolCall(
                call_id="delete",
                name="edit_workbook",
                arguments={
                    "source": "input.xlsx",
                    "destination": "output.xlsx",
                    "deletions": [{"sheet": "Sheet", "start": 3, "count": 1}],
                },
            ),
        ),
        finish_tick=1,
        sources=(initial,),
        created_at=NOW,
        code_revision="fixture",
    )
    record = load_record_execution(project.artifacts, receipt)
    predicate = TaskVerifier(
        description="External fixture reference projection",
        outcome_schema={"type": "object"},
        required_paths=("/final_state/sheets",),
        rules=(
            OutcomeRule(path="/final_state/sheets", operator="equals", value=expected["sheets"]),
        ),
    )
    spec = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Decode the separate reference workbook bytes",
            subject="native_observation",
            predicate=predicate,
        ),
        sources=(initial, reference),
        created_at=NOW,
        code_revision="fixture",
    )
    witness = record_label_witness(
        project.artifacts,
        specification=spec,
        outcome=record.outcome,
        receipt=receipt,
        record_runtime=replay,
        created_at=NOW,
        code_revision="fixture",
    )
    challenge = VerifierChallenge(
        id="native",
        outcome=record.outcome,
        expected="pass",
        authority="native_execution",
        witness=witness,
        rationale="Actual workbook matches the separately decoded reference.",
    )
    assert (
        verify_label_witness(
            project.artifacts, challenge, label="pass", compared=(), record_runtime=replay
        )
        is None
    )
    assert (
        verify_label_witness(
            project.artifacts, challenge, label="pass", compared=(predicate,), record_runtime=replay
        )
        is not None
    )
    assert verify_label_witness(project.artifacts, challenge, label="pass", compared=()) is not None

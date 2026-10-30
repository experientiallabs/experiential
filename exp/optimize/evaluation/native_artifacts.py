"""Materialize simulator-declared edits from audited checkpoints without native rescue."""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    canonical_json_bytes,
    sha256_json,
    stable_id,
    strict_equal,
)
from exp.common.core.locks import CoordinationLock
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.environments.artifact_edits import (
    MAX_ARTIFACT_BYTES,
    ArtifactEditAdapter,
    NativeArtifactBinding,
    NativeArtifactPlan,
)
from exp.simulation.world_model.checkpoints import load_world_checkpoint


class NativeMaterializationSpec(ContractModel):
    """Explicit native handoff request over already frozen evidence.

    Attributes:
        checkpoint: Verified committed candidate checkpoint containing the proposal.
        binding: Contract that must match the checkpoint's original public context.
        destination: Optional new directory receiving matched output artifacts.
    """

    checkpoint: ArtifactInput
    binding: NativeArtifactBinding
    destination: Path | None = None


def materialize_native_artifacts(
    artifacts: ArtifactStore,
    spec: NativeMaterializationSpec,
    adapter: ArtifactEditAdapter,
    *,
    code_revision: str,
    dry_run: bool = False,
) -> ArtifactInput | None:
    """Validate, transform, compare and atomically publish one native artifact receipt.

    A completed receipt replays without executing the adapter. Failed pure transformations
    publish nothing and may be retried against the same immutable inputs. Candidate state
    is never replaced by native observations. Divergent results remain diagnostic evidence
    and cannot be exported as a matched submission.
    """
    checkpoint = load_world_checkpoint(artifacts, spec.checkpoint)
    binding = spec.binding
    if not strict_equal(
        checkpoint.public_context.get("native_artifact_binding"), binding.model_dump(mode="json")
    ):
        raise ValueError("native binding was not frozen in the source task context; rebuild it")
    if checkpoint.prefix_position == 0:
        raise ValueError("native edits require a committed simulator transition")
    plan = NativeArtifactPlan.model_validate(
        checkpoint.current.business_state.get(binding.plan_key)
    )
    if plan.input_sha256 != binding.input_artifact.sha256:
        raise ValueError("native plan names a different input artifact; refuse execution")
    if sha256_json(adapter.definition) != binding.adapter_sha256:
        raise ValueError("native adapter implementation or dependencies changed; rebuild the task")
    manifest = artifacts.read(binding.input_artifact.artifact_id).manifest
    if artifact_input(manifest) != binding.input_artifact:
        raise ValueError("native input manifest differs from the frozen binding")
    if sum(row.size_bytes for row in manifest.files) > MAX_ARTIFACT_BYTES:
        raise ValueError("native input exceeds the artifact byte allowance")
    files = {
        row.path: artifacts.read_bytes(manifest.artifact_id, row.path) for row in manifest.files
    }
    if sum(map(len, files.values())) > MAX_ARTIFACT_BYTES:
        raise ValueError("native input exceeds the artifact byte allowance")
    identity = stable_id(
        "native-materialization",
        {
            "checkpoint": spec.checkpoint.model_dump(mode="json"),
            "binding": binding.model_dump(mode="json"),
            "code_revision": code_revision,
        },
    )
    if dry_run:
        return None
    with CoordinationLock(artifacts.project_directory / (identity + ".lock"), timeout=0):
        if artifacts.exists(identity):
            reference = artifact_input(artifacts.read(identity).manifest)
        else:
            native = adapter.apply(dict(files), plan.operations, binding.output_paths)
            if sha256_json(adapter.definition) != binding.adapter_sha256:
                raise ValueError("native adapter changed during execution; refuse publication")
            if set(native.files) != set(binding.output_paths):
                raise ValueError("adapter did not produce exactly the declared native output files")
            if any(not isinstance(value, bytes) for value in native.files.values()):
                raise ValueError("native outputs must be detached bytes")
            if sum(map(len, native.files.values())) > MAX_ARTIFACT_BYTES:
                raise ValueError("native output exceeds the artifact byte allowance")
            report: JsonObject = {
                "status": "matched"
                if strict_equal(plan.expected_projection, native.projection)
                else "diverged",
                "qualification": "diagnostic_only",
                "authority": "native_artifact_transformation",
                "checkpoint": spec.checkpoint.model_dump(mode="json"),
                "binding": binding.model_dump(mode="json"),
                "adapter": adapter.definition,
                "plan": plan.model_dump(mode="json"),
                "native_projection": native.projection,
                "candidate_state_changed": False,
                "provider_calls": 0,
            }
            stored = artifacts.write(
                artifact_id=identity,
                artifact_type="native-artifact-materialization",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(
                        sorted(
                            (spec.checkpoint, binding.input_artifact),
                            key=lambda ref: ref.artifact_id,
                        )
                    ),
                ),
                files={
                    "receipt.json": canonical_json_bytes(report),
                    **{"outputs/" + path: value for path, value in native.files.items()},
                },
            )
            reference = artifact_input(stored)
        if spec.destination is not None:
            export_native_artifacts(artifacts, reference, spec.destination)
        return reference


def export_native_artifacts(
    artifacts: ArtifactStore, reference: ArtifactInput, destination: Path
) -> None:
    """Export only matched, digest-verified outputs into a new directory atomically."""
    manifest = artifacts.read(reference.artifact_id).manifest
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != "native-artifact-materialization"
    ):
        raise ValueError("native output reference is not an exact materialization receipt")
    receipt = json.loads(artifacts.read_bytes(reference.artifact_id, "receipt.json"))
    if receipt["status"] != "matched":
        raise ValueError(
            "native output contradicts candidate prediction; inspect the divergence receipt"
        )
    if destination.exists() or destination.is_symlink():
        raise ValueError("native output destination exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".native-outputs-", dir=destination.parent))
    try:
        for row in manifest.files:
            if not row.path.startswith("outputs/"):
                continue
            path = staging / row.path.removeprefix("outputs/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(artifacts.read_bytes(reference.artifact_id, row.path))
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

"""Persist executable-world images as verified binary artifacts, never base64 text evidence."""

import base64
import json
from datetime import datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.environments.mixed import MixedCheckpoint
from exp.runtime.environments.world_contracts import ExecutableCheckpoint

WorldCheckpoint = ExecutableCheckpoint | MixedCheckpoint


def persist_world_image(
    store: ArtifactStore,
    checkpoint: WorldCheckpoint,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Store native bytes beside their typed checkpoint metadata under one manifest.

    Args:
        store: Trusted immutable artifact owner.
        checkpoint: Executable world or paired native/simulated world.
        created_at: Publication time.
        code_revision: Producing implementation identity.

    Returns:
        Digest-verified reference. Exact repeats reuse the already published image.
    """
    identity = stable_id(
        "executable-image", {"checkpoint": sha256_json(checkpoint), "code": code_revision}
    )
    mixed = isinstance(checkpoint, MixedCheckpoint)
    wire = checkpoint.model_dump(mode="json")
    parts: dict[str, bytes] = {}
    for owner, world in _worlds(checkpoint).items():
        target = wire[owner] if mixed else wire
        references = {}
        for index, (name, payload) in enumerate(sorted(world.image.parts.items())):
            path = f"images/{owner}/{index}.bin"
            parts[path] = payload
            references[name] = {"path": path, "sha256": sha256_bytes(payload)}
        target["image"]["parts"] = references
    parts["checkpoint.json"] = canonical_json_bytes(
        {"kind": "mixed" if mixed else "executable", "checkpoint": wire}
    )
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=created_at, code_revision=code_revision
    )
    parts["envelope.json"] = canonical_json_bytes(envelope)
    _, manifest = store.write_or_replay(
        artifact_id=identity,
        artifact_type="executable-world-image",
        envelope=envelope,
        envelope_type=ArtifactEnvelope,
        envelope_path="envelope.json",
        files=parts,
    )
    return artifact_input(manifest)


def load_world_image(store: ArtifactStore, reference: ArtifactInput) -> WorldCheckpoint:
    """Reconstruct typed world state only from its manifest-bound binary inventory."""
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != "executable-world-image":
        raise ValueError("world image differs from its frozen artifact reference")
    payload = json.loads(store.read_bytes(reference.artifact_id, "checkpoint.json"))
    if not isinstance(payload, dict) or payload.get("kind") not in ("mixed", "executable"):
        raise ValueError("unknown executable image format; capture a fresh checkpoint")
    wire = payload.get("checkpoint")
    if not isinstance(wire, dict):
        raise ValueError("world image is missing typed checkpoint metadata")
    targets = [wire.get("native"), wire.get("candidate")] if payload["kind"] == "mixed" else [wire]
    used = {"checkpoint.json", "envelope.json"}
    for target in targets:
        if target is None:
            continue
        if not isinstance(target, dict) or not isinstance(target.get("image"), dict):
            raise ValueError("world image metadata is incomplete")
        image = target["image"]
        if not isinstance(image.get("parts"), dict):
            raise ValueError("world image has no part inventory")
        decoded: JsonObject = {}
        for name, binding in image["parts"].items():
            if not isinstance(binding, dict) or not isinstance(binding.get("path"), str):
                raise ValueError("world image part has no immutable binary path")
            path = binding["path"]
            if path in used:
                raise ValueError("world image repeats a binary part")
            raw = store.read_bytes(reference.artifact_id, path)
            if sha256_bytes(raw) != binding.get("sha256"):
                raise ValueError("world image part differs from its declared digest")
            used.add(path)
            decoded[name] = base64.b64encode(raw).decode("ascii")
        image["parts"] = decoded
    if used != {entry.path for entry in manifest.files}:
        raise ValueError("world image manifest differs from its part inventory")
    if payload["kind"] == "mixed":
        return MixedCheckpoint.model_validate(wire)
    return ExecutableCheckpoint.model_validate(wire)


def _worlds(checkpoint: WorldCheckpoint) -> dict[str, ExecutableCheckpoint]:
    """Enumerate exactly the backends present in a native or mixed checkpoint."""
    if isinstance(checkpoint, ExecutableCheckpoint):
        return {"world": checkpoint}
    return {
        name: world
        for name, world in (("native", checkpoint.native), ("candidate", checkpoint.candidate))
        if world is not None
    }

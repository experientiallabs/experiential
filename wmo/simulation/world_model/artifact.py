"""Immutable grounded world-model artifact construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.models import ModelSnapshot
from wmo.common.project import (
    ArtifactAlreadyExistsError,
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.simulation.retrieval import load_rag_index

GROUNDED_WORLD_MODEL_ARTIFACT_TYPE = "grounded-world-model"
WORLD_MODEL_ARTIFACT_PATH = "world-model.json"
GROUNDED_WORLD_MODEL_PROMPT_VERSION = "grounded-world-model-v1"
GROUNDED_WORLD_MODEL_SYSTEM_PROMPT = """Protocol version: grounded-world-model-v1.
Predict the next visible user or environment message after the supplied agent action. Use only the
retrieved real transitions as grounding. Do not execute or invent tools. Return only this JSON
object: {"message":"the next visible user or environment message","terminal":false}
Set terminal to true only when the scenario has reached a visible terminal state. Do not include
markdown fences or other keys."""


def grounded_world_model_prompt_sha256() -> str:
    """Return the digest of the complete grounded world-model system prompt.

    Returns:
        Stable SHA-256 prompt identity.
    """
    return sha256_json(
        {
            "prompt_version": GROUNDED_WORLD_MODEL_PROMPT_VERSION,
            "system_prompt": GROUNDED_WORLD_MODEL_SYSTEM_PROMPT,
        }
    )


def grounded_world_model_content(
    *,
    serving_rag: ArtifactInput,
    model_alias: str,
    model: ModelSnapshot,
    prompt_version: str,
    prompt_sha256: str,
    top_k: int,
    code_revision: str,
) -> dict[str, object]:
    """Return the complete content-addressed identity of a grounded world model.

    Args:
        serving_rag: Exact serving RAG manifest reference.
        model_alias: Configured local world-model alias.
        model: Frozen provider model identity.
        prompt_version: Grounded prediction protocol version.
        prompt_sha256: Exact prompt-content digest.
        top_k: Number of real transitions retrieved per prediction.
        code_revision: Exact code revision that creates the artifact.

    Returns:
        Canonical content mapping used to derive the artifact identity.
    """
    return {
        "serving_rag": serving_rag.model_dump(mode="json"),
        "model_alias": model_alias,
        "model": model.model_dump(mode="json"),
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "top_k": top_k,
        "code_revision": code_revision,
    }


class GroundedWorldModelArtifact(ArtifactEnvelope):
    """Executable text world model bound to one immutable serving RAG snapshot."""

    world_model_id: ArtifactId
    serving_rag: ArtifactInput
    model_alias: ArtifactId
    model: ModelSnapshot
    prompt_version: str = Field(min_length=1, max_length=256)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(gt=0)

    @model_validator(mode="after")
    def _require_single_rag_input(self) -> GroundedWorldModelArtifact:
        """Bind the artifact envelope to exactly its serving RAG input.

        Returns:
            The validated grounded world-model artifact.

        Raises:
            ValueError: Envelope inputs differ from the selected serving RAG.
        """
        if self.inputs != (self.serving_rag,):
            raise ValueError("grounded world model input must be its exact serving RAG artifact")
        return self


@dataclass(frozen=True)
class PersistedGroundedWorldModel:
    """Completed grounded world-model artifact and verified manifest."""

    artifact: GroundedWorldModelArtifact
    manifest: ArtifactManifest


def persist_grounded_world_model(
    store: ArtifactStore,
    serving_rag: ArtifactInput,
    *,
    model_alias: str,
    model: ModelSnapshot,
    created_at: datetime,
    code_revision: str,
    top_k: int,
) -> PersistedGroundedWorldModel:
    """Persist a reusable world model over one verified immutable serving index.

    Args:
        store: Project-local immutable artifact store.
        serving_rag: Exact manifest reference for the serving retrieval index.
        model_alias: Project-selected world-model alias.
        model: Resolved secret-free model identity snapshot.
        created_at: Artifact materialization time.
        code_revision: Exact WMO revision producing the artifact.
        top_k: Number of grounded demonstrations supplied per step.

    Returns:
        Completed artifact and exact verified manifest.

    Raises:
        ValueError: The RAG input is stale or malformed.
    """
    loaded = load_rag_index(store, serving_rag.artifact_id)
    if artifact_input(loaded.manifest) != serving_rag:
        raise ValueError("serving RAG manifest differs from the supplied artifact input")
    content = grounded_world_model_content(
        serving_rag=serving_rag,
        model_alias=model_alias,
        model=model,
        prompt_version=GROUNDED_WORLD_MODEL_PROMPT_VERSION,
        prompt_sha256=grounded_world_model_prompt_sha256(),
        top_k=top_k,
        code_revision=code_revision,
    )
    artifact = GroundedWorldModelArtifact(
        schema_version=1,
        created_at=created_at,
        inputs=(serving_rag,),
        code_revision=code_revision,
        source=None,
        world_model_id=stable_id("grounded-world-model", content),
        serving_rag=serving_rag,
        model_alias=model_alias,
        model=model,
        prompt_version=GROUNDED_WORLD_MODEL_PROMPT_VERSION,
        prompt_sha256=grounded_world_model_prompt_sha256(),
        top_k=top_k,
    )
    files = {WORLD_MODEL_ARTIFACT_PATH: canonical_json_bytes(artifact)}
    try:
        manifest = store.write(
            artifact_id=artifact.world_model_id,
            artifact_type=GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
            envelope=artifact,
            files=files,
        )
    except ArtifactAlreadyExistsError:
        stored = store.read(artifact.world_model_id)
        replay = GroundedWorldModelArtifact.model_validate_json(
            store.read_bytes(artifact.world_model_id, WORLD_MODEL_ARTIFACT_PATH)
        )
        if replay.model_copy(update={"created_at": artifact.created_at}) != artifact:
            raise ValueError("existing grounded world-model artifact differs from replay") from None
        manifest = stored.manifest
    return PersistedGroundedWorldModel(artifact=artifact, manifest=manifest)

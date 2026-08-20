"""Immutable precomputed router embeddings for provider-free offline fitting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    canonical_json_bytes,
    envelope_matches_manifest,
    stable_id,
)
from exp.common.core.files import write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.common.models import BillingSource, Embedding, EmbeddingClient, ModelSnapshot
from exp.common.project import ArtifactStore
from exp.common.routing.features import RouterFeatureExtractor
from exp.common.tasks import TaskCase


class RouterEmbeddingReservation(ContractModel):
    """Conservative complete request ceiling for frozen router feature embeddings."""

    model: ModelSnapshot
    input_usd_per_million_tokens: float = Field(ge=0)
    maximum_attempts_per_feature: int = Field(gt=0)
    maximum_input_tokens_per_feature: int = Field(gt=0)
    feature_count: int = Field(gt=0)
    estimated_cost_usd: float = Field(ge=0)

    @field_validator("input_usd_per_million_tokens", "estimated_cost_usd")
    @classmethod
    def _require_finite_costs(cls, value: float) -> float:
        """Reject non-finite prices before budget comparison.

        Args:
            value: Nonnegative reservation price or total.

        Returns:
            The unchanged finite value.

        Raises:
            ValueError: The value is infinite or NaN.
        """
        if not math.isfinite(value):
            raise ValueError("router embedding reservation costs must be finite")
        return value

    @model_validator(mode="after")
    def _require_conservative_total(self) -> RouterEmbeddingReservation:
        """Require the persisted total to equal the complete retry-bound ceiling.

        Returns:
            Validated reservation.

        Raises:
            ValueError: The stated total omits a feature, token, price, or retry factor.
        """
        expected = (
            self.input_usd_per_million_tokens
            * self.maximum_attempts_per_feature
            * self.maximum_input_tokens_per_feature
            * self.feature_count
            / 1_000_000
        )
        if not math.isclose(self.estimated_cost_usd, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "router embedding estimated cost differs from its complete reservation"
            )
        return self


class FrozenEmbedding(ContractModel):
    """One exact feature-text digest and its completed vector."""

    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: tuple[float, ...] = Field(min_length=1)

    @field_validator("values")
    @classmethod
    def _finite_values(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        """Reject non-finite persisted vector components.

        Args:
            values: Persisted vector components.

        Returns:
            The unchanged finite components.

        Raises:
            ValueError: Any component is infinite or NaN.
        """
        if not all(math.isfinite(value) for value in values):
            raise ValueError("frozen router embeddings must be finite")
        return values


class FrozenEmbeddingSet(ArtifactEnvelope):
    """Completed vectors bound to one exact embedder snapshot and feature texts."""

    schema_version: Literal[1, 2, 3] = 3
    embedding_set_id: ArtifactId
    embedder_alias: ArtifactId
    embedder: ModelSnapshot
    embeddings: tuple[FrozenEmbedding, ...]

    @model_validator(mode="before")
    @classmethod
    def _migrate_v1_embedder_billing_source(cls, value: object) -> object:
        """Upgrade only schema-v1 embedding payloads with conservative attribution.

        Args:
            value: Candidate frozen embedding payload.

        Returns:
            Copied schema-v1 payload with explicit customer-managed embedder attribution.
            Current payloads are unchanged and still fail when the field is absent.
        """
        if not isinstance(value, dict):
            return value
        schema_version = value.get("schema_version")
        if type(schema_version) is not int:
            raise ValueError("router embedding schema_version must be an integer")
        if schema_version != 1:
            return value
        embedder = value.get("embedder")
        if not isinstance(embedder, dict):
            return value
        migrated = dict(value)
        migrated_embedder = dict(embedder)
        if "billing_source" in migrated_embedder:
            raise ValueError("schema-v1 router embedder must not declare current billing_source")
        migrated_embedder["billing_source"] = BillingSource.CUSTOMER_MANAGED.value
        migrated["embedder"] = migrated_embedder
        migrated["schema_version"] = 3
        return migrated

    @field_validator("embeddings")
    @classmethod
    def _unique_texts(cls, values: tuple[FrozenEmbedding, ...]) -> tuple[FrozenEmbedding, ...]:
        """Require unique feature identities and one shared vector dimension.

        Args:
            values: Persisted feature digests and vectors.

        Returns:
            The unchanged validated vectors.

        Raises:
            ValueError: The set is empty, repeats a digest, or mixes dimensions.
        """
        if not values:
            raise ValueError("a frozen embedding set cannot be empty")
        digests = tuple(value.text_sha256 for value in values)
        if len(set(digests)) != len(digests):
            raise ValueError("a frozen embedding set repeats a feature digest")
        dimensions = {len(value.values) for value in values}
        if len(dimensions) != 1:
            raise ValueError("all frozen router embeddings need one dimension")
        return values


class ReservedFrozenEmbeddingSet(FrozenEmbeddingSet):
    """Versioned automatic-router vectors with their exact provider reservation."""

    embedding_dimension: int = Field(gt=0)
    reservation: RouterEmbeddingReservation

    @model_validator(mode="after")
    def _require_v2_dimension(self) -> ReservedFrozenEmbeddingSet:
        """Verify the v2 schema marker and declared vector dimension.

        Returns:
            The unchanged validated artifact.

        Raises:
            ValueError: The schema marker or a vector dimension differs.
        """
        if self.schema_version != 2:
            raise ValueError("reserved router embeddings require schema_version=2")
        if any(len(item.values) != self.embedding_dimension for item in self.embeddings):
            raise ValueError("router embedding vector differs from its declared dimension")
        return self


class FrozenEmbeddingClient(EmbeddingClient):
    """Resolve exact precomputed feature texts without network or environment access."""

    def __init__(self, artifact: FrozenEmbeddingSet) -> None:
        self._vectors = {
            item.text_sha256: Embedding(values=item.values) for item in artifact.embeddings
        }

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return completed vectors for exact precomputed feature texts.

        Args:
            texts: Canonical request-visible feature strings to resolve.

        Returns:
            Frozen embeddings in the caller's input order.

        Raises:
            ValueError: A requested feature digest was not precomputed.
        """
        result = []
        for value in texts:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            try:
                result.append(self._vectors[digest])
            except KeyError as exc:
                raise ValueError(f"frozen embedding set lacks feature digest {digest}") from exc
        return tuple(result)


def load_frozen_embedding_set(
    store: ArtifactStore, artifact_id: ArtifactId
) -> FrozenEmbeddingSet | ReservedFrozenEmbeddingSet:
    """Load one manifest-verified completed router embedding artifact.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Router-embedding artifact identity.

    Returns:
        Parsed frozen embedding set.

    Raises:
        ValueError: The artifact type, payload, or bound identity is invalid.
    """
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "router-embeddings":
        raise ValueError(f"artifact {artifact_id} is not a router embedding set")
    payload = store.read_bytes(artifact_id, "embeddings.json")
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("router embedding payload is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("router embedding payload must be a JSON object")
    schema_version = raw.get("schema_version")
    value: FrozenEmbeddingSet | ReservedFrozenEmbeddingSet = (
        ReservedFrozenEmbeddingSet.model_validate_json(payload)
        if schema_version == 2
        else FrozenEmbeddingSet.model_validate_json(payload)
    )
    if value.schema_version not in {1, 2, 3}:
        raise ValueError("router embedding set schema version is unsupported")
    if value.embedding_set_id != artifact_id:
        raise ValueError("router embedding set identity differs from its artifact")
    if not envelope_matches_manifest(value, stored.manifest):
        raise ValueError("router embedding payload differs from its artifact manifest")
    for item in value.embeddings:
        Embedding(values=item.values)
    if isinstance(value, ReservedFrozenEmbeddingSet):
        if len(value.inputs) != 1:
            raise ValueError("router embedding set needs one exact task-set input")
        expected_id = _router_embedding_set_id(
            task_set_input=value.inputs[0],
            embedder_alias=value.embedder_alias,
            embedder=value.embedder,
            reservation=value.reservation,
            feature_digests=tuple(item.text_sha256 for item in value.embeddings),
        )
        if expected_id != artifact_id:
            raise ValueError("router embedding set content identity is invalid")
    return value


def router_feature_token_upper_bound(text: str) -> int:
    """Return the conservative UTF-8 token ceiling used for one router feature.

    Args:
        text: Exact request-visible router feature.

    Returns:
        A byte-count upper bound with fixed request framing allowance.
    """
    return len(text.encode("utf-8")) + 4


def _router_embedding_set_id(
    *,
    task_set_input: ArtifactInput,
    embedder_alias: ArtifactId,
    embedder: ModelSnapshot,
    reservation: RouterEmbeddingReservation,
    feature_digests: tuple[str, ...],
) -> ArtifactId:
    """Derive the canonical content-addressed frozen embedding request identity.

    Args:
        task_set_input: Exact verified task-set pointer.
        embedder_alias: Stable selected embedding alias.
        embedder: Exact embedding model snapshot.
        reservation: Exact price, retry, input, and feature ceiling.
        feature_digests: Ordered request-visible feature digests.

    Returns:
        Stable artifact identity shared by production and loading.
    """
    return stable_id(
        "router-embeddings",
        {
            "task_set": task_set_input.model_dump(mode="json"),
            "embedder_alias": embedder_alias,
            "embedder": embedder.model_dump(mode="json"),
            "reservation": reservation.model_dump(mode="json"),
            "feature_digests": feature_digests,
        },
    )


def router_embedding_reservation(
    *,
    model: ModelSnapshot,
    input_usd_per_million_tokens: float,
    maximum_attempts_per_feature: int,
    maximum_input_tokens_per_feature: int,
    feature_count: int,
) -> RouterEmbeddingReservation:
    """Create the exact conservative reservation for one frozen feature set.

    Args:
        model: Exact embedding model identity.
        input_usd_per_million_tokens: Explicit catalog input price.
        maximum_attempts_per_feature: Runtime retry ceiling reserved for each feature.
        maximum_input_tokens_per_feature: Product token ceiling for each rendered feature.
        feature_count: Number of unique feature texts dispatched.

    Returns:
        Validated total reservation suitable for shared budget admission.
    """
    estimated = (
        input_usd_per_million_tokens
        * maximum_attempts_per_feature
        * maximum_input_tokens_per_feature
        * feature_count
        / 1_000_000
    )
    return RouterEmbeddingReservation(
        model=model,
        input_usd_per_million_tokens=input_usd_per_million_tokens,
        maximum_attempts_per_feature=maximum_attempts_per_feature,
        maximum_input_tokens_per_feature=maximum_input_tokens_per_feature,
        feature_count=feature_count,
        estimated_cost_usd=estimated,
    )


def persist_router_embeddings(
    store: ArtifactStore,
    *,
    task_set_input: ArtifactInput,
    tasks: Sequence[TaskCase],
    embedder_alias: ArtifactId,
    embedder: ModelSnapshot,
    client: EmbeddingClient,
    reservation: RouterEmbeddingReservation,
    active_input_usd_per_million_tokens: float,
    active_maximum_attempts_per_feature: int,
    created_at: datetime,
    code_revision: str,
) -> ReservedFrozenEmbeddingSet:
    """Dispatch once and persist exact request-visible router feature vectors.

    Args:
        store: Project-local immutable artifact store.
        task_set_input: Exact verified task-set pointer.
        tasks: Fit and held-out tasks consumed by offline and online feature extraction.
        embedder_alias: Stable local alias for the selected embedding model.
        embedder: Exact resolved embedding model identity.
        client: Focused embedding client.
        reservation: Pre-admitted retry-bound reservation for every unique feature.
        active_input_usd_per_million_tokens: Current exact catalog input price.
        active_maximum_attempts_per_feature: Current runtime retry ceiling.
        created_at: Artifact completion time.
        code_revision: Exact producer revision.

    Returns:
        Persisted frozen embedding artifact, reusing exact replay without dispatch.

    Raises:
        ValueError: Reservation, response count, or replay content is inconsistent.
    """
    extractor = RouterFeatureExtractor()
    texts = tuple(dict.fromkeys(extractor.from_task(task) for task in tasks))
    if not texts:
        raise ValueError("router embedding production requires at least one task feature")
    if reservation.model != embedder:
        raise ValueError("router embedding reservation model differs from the active embedder")
    if reservation.feature_count != len(texts):
        raise ValueError("router embedding reservation differs from its feature scope")
    if reservation.input_usd_per_million_tokens != active_input_usd_per_million_tokens:
        raise ValueError("router embedding reservation price differs from the active catalog")
    if reservation.maximum_attempts_per_feature != active_maximum_attempts_per_feature:
        raise ValueError("router embedding reservation retry bound differs from the client")
    if any(
        router_feature_token_upper_bound(text) > reservation.maximum_input_tokens_per_feature
        for text in texts
    ):
        raise ValueError("router feature exceeds its reserved input-token ceiling")
    feature_digests = tuple(hashlib.sha256(text.encode()).hexdigest() for text in texts)
    embedding_set_id = _router_embedding_set_id(
        task_set_input=task_set_input,
        embedder_alias=embedder_alias,
        embedder=embedder,
        reservation=reservation,
        feature_digests=feature_digests,
    )
    destination = store.project_directory / "artifacts" / embedding_set_id
    if destination.exists():
        existing = load_frozen_embedding_set(store, embedding_set_id)
        if (
            not isinstance(existing, ReservedFrozenEmbeddingSet)
            or existing.inputs != (task_set_input,)
            or existing.embedder_alias != embedder_alias
            or existing.embedder != embedder
            or existing.reservation != reservation
            or tuple(item.text_sha256 for item in existing.embeddings) != feature_digests
        ):
            raise ValueError("existing router embeddings differ from exact replay inputs")
        return existing
    _record_embedding_dispatch_intent(
        store,
        embedding_set_id=embedding_set_id,
        task_set_input=task_set_input,
        embedder_alias=embedder_alias,
        embedder=embedder,
        reservation=reservation,
        feature_digests=feature_digests,
    )
    vectors = client.embed(texts)
    if len(vectors) != len(texts):
        raise ValueError("embedding provider returned a different number of router vectors")
    artifact = ReservedFrozenEmbeddingSet(
        schema_version=2,
        created_at=created_at,
        inputs=(task_set_input,),
        code_revision=code_revision,
        embedding_set_id=embedding_set_id,
        embedder_alias=embedder_alias,
        embedder=embedder,
        embeddings=tuple(
            FrozenEmbedding(
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                values=vector.values,
            )
            for text, vector in zip(texts, vectors, strict=True)
        ),
        embedding_dimension=len(vectors[0].values),
        reservation=reservation,
    )
    try:
        stored, _ = store.write_or_replay(
            artifact_id=embedding_set_id,
            artifact_type="router-embeddings",
            envelope=artifact,
            envelope_path="embeddings.json",
            envelope_type=ReservedFrozenEmbeddingSet,
            files={"embeddings.json": canonical_json_bytes(artifact)},
        )
    except ValueError as exc:
        raise ValueError("existing router embeddings differ from deterministic replay") from exc
    return stored


def _record_embedding_dispatch_intent(
    store: ArtifactStore,
    *,
    embedding_set_id: ArtifactId,
    task_set_input: ArtifactInput,
    embedder_alias: ArtifactId,
    embedder: ModelSnapshot,
    reservation: RouterEmbeddingReservation,
    feature_digests: tuple[str, ...],
) -> None:
    """Persist a durable exact request intent before the embedding provider boundary.

    Args:
        store: Project-local artifact store that owns the coordination record.
        embedding_set_id: Content-addressed completed embedding identity.
        task_set_input: Exact task-set manifest selected for features.
        embedder_alias: Stable local embedding alias.
        embedder: Exact provider model snapshot.
        reservation: Frozen retry-bound provider reservation.
        feature_digests: Ordered exact request-visible feature digests.

    Raises:
        ValueError: A prior accepted intent lacks a completed artifact or differs semantically.
    """
    intent = {
        "schema_version": 1,
        "embedding_set_id": embedding_set_id,
        "task_set": task_set_input.model_dump(mode="json"),
        "embedder_alias": embedder_alias,
        "embedder": embedder.model_dump(mode="json"),
        "reservation": reservation.model_dump(mode="json"),
        "feature_digests": list(feature_digests),
    }
    payload = canonical_json_bytes(intent)
    path = (
        store.project_directory
        / "coordination"
        / "router-embedding-dispatches"
        / f"{embedding_set_id}.json"
    )
    _prepare_embedding_coordination_path(store.project_directory, path)
    with file_write_lock(path, what="router embedding provider dispatch"):
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("router embedding dispatch intent differs from exact inputs")
            raise ValueError(
                "router embedding dispatch may have completed without a durable artifact; "
                "reconcile provider spend before retrying"
            )
        write_bytes_atomic(path, payload)


def _prepare_embedding_coordination_path(project_directory: Path, path: Path) -> None:
    """Create only real project-local ancestors for one provider dispatch intent.

    Args:
        project_directory: Canonical project-owned state directory.
        path: Intended coordination record below the project directory.

    Raises:
        ValueError: The path escapes the project or any project-relative component is a symlink
            or non-directory.
    """
    try:
        relative = path.relative_to(project_directory)
    except ValueError as exc:
        raise ValueError("router embedding dispatch intent escapes its project") from exc
    current = project_directory
    if current.is_symlink() or not current.is_dir():
        raise ValueError("router embedding project directory must be a real directory")
    for component in relative.parts[:-1]:
        current /= component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError("router embedding coordination ancestors must be real directories")
            continue
        current.mkdir(mode=0o700)
        if current.is_symlink() or not current.is_dir():
            raise ValueError("router embedding coordination directory is unsafe")
    if path.is_symlink():
        raise ValueError("router embedding dispatch intent cannot be a symbolic link")

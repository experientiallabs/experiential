"""Provider-free frozen embedding artifact tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import ArtifactInput
from wmo.common.models import Embedding, ModelSnapshot
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.common.routing.embeddings import (
    FrozenEmbedding,
    FrozenEmbeddingClient,
    FrozenEmbeddingSet,
    ReservedFrozenEmbeddingSet,
    load_frozen_embedding_set,
    persist_router_embeddings,
    router_embedding_reservation,
    router_feature_token_upper_bound,
)
from wmo.common.tasks import TaskCase

_DIGEST = "a" * 64


def test_frozen_embedding_client_is_exact_and_never_imputes() -> None:
    """Completed vectors resolve by exact feature bytes and missing inputs fail loudly."""
    text = '{"initial_user_intent":"hello"}'
    artifact = FrozenEmbeddingSet(
        schema_version=1,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
        embedding_set_id="embedding-set-a",
        embedder_alias="embedder-a",
        embedder=ModelSnapshot(
            provider="fixture",
            model_id="fixture-embedder",
            capabilities_sha256=_DIGEST,
            connection_sha256=_DIGEST,
        ),
        embeddings=(
            FrozenEmbedding(
                text_sha256=hashlib.sha256(text.encode()).hexdigest(), values=(1.0, 0.0)
            ),
        ),
    )
    client = FrozenEmbeddingClient(artifact)

    assert client.embed((text,))[0].values == (1.0, 0.0)
    with pytest.raises(ValueError, match="lacks feature digest"):
        client.embed((text + " ",))


def test_router_embedding_reservation_is_persisted_and_replay_dispatches_zero_calls(
    tmp_path: Path,
) -> None:
    """Router features consume a conservative shared-budget reservation exactly once.

    Args:
        tmp_path: Temporary initialized project root.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    task_input = _artifact_input()
    task = TaskCase(
        task_id="task-a",
        lineage_group_id="lineage-a",
        partition="fit",
        instruction="Help the customer",
        workload_weight=1,
        source_trace_ids=("trace-a",),
    )
    model = ModelSnapshot(
        provider="fixture",
        model_id="embedder",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=1_000,
        feature_count=1,
    )
    client = _EmbeddingClient()

    first = persist_router_embeddings(
        project.artifacts,
        task_set_input=task_input,
        tasks=(task,),
        embedder_alias="embedder",
        embedder=model,
        client=client,
        reservation=reservation,
        active_input_usd_per_million_tokens=2,
        active_maximum_attempts_per_feature=2,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )
    replay = persist_router_embeddings(
        project.artifacts,
        task_set_input=task_input,
        tasks=(task,),
        embedder_alias="embedder",
        embedder=model,
        client=client,
        reservation=reservation,
        active_input_usd_per_million_tokens=2,
        active_maximum_attempts_per_feature=2,
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
        code_revision="test",
    )

    assert reservation.estimated_cost_usd == pytest.approx(0.004)
    assert first == replay
    assert client.calls == 1
    assert replay.reservation == reservation


def test_router_embedding_write_crash_blocks_unknown_spend_redispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A durable accepted intent prevents a second call after artifact publication fails.

    Args:
        monkeypatch: Artifact-write failure injection.
        tmp_path: Temporary initialized project root.
    """
    project, task, model = _project_task_model(tmp_path)
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=1_000,
        feature_count=1,
    )
    client = _EmbeddingClient()
    original_write = project.artifacts.write_or_replay

    def fail_after_provider(**_kwargs: object) -> object:
        """Model a crash after provider success but before artifact publication.

        Args:
            **_kwargs: Completed artifact write arguments.

        Raises:
            OSError: Always, at the injected crash boundary.
        """
        raise OSError("injected embedding artifact publication crash")

    monkeypatch.setattr(project.artifacts, "write_or_replay", fail_after_provider)
    with pytest.raises(OSError, match="publication crash"):
        persist_router_embeddings(
            project.artifacts,
            task_set_input=_artifact_input(),
            tasks=(task,),
            embedder_alias="embedder",
            embedder=model,
            client=client,
            reservation=reservation,
            active_input_usd_per_million_tokens=2,
            active_maximum_attempts_per_feature=2,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test",
        )
    monkeypatch.setattr(project.artifacts, "write_or_replay", original_write)

    with pytest.raises(ValueError, match="reconcile provider spend"):
        persist_router_embeddings(
            project.artifacts,
            task_set_input=_artifact_input(),
            tasks=(task,),
            embedder_alias="embedder",
            embedder=model,
            client=client,
            reservation=reservation,
            active_input_usd_per_million_tokens=2,
            active_maximum_attempts_per_feature=2,
            created_at=datetime(2026, 8, 13, tzinfo=UTC),
            code_revision="test",
        )

    assert client.calls == 1


def test_router_embedding_rejects_symlinked_coordination_ancestor_before_dispatch(
    tmp_path: Path,
) -> None:
    """Do not follow a project-relative coordination symlink outside the project.

    Args:
        tmp_path: Temporary initialized project and external target root.
    """
    project, task, model = _project_task_model(tmp_path / "project")
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    coordination = project.artifacts.project_directory / "coordination"
    coordination.mkdir()
    (coordination / "router-embedding-dispatches").symlink_to(
        external,
        target_is_directory=True,
    )
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=1_000,
        feature_count=1,
    )
    client = _EmbeddingClient()
    before = tuple(sorted(path.name for path in external.iterdir()))

    with pytest.raises(ValueError, match="coordination ancestors must be real directories"):
        persist_router_embeddings(
            project.artifacts,
            task_set_input=_artifact_input(),
            tasks=(task,),
            embedder_alias="embedder",
            embedder=model,
            client=client,
            reservation=reservation,
            active_input_usd_per_million_tokens=2,
            active_maximum_attempts_per_feature=2,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test",
        )

    assert client.calls == 0
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert tuple(sorted(path.name for path in external.iterdir())) == before


def test_router_embedding_under_reservation_fails_before_dispatch(tmp_path: Path) -> None:
    """A rendered feature larger than its ceiling cannot reach the provider.

    Args:
        tmp_path: Temporary initialized project root.
    """
    project, task, model = _project_task_model(tmp_path)
    client = _EmbeddingClient()
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=1,
        feature_count=1,
    )

    with pytest.raises(ValueError, match="reserved input-token ceiling"):
        persist_router_embeddings(
            project.artifacts,
            task_set_input=_artifact_input(),
            tasks=(task,),
            embedder_alias="embedder",
            embedder=model,
            client=client,
            reservation=reservation,
            active_input_usd_per_million_tokens=2,
            active_maximum_attempts_per_feature=2,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test",
        )

    assert router_feature_token_upper_bound("x") == 5
    assert client.calls == 0


@pytest.mark.parametrize("drift", ["model", "price", "retry"])
def test_router_embedding_active_economics_drift_fails_before_dispatch(
    tmp_path: Path, drift: str
) -> None:
    """Model, price, and retry drift invalidate a frozen reservation without dispatch.

    Args:
        tmp_path: Temporary initialized project root.
        drift: Active runtime property changed after reservation.
    """
    project, task, model = _project_task_model(tmp_path)
    active_model = model
    active_price = 2.0
    active_attempts = 2
    if drift == "model":
        active_model = model.model_copy(update={"model_id": "changed"})
    elif drift == "price":
        active_price = 3.0
    else:
        active_attempts = 3
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=10_000,
        feature_count=1,
    )
    client = _EmbeddingClient()

    with pytest.raises(
        ValueError,
        match={
            "model": "model differs",
            "price": "price differs",
            "retry": "retry bound differs",
        }[drift],
    ):
        persist_router_embeddings(
            project.artifacts,
            task_set_input=_artifact_input(),
            tasks=(task,),
            embedder_alias="embedder",
            embedder=active_model,
            client=client,
            reservation=reservation,
            active_input_usd_per_million_tokens=active_price,
            active_maximum_attempts_per_feature=active_attempts,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test",
        )

    assert client.calls == 0


def test_router_embedding_loader_recomputes_content_identity(tmp_path: Path) -> None:
    """A valid manifest cannot bless a forged content-addressed embedding envelope.

    Args:
        tmp_path: Temporary initialized project root.
    """
    project, _task, model = _project_task_model(tmp_path)
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=10_000,
        feature_count=1,
    )
    forged = ReservedFrozenEmbeddingSet(
        schema_version=2,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        inputs=(_artifact_input(),),
        code_revision="test",
        embedding_set_id="forged-router-embeddings",
        embedder_alias="embedder",
        embedder=model,
        embeddings=(FrozenEmbedding(text_sha256=_DIGEST, values=(1.0, 0.0)),),
        embedding_dimension=2,
        reservation=reservation,
    )
    project.artifacts.write_json(
        artifact_id=forged.embedding_set_id,
        artifact_type="router-embeddings",
        envelope=forged,
        files={"embeddings.json": forged},
    )

    with pytest.raises(ValueError, match="content identity"):
        load_frozen_embedding_set(project.artifacts, forged.embedding_set_id)


def test_router_embedding_loader_rejects_feature_order_drift(tmp_path: Path) -> None:
    """Ordered feature identity cannot be changed under an otherwise valid manifest.

    Args:
        tmp_path: Temporary root for source and forged projects.
    """
    source, task, model = _project_task_model(tmp_path / "source")
    second = task.model_copy(
        update={
            "task_id": "task-b",
            "lineage_group_id": "lineage-b",
            "instruction": "Answer a different question",
            "source_trace_ids": ("trace-b",),
        }
    )
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=10_000,
        feature_count=2,
    )
    artifact = persist_router_embeddings(
        source.artifacts,
        task_set_input=_artifact_input(),
        tasks=(task, second),
        embedder_alias="embedder",
        embedder=model,
        client=_EmbeddingClient(),
        reservation=reservation,
        active_input_usd_per_million_tokens=2,
        active_maximum_attempts_per_feature=2,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )
    forged = ProjectStore(tmp_path / "forged", "project-a")
    forged.initialize(ProjectConfig(project_id="project-a"))
    reordered = artifact.model_copy(update={"embeddings": tuple(reversed(artifact.embeddings))})
    forged.artifacts.write_json(
        artifact_id=artifact.embedding_set_id,
        artifact_type="router-embeddings",
        envelope=reordered,
        files={"embeddings.json": reordered},
    )

    with pytest.raises(ValueError, match="content identity"):
        load_frozen_embedding_set(forged.artifacts, artifact.embedding_set_id)


def test_router_embedding_loader_rejects_manifest_envelope_drift(tmp_path: Path) -> None:
    """Payload provenance must exactly equal its verified immutable manifest.

    Args:
        tmp_path: Temporary initialized project root.
    """
    project, _task, model = _project_task_model(tmp_path)
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=10_000,
        feature_count=1,
    )
    payload = ReservedFrozenEmbeddingSet(
        schema_version=2,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        inputs=(_artifact_input(),),
        code_revision="payload",
        embedding_set_id="manifest-drift",
        embedder_alias="embedder",
        embedder=model,
        embeddings=(FrozenEmbedding(text_sha256=_DIGEST, values=(1.0, 0.0)),),
        embedding_dimension=2,
        reservation=reservation,
    )
    envelope = payload.model_copy(update={"code_revision": "manifest"})
    project.artifacts.write_json(
        artifact_id=payload.embedding_set_id,
        artifact_type="router-embeddings",
        envelope=envelope,
        files={"embeddings.json": payload},
    )

    with pytest.raises(ValueError, match="differs from its artifact manifest"):
        load_frozen_embedding_set(project.artifacts, payload.embedding_set_id)


def test_router_embedding_loader_rejects_unreserved_schema(tmp_path: Path) -> None:
    """Persisted embeddings must carry the reserved schema, not an unreserved payload.

    Args:
        tmp_path: Temporary initialized project root.
    """
    project, _task, model = _project_task_model(tmp_path)
    payload = FrozenEmbeddingSet(
        schema_version=1,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        inputs=(_artifact_input(),),
        code_revision="unreserved",
        embedding_set_id="unreserved-embeddings",
        embedder_alias="embedder",
        embedder=model,
        embeddings=(FrozenEmbedding(text_sha256=_DIGEST, values=(1.0, 0.0)),),
    )
    project.artifacts.write_json(
        artifact_id=payload.embedding_set_id,
        artifact_type="router-embeddings",
        envelope=payload,
        files={"embeddings.json": payload},
    )

    with pytest.raises(ValueError, match="schema version is unsupported"):
        load_frozen_embedding_set(project.artifacts, payload.embedding_set_id)


@pytest.mark.parametrize(
    ("vectors", "dimension", "message"),
    [
        (((float("nan"), 0.0),), 2, "finite"),
        (((0.5, 0.0),), 2, "unit norm"),
        (((1.0, 0.0), (0.0, 1.0, 0.0)), 2, "one dimension"),
    ],
)
def test_router_embedding_loader_rejects_malformed_vectors(
    tmp_path: Path,
    vectors: tuple[tuple[float, ...], ...],
    dimension: int,
    message: str,
) -> None:
    """Loader validation rejects NaN, non-unit, and dimension-corrupt vectors.

    Args:
        tmp_path: Temporary initialized project root.
        vectors: Raw persisted vectors to validate.
        dimension: Claimed vector dimension.
        message: Expected validation failure fragment.
    """
    project, _task, model = _project_task_model(tmp_path)
    artifact_id = f"malformed-{message.replace(' ', '-')}"
    reservation = router_embedding_reservation(
        model=model,
        input_usd_per_million_tokens=2,
        maximum_attempts_per_feature=2,
        maximum_input_tokens_per_feature=10_000,
        feature_count=len(vectors),
    )
    envelope = ReservedFrozenEmbeddingSet(
        schema_version=2,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
        embedding_set_id=artifact_id,
        embedder_alias="embedder",
        embedder=model,
        embeddings=(FrozenEmbedding(text_sha256=_DIGEST, values=(1.0, 0.0)),),
        embedding_dimension=2,
        reservation=reservation,
    )
    payload = envelope.model_dump(mode="json")
    payload["embedding_dimension"] = dimension
    payload["embeddings"] = [
        {"text_sha256": hashlib.sha256(str(index).encode()).hexdigest(), "values": values}
        for index, values in enumerate(vectors)
    ]
    project.artifacts.write(
        artifact_id=artifact_id,
        artifact_type="router-embeddings",
        envelope=envelope,
        files={"embeddings.json": json.dumps(payload, allow_nan=True).encode()},
    )

    with pytest.raises(ValueError, match=message):
        load_frozen_embedding_set(project.artifacts, artifact_id)


def _project_task_model(tmp_path: Path) -> tuple[ProjectStore, TaskCase, ModelSnapshot]:
    """Create one initialized project, task, and exact embedder fixture.

    Args:
        tmp_path: Temporary project root.

    Returns:
        Initialized store, one fit task, and one model snapshot.
    """
    project = ProjectStore(tmp_path, "project-a")
    project.initialize(ProjectConfig(project_id="project-a"))
    task = TaskCase(
        task_id="task-a",
        lineage_group_id="lineage-a",
        partition="fit",
        instruction="Help the customer",
        workload_weight=1,
        source_trace_ids=("trace-a",),
    )
    model = ModelSnapshot(
        provider="fixture",
        model_id="embedder",
        capabilities_sha256=_DIGEST,
        connection_sha256=_DIGEST,
    )
    return project, task, model


def _artifact_input() -> ArtifactInput:
    """Return a deterministic task-set pointer fixture."""
    return ArtifactInput(artifact_id="task-set-a", sha256=_DIGEST)


class _EmbeddingClient:
    """Count embedding dispatches and return unit vectors."""

    def __init__(self) -> None:
        """Initialize zero provider dispatches."""
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return one unit vector per input text and count the dispatch.

        Args:
            texts: Router feature texts.

        Returns:
            Unit vectors aligned to the input order.
        """
        self.calls += 1
        return tuple(Embedding(values=(1.0, 0.0)) for _text in texts)

"""Executable grounded world-model runtime tests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes, sha256_json
from wmo.common.models import (
    AssistantAction,
    BillingSource,
    Embedding,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.common.project import ProjectConfig, ProjectStore, artifact_input
from wmo.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from wmo.simulation.engines.text.prompt import (
    WORLD_MODEL_TEXT_SYSTEM_PROMPT,
    text_prompt_sha256,
)
from wmo.simulation.retrieval import (
    RAGAction,
    RAGEmbedderBinding,
    RAGLineageBinding,
    load_fit_rag_retriever,
    load_rag_index,
    persist_trace_rag,
)
from wmo.simulation.world_model import (
    bind_fit_grounded_world_model,
    load_grounded_world_model,
    persist_grounded_world_model,
)
from wmo.simulation.world_model.artifact import (
    GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
    GROUNDED_WORLD_MODEL_SYSTEM_PROMPT,
    WORLD_MODEL_ARTIFACT_PATH,
    grounded_world_model_prompt_sha256,
)


class _Embedder:
    """Stable local embedding client shared by index build and runtime query."""

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Return deterministic unit vectors.

        Args:
            texts: Canonical query or transition texts.

        Returns:
            Stable unit vectors in input order.
        """
        embedded = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            raw = [float(value + 1) for value in digest[:8]]
            norm = math.sqrt(sum(value * value for value in raw))
            embedded.append(Embedding(values=tuple(value / norm for value in raw)))
        return tuple(embedded)


class _WorldClient:
    """Capture the grounded request and return one strict protocol transition."""

    def __init__(self, snapshot: ModelSnapshot) -> None:
        """Record requests under one exact fixture model identity.

        Args:
            snapshot: Frozen world-model identity returned with every response.
        """
        self.snapshot = snapshot
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one visible environment observation.

        Args:
            request: Grounded world-model completion request to capture.

        Returns:
            Strict typed transition response under the configured fixture identity.
        """
        self.requests.append(request)
        return ModelResponse(
            output=AssistantAction(content='{"message":"Use the saved email.","terminal":false}'),
            model=self.snapshot,
            economics=OperationEconomics(),
        )


def test_build_and_simulation_share_one_grounded_prompt_identity() -> None:
    """The persisted world-model protocol exactly matches active simulation framing."""
    assert GROUNDED_WORLD_MODEL_SYSTEM_PROMPT == WORLD_MODEL_TEXT_SYSTEM_PROMPT
    assert grounded_world_model_prompt_sha256() == text_prompt_sha256()


def test_loaded_world_model_retrieves_real_evidence_before_prediction(tmp_path: Path) -> None:
    """A completed build artifact executes with immutable observed-transition grounding.

    Args:
        tmp_path: Temporary project root containing the RAG and world-model artifacts.
    """
    store = ProjectStore(tmp_path / ".wmo", "support")
    store.initialize(ProjectConfig(project_id="support"))
    created_at = datetime(2026, 8, 13, tzinfo=UTC)
    trace = _trace(created_at)
    trace_payload = trace.model_dump_json().encode() + b"\n"
    dataset = TraceDataset(
        schema_version=1,
        created_at=created_at,
        code_revision="fixture-revision",
        source=trace.source.identity,
        dataset_id="trace-source",
        semantic_convention_version="1.37.0",
        traces_path="traces.jsonl",
        traces_sha256=hashlib.sha256(trace_payload).hexdigest(),
        issues_path="normalization-issues.json",
        issues_sha256=hashlib.sha256(b"[]").hexdigest(),
        invalid_trace_count=0,
        trace_ids=(trace.trace_id,),
    )
    trace_manifest = store.artifacts.write(
        artifact_id=dataset.dataset_id,
        artifact_type="trace-dataset",
        envelope=dataset,
        files={
            "trace-dataset.json": dataset.model_dump_json().encode(),
            "traces.jsonl": trace_payload,
            "normalization-issues.json": b"[]",
        },
    )
    capabilities = ModelCapabilities(supports_embeddings=True)
    embedding_snapshot = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="embed",
        capabilities_sha256=sha256_json(capabilities),
        connection_sha256=sha256_json({"connection": "fixture"}),
    )
    binding = RAGEmbedderBinding(client=_Embedder(), snapshot=embedding_snapshot)
    rag = persist_trace_rag(
        store.artifacts,
        (artifact_input(trace_manifest),),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="lineage-a", partition="fit"),),
        created_at=created_at,
        code_revision="fixture-revision",
        embedder=binding,
        default_top_k=5,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    world_snapshot = ModelSnapshot(
        billing_source=BillingSource.CUSTOMER_MANAGED,
        provider="fixture",
        model_id="world",
        capabilities_sha256=sha256_json(ModelCapabilities()),
        connection_sha256=sha256_json({"connection": "world"}),
    )
    artifact = persist_grounded_world_model(
        store.artifacts,
        artifact_input(rag.manifest),
        model_alias="world",
        model=world_snapshot,
        created_at=created_at,
        code_revision="fixture-revision",
        top_k=5,
    )
    assert artifact.artifact.top_k == 5
    client = _WorldClient(world_snapshot)

    runtime = load_grounded_world_model(
        store.artifacts,
        artifact.artifact.world_model_id,
        client=client,
        embedder=binding,
    )
    transition = runtime.step(
        task="Reset my password",
        action=RAGAction(kind="message", content="What email is associated with the account?"),
    )

    assert transition.message == "Use the saved email."
    assert transition.terminal is False
    assert len(client.requests) == 1
    content = client.requests[0].messages[1].content
    assert content is not None
    assert "grounded_examples" in content
    assert "customer@example.test" in content
    unchanged = load_rag_index(store.artifacts, rag.index.rag_id)
    assert unchanged.transitions == rag.transitions
    assert unchanged.vectors == rag.vectors

    fit_rag = persist_trace_rag(
        store.artifacts,
        (artifact_input(trace_manifest),),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id="lineage-a", partition="fit"),),
        created_at=created_at,
        code_revision="fixture-revision",
        included_partitions=frozenset({"fit"}),
        embedder=binding,
        default_top_k=5,
    )
    fit_retriever = load_fit_rag_retriever(
        store.artifacts,
        artifact_input(fit_rag.manifest),
        embedder=binding,
    )
    fit_runtime = bind_fit_grounded_world_model(
        store.artifacts,
        artifact_input(artifact.manifest),
        client=client,
        fit_retriever=fit_retriever,
    )
    assert runtime.retriever.rag_input == artifact_input(rag.manifest)
    assert fit_runtime.retriever.rag_input == artifact_input(fit_rag.manifest)
    assert fit_runtime.artifact_input == artifact_input(artifact.manifest)
    with pytest.raises(ValueError, match="fit-only"):
        bind_fit_grounded_world_model(
            store.artifacts,
            artifact_input(artifact.manifest),
            client=client,
            fit_retriever=runtime.retriever,
        )
    with pytest.raises(ValueError, match="manifest differs"):
        bind_fit_grounded_world_model(
            store.artifacts,
            artifact_input(artifact.manifest).model_copy(update={"sha256": "0" * 64}),
            client=client,
            fit_retriever=fit_retriever,
        )

    inconsistent = artifact.artifact.model_copy(update={"world_model_id": "inconsistent-world"})
    store.artifacts.write(
        artifact_id=inconsistent.world_model_id,
        artifact_type=GROUNDED_WORLD_MODEL_ARTIFACT_TYPE,
        envelope=inconsistent,
        files={WORLD_MODEL_ARTIFACT_PATH: canonical_json_bytes(inconsistent)},
    )
    with pytest.raises(ValueError, match="complete content"):
        load_grounded_world_model(
            store.artifacts,
            inconsistent.world_model_id,
            client=client,
            embedder=binding,
        )


def _trace(created_at: datetime) -> Trace:
    """Create one real assistant-to-user transition.

    Args:
        created_at: Fixture timestamp shared by the trace spans.

    Returns:
        Canonical two-span production trace.
    """
    source = TraceSource(
        identity=SourceIdentity(kind="otlp", source_id="fixture-source"),
        semantic_convention_version="1.37.0",
    )
    return Trace(
        trace_id="0" * 31 + "1",
        source=source,
        task="Reset my password",
        spans=(
            TraceSpan(
                span_id="0" * 15 + "1",
                parent_span_id=None,
                name="agent.model_call",
                started_at=created_at,
                ended_at=created_at,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.output.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "What email is associated with the account?",
                            }
                        ]
                    ),
                    "gen_ai.input.messages": '[{"role":"user","content":"Reset my password"}]',
                },
            ),
            TraceSpan(
                span_id="0" * 15 + "2",
                parent_span_id=None,
                name="agent.model_call",
                started_at=created_at,
                ended_at=created_at,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.input.messages": json.dumps(
                        [
                            {
                                "role": "assistant",
                                "content": "What email is associated with the account?",
                            },
                            {"role": "user", "content": "customer@example.test"},
                        ]
                    ),
                },
            ),
        ),
    )

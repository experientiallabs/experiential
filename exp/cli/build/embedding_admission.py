"""Physical admission for the existing embedding/index phase, including async transports."""

import asyncio
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.models import ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.spending import operation_spend_ceiling, save_spend_receipt
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import (
    AdmittedJsonTransport,
    require_admitted_transport,
)
from exp.runtime.models.providers.async_transport import run_then_close_pooled_client
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport, RetryPolicy
from exp.runtime.models.registry import ProviderTransport, RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import (
    current_spend_ceiling,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.build import ProjectBuild
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.embedding_inputs import (
    embedding_batches,
    embedding_chunk_bytes,
    plan_rag_embedding_inputs,
)
from exp.simulation.retrieval.transitions import extract_real_transitions


class _SyncTransport(JsonHttpTransport):
    """Run async inners on a private loop; preserve caller-injected sync inners unchanged."""

    def __init__(self, inner: ProviderTransport) -> None:
        self.inner = inner

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        result = self.inner.get(url, headers=headers, timeout_seconds=timeout_seconds)
        if isinstance(result, JsonHttpResponse):
            return result
        return asyncio.run(run_then_close_pooled_client(result))

    def post(
        self, url: str, *, headers: Mapping[str, str], payload: JsonObject, timeout_seconds: float
    ) -> JsonHttpResponse:
        result = self.inner.post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        if isinstance(result, JsonHttpResponse):
            return result
        return asyncio.run(run_then_close_pooled_client(result))


@contextmanager
def admitted_embeddings(
    store: ProjectStore,
    completed: ProjectBuild,
    runtime: RuntimeModelCatalog,
    *,
    alias: str,
    snapshot: ModelSnapshot,
    maximum_input_tokens: int | None,
    estimate: float | None,
    provider_spend_authorized: bool,
) -> Iterator[RuntimeModelCatalog]:
    """Own physical admission over unadmitted inner transports and the shared input schedule.

    The default retry schedule bounds physical slots; a custom retry policy may stop earlier
    at this bound. Already-admitted injections are refused rather than nesting physical owners.
    Explicit consent to an unknown local estimate permits only this finite unpriced schedule;
    an inherited operation's unpriced allowance is never changed here.
    """
    if not provider_spend_authorized:
        raise ValueError("grounded build provider work requires explicit authorization")
    mining = completed.artifacts.mining
    bindings = tuple(
        RAGLineageBinding(
            trace_id=trace,
            lineage_id=group.lineage_group_id,
            partition=mining.partition.partition_for(group.lineage_group_id),
        )
        for group in mining.analysis.leakage_groups
        for trace in group.source_trace_ids
    )
    transitions = extract_real_transitions(
        completed.artifacts.trace_dataset.traces,
        bindings,
        included_partitions=frozenset({"fit", "held_out"}),
    )
    plan = plan_rag_embedding_inputs(
        tuple(row.key_text for row in transitions),
        maximum_chunk_bytes=embedding_chunk_bytes(maximum_input_tokens),
    )
    attempts = max(1, len(tuple(embedding_batches(plan.texts)))) * RetryPolicy().maximum_attempts
    phase = {
        "protocol": "build-embedding-admission-v1",
        "alias": alias,
        "code_revision": completed.review.code_revision,
        "catalog_sha256": sha256_json(load_model_catalog(store.model_catalog_path)),
        "trace_dataset": artifact_input(completed.artifacts.trace_dataset.manifest).model_dump(
            mode="json"
        ),
        "task_set": artifact_input(
            store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
        ).model_dump(mode="json"),
        "embedder": snapshot.model_dump(mode="json"),
        "input_sha256": sha256_json(list(plan.texts)),
        "maximum_physical_attempts": attempts,
    }
    identity = sha256_json(phase)
    operation = "build-embeddings-" + identity[:24]
    inherited = current_spend_ceiling()
    ceiling = inherited or operation_spend_ceiling(store, operation)
    if inherited is None:
        ceiling.authorize(
            operation,
            ceiling_usd=estimate or 0.0,
            unpriced_attempts=attempts if estimate is None else 0,
        )
    ledger = DispatchLedger(
        store.paths.runtime_directory / operation / "physical.sqlite", attempts, identity
    )
    intent_id = "build-embedding-intent-" + identity[:24]
    if store.artifacts.exists(intent_id):
        intent = store.artifacts.read(intent_id).manifest
        if json.loads(store.artifacts.read_bytes(intent_id, "request.json")) != phase:
            raise ValueError("embedding admission intent differs from the registered phase")
    else:
        intent = store.artifacts.write_json(
            artifact_id="build-embedding-intent-" + identity[:24],
            artifact_type="provider-request-intent",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=completed.artifacts.trace_dataset.manifest.created_at,
                code_revision=completed.review.code_revision,
                inputs=(
                    artifact_input(
                        store.artifacts.read(completed.artifacts.task_set.task_set_id).manifest
                    ),
                    artifact_input(completed.artifacts.trace_dataset.manifest),
                ),
            ),
            files={"request.json": phase},
        )
    deadline = datetime.now(UTC) + timedelta(hours=1)

    def wrap(inner: ProviderTransport) -> ProviderTransport:
        if isinstance(inner, AdmittedJsonTransport):
            raise ValueError("build embedding admission requires an unadmitted inner transport")
        return AdmittedJsonTransport(
            _SyncTransport(inner),
            ledger=ledger,
            artifacts=store.artifacts,
            code_revision=completed.review.code_revision,
            deadline=deadline,
            spend_ceiling=ceiling,
        )

    require_admitted_transport(
        load_model_catalog(store.model_catalog_path), alias, consumer="build"
    )
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        try:
            with canonical_call(artifact_input(intent)):
                yield runtime.with_transport_wrapper(wrap)
        finally:
            if inherited is None:
                save_spend_receipt(store, operation, ceiling, "build embedding/index phase")

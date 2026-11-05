"""A real native embedding client, not manually written physical receipts, proves the index."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from exp.common.models import ModelCapabilities, load_model_catalog
from exp.common.project import artifact_input
from exp.optimize.evaluation.executable.catalog_binding import register_runtime_catalog
from exp.optimize.evaluation.factory.author_retries import register_author_execution_window
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume
from exp.optimize.evaluation.variance.family_embedding import (
    FamilyEmbeddingCompletion,
    materialize_family_embedding,
    propose_family_embedding_intent,
    register_family_embedding_completion,
    register_family_embedding_intent,
)
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.retrieval.embedding import RAGEmbedderBinding

# Synthetic float32 unit vectors expose the SDK's normalization before RAG reduction.
RAW_VECTOR = (
    0.0025619540829211473,
    0.0026445977855473757,
    -0.1999150663614273,
    0.02148735709488392,
    0.917344868183136,
    -0.016528736799955368,
    0.027520345523953438,
    0.34206220507621765,
)


@pytest.mark.parametrize(
    ("different_echo", "valid_echo"), [(False, True), (True, True), (True, False)]
)
def test_native_client_request_model_echo_and_exact_response_reduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, different_echo: bool, valid_echo: bool
) -> None:
    """Request identity differs from provider echo; exact native parsing still qualifies."""
    import json

    project, spec, original, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, original, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    assert spec.build.reviewer is not None
    ceiling = operation_spend_ceiling(project, parent.original_operation)
    deadline = datetime.now(UTC) + timedelta(minutes=30)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        window = register_author_execution_window(
            project,
            spec.build.reviewer,
            operation=ceiling,
            window_id="wire-fixture",
            execution_deadline=deadline,
            code_revision="wire-fixture",
            sources=(parent.original_inputs,),
        )
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["embed"].model_copy(
        update={
            "model": "synthetic-requested-model",
            "served_model_id": "synthetic-provider-echo"
            if different_echo
            else "synthetic-requested-model",
            "revision": "fixture;representation-sha256=" + "a" * 64,
            "capabilities": ModelCapabilities(
                supports_embeddings=True, context_window_tokens=32768
            ),
        }
    )
    target_catalog = catalog.model_copy(update={"models": {**catalog.models, "target": record}})
    target = register_runtime_catalog(
        project, target_catalog, embedder_alias="target", code_revision="wire-fixture"
    )
    intent = propose_family_embedding_intent(
        project,
        previous_binding=previous,
        target_catalog=target,
        representation_sha256="a" * 64,
        dimension=8,
        execution_window=window,
        code_revision="wire-fixture",
    )
    reference = register_family_embedding_intent(project, intent)
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["model"] == record.model
        assert (body["model"] != record.served_model_id) is different_echo
        return httpx.Response(
            200,
            json={
                "model": record.served_model_id if valid_echo else "wrong-provider-echo",
                "data": [
                    {"index": i, "embedding": list(RAW_VECTOR)} for i in range(len(body["input"]))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    ledger = DispatchLedger(tmp_path / "wire-physical.sqlite", 100, "b" * 64)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client, operation_spend_scope():
        install_spend_ceiling(ceiling)
        transport = AdmittedJsonTransport(
            HttpxJsonTransport(client),
            ledger=ledger,
            artifacts=project.artifacts,
            code_revision="wire-fixture",
            deadline=deadline,
            spend_ceiling=ceiling,
        )
        resolved = RuntimeModelCatalog(target_catalog, transport_factory=lambda: transport).resolve(
            "target"
        )
        assert resolved.embedding_client is not None
        assert resolved.served_model_id == record.served_model_id
        with canonical_call(reference):
            scope = materialize_family_embedding(
                project,
                reference,
                RAGEmbedderBinding(
                    client=resolved.embedding_client,
                    snapshot=resolved.snapshot,
                    maximum_attempts=3,
                    maximum_batch_inputs=64,
                    maximum_input_tokens=resolved.capabilities.context_window_tokens,
                ),
            )
    responses = tuple(
        artifact_input(project.artifacts.read(identity).manifest)
        for identity in project.artifacts.list_ids()
        if identity.startswith("provider-attempt-") and identity.endswith("-response")
    )
    assert requests and len(requests) == ledger.used() == len(responses)
    completion = FamilyEmbeddingCompletion(
        intent=reference,
        scope=scope,
        dispatch_evidence=responses,
        producer_revision="wire-fixture",
    )
    if not valid_echo:
        with pytest.raises(ValueError, match="response echo"):
            register_family_embedding_completion(project, completion)
        return
    completed = register_family_embedding_completion(project, completion)
    assert completed.artifact_id.startswith("family-embedding-completion-")

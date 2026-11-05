"""Executor compatibility preserves the old intent, exact source delta and physical history."""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, sha256_bytes, sha256_json, stable_id
from exp.common.models import ModelCapabilities, load_model_catalog
from exp.common.project import artifact_input
from exp.common.release_revision import _package_source_digest, installed_release_revision
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.catalog_binding import register_runtime_catalog
from exp.optimize.evaluation.factory.author_retries import register_author_execution_window
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume
from exp.optimize.evaluation.variance.family_embedding import (
    INTENT_TYPE,
    load_family_embedding_intent,
    plan_family_embedding_migration,
    propose_family_embedding_intent,
)
from exp.optimize.evaluation.variance.family_embedding_executor import (
    SourceDelta,
    _files,
    _physical_cap,
    _validate,
    load_family_embedding_executor_revision,
    propose_family_embedding_executor_revision,
    register_family_embedding_executor_revision,
)
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.retrieval.embedding_inputs import plan_rag_embedding_inputs
from exp.simulation.retrieval.reembedding import load_reembedding_input


def test_executor_revision_keeps_original_intent_ledger_and_closed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native compatibility admits exact new source, refusing unknown deltas and history resets."""
    project, spec, original, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, original, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    assert spec.build.reviewer is not None
    ceiling = operation_spend_ceiling(project, parent.original_operation)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        window = register_author_execution_window(
            project,
            spec.build.reviewer,
            operation=ceiling,
            window_id="source-fixture",
            execution_deadline=datetime.now(UTC) + timedelta(minutes=30),
            code_revision="historical-executor",
            sources=(parent.original_inputs,),
        )
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["embed"].model_copy(
        update={
            "model": "requested-model",
            "served_model_id": "friendly-echo",
            "revision": "fixture;representation-sha256=" + "a" * 64,
            "capabilities": ModelCapabilities(
                supports_embeddings=True, context_window_tokens=32768
            ),
        }
    )
    catalog = catalog.model_copy(update={"models": {**catalog.models, "target": record}})
    target = register_runtime_catalog(
        project, catalog, embedder_alias="target", code_revision="historical-executor"
    )
    template = propose_family_embedding_intent(
        project,
        previous_binding=previous,
        target_catalog=target,
        representation_sha256="a" * 64,
        dimension=8,
        execution_window=window,
        code_revision="historical-executor",
        maximum_batch_inputs=16,
    )
    # A labelled synthetic historical package retains all unrelated production bytes.
    current_package = Path(grounded.__file__).resolve().parents[3]
    historical_package = tmp_path / "historical" / "exp"
    for name in _files(current_package):
        destination = historical_package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(current_package / name, destination)
    (historical_package / "optimize/evaluation/variance/family_embedding.py").write_text(
        "# synthetic historical embedding implementation fixture\n"
    )
    (historical_package / "optimize/evaluation/variance/family_embedding_executor.py").unlink()
    old_runtime = historical_package / "optimize/evaluation/executable/grounded.py"
    assert old_runtime.read_bytes() == Path(grounded.__file__).read_bytes()
    intent = template.model_copy(
        update={
            "runtime_file": old_runtime,
            "source_fingerprint": _package_source_digest(historical_package),
        }
    )
    inputs = tuple(
        sorted(
            set(
                (
                    intent.previous_binding,
                    intent.source_build,
                    intent.source_scope.index,
                    intent.source_scope.world_model,
                    intent.target_catalog.catalog,
                    intent.execution_window,
                )
            ),
            key=lambda r: r.artifact_id,
        )
    )
    manifest = project.artifacts.write_json(
        artifact_id=stable_id(INTENT_TYPE, intent),
        artifact_type=INTENT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=intent.producer_revision,
            inputs=inputs,
        ),
        files={"intent.json": intent},
    )
    reference = artifact_input(manifest)
    old = load_reembedding_input(project.artifacts, intent.source_scope.index)
    texts = plan_rag_embedding_inputs(
        tuple(r.key_text for r in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    ).texts[:16]
    cap = _physical_cap(project, intent)
    endpoint = catalog.connections[record.connection].base_url
    assert endpoint is not None
    physical = DispatchLedger(tmp_path / "original-physical.sqlite", cap, sha256_json(intent))
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        admitted = AdmittedJsonTransport(
            ScriptedJsonTransport(
                [
                    JsonHttpResponse(
                        status_code=200,
                        body={
                            "model": "friendly-echo",
                            "data": [
                                {"index": i, "embedding": [1.0] + [0.0] * 7}
                                for i in range(len(texts))
                            ],
                        },
                    )
                ]
            ),
            ledger=physical,
            artifacts=project.artifacts,
            code_revision=intent.producer_revision,
            deadline=intent.deadline,
            spend_ceiling=ceiling,
        )
        with canonical_call(reference):
            admitted.post(
                endpoint.rstrip("/") + "/embeddings",
                headers={},
                payload={"model": record.model, "input": list(texts)},
                timeout_seconds=10,
            )
    before = ceiling.summary()
    original_bytes = project.artifacts.read_bytes(reference.artifact_id, "intent.json")
    revision = propose_family_embedding_executor_revision(
        project,
        intent=reference,
        physical_ledger=tmp_path / "original-physical.sqlite",
        code_revision=installed_release_revision(),
    )
    successor = register_family_embedding_executor_revision(project, revision)
    assert register_family_embedding_executor_revision(project, revision) == successor
    assert load_family_embedding_executor_revision(project, successor) == revision
    assert revision.intent == reference and revision.physical_cap == cap
    assert (
        revision.phase_config_sha256 == sha256_json(intent) and len(revision.retained_attempts) == 1
    )
    assert revision.runtime_file == Path(grounded.__file__).resolve()
    assert revision.runtime_sha256 == sha256_bytes(old_runtime.read_bytes())
    assert load_family_embedding_intent(project, reference, executor_revision=successor) == intent
    with pytest.raises(ValueError, match="actual source changed"):
        load_family_embedding_intent(project, reference)
    plan = plan_family_embedding_migration(project, reference, executor_revision=successor)
    assert (
        plan.executor_revision == successor and plan.retained_attempts == before.unpriced_attempts
    )
    assert physical.used() == 1 and ceiling.summary() == before
    assert project.artifacts.read_bytes(reference.artifact_id, "intent.json") == original_bytes
    for bad in (
        revision.model_copy(update={"physical_cap": cap + 1}),
        revision.model_copy(update={"phase_config_sha256": "0" * 64}),
        revision.model_copy(update={"runtime_file": old_runtime}),
        revision.model_copy(update={"retained_attempts": ("unrelated-attempt",)}),
        revision.model_copy(
            update={
                "changes": {
                    **revision.changes,
                    "runtime/unreviewed.py": SourceDelta(before=None, after="0" * 64),
                }
            }
        ),
    ):
        with pytest.raises(ValueError):
            _validate(project, bad)

"""Exact-family embedding revisions preserve native authority and evidence membership."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_bytes, sha256_json
from exp.common.models import Embedding, ModelCapabilities, load_model_catalog
from exp.common.project import artifact_input
from exp.optimize.evaluation.executable.catalog_binding import register_runtime_catalog
from exp.optimize.evaluation.factory.author_retries import register_author_execution_window
from exp.optimize.evaluation.rules.rule_tags import RuleTaggingSettings
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_pilot import load_pilot_build
from exp.optimize.evaluation.variance.bootstrap_pilot_test import pilot_fixture, register_pilot
from exp.optimize.evaluation.variance.bootstrap_resume import (
    bootstrap_execution_spec,
    load_bootstrap_resume,
    register_bootstrap_resume,
)
from exp.optimize.evaluation.variance.bootstrap_selection import select_bootstrap_families
from exp.optimize.evaluation.variance.family_embedding import (
    FamilyEmbeddingCompletion,
    check_rules_revision,
    check_scope_preservation,
    load_family_embedding_intent,
    materialize_family_embedding,
    plan_family_embedding_migration,
    propose_family_embedding_intent,
    register_family_embedding_completion,
    register_family_embedding_intent,
)
from exp.runtime.models.call_evidence import canonical_call
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.openai_compatible import openai_embedding_response
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.retrieval.embedding import HashingRAGEmbedder, RAGEmbedderBinding


def test_native_intent_and_pending_inventory_preserve_original_scope_and_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real native pilot plans all pending roles without inference or a new root."""
    project, spec, original, pilot = pilot_fixture(tmp_path, monkeypatch)
    previous = register_pilot(project, spec, original, pilot)
    parent = load_bootstrap_resume(project, previous, spec=spec, code_revision="pilot-executor")
    baseline = bootstrap_execution_spec(project, parent, spec)
    assert spec.build.reviewer is not None
    ledger = operation_spend_ceiling(project, parent.original_operation)
    before = ledger.summary()
    deadline = datetime.now(UTC) + timedelta(minutes=30)
    with operation_spend_scope():
        install_spend_ceiling(ledger)
        window = register_author_execution_window(
            project,
            spec.build.reviewer,
            operation=ledger,
            window_id="fixture-qwen",
            execution_deadline=deadline,
            code_revision="fixture-new",
            sources=(parent.original_inputs,),
        )
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["embed"].model_copy(
        update={
            "model": "synthetic-target",
            "served_model_id": "synthetic-target-wire",
            "revision": "fixture;representation-sha256=" + "a" * 64,
            "capabilities": ModelCapabilities(
                supports_embeddings=True, context_window_tokens=32768
            ),
        }
    )
    target = catalog.model_copy(update={"models": {**catalog.models, "target": record}})
    binding = register_runtime_catalog(
        project, target, embedder_alias="target", code_revision="fixture-new"
    )
    assert baseline.bootstrap.rules is not None
    rules = baseline.bootstrap.rules.model_copy(
        update={
            "profile": "trajectory_tags",
            "tagging": RuleTaggingSettings(decision_alias="decisions"),
            "author": baseline.bootstrap.rules.author.model_copy(update={"deadline": deadline}),
        }
    )
    assert baseline.bootstrap.rules.memory is not None
    assert rules.tagging is not None
    rules = rules.model_copy(
        update={
            "tagging": rules.tagging.model_copy(
                update={"decision_alias": baseline.bootstrap.rules.memory.decision_alias}
            )
        }
    )
    intent = propose_family_embedding_intent(
        project,
        previous_binding=previous,
        target_catalog=binding,
        representation_sha256="a" * 64,
        dimension=8,
        execution_window=window,
        code_revision="fixture-new",
        rules_revision=rules,
    )
    reference = register_family_embedding_intent(project, intent)
    assert register_family_embedding_intent(project, intent) == reference
    assert load_family_embedding_intent(project, reference) == intent
    plan = plan_family_embedding_migration(project, reference)
    assert plan.original_operation == parent.original_operation
    assert plan.embedding_inputs > 0
    assert plan.physical_pending == sum(row.attempts for row in plan.costs)
    assert len(plan.costs) >= 5
    assert ledger.summary() == before
    assert intent.source_scope.family_id == pilot.family_id
    with pytest.raises(ValueError, match="scope, build or operation"):
        register_family_embedding_intent(
            project,
            intent.model_copy(
                update={"source_scope": intent.source_scope.model_copy(update={"tools": ()})}
            ),
        )
    with pytest.raises(ValueError, match="source changed"):
        register_family_embedding_intent(
            project, intent.model_copy(update={"source_fingerprint": "bad"})
        )
    with pytest.raises(ValueError, match="membership or controls"):
        check_scope_preservation(
            intent.source_scope, intent.source_scope.model_copy(update={"members": ()})
        )
    check_scope_preservation(
        intent.source_scope,
        intent.source_scope.model_copy(
            update={
                "index": intent.source_scope.world_model,
                "world_model": intent.source_scope.index,
            }
        ),
    )
    altered = rules.model_copy(update={"draws_per_trace": rules.draws_per_trace + 1})
    with pytest.raises(ValueError, match="unrelated frozen settings"):
        check_rules_revision(baseline.bootstrap.rules, altered, deadline)
    assert sha256_json(baseline) == parent.execution_spec_sha256

    class FixtureEmbedding:
        """Synthetic offline vectors, explicitly not real Qwen inference or runtime QA."""

        def __init__(self) -> None:
            self.requests: list[tuple[tuple[str, ...], tuple[Embedding, ...]]] = []

        def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
            values = HashingRAGEmbedder(8).embed(texts)
            self.requests.append((tuple(texts), values))
            return openai_embedding_response(
                {
                    "data": [
                        {"index": i, "embedding": list(value.values)}
                        for i, value in enumerate(values)
                    ]
                },
                expected_count=len(texts),
            )

    fixture = FixtureEmbedding()
    with pytest.raises(ValueError, match="chunk semantics"):
        materialize_family_embedding(
            project,
            reference,
            RAGEmbedderBinding(
                client=fixture,
                snapshot=intent.target_embedder,
                maximum_attempts=3,
                maximum_batch_inputs=64,
                maximum_input_tokens=64,
            ),
        )
    assert fixture.requests == []
    scope = materialize_family_embedding(
        project,
        reference,
        RAGEmbedderBinding(
            client=fixture,
            snapshot=intent.target_embedder,
            maximum_attempts=3,
            maximum_batch_inputs=64,
        ),
    )
    responses = [
        JsonHttpResponse(
            status_code=200,
            body={
                "model": record.served_model_id,
                "data": [
                    {"index": i, "embedding": list(value.values)} for i, value in enumerate(values)
                ],
            },
        )
        for _, values in fixture.requests
    ]
    transport = AdmittedJsonTransport(
        ScriptedJsonTransport(responses),
        ledger=DispatchLedger(tmp_path / "synthetic-physical.sqlite", len(responses), "b" * 64),
        artifacts=project.artifacts,
        code_revision="fixture-new",
        deadline=deadline,
    )
    with canonical_call(reference):
        for texts, _ in fixture.requests:
            transport.post(
                "https://fixture.invalid/v1/embeddings",
                headers={},
                payload={"model": record.model, "input": list(texts)},
                timeout_seconds=30,
            )
    refs = [
        artifact_input(project.artifacts.read(identity).manifest)
        for identity in project.artifacts.list_ids()
        if identity.startswith("provider-attempt-") and identity.endswith("-response")
    ]
    completion = FamilyEmbeddingCompletion(
        intent=reference,
        scope=scope,
        dispatch_evidence=tuple(refs),
        producer_revision="fixture-new",
    )
    completed = register_family_embedding_completion(project, completion)
    # The catalog may adopt only this exact verified completion, without new physical work.
    from exp.optimize.evaluation.variance.build_records import REPORT_TYPE, save_artifact
    from exp.optimize.evaluation.variance.build_test import TASK_BANK
    from exp.optimize.evaluation.variance.environment_catalog import (
        EnvironmentBaseSpec,
        EnvironmentCatalogSpec,
        TagTaskSetSpec,
        plan_environment_catalog,
    )
    from exp.simulation.mining.questions import resolve_question_bank
    from exp.simulation.mining.tag_selection import TagRule

    catalog_spec = EnvironmentCatalogSpec(
        task_sets=(
            TagTaskSetSpec(
                rule=TagRule(
                    name="tagged-probes",
                    require_all=("shape.q0",),
                    threshold=0.05,
                )
            ),
        ),
        bases=(
            EnvironmentBaseSpec(
                name="completed-base",
                family_id=scope.family_id,
                embedding_completion=completed,
            ),
        ),
        banks={"task": resolve_question_bank(TASK_BANK)},
    )
    ledger_before_catalog = ledger.summary()
    catalog = plan_environment_catalog(project, intent.source_build, catalog_spec)
    assert catalog.bases[0].scope == completion.scope
    assert catalog.bases[0].index_embedder == intent.target_embedder
    assert catalog.bases[0].producer_revision == completion.producer_revision
    assert catalog.exposure == load_pilot_build(project, pilot).exposure
    assert catalog.bases[0].scope.members == intent.source_scope.members
    assert all(row.task_start_qa == "pending" for row in catalog.instances)
    assert ledger.summary() == ledger_before_catalog
    wrong_source = save_artifact(
        project.artifacts,
        REPORT_TYPE,
        {"report.json": load_pilot_build(project, pilot)},
        (intent.source_build,),
        "test",
    )
    with pytest.raises(ValueError, match="another source build or base"):
        plan_environment_catalog(project, wrong_source, catalog_spec)
    successor = register_bootstrap_resume(
        project,
        original_inputs=parent.original_inputs,
        code_revision="fixture-new",
        previous_binding=previous,
        pilot_revision=parent.pilot_revision,
        permitted_changes=(
            "pilot_selection_revision",
            "family_embedding_revision",
            "execution_watchdog_renewal",
        ),
        execution_deadline=deadline,
        execution_window=window,
        embedding_revision=completed,
    )
    actual = load_bootstrap_resume(project, successor, spec=spec, code_revision="fixture-new")
    migrated = bootstrap_execution_spec(project, actual, spec)
    assert actual.original_operation == parent.original_operation
    assert actual.build_revision == pilot.build_producer_revision
    assert migrated.bootstrap.runtime.catalog_binding == binding
    assert migrated.bootstrap.runtime.runtime_file == intent.runtime_file
    from exp.optimize.evaluation.variance.family_embedding import apply_family_embedding_spec

    historical_runtime = tmp_path / "historical-grounded.py"
    historical_runtime.write_text("# genuine preserved old runtime fixture bytes\n")
    assert historical_runtime.read_bytes() != intent.runtime_file.read_bytes()
    stale = baseline.model_copy(
        update={
            "bootstrap": baseline.bootstrap.model_copy(
                update={
                    "runtime": baseline.bootstrap.runtime.model_copy(
                        update={
                            "runtime_file": historical_runtime,
                            "runtime_sha256": sha256_bytes(historical_runtime.read_bytes()),
                        }
                    )
                }
            )
        }
    )
    rebound = apply_family_embedding_spec(project, completed, stale)
    assert rebound.bootstrap.runtime.runtime_file == intent.runtime_file
    assert rebound.bootstrap.runtime.runtime_sha256 == intent.runtime_sha256
    assert migrated.bootstrap.rules is not None
    assert migrated.bootstrap.rules.profile == "trajectory_tags"
    historical_build = load_pilot_build(project, pilot)
    rows = select_bootstrap_families(
        project,
        historical_build,
        migrated.bootstrap,
        migrated.limits,
        task_ids=migrated.task_ids,
        embedding_revision=completed,
    )
    assert len(rows) == 1
    selected = rows[0]
    assert selected.scope == scope
    assert selected.simulator.index == scope.index
    assert selected.simulator.world_model == scope.world_model
    assert selected.spec.runtime.world_model == scope.world_model
    assert selected.spec.runtime.catalog_binding == binding
    assert historical_build.environment_families is not None
    assert historical_build.environment_families.scope(pilot.family_id) == intent.source_scope
    with pytest.raises(ValueError, match="registered runtime catalog"):
        select_bootstrap_families(
            project,
            historical_build,
            baseline.bootstrap,
            migrated.limits,
            task_ids=migrated.task_ids,
            embedding_revision=completed,
        )
    from exp.optimize.evaluation.variance.bootstrap_plan import plan_bootstrap_preparation

    native_plan = plan_bootstrap_preparation(
        project, spec, code_revision="fixture-new", resume_binding=successor
    )
    assert native_plan.identity == parent.original_operation
    assert native_plan.build_saved and native_plan.remaining_tasks == migrated.task_ids
    assert ledger.summary() == before

    with pytest.raises(ValueError, match="complete successful"):
        register_family_embedding_completion(
            project, completion.model_copy(update={"dispatch_evidence": (reference,)})
        )

    # A later source producer must inherit this completed asset, not execute its old intent.
    from exp.optimize.evaluation.variance import family_embedding as embedding_owner
    from exp.optimize.evaluation.variance.bootstrap_protocols import (
        propose_bootstrap_protocol_revision,
        register_bootstrap_protocol_revision,
    )

    monkeypatch.setattr(embedding_owner, "source_fingerprint", lambda: "c" * 64)
    proposal = propose_bootstrap_protocol_revision(
        project, previous_binding=successor, code_revision="rules-source-executor"
    )
    protocol = register_bootstrap_protocol_revision(
        project, proposal, code_revision="rules-source-executor"
    )
    continued = register_bootstrap_resume(
        project,
        original_inputs=actual.original_inputs,
        code_revision="rules-source-executor",
        previous_binding=successor,
        review_revision=actual.review_revision,
        pilot_revision=actual.pilot_revision,
        permitted_changes=(*actual.permitted_changes, "preparation_protocol_revision"),
        execution_deadline=actual.execution_deadline,
        execution_window=actual.execution_window,
        embedding_revision=completed,
        protocol_revision=protocol,
    )
    revised = load_bootstrap_resume(
        project, continued, spec=spec, code_revision="rules-source-executor"
    )
    assert bootstrap_execution_spec(project, revised, spec) == migrated
    assert revised.embedding_revision == completed
    from exp.optimize.evaluation.variance.bootstrap_protocols import validate_protocol_successor

    for wrong_embedding in (None, reference, protocol):
        with pytest.raises(ValueError, match="unreviewed settings"):
            validate_protocol_successor(
                project,
                revised.model_copy(update={"embedding_revision": wrong_embedding}),
                actual,
            )
    assert revised.producer_revision != completion.producer_revision
    assert revised.original_operation == actual.original_operation
    assert revised.grants == actual.grants and revised.stop == actual.stop
    continued_plan = plan_bootstrap_preparation(
        project, spec, code_revision="rules-source-executor", resume_binding=continued
    )
    assert continued_plan.identity == actual.original_operation
    assert continued_plan.build_saved and continued_plan.accounted_unpriced_attempts == 0
    assert ledger.summary() == before
    with pytest.raises(ValueError, match="source changed"):
        load_family_embedding_intent(project, reference)

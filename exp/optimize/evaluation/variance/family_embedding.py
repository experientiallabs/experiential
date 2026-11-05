"""Append-only exact-family embedding revisions under original preparation authority."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import AwareDatetime, Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import Embedding, ModelSnapshot, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.release_revision import installed_execution_revision
from exp.optimize.evaluation.executable.catalog_binding import (
    RuntimeCatalogBinding,
    load_runtime_catalog,
)
from exp.optimize.evaluation.executable.costs import CostComponent
from exp.optimize.evaluation.executable.model_costs import catalog_model_cost
from exp.optimize.evaluation.factory.author_retries import load_author_execution_window
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.bootstrap_pilot import load_pilot_build
from exp.optimize.evaluation.variance.bootstrap_protocols import _predecessor
from exp.optimize.evaluation.variance.build_records import FamilyScope, load_exposure_plan
from exp.optimize.evaluation.variance.pipeline_inputs import rules_costs
from exp.runtime.models.providers.openai_compatible import openai_embedding_response
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import MAXIMUM_UNPRICED_ATTEMPTS
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.embedding import RAGEmbedderBinding, embed_rag_texts
from exp.simulation.retrieval.embedding_checkpoint import RAGEmbeddingCheckpoint
from exp.simulation.retrieval.embedding_inputs import (
    embedding_batches,
    embedding_chunk_bytes,
    plan_rag_embedding_inputs,
)
from exp.simulation.retrieval.reembedding import load_reembedding_input
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
    from exp.optimize.evaluation.variance.bootstrap_resume import BootstrapResumeBinding
    from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
    from exp.optimize.evaluation.variance.family_embedding_executor import (
        FamilyEmbeddingExecutorRevision,
    )

INTENT_TYPE = "family-embedding-intent"
COMPLETION_TYPE = "family-embedding-completion"


def source_fingerprint() -> str:
    """Pin actual installed source bytes independently of a caller's revision label."""
    return installed_execution_revision().split(";source-sha256=", 1)[1]


class FamilyEmbeddingIntent(ContractModel):
    """A pending exact-family migration, not vectors, readiness or a new budget.

    Attributes:
        previous_binding: Exact preparation predecessor.
        original_operation: Unchanged preparation root.
        ledger_sha256: Unchanged original ledger identity.
        source_build: Actual historical variance-build report.
        source_scope: Whole admitted family, never a trimmed subset.
        target_catalog: Native private runtime config descriptor.
        target_embedder: Exact target model/capability/connection snapshot.
        representation_sha256: Model/tokenizer/pooling/prompt/dimension profile digest.
        dimension: Expected real target vector width.
        execution_window: Separately registered native finite operational window.
        deadline: Exact stop from that native window.
        rules_revision: Exact separately reviewed discovery/tagging profile, when requested.
        maximum_batch_inputs: Native embedding request input bound.
        maximum_batch_bytes: Native UTF-8 byte bound.
        attempts_per_batch: Finite physical request allowance.
        source_fingerprint: Actual installed source digest.
        runtime_file: Exact maintained executor runtime path, preserved in historical replay.
        runtime_sha256: Exact maintained grounded runtime source bytes.
        producer_revision: Actual producer of new native artifacts.
    """

    previous_binding: ArtifactInput
    original_operation: str
    ledger_sha256: Sha256
    source_build: ArtifactInput
    source_scope: FamilyScope
    target_catalog: RuntimeCatalogBinding
    target_embedder: ModelSnapshot
    representation_sha256: Sha256
    dimension: int = Field(ge=1)
    execution_window: ArtifactInput
    deadline: AwareDatetime
    rules_revision: RulesStageSpec | None = None
    maximum_batch_inputs: int = Field(default=64, ge=1, le=100)
    maximum_batch_bytes: int = Field(default=262144, ge=2048)
    attempts_per_batch: int = Field(default=3, ge=1, le=10)
    source_fingerprint: str
    runtime_file: Path
    runtime_sha256: Sha256
    producer_revision: str


class FamilyEmbeddingPlan(ContractModel):
    """Combined native physical/price inventory, with original authority preserved.

    Attributes:
        intent: Exact pending native intent.
        original_operation: Existing preparation root.
        retained_attempts: Previously charged/retained unpriced claims.
        embedding_inputs: Target cold input count, without any old vector reuse.
        costs: Complete re-embedding and pending preparation components.
        unpriced_attempts: Cumulative retained and pending unpriced allowance.
        physical_pending: Pending physical attempts across all priced/unpriced components.
        known_usd: Existing accounted and known-rate pending amount.
        stop: Native finite stop, never renewed by planning.
        executor_revision: Exact same-intent source/executor compatibility, when required.
    """

    intent: ArtifactInput
    original_operation: str
    retained_attempts: int
    embedding_inputs: int
    costs: tuple[CostComponent, ...]
    unpriced_attempts: int
    physical_pending: int
    known_usd: float
    stop: AwareDatetime
    executor_revision: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class FamilyEmbeddingCompletion(ContractModel):
    """New index/world evidence for only the whole selected family.

    Attributes:
        intent: Native pending migration.
        scope: Same admitted family with only its index/world pointers changed.
        dispatch_evidence: Native physical response records proving target inference occurred.
        producer_revision: Actual new artifact producer, never the historical build producer.
        executor_revision: Explicit reviewed same-intent new executor, when required.
    """

    intent: ArtifactInput
    scope: FamilyScope
    dispatch_evidence: tuple[ArtifactInput, ...] = Field(min_length=1)
    producer_revision: str
    executor_revision: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


def check_rules_revision(old: RulesStageSpec, new: RulesStageSpec, deadline: datetime) -> None:
    """Permit only the requested trajectory-discovery/tag profile and native finite renewal."""
    before, after = old.model_dump(mode="json"), new.model_dump(mode="json")
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if (
        changed - {"profile", "tagging", "memory", "author"}
        or after["profile"] != "trajectory_tags"
    ):
        raise ValueError("rules revision changes unrelated frozen settings")
    if new.author.model_dump(exclude={"deadline"}) != old.author.model_dump(exclude={"deadline"}):
        raise ValueError("rules revision changes its original model or allowance")
    if new.author.deadline not in {old.author.deadline, deadline}:
        raise ValueError("rules revision has an unregistered deadline")
    old_memory = before.get("memory")
    tagging = after.get("tagging")
    if not isinstance(old_memory, dict) or not isinstance(tagging, dict):
        raise ValueError("rules revision needs its exact existing native decision alias")
    if tagging.get("decision_alias") != old_memory.get("decision_alias"):
        raise ValueError("rules revision changes the decision model route")
    if after.get("memory") not in (None, old_memory):
        raise ValueError("rules revision changes unrelated memory controls")


def _executor(
    project: ProjectStore,
    reference: ArtifactInput | None,
    intent: FamilyEmbeddingIntent,
    *,
    historical: bool = False,
) -> FamilyEmbeddingExecutorRevision | None:
    """Require exact native compatibility for a real new executor, never a historical bypass."""
    if reference is None:
        return None
    from exp.optimize.evaluation.variance.family_embedding_executor import (
        load_family_embedding_executor_revision,
    )

    revision = load_family_embedding_executor_revision(project, reference, historical=historical)
    current = artifact_input(project.artifacts.read(stable_id(INTENT_TYPE, intent)).manifest)
    if revision.intent != current:
        raise ValueError("executor revision belongs to another embedding intent")
    return revision


def _context(
    project: ProjectStore,
    intent: FamilyEmbeddingIntent,
    *,
    historical: bool = False,
    executor_revision: ArtifactInput | None = None,
) -> tuple[BootstrapResumeBinding, BootstrapPreparationSpec, VarianceBuildReport]:
    """Verify old authority, exact whole-family evidence and the real native execution window."""
    parent, original = _predecessor(project, intent.previous_binding)
    pilot = parent.pilot_revision
    if pilot is None:
        raise ValueError("family embedding requires an existing admitted pilot")
    report = load_pilot_build(project, pilot)
    assert report.environment_families is not None
    scope = report.environment_families.scope(pilot.family_id)
    if (
        intent.original_operation != parent.original_operation
        or intent.ledger_sha256 != parent.ledger_sha256
        or intent.source_build != pilot.saved_build
        or intent.source_scope != scope
        or sha256_json(scope) != pilot.scope_sha256
    ):
        raise ValueError("family embedding changes original scope, build or operation")
    window = load_author_execution_window(project, intent.execution_window)
    from exp.optimize.evaluation.variance.bootstrap_resume import _execution_spec

    reviewer = _execution_spec(original, None, parent.review_revision).build.reviewer
    assert reviewer is not None
    window.require_settings(reviewer)
    if (
        window.operation_ledger_sha256 != parent.ledger_sha256
        or window.operation_ceiling_usd != parent.ceiling_usd
        or window.operation_unpriced_attempts != parent.unpriced_attempts
        or parent.original_inputs not in window.sources
        or window.execution_deadline != intent.deadline
    ):
        raise ValueError("family embedding window changes original authority")
    executor = _executor(project, executor_revision, intent, historical=historical)
    if not historical and executor is None and intent.source_fingerprint != source_fingerprint():
        raise ValueError("family embedding actual source changed")
    catalog = load_runtime_catalog(project, intent.target_catalog)
    snapshot, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(
        intent.target_catalog.embedder_alias
    )
    if snapshot != intent.target_embedder or capabilities.supports_embeddings is not True:
        raise ValueError("family embedding target snapshot changed")
    if intent.representation_sha256 not in (snapshot.revision or ""):
        raise ValueError("target model revision does not bind its representation profile")
    original_catalog = load_model_catalog(project.model_catalog_path)
    target_record = catalog.models[intent.target_catalog.embedder_alias]
    if catalog.model_dump(exclude={"models", "connections"}) != original_catalog.model_dump(
        exclude={"models", "connections"}
    ) or set(catalog.connections) != set(original_catalog.connections) | {target_record.connection}:
        raise ValueError("runtime catalog changes unrelated catalog controls")
    if (
        any(catalog.models.get(key) != value for key, value in original_catalog.models.items())
        or any(
            catalog.connections.get(key) != value
            for key, value in original_catalog.connections.items()
        )
        or set(catalog.models)
        != set(original_catalog.models) | {intent.target_catalog.embedder_alias}
    ):
        raise ValueError("runtime catalog changes unrelated original model routes")
    from exp.optimize.evaluation.variance.bootstrap_resume import bootstrap_execution_spec

    baseline = bootstrap_execution_spec(project, parent, original)
    if intent.rules_revision is not None:
        assert baseline.bootstrap.rules is not None
        check_rules_revision(baseline.bootstrap.rules, intent.rules_revision, intent.deadline)
    from exp.optimize.evaluation.executable import grounded

    if (
        not historical
        and executor is None
        and (
            intent.runtime_file != Path(grounded.__file__).resolve()
            or intent.runtime_sha256 != sha256_bytes(intent.runtime_file.read_bytes())
        )
    ):
        raise ValueError("family embedding maintained runtime source changed")
    exposure = load_exposure_plan(project, report.exposure)
    exposure.require(scope.members, "world_learning")
    exposure.require(scope.rules_traces, "world_learning")
    return parent, original, report


def propose_family_embedding_intent(
    project: ProjectStore,
    *,
    previous_binding: ArtifactInput,
    target_catalog: RuntimeCatalogBinding,
    representation_sha256: Sha256,
    dimension: int,
    execution_window: ArtifactInput,
    code_revision: str,
    rules_revision: RulesStageSpec | None = None,
    maximum_batch_inputs: int = 64,
    maximum_batch_bytes: int = 262144,
    attempts_per_batch: int = 3,
) -> FamilyEmbeddingIntent:
    """Prepare a concrete exact-family migration without dispatch, publication or new authority."""
    from exp.optimize.evaluation.executable import grounded

    parent, _ = _predecessor(project, previous_binding)
    pilot = parent.pilot_revision
    if pilot is None:
        raise ValueError("family migration needs a saved pilot")
    report = load_pilot_build(project, pilot)
    assert report.environment_families is not None
    catalog = load_runtime_catalog(project, target_catalog)
    model, _ = RuntimeModelCatalog(catalog, environment={}).snapshot(target_catalog.embedder_alias)
    intent = FamilyEmbeddingIntent(
        previous_binding=previous_binding,
        original_operation=parent.original_operation,
        ledger_sha256=parent.ledger_sha256,
        source_build=pilot.saved_build,
        source_scope=report.environment_families.scope(pilot.family_id),
        target_catalog=target_catalog,
        target_embedder=model,
        representation_sha256=representation_sha256,
        dimension=dimension,
        execution_window=execution_window,
        deadline=load_author_execution_window(project, execution_window).execution_deadline,
        rules_revision=rules_revision,
        maximum_batch_inputs=maximum_batch_inputs,
        maximum_batch_bytes=maximum_batch_bytes,
        attempts_per_batch=attempts_per_batch,
        source_fingerprint=source_fingerprint(),
        runtime_file=Path(grounded.__file__).resolve(),
        runtime_sha256=sha256_bytes(Path(grounded.__file__).read_bytes()),
        producer_revision=code_revision,
    )
    _context(project, intent)
    return intent


def register_family_embedding_intent(
    project: ProjectStore, intent: FamilyEmbeddingIntent
) -> ArtifactInput:
    """Append the exact pending intent; do not grant budget or declare a completed stage."""
    _context(project, intent)
    identity = stable_id(INTENT_TYPE, intent)
    if project.artifacts.exists(identity):
        return artifact_input(project.artifacts.read(identity).manifest)
    inputs = (
        intent.previous_binding,
        intent.source_build,
        intent.source_scope.index,
        intent.source_scope.world_model,
        intent.target_catalog.catalog,
        intent.execution_window,
    )
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=INTENT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=intent.producer_revision,
            inputs=tuple(sorted(set(inputs), key=lambda r: r.artifact_id)),
        ),
        files={"intent.json": intent},
    )
    return artifact_input(manifest)


def load_family_embedding_intent(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    historical: bool = False,
    executor_revision: ArtifactInput | None = None,
) -> FamilyEmbeddingIntent:
    """Load an immutable native intent, refusing guessed identities and source changes."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    intent = FamilyEmbeddingIntent.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "intent.json")
    )
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != INTENT_TYPE
        or reference.artifact_id != stable_id(INTENT_TYPE, intent)
        or manifest.code_revision != intent.producer_revision
    ):
        raise ValueError("family embedding intent identity or producer changed")
    expected = tuple(
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
    if manifest.inputs != expected:
        raise ValueError("family embedding intent ancestry changed")
    _context(project, intent, historical=historical, executor_revision=executor_revision)
    return intent


def plan_family_embedding_migration(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    executor_revision: ArtifactInput | None = None,
) -> FamilyEmbeddingPlan:
    """Price every pending native role and exact-family target input under unchanged authority."""
    from exp.optimize.evaluation.variance.bootstrap_plan import _role_costs, _validation_costs
    from exp.optimize.evaluation.variance.bootstrap_resume import bootstrap_execution_spec

    intent = load_family_embedding_intent(project, reference, executor_revision=executor_revision)
    parent, original, _ = _context(project, intent, executor_revision=executor_revision)
    execution = bootstrap_execution_spec(project, parent, original)
    scope = intent.source_scope
    old = load_reembedding_input(project.artifacts, scope.index)
    inputs = plan_rag_embedding_inputs(
        tuple(row.key_text for row in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    )
    checkpoint = RAGEmbeddingCheckpoint(
        project.paths, intent.target_embedder, maximum_chunk_bytes=old.index.embedding_chunk_bytes
    )
    cached = checkpoint.load(inputs.texts)
    pending = tuple(text for text in inputs.texts if text not in cached)
    batches = tuple(
        embedding_batches(
            pending,
            maximum_inputs=intent.maximum_batch_inputs,
            maximum_bytes=intent.maximum_batch_bytes,
        )
    )
    attempts = len(batches) * intent.attempts_per_batch
    catalog = load_runtime_catalog(project, intent.target_catalog)
    costs = [
        catalog_model_cost(
            catalog,
            intent.target_catalog.embedder_alias,
            attempts,
            0,
            input_tokens=max((sum(len(t.encode()) for t in batch) for batch in batches), default=0),
        )
    ]
    settings = execution.bootstrap
    runtime = settings.runtime.model_copy(
        update={
            "world_model": scope.world_model,
            "catalog_binding": intent.target_catalog,
            "catalog_sha256": intent.target_catalog.catalog_sha256,
        }
    )
    rules = intent.rules_revision or settings.rules
    assert rules is not None
    settings = settings.model_copy(update={"runtime": runtime, "rules": rules})
    rows, _ = rules_costs(project, rules, len(scope.rules_traces), len(scope.tools))
    costs.extend(rows)
    costs.extend(_role_costs(project, settings, 1))
    costs.extend(_validation_costs(project, settings, 1))
    summary = operation_spend_ceiling(project, parent.original_operation).summary()
    return FamilyEmbeddingPlan(
        intent=reference,
        original_operation=parent.original_operation,
        retained_attempts=summary.unpriced_attempts,
        embedding_inputs=len(inputs.texts),
        costs=tuple(costs),
        physical_pending=sum(r.attempts for r in costs),
        unpriced_attempts=summary.unpriced_attempts
        + sum(r.attempts for r in costs if r.usd is None),
        known_usd=summary.accounted_usd + sum(r.usd or 0.0 for r in costs),
        stop=intent.deadline,
        executor_revision=executor_revision,
    )


@contextmanager
def admitted_family_embedding_migration(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    executor_revision: ArtifactInput | None = None,
) -> Iterator[FamilyEmbeddingPlan]:
    """Hold original-operation ownership from fresh full admission through materialization."""
    intent = load_family_embedding_intent(project, reference, executor_revision=executor_revision)
    with CoordinationLock(
        project.artifacts.project_directory / (intent.original_operation + ".lock"), timeout=0
    ):
        plan = plan_family_embedding_migration(
            project, reference, executor_revision=executor_revision
        )
        parent, _, _ = _context(project, intent, executor_revision=executor_revision)
        if datetime.now(UTC) >= plan.stop:
            raise ValueError("family embedding native execution window expired")
        if plan.unpriced_attempts > min(
            parent.preparation_unpriced_attempts,
            parent.unpriced_attempts,
            MAXIMUM_UNPRICED_ATTEMPTS,
        ) or plan.known_usd > min(parent.preparation_usd, parent.ceiling_usd):
            raise ValueError(
                "combined family embedding/preparation inventory exceeds original grant"
            )
        yield plan


def admit_family_embedding_migration(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    executor_revision: ArtifactInput | None = None,
) -> FamilyEmbeddingPlan:
    """Check full pending admission without dispatch; execution must reacquire the owner context."""
    with admitted_family_embedding_migration(
        project, reference, executor_revision=executor_revision
    ) as plan:
        return plan


def check_scope_preservation(original: FamilyScope, revised: FamilyScope) -> None:
    """Allow index/world replacement only, preserving every task, member, tag, tool and scope."""
    if original.model_dump(exclude={"index", "world_model"}) != revised.model_dump(
        exclude={"index", "world_model"}
    ):
        raise ValueError("embedding migration changes original family membership or controls")


def validate_family_embedding_completion(
    project: ProjectStore,
    intent: FamilyEmbeddingIntent,
    completion: FamilyEmbeddingCompletion,
    *,
    historical: bool = False,
) -> None:
    """Require actual exact-scope target vectors and native target dispatch provenance."""
    executor = _executor(project, completion.executor_revision, intent, historical=historical)
    producer = intent.producer_revision if executor is None else executor.producer_revision
    if completion.producer_revision != producer:
        raise ValueError("completion does not name its actual admitted executor")
    check_scope_preservation(intent.source_scope, completion.scope)
    old = load_rag_index(project.artifacts, intent.source_scope.index.artifact_id)
    new = load_rag_index(project.artifacts, completion.scope.index.artifact_id)
    if artifact_input(new.manifest) != completion.scope.index:
        raise ValueError("new family index manifest changed")
    if (
        new.index.embedder != intent.target_embedder
        or new.index.embedding_dimension != intent.dimension
        or new.transitions != old.transitions
        or new.index.model_dump(
            exclude={
                "rag_id",
                "created_at",
                "inputs",
                "code_revision",
                "embedder",
                "embedding_dimension",
                "vectors_sha256",
            }
        )
        != old.index.model_dump(
            exclude={
                "rag_id",
                "created_at",
                "inputs",
                "code_revision",
                "embedder",
                "embedding_dimension",
                "vectors_sha256",
            }
        )
    ):
        raise ValueError("new family vectors change admitted evidence or representation")
    before = load_grounded_world_model_artifact(project.artifacts, intent.source_scope.world_model)
    after = load_grounded_world_model_artifact(project.artifacts, completion.scope.world_model)
    omitted = {"world_model_id", "created_at", "inputs", "code_revision", "serving_rag"}
    if after.serving_rag != completion.scope.index or before.model_dump(
        exclude=omitted
    ) != after.model_dump(exclude=omitted):
        raise ValueError("new family world changes predictor, knowledge or controls")
    if (
        new.manifest.code_revision != completion.producer_revision
        or after.code_revision != completion.producer_revision
    ):
        raise ValueError("new family artifacts must retain their actual producer")
    catalog = load_runtime_catalog(project, intent.target_catalog)
    target = catalog.models[intent.target_catalog.embedder_alias]
    wire_model = target.model
    import json

    proved: dict[str, Embedding] = {}
    inventory = plan_rag_embedding_inputs(
        tuple(row.key_text for row in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    )
    text_inventory = {(sha256_bytes(text.encode()), len(text)): text for text in inventory.texts}
    for reference in completion.dispatch_evidence:
        manifest = project.artifacts.read(reference.artifact_id).manifest
        if manifest.artifact_type != "provider-physical-attempt":
            raise ValueError("family embedding needs complete successful native physical responses")
        response = json.loads(project.artifacts.read_bytes(reference.artifact_id, "receipt.json"))
        if (
            artifact_input(manifest) != reference
            or manifest.artifact_type != "provider-physical-attempt"
            or manifest.code_revision != completion.producer_revision
            or not reference.artifact_id.endswith("-response")
            or len(manifest.inputs) != 1
            or response.get("http_status") != 200
            or response.get("returned_after_deadline") is not False
        ):
            raise ValueError("family embedding needs complete successful native physical responses")
        request_ref = manifest.inputs[0]
        request_manifest = project.artifacts.read(request_ref.artifact_id).manifest
        request = json.loads(project.artifacts.read_bytes(request_ref.artifact_id, "receipt.json"))
        body = request.get("request_body")
        if (
            artifact_input(request_manifest) != request_ref
            or request_manifest.artifact_type != "provider-physical-attempt"
            or request_manifest.code_revision != completion.producer_revision
            or request_manifest.inputs != (completion.intent,)
            or request.get("canonical_request") != completion.intent.model_dump(mode="json")
            or not isinstance(body, dict)
            or body.get("model") != wire_model
        ):
            raise ValueError("family embedding dispatch is not bound to this target and intent")
        cited = body.get("input")
        result = response.get("response_body")
        if not isinstance(result, dict) or result.get("model") != (
            target.served_model_id or target.model
        ):
            raise ValueError("embedding response echo differs from the declared served model")
        if (
            not isinstance(cited, list)
            or not isinstance(result, dict)
            or not isinstance(result.get("data"), list)
        ):
            raise ValueError("family embedding response lacks exact input/output records")
        texts: list[str] = []
        for entry in cited:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"text_sha256", "characters"}
                or not isinstance(entry.get("text_sha256"), str)
                or type(entry.get("characters")) is not int
            ):
                raise ValueError("family embedding request has no exact native text citation")
            text = text_inventory.get((entry["text_sha256"], entry["characters"]))
            if text is None:
                raise ValueError("family embedding request cites an unadmitted input")
            texts.append(text)
        if request.get("request_sha256") != sha256_json({**body, "input": texts}):
            raise ValueError("family embedding reconstructed request digest differs")
        values = openai_embedding_response(result, expected_count=len(texts))
        for text, value in zip(texts, values, strict=True):
            if text in proved and proved[text] != value:
                raise ValueError("family embedding native responses conflict")
            proved[text] = value

    class DeliveredEmbeddings:
        """Read only complete native target responses for numerical provenance verification."""

        def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
            if any(text not in proved for text in texts):
                raise ValueError("family embedding input has no native target response")
            return tuple(proved[text] for text in texts)

    verified = embed_rag_texts(
        RAGEmbedderBinding(client=DeliveredEmbeddings(), snapshot=intent.target_embedder),
        tuple(row.key_text for row in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    )
    if verified != tuple(row.values for row in new.vectors):
        raise ValueError("family vector values are not the native target response reduction")


def register_family_embedding_completion(
    project: ProjectStore,
    completion: FamilyEmbeddingCompletion,
) -> ArtifactInput:
    """Publish a genuine family-only completion, without relabeling the original build report."""
    intent = load_family_embedding_intent(
        project, completion.intent, executor_revision=completion.executor_revision
    )
    executor = _executor(project, completion.executor_revision, intent)
    producer = intent.producer_revision if executor is None else executor.producer_revision
    if completion.producer_revision != producer:
        raise ValueError("family embedding completion producer changed")
    validate_family_embedding_completion(project, intent, completion)
    identity = stable_id(COMPLETION_TYPE, completion)
    if project.artifacts.exists(identity):
        return artifact_input(project.artifacts.read(identity).manifest)
    inputs = tuple(
        sorted(
            set(
                (
                    completion.intent,
                    completion.scope.index,
                    completion.scope.world_model,
                    *completion.dispatch_evidence,
                    *(
                        ()
                        if completion.executor_revision is None
                        else (completion.executor_revision,)
                    ),
                )
            ),
            key=lambda r: r.artifact_id,
        )
    )
    manifest = project.artifacts.write_json(
        artifact_id=identity,
        artifact_type=COMPLETION_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=completion.producer_revision,
            inputs=inputs,
        ),
        files={"completion.json": completion},
    )
    return artifact_input(manifest)


def load_family_embedding_completion(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    historical: bool = False,
) -> tuple[FamilyEmbeddingIntent, FamilyEmbeddingCompletion]:
    """Verify the separate family overlay without changing historical build/pilot artifacts."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    completion = FamilyEmbeddingCompletion.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "completion.json")
    )
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != COMPLETION_TYPE
        or stable_id(COMPLETION_TYPE, completion) != reference.artifact_id
        or manifest.code_revision != completion.producer_revision
    ):
        raise ValueError("family embedding completion identity changed")
    expected = tuple(
        sorted(
            set(
                (
                    completion.intent,
                    completion.scope.index,
                    completion.scope.world_model,
                    *completion.dispatch_evidence,
                    *(
                        ()
                        if completion.executor_revision is None
                        else (completion.executor_revision,)
                    ),
                )
            ),
            key=lambda r: r.artifact_id,
        )
    )
    if manifest.inputs != expected:
        raise ValueError("family embedding completion ancestry changed")
    intent = load_family_embedding_intent(
        project,
        completion.intent,
        historical=historical,
        executor_revision=completion.executor_revision,
    )
    validate_family_embedding_completion(project, intent, completion, historical=historical)
    return intent, completion


def apply_family_embedding_spec(
    project: ProjectStore,
    reference: ArtifactInput,
    spec: BootstrapPreparationSpec,
    *,
    historical: bool = False,
) -> BootstrapPreparationSpec:
    """Apply only the exact registered catalog/runtime/rules changes to original pilot settings."""
    intent, completion = load_family_embedding_completion(project, reference, historical=historical)
    executor = _executor(project, completion.executor_revision, intent, historical=historical)
    settings = spec.bootstrap
    runtime = settings.runtime.model_copy(
        update={
            "catalog_binding": intent.target_catalog,
            "catalog_sha256": intent.target_catalog.catalog_sha256,
            "runtime_file": intent.runtime_file if executor is None else executor.runtime_file,
            "runtime_sha256": intent.runtime_sha256
            if executor is None
            else executor.runtime_sha256,
        }
    )
    rules = intent.rules_revision or settings.rules
    if rules is not None:
        rules = rules.model_copy(
            update={"author": rules.author.model_copy(update={"deadline": intent.deadline})}
        )
    return type(spec).model_validate(
        spec.model_copy(
            update={"bootstrap": settings.model_copy(update={"runtime": runtime, "rules": rules})}
        ).model_dump(mode="json")
    )


def validate_family_embedding_successor(
    project: ProjectStore,
    binding: BootstrapResumeBinding,
    parent: BootstrapResumeBinding,
    *,
    historical: bool = False,
) -> None:
    """Bind a genuine completed exact-family migration to unchanged original authority."""
    assert binding.embedding_revision is not None
    intent, completion = load_family_embedding_completion(
        project, binding.embedding_revision, historical=historical
    )
    expected = (set(parent.permitted_changes) - {"preparation_protocol_revision"}) | {
        "family_embedding_revision",
        "execution_watchdog_renewal",
    }
    if (
        intent.previous_binding != binding.previous_binding
        or intent.original_operation != binding.original_operation
        or intent.ledger_sha256 != binding.ledger_sha256
        or binding.producer_revision != completion.producer_revision
        or binding.review_revision != parent.review_revision
        or binding.pilot_revision != parent.pilot_revision
        or binding.build_identity != parent.build_identity
        or binding.execution_window != intent.execution_window
        or binding.stop != intent.deadline
        or binding.protocol_revision is not None
        or set(binding.permitted_changes) != expected
    ):
        raise ValueError(
            "family embedding successor changes unreviewed original authority or settings"
        )


def materialize_family_embedding(
    project: ProjectStore,
    reference: ArtifactInput,
    embedder: RAGEmbedderBinding,
    *,
    executor_revision: ArtifactInput | None = None,
) -> FamilyScope:
    """Run only native exact-family index/world materialization after caller admission.

    The caller owns original-operation admission, spend scope, admitted transport and
    canonical_call(reference). This function neither creates authority nor copies a report.
    """
    from exp.simulation.retrieval.build import persist_trace_rag
    from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
    from exp.simulation.world_model.artifact import persist_grounded_world_model

    intent = load_family_embedding_intent(project, reference, executor_revision=executor_revision)
    _, _, report = _context(project, intent, executor_revision=executor_revision)
    executor = _executor(project, executor_revision, intent)
    producer = intent.producer_revision if executor is None else executor.producer_revision
    if (
        embedder.snapshot != intent.target_embedder
        or embedder.maximum_attempts != intent.attempts_per_batch
        or embedder.maximum_batch_inputs != intent.maximum_batch_inputs
        or embedder.maximum_batch_bytes != intent.maximum_batch_bytes
    ):
        raise ValueError("family embedding client differs from native admitted intent")
    old = load_rag_index(project.artifacts, intent.source_scope.index.artifact_id)
    if embedding_chunk_bytes(embedder.maximum_input_tokens) != old.index.embedding_chunk_bytes:
        raise ValueError("family embedding client changes admitted chunk semantics")
    original = load_grounded_world_model_artifact(
        project.artifacts, intent.source_scope.world_model
    )
    new = persist_trace_rag(
        project.artifacts,
        tuple(source.artifact_input for source in old.index.sources),
        load_completed_build_rag_lineage_bindings(project.artifacts, report.build),
        created_at=datetime.now(UTC),
        code_revision=producer,
        embedder=embedder,
        default_top_k=old.index.default_top_k,
        included_partitions=frozenset(old.index.included_partitions),
        corpus_scope=old.index.corpus_scope,
        trace_tags=old.index.trace_tags,
    )
    world = persist_grounded_world_model(
        project.artifacts,
        artifact_input(new.manifest),
        model_alias=original.model_alias,
        model=original.model,
        created_at=datetime.now(UTC),
        code_revision=producer,
        top_k=original.top_k,
        knowledge=original.knowledge,
        selection_policy_sha256=original.selection_policy_sha256,
        original_evidence=original.original_evidence,
        construction_scope=original.construction_scope,
    )
    return intent.source_scope.model_copy(
        update={
            "index": artifact_input(new.manifest),
            "world_model": artifact_input(world.manifest),
        }
    )

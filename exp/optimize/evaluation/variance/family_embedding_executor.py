"""Reviewed executor successors for one immutable family-embedding intent."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.release_revision import _package_source_digest, installed_release_revision
from exp.optimize.evaluation.executable.catalog_binding import load_runtime_catalog
from exp.optimize.evaluation.factory.author_retries import operation_identity
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.retrieval.embedding_inputs import embedding_batches, plan_rag_embedding_inputs
from exp.simulation.retrieval.reembedding import load_reembedding_input

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.family_embedding import FamilyEmbeddingIntent

REVISION_TYPE = "family-embedding-executor-revision"
OWNER_TYPE = "family-embedding-physical-owner"
_ALLOWED_SOURCE = frozenset(
    {
        "optimize/evaluation/variance/family_embedding.py",
        "optimize/evaluation/variance/family_embedding_executor.py",
    }
)


class SourceDelta(ContractModel):
    """One exact reviewed production-file difference."""

    before: Sha256 | None
    after: Sha256 | None


class FamilyEmbeddingExecutorRevision(ContractModel):
    """Source/executor compatibility only: the original intent and all authority stay fixed.

    Attributes:
        protocol: Version of this narrow compatibility contract.
        intent: Exact original embedding intent, never copied or replaced.
        original_operation: Existing preparation operation.
        ledger_sha256: Existing accounting identity.
        grants_sha256: Exact existing authorization history.
        previous_producer_revision: Original intent producer.
        producer_revision: Actual executor producing new artifacts.
        previous_source_sha256: Original frozen package fingerprint.
        source_sha256: Reviewed new package fingerprint.
        changes: Exact closed production-source differences.
        runtime_file: Actual new executor runtime file.
        runtime_sha256: Unchanged grounded-runtime bytes.
        physical_ledger: Original physical child, reopened without creation.
        physical_cap: Its unchanged cumulative cap.
        phase_config_sha256: Its original intent digest.
        retained_attempts: Original charged physical prefix.
        wire_contract: Canonical native request-model semantics; response echoes stay separate.
    """

    protocol: Literal["family-embedding-executor-v2"] = "family-embedding-executor-v2"
    intent: ArtifactInput
    original_operation: str
    ledger_sha256: Sha256
    grants_sha256: Sha256
    previous_producer_revision: str
    producer_revision: str
    previous_source_sha256: Sha256
    source_sha256: Sha256
    changes: dict[str, SourceDelta]
    runtime_file: Path
    runtime_sha256: Sha256
    physical_ledger: Path
    physical_cap: int = Field(ge=1, le=100000)
    phase_config_sha256: Sha256
    retained_attempts: tuple[str, ...] = Field(min_length=1)
    wire_contract: Literal["native-openai-embedding-record-model-v1"] = (
        "native-openai-embedding-record-model-v1"
    )


class FamilyEmbeddingPhysicalOwner(ContractModel):
    """One immutable original child owner across every executor revision of an intent."""

    protocol: Literal["family-embedding-physical-owner-v1"] = "family-embedding-physical-owner-v1"
    intent: ArtifactInput
    original_operation: str
    ledger_sha256: Sha256
    physical_ledger: Path
    physical_cap: int = Field(ge=1, le=100000)
    phase_config_sha256: Sha256
    retained_attempts: tuple[str, ...] = Field(min_length=1)
    producer_revision: str


def _owner_id(intent: ArtifactInput) -> str:
    """Bind original physical ownership to the intent, independently of source revisions."""
    return stable_id(OWNER_TYPE, intent)


def _owner(project: ProjectStore, intent: ArtifactInput) -> FamilyEmbeddingPhysicalOwner | None:
    """Read the exact immutable original owner, without admitting another child path."""
    identity = _owner_id(intent)
    if not project.artifacts.exists(identity):
        return None
    stored = project.artifacts.read(identity)
    value = FamilyEmbeddingPhysicalOwner.model_validate_json(stored.payloads["owner.json"])
    if (
        stored.manifest.artifact_type != OWNER_TYPE
        or stored.manifest.artifact_id != identity
        or stored.manifest.code_revision != value.producer_revision
        or stored.manifest.inputs != (intent,)
        or value.intent != intent
        or value.physical_ledger != value.physical_ledger.resolve()
    ):
        raise ValueError("original embedding physical owner identity or provenance changed")
    return value


def _require_owner(
    project: ProjectStore,
    revision: FamilyEmbeddingExecutorRevision,
) -> FamilyEmbeddingPhysicalOwner | None:
    """Reject copied/backup counters, even with identical registration and old claim prefix."""
    owner = _owner(project, revision.intent)
    if owner is not None and (
        owner.original_operation != revision.original_operation
        or owner.ledger_sha256 != revision.ledger_sha256
        or owner.physical_ledger != revision.physical_ledger.resolve()
        or owner.physical_cap != revision.physical_cap
        or owner.phase_config_sha256 != revision.phase_config_sha256
        or owner.retained_attempts != revision.retained_attempts
    ):
        raise ValueError("executor revision changes the original physical owner")
    return owner


def _bind_owner(project: ProjectStore, revision: FamilyEmbeddingExecutorRevision) -> None:
    """Register once while the caller holds the original operation's native lock."""
    if _require_owner(project, revision) is not None:
        return
    owner = FamilyEmbeddingPhysicalOwner(
        intent=revision.intent,
        original_operation=revision.original_operation,
        ledger_sha256=revision.ledger_sha256,
        physical_ledger=revision.physical_ledger.resolve(),
        physical_cap=revision.physical_cap,
        phase_config_sha256=revision.phase_config_sha256,
        retained_attempts=revision.retained_attempts,
        producer_revision=revision.producer_revision,
    )
    project.artifacts.write_json(
        artifact_id=_owner_id(revision.intent),
        artifact_type=OWNER_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision=revision.producer_revision,
            inputs=(revision.intent,),
        ),
        files={"owner.json": owner},
    )


def _files(package: Path) -> dict[str, Sha256]:
    """Inventory the same production files included by the native package fingerprint."""
    result = {}
    ignored = {"__pycache__", "tests", "target", "node_modules"}
    for directory, directories, names in package.walk():
        directories[:] = [
            name for name in directories if name not in ignored and not name.startswith(".")
        ]
        for name in names:
            path = directory / name
            if name.startswith(".") or name.endswith("_test.py") or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("executor compatibility requires regular immutable source files")
            result[path.relative_to(package).as_posix()] = sha256_bytes(path.read_bytes())
    return result


def _changes(before: Path, after: Path) -> dict[str, SourceDelta]:
    old, new = _files(before), _files(after)
    return {
        name: SourceDelta(before=old.get(name), after=new.get(name))
        for name in sorted(old.keys() | new.keys())
        if old.get(name) != new.get(name)
    }


def _original(project: ProjectStore, reference: ArtifactInput) -> FamilyEmbeddingIntent:
    """Read immutable historical inputs intrinsically; this does not admit current execution."""
    from exp.optimize.evaluation.variance.family_embedding import (
        INTENT_TYPE,
        FamilyEmbeddingIntent,
        _context,
    )

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
        raise ValueError("executor revision original intent identity changed")
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
            key=lambda row: row.artifact_id,
        )
    )
    if manifest.inputs != expected:
        raise ValueError("executor revision original intent ancestry changed")
    _context(project, intent, historical=True)
    return intent


def _physical_cap(project: ProjectStore, intent: FamilyEmbeddingIntent) -> int:
    old = load_reembedding_input(project.artifacts, intent.source_scope.index)
    inputs = plan_rag_embedding_inputs(
        tuple(row.key_text for row in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    )
    return (
        sum(
            1
            for _ in embedding_batches(
                inputs.texts,
                maximum_inputs=intent.maximum_batch_inputs,
                maximum_bytes=intent.maximum_batch_bytes,
            )
        )
        * intent.attempts_per_batch
    )


def _retained_request(project: ProjectStore, intent: FamilyEmbeddingIntent, attempt: str) -> None:
    """Bind retained physical metadata to this exact admitted input corpus and declared wire."""
    import json

    catalog = load_runtime_catalog(project, intent.target_catalog)
    target = catalog.models[intent.target_catalog.embedder_alias]
    connection = catalog.connections[target.connection]
    receipt = json.loads(
        project.artifacts.read_bytes("provider-attempt-" + attempt + "-intent", "receipt.json")
    )
    body = receipt.get("request_body")
    if (
        connection.provider != "openai-compatible"
        or not connection.base_url
        or not isinstance(body, dict)
        or body.get("model") != target.model
        or receipt.get("endpoint_sha256")
        != sha256_json({"url": connection.base_url.rstrip("/") + "/embeddings"})
    ):
        raise ValueError("executor revision retained request uses another target service or model")
    old = load_reembedding_input(project.artifacts, intent.source_scope.index)
    inputs = plan_rag_embedding_inputs(
        tuple(row.key_text for row in old.transitions),
        maximum_chunk_bytes=old.index.embedding_chunk_bytes,
    )
    inventory = {(sha256_bytes(text.encode()), len(text)): text for text in inputs.texts}
    cited = body.get("input")
    if not isinstance(cited, list):
        raise ValueError("executor revision retained request has no exact native input inventory")
    texts = []
    for entry in cited:
        if not isinstance(entry, dict):
            raise ValueError("executor revision retained input lacks native citation metadata")
        text = inventory.get((entry.get("text_sha256"), entry.get("characters")))
        if text is None:
            raise ValueError("executor revision retained input is outside its admitted scope")
        texts.append(text)
    if receipt.get("request_sha256") != sha256_json({**body, "input": texts}):
        raise ValueError("executor revision retained request cannot be reconstructed exactly")


def _validate(
    project: ProjectStore,
    revision: FamilyEmbeddingExecutorRevision,
    *,
    historical: bool = False,
) -> FamilyEmbeddingIntent:
    from exp.optimize.evaluation.executable import grounded

    intent = _original(project, revision.intent)
    _require_owner(project, revision)
    old_package = intent.runtime_file.parents[3]
    new_package = Path(__file__).resolve().parents[3]
    ledger = operation_spend_ceiling(project, intent.original_operation)
    summary = ledger.summary()
    if (
        revision.original_operation != intent.original_operation
        or revision.ledger_sha256 != intent.ledger_sha256
        or operation_identity(ledger) != intent.ledger_sha256
        or revision.grants_sha256 != sha256_json(summary.model_dump(mode="json")["authorizations"])
        or revision.previous_producer_revision != intent.producer_revision
        or revision.producer_revision == intent.producer_revision
        or revision.previous_source_sha256 != intent.source_fingerprint
        or _package_source_digest(old_package) != intent.source_fingerprint
        or set(revision.changes) != _ALLOWED_SOURCE
        or revision.runtime_sha256 != intent.runtime_sha256
        or sha256_bytes(revision.runtime_file.read_bytes()) != intent.runtime_sha256
        or revision.physical_cap != _physical_cap(project, intent)
        or revision.phase_config_sha256 != sha256_json(intent)
    ):
        raise ValueError(
            "executor revision changes original inputs, source, accounting or physical cap"
        )
    if not historical and (
        revision.producer_revision != installed_release_revision()
        or revision.source_sha256 != _package_source_digest(new_package)
        or revision.changes != _changes(old_package, new_package)
        or revision.runtime_file != Path(grounded.__file__).resolve()
        or sha256_bytes(Path(grounded.__file__).read_bytes()) != intent.runtime_sha256
    ):
        raise ValueError("executor revision differs from the exact reviewed current source")
    child = DispatchLedger(
        revision.physical_ledger,
        revision.physical_cap,
        revision.phase_config_sha256,
        create=False,
    )
    import sqlite3

    with sqlite3.connect(f"file:{revision.physical_ledger}?mode=ro", uri=True) as database:
        actual = tuple(
            row[0]
            for row in database.execute("SELECT attempt_id FROM invocation_slots ORDER BY seq")
        )
    if actual[: len(revision.retained_attempts)] != revision.retained_attempts:
        raise ValueError("executor revision changes the retained physical claim prefix")
    if child.used() != len(actual):
        raise ValueError("executor revision physical accounting changed")
    ledger.require_accounted_attempts(actual)
    for attempt in revision.retained_attempts:
        manifest = project.artifacts.read("provider-attempt-" + attempt + "-intent").manifest
        if (
            manifest.inputs != (revision.intent,)
            or manifest.code_revision != intent.producer_revision
        ):
            raise ValueError("executor revision retains physical claims from another intent")
        _retained_request(project, intent, attempt)
    return intent


def propose_family_embedding_executor_revision(
    project: ProjectStore,
    *,
    intent: ArtifactInput,
    physical_ledger: Path,
    code_revision: str,
) -> FamilyEmbeddingExecutorRevision:
    """Describe one concrete same-intent/same-ledger reviewed executor; no dispatch or writes."""
    import sqlite3

    from exp.optimize.evaluation.executable import grounded

    original = _original(project, intent)
    physical_ledger = physical_ledger.resolve()
    owner = _owner(project, intent)
    if owner is not None and owner.physical_ledger != physical_ledger:
        raise ValueError("executor proposal changes the original physical owner")
    with sqlite3.connect(f"file:{physical_ledger}?mode=ro", uri=True) as database:
        cap, phase = database.execute("SELECT cap,phase_config_sha256 FROM registration").fetchone()
        current = tuple(
            row[0]
            for row in database.execute("SELECT attempt_id FROM invocation_slots ORDER BY seq")
        )
        retained = current if owner is None else owner.retained_attempts
    ledger = operation_spend_ceiling(project, original.original_operation)
    revision = FamilyEmbeddingExecutorRevision(
        intent=intent,
        original_operation=original.original_operation,
        ledger_sha256=original.ledger_sha256,
        grants_sha256=sha256_json(ledger.summary().model_dump(mode="json")["authorizations"]),
        previous_producer_revision=original.producer_revision,
        producer_revision=code_revision,
        previous_source_sha256=original.source_fingerprint,
        source_sha256=_package_source_digest(Path(__file__).resolve().parents[3]),
        changes=_changes(original.runtime_file.parents[3], Path(__file__).resolve().parents[3]),
        runtime_file=Path(grounded.__file__).resolve(),
        runtime_sha256=original.runtime_sha256,
        physical_ledger=physical_ledger,
        physical_cap=cap,
        phase_config_sha256=phase,
        retained_attempts=retained,
    )
    _validate(project, revision)
    return revision


def register_family_embedding_executor_revision(
    project: ProjectStore,
    revision: FamilyEmbeddingExecutorRevision,
) -> ArtifactInput:
    """Append exact compatibility under the caller-held original-operation lock.

    The same lock must cover proposal and registration. The deterministic original-owner
    artifact is also published atomically, so a competing copied counter never replaces it.
    """
    _validate(project, revision)
    _bind_owner(project, revision)
    _require_owner(project, revision)
    owner_ref = artifact_input(project.artifacts.read(_owner_id(revision.intent)).manifest)
    identity = stable_id(REVISION_TYPE, revision)
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=REVISION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision.producer_revision,
                inputs=(revision.intent, owner_ref),
            ),
            files={"revision.json": revision},
        )
    reference = artifact_input(project.artifacts.read(identity).manifest)
    load_family_embedding_executor_revision(project, reference)
    return reference


def load_family_embedding_executor_revision(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    historical: bool = False,
) -> FamilyEmbeddingExecutorRevision:
    """Require exact immutable compatibility and current source for every real execution."""
    manifest = project.artifacts.read(reference.artifact_id).manifest
    revision = FamilyEmbeddingExecutorRevision.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "revision.json")
    )
    owner = _owner(project, revision.intent)
    if owner is None:
        raise ValueError("executor revision has no original physical owner registration")
    owner_ref = artifact_input(project.artifacts.read(_owner_id(revision.intent)).manifest)
    if (
        artifact_input(manifest) != reference
        or manifest.artifact_type != REVISION_TYPE
        or reference.artifact_id != stable_id(REVISION_TYPE, revision)
        or manifest.code_revision != revision.producer_revision
        or manifest.inputs != (revision.intent, owner_ref)
    ):
        raise ValueError("executor revision artifact identity or provenance changed")
    _validate(project, revision, historical=historical)
    return revision

"""Disclosure classifications and qualification exclusions for exact immutable evidence."""

from datetime import UTC, datetime
from typing import Literal

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.core.locks import CoordinationLock
from exp.common.project import ArtifactStore, artifact_input


class SourceAccess(ContractModel):
    """Operator classification of one exact immutable evidence source.

    Attributes:
        source: Exact artifact whose complete contents have been classified.
        scope: Development permits diagnostics; protected always refuses disclosure.
        rationale: Operator's reason, without copying protected evidence.
    """

    source: ArtifactInput
    scope: Literal["development", "protected"]
    rationale: str


class QualificationExclusion(ContractModel):
    """Exclude exact evidence from qualification while retaining diagnostic access.

    Attributes:
        source: Immutable evidence whose qualification is excluded.
        audit: Immutable audit supporting the operator's exclusion.
        rationale: Reason for exclusion, without copying source contents.
    """

    source: ArtifactInput
    audit: ArtifactInput
    rationale: str


def exclude_source_from_qualification(
    store: ArtifactStore,
    source: ArtifactInput,
    *,
    audit: ArtifactInput,
    rationale: str,
    code_revision: str,
) -> ArtifactInput:
    """Record a permanent qualification exclusion, separately from disclosure policy.

    This does not protect source contents from diagnostic access or approve any other source.
    Requalification requires different immutable evidence, not changing this declaration.
    """
    if not rationale.strip():
        raise ValueError("qualification exclusion requires an operator rationale")
    for reference in (source, audit):
        if artifact_input(store.read(reference.artifact_id).manifest) != reference:
            raise ValueError("qualification exclusion reference identity differs")
    identity = _exclusion_identity(source)
    with CoordinationLock(store.project_directory / "source-access.lock", timeout=0):
        if not store.exists(identity):
            store.write_json(
                artifact_id=identity,
                artifact_type="evidence-qualification-exclusion",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=tuple(sorted((source, audit), key=lambda ref: ref.artifact_id)),
                ),
                files={
                    "exclusion.json": QualificationExclusion(
                        source=source, audit=audit, rationale=rationale
                    ).model_dump(mode="json")
                },
            )
        existing = _read_exclusion(store, source)
        if existing.audit != audit or existing.rationale != rationale:
            raise ValueError("qualification exclusion already records a different immutable audit")
        return artifact_input(store.read(identity).manifest)


def refuse_known_ineligible_sources(
    store: ArtifactStore, sources: tuple[ArtifactInput, ...], *, maximum_artifacts: int
) -> None:
    """Refuse known exclusions on verified roots and ancestors before qualification.

    Unknown classification is not approval. Disclosure, development admission and memory
    knowledge-scope validation remain separate checks owned by their existing callers.
    Verified artifact snapshots may load payload bytes; source content is not semantically
    decoded or admitted for qualification by this ancestry check.
    """
    if maximum_artifacts < 1:
        raise ValueError("qualification ancestry bound must be positive")
    pending = list(sources)
    seen: set[ArtifactInput] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        if len(seen) > maximum_artifacts:
            raise ValueError("qualification sources exceed the complete dependency bound")
        manifest = store.read(source.artifact_id).manifest
        if artifact_input(manifest) != source:
            raise ValueError("qualification source dependency identity differs")
        if store.exists(_exclusion_identity(source)):
            _read_exclusion(store, source)
            raise ValueError(
                f"source excluded from qualification: {source.artifact_id}; "
                "retain it for diagnostics and prepare independently qualified evidence"
            )
        pending.extend(manifest.inputs)


def _read_exclusion(store: ArtifactStore, source: ArtifactInput) -> QualificationExclusion:
    """Verify the exact immutable exclusion and its audit reference."""
    identity = _exclusion_identity(source)
    declaration = store.read(identity).manifest
    exclusion = QualificationExclusion.model_validate_json(
        store.read_bytes(identity, "exclusion.json")
    )
    if (
        declaration.artifact_type != "evidence-qualification-exclusion"
        or exclusion.source != source
        or not exclusion.rationale.strip()
        or declaration.inputs
        != tuple(sorted((source, exclusion.audit), key=lambda ref: ref.artifact_id))
        or artifact_input(store.read(exclusion.audit.artifact_id).manifest) != exclusion.audit
    ):
        raise ValueError("qualification exclusion declaration differs")
    return exclusion


def _exclusion_identity(source: ArtifactInput) -> str:
    """Bind exclusion lookup to exact evidence, independently of diagnostic classification."""
    return stable_id("source-qualification-exclusion", {"source": source.model_dump(mode="json")})


def declare_source_access(
    store: ArtifactStore,
    source: ArtifactInput,
    *,
    scope: Literal["development", "protected"],
    rationale: str,
    code_revision: str,
) -> ArtifactInput:
    """Record an explicit classification; a protected source cannot later be declassified.

    Classifications are trusted operator declarations, not automatic content inspection.
    They confer no authority on a source's dependencies, which need their own declarations.
    """
    with CoordinationLock(store.project_directory / "source-access.lock", timeout=0):
        return _declare_source_access(
            store, source, scope=scope, rationale=rationale, code_revision=code_revision
        )


def _declare_source_access(
    store: ArtifactStore,
    source: ArtifactInput,
    *,
    scope: Literal["development", "protected"],
    rationale: str,
    code_revision: str,
) -> ArtifactInput:
    """Publish a classification while exclusively owning the disclosure boundary."""
    if not rationale.strip():
        raise ValueError("source classification requires an operator rationale")
    if artifact_input(store.read(source.artifact_id).manifest) != source:
        raise ValueError("source classification identity differs")
    protected = _identity(source, "protected")
    if scope == "development" and store.exists(protected):
        raise ValueError("protected source cannot be admitted for diagnostics")
    identity = _identity(source, scope)
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="evidence-source-access",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(source,),
            ),
            files={
                "access.json": SourceAccess(
                    source=source, scope=scope, rationale=rationale
                ).model_dump(mode="json")
            },
        )
    return artifact_input(store.read(identity).manifest)


def require_development_sources(
    store: ArtifactStore, sources: tuple[ArtifactInput, ...], *, maximum_artifacts: int
) -> tuple[ArtifactInput, ...]:
    """Verify every ancestor and its development declaration before reading model input.

    Unknown and protected sources refuse even when reached through an admitted root.
    Protection takes precedence over an earlier development declaration. No source text is
    included in errors, and no provider or credential is resolved here.
    """
    pending = list(sources)
    seen: set[ArtifactInput] = set()
    declarations: set[ArtifactInput] = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        if len(seen) > maximum_artifacts:
            raise ValueError("source admission exceeds the complete dependency bound")
        manifest = store.read(source.artifact_id).manifest
        if artifact_input(manifest) != source:
            raise ValueError("diagnostic source dependency identity differs")
        if store.exists(_identity(source, "protected")):
            raise ValueError(f"protected diagnostic source: {source.artifact_id}")
        identity = _identity(source, "development")
        if not store.exists(identity):
            raise ValueError(
                f"unclassified diagnostic source: {source.artifact_id}; classify this exact "
                "source with declare_source_access before diagnostic execution"
            )
        declaration = store.read(identity).manifest
        access = SourceAccess.model_validate_json(store.read_bytes(identity, "access.json"))
        if (
            declaration.artifact_type != "evidence-source-access"
            or declaration.inputs != (source,)
            or access.source != source
            or access.scope != "development"
        ):
            raise ValueError("diagnostic source classification differs")
        declarations.add(artifact_input(declaration))
        pending.extend(manifest.inputs)
    return tuple(sorted(declarations, key=lambda ref: ref.artifact_id))


def _identity(source: ArtifactInput, scope: Literal["development", "protected"]) -> str:
    """Bind classification to exact content, never an artifact name alone."""
    return stable_id("source-access", {"source": source.model_dump(mode="json"), "scope": scope})

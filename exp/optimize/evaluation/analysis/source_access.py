"""Explicit, immutable source classifications before diagnostic evidence leaves its store."""

from datetime import UTC, datetime
from typing import Literal

from filelock import FileLock

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
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
    with FileLock(store.project_directory / "source-access.lock", timeout=0):
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

# Portable Project bundles

Experiential can export one Project's selected durable state as a deterministic bundle and restore it under
a different local Experiential root. The supported package-root API is limited to
`export_project_bundle`, `restore_project_bundle`, and `ExportedProjectBundle`.

## Selected immutable state

The bundle contains `project.json` plus the exact transitive closure of every `ArtifactInput`
selected by the Project configuration. Each artifact directory contains its verified
`manifest.json` and the complete file set named by that manifest. Unselected artifacts that happen
to exist below the same source root are not included.

Artifact manifests remain authoritative for source identity, producer revision, dependency
lineage, schema version, and payload digests. The bundle manifest does not copy those fields into a
second provenance record. It binds the Project identity and schema, sorted selected pointers,
completed durable stages, every member digest and expanded size, the bundle producer revision, and
the explicit value `runtime_state = "excluded"`.

An unchanged Project state and unchanged bundle producer revision produce identical bytes and the
same SHA-256 digest. Callers should retain that digest next to the stored bundle and pass it as
`expected_sha256` during restore.

## Project-scoped model catalog

A provider-free Project does not need model metadata. When a later selected stage needs model
metadata, `ProjectConfig.model_catalog` points to one immutable `project-model-catalog` artifact
whose aliases bind secret-free model and capability snapshots. Model roles and that pointer must be
selected together.

Export and restore use only this explicit Project pointer. They never read, infer, copy, or restore
the root-global `models.toml` file. Catalog artifacts contain neither credentials nor Platform
connection identifiers.

## Restore boundary

Restore first verifies the caller-supplied bundle digest, canonical regular-file archive metadata,
portable relative paths, member and artifact digests, Project schema, selected closure, durable
source identities, catalog bindings, and hard size limits. It rejects symlinks, traversal,
duplicate or case-colliding names, compression, unsupported schemas, secret-bearing content, local
absolute paths, extra artifacts, and incomplete content.

Verified state is materialized beneath a private staging root and becomes visible only by renaming
the completed Project directory into an absent `projects/<project_id>` destination. Failure removes
the staging root and leaves no partially selected Project.

## Stage events and runtime state

Transport-neutral Project events use the stages `preparing_traces`, `building_world_model`,
`optimizing_router`, and `completing_report`. Their event kinds are `started`, bounded `progress`,
`completed`, and `failed`. Completion carries exact immutable output pointers; failure carries a
typed redacted code and retryability, not provider text. Experiential defines these domain records but does
not provide an event bus, persistence service, or delivery guarantee.

The bundle contains completed immutable build state only. A currently running operation belongs to
the hosting system's job record. The mutable routed-interaction journal under the Project runtime
directory is a separate serving concern: restore does not create it, and serving may attach or
start runtime state only after the immutable Project has been verified and restored.

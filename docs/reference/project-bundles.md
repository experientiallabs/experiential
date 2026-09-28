# Portable Project bundles

Experiential can export one Project's selected durable state as a deterministic bundle and restore it under
a different local Experiential root. The supported package-root API is limited to
`export_project_bundle`, `restore_project_bundle`, and `ExportedProjectBundle`.

## Selected immutable state

The bundle contains `project.json` plus the exact transitive closure of every `ArtifactInput`
selected by the Project configuration. Each archive artifact directory contains its verified
`manifest.json` and the complete file set named by that manifest. Unselected artifact records in the source database are not included.

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

Verified state is materialized in a private staging database. Referenced large files are published
before one SQLite transaction installs the project's configuration and artifact records in
`gateway/traffic.db`. Other projects, capture rows, and trace imports are unchanged. A failed commit
rolls back the selected project records. A durable SQLite intent binds any published files to the
exact bundle digest, so a retry verifies those bytes and completes the same restore. No partial
project is selected, and a different bundle cannot adopt the interrupted publication.

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

## Upgrading folder-based projects to 0.8

Experiential 0.8 rejects project roots containing the old `project.toml` or `artifacts` layout.
It does not delete that evidence, automatically migrate it, or run parallel readers for both
formats. Keep the existing root and its matching Experiential release available. Use that release
to export the selected completed Project, retain the bundle's SHA-256 digest, and restore with 0.8
into a fresh root. Finish interrupted legacy runs with their matching release before exporting;
the bundle does not transfer active checkpoints or in-flight request accounting.

Copying only `projects/<project-id>` is insufficient for a SQLite-backed Project. A full workspace
backup must preserve a consistent `gateway/traffic.db` snapshot and its referenced project blobs,
along with any other local state required by the running application. Stop writers before copying
the workspace, or use SQLite's online backup API with coordinated blob retention. Do not copy a
live database file alone while its committed state may still be in the WAL. A completed Project
bundle is the supported portable export for the selected immutable graph.

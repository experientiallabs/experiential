# Catalog manifests and per-model artifacts

A gateway catalog can be delivered in two forms that serve identically:

1. The whole authored catalog document (`ModelCatalog`), read through
   `exp.common.models.gateway_catalog`. This path is unchanged.
2. A **manifest** (`CatalogManifest`): one consistent cut that maps each model
   id to the sha256 of one **model artifact** (`ModelArtifact`). An artifact is a
   self-contained slice of the same document: the connections, deployment
   records, certified pools and model chains that one model needs.

`exp.common.models.catalog_manifest` is the reader of the second form. It never
writes artifacts, and it never re-normalizes a document or re-derives an
identity to verify it. The capability is advertised as
`MANIFEST_READER_VERSION` (currently `1`) so a host can tell which builds read
manifests.

## Verification

`read_model_artifact(data, artifact_sha256=...)` checks, in order:

| Check | Refusal reason |
|---|---|
| sha256 of the exact stored bytes equals the manifest's hash | `digest_mismatch` |
| the bytes are a JSON object | `unreadable` |
| `artifact_schema_version` is an integer in `SUPPORTED_MODEL_ARTIFACT_SCHEMA_VERSIONS` (the current schema and, once one exists, the previous) | `unsupported_schema` |
| every name in `required_to_serve` is in `KNOWN_SERVING_FEATURES` | `unknown_required_feature` |
| the forward-compatible parse validates | `unreadable` |

Unknown optional fields are dropped and reported in
`LoadedModelArtifact.dropped_fields`. A writer that adds a field an older
reader must not silently drop (for example one that changes what a request
costs) names a feature in `required_to_serve`; an older reader then refuses the
artifact instead of serving it wrongly.

`read_catalog_manifest` parses a manifest forward-compatibly and refuses a
manifest schema marker that is not the integer `CATALOG_MANIFEST_SCHEMA_VERSION`.

## Dependencies

Each artifact lists `dependencies`: `(kind, key, sha256)` triples naming what it
was compiled from. Kinds in use are `model`, `account`, `name` and `card`.

- `model` dependencies are verified: each must name a manifest entry carrying
  exactly that artifact hash, and every model-reference rung in the artifact's
  chains must be declared as one. A chain that enters another model therefore
  always sees the version of that model it was compiled against.
- Other kinds are opaque to the reader. They are part of the artifact bytes,
  so a change to any of them changes the artifact hash.

## Assembly

`assemble_manifest_catalog(manifest, artifacts)` merges the verified artifacts
into one `ModelCatalog` carrying `manifest.catalog_schema_version`. Problems
with one entry never fail the whole manifest; they become `ArtifactRefusal`
rows that a host reports (for example as a negative acknowledgement):

| Reason | Meaning |
|---|---|
| `missing_bytes` | the artifact was not loaded |
| `model_mismatch` | the artifact's `model_id` is not the entry's, it carries no deployment or chain, or a record, pool or chain in it belongs to another model (a deployment belongs to its pool's exact model, else its `gateway.exact_model_id`, else its own alias) |
| `dependency_mismatch` | a `model` dependency is absent or at another hash |
| `connection_conflict` | a connection name is defined differently by several artifacts (every definer is refused; identical copies merge) |
| `deployment_conflict`, `pool_conflict`, `chain_conflict` | two artifacts claim one deployment alias, pool id or chain key (both are refused) |
| `invalid_artifact` | the artifact, together with the models it depends on, fails catalog validation or normalization (for example a chain naming a pool it does not carry, or a provider that needs explicit capabilities without them) |
| `dependency_refused` | a model this artifact depends on was refused (closed to a fixpoint) |

The merged survivors are validated and normalized once. Only when that fails
is each survivor validated with its dependency closure, in dependency order,
so a malformed artifact is refused alone and never aborts the manifest.
Results are deterministic (entries are processed by model id). The catalog is
`None` only when no connection survives. Assembling a manifest written from a
catalog yields an equal `ModelCatalog`, so its normalized identity equals the
whole document's; `CatalogManifest.catalog_sha256` records that identity when
the cut reproduces one.

`ParsedArtifactCache` keeps parsed artifacts by hash across manifest versions
(the bytes in hand are hashed on every `load`, so changed bytes for a cached
hash are refused as `digest_mismatch`):
an unchanged model keeps its hash, so it is parsed once and shared, and
`retain` evicts hashes no held manifest names.

## Manifest pins

A request pins one manifest version. `CatalogRouteResolver` accepts
`manifest_pins`, a mapping from `(revision_id, catalog_sha256)` keys to manifest
versions (also on `swap_catalogs`). A pinned key was assembled by the host from
byte-verified artifacts, so it is indexed without the identity check that every
other same-schema key still gets; the version is stored on the
key's indexed view, so every `GatewayRoute` resolved on it carries
`manifest_version` from the same view it was resolved against, and
`manifest_version(revision_id, catalog_sha256)` reads that view. Without pins the
resolver behaves exactly as before.

The pin is a host assertion: assembly never re-derives the assembled
catalog's identity, so it cannot prove the cut reproduces the key's digest (a
cut that holds or canaries a model intentionally does not). A host pins only
keys it assembled itself from a manifest it verified.

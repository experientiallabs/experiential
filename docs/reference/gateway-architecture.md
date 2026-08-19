# Gateway foundation architecture contract

## Current scope

This reference describes the inert contracts available on main after the gateway foundation
change. It does not claim that a gateway server, identity store, provider stream, public alias,
or no-argument `wmo run` mode exists yet.

The foundation establishes one vocabulary for later independently reviewed implementations:

- a public gateway alias targets either one exact-model pool or one immutable project activation
- a project resolver selects one exact logical model, and later operational execution may choose
  only among deployments certified for that model
- authentication, grants, persistence, provider execution, and wire encoding depend on injected
  interfaces instead of importing optimizer owners into runtime
- tool calls keep their existing parsed argument object and may also retain exact provider JSON
- provider stream events carry raw tool-argument fragments without requiring each fragment to be
  valid JSON

No root CLI set, endpoint set, release claim, or provider behavior changes in this foundation.

## Ownership

`wmo/common/models/catalog.py` remains the sole authored provider and model catalog. A model record
may carry optional gateway-only protocol capabilities, exact-model certification, and integer
attribution prices. These fields are deployment metadata. They do not enter
`ModelCapabilities`, whose existing identity digest remains frozen for compatibility with built
router artifacts.

`wmo/common/models/gateway_catalog.py` derives an immutable secret-free deployment view. Existing
model aliases become separate singleton pools. Runtime gateway contracts and injected interfaces
live under `wmo/runtime/gateway/`. Runtime continues to be forbidden from importing simulation,
optimization, or CLI modules by `wmo/repo_import_boundaries_test.py`.

## Conservative singleton migration

The derived identity for an existing model includes:

1. the normalized secret-free connection identity digest
2. the exact provider model spelling and optional revision
3. a digest of the complete authored capability declaration

Capability variants and connections to different compatible endpoints therefore do not collapse
into one exact model. Separate aliases remain separate singleton pools even when all three inputs
are identical. Grouping them later requires a deliberate equivalence contract.

Provider `tinker` records and any model carrying SFT provenance are excluded. Sampling handles are
run-bound training provenance, not general gateway deployments.

## Catalog authoring boundary

A valid catalog may contain provider connections and zero model records. The role-free connection
authoring service can create this state for a runtime consumer without inventing optimizer roles.
The existing `ProviderSetup` service is unchanged and continues to require world-model, judge, and
embedding roles plus an embedding-capable embedder for build and optimize workflows.

The catalog stores environment variable names, never credential values. Gateway prices use integer
micro-USD rates and keep unknown values as `None`. The existing optimizer float pricing and pricing
snapshot contracts remain unchanged.

## Raw tool arguments

`ToolCall.arguments` remains the parsed JSON object consumed by environments, judging, simulation,
and training. `ToolCall.raw_arguments` is optional. When present, it must decode to the same object
and is used for exact provider-order wire replay. When absent, it is omitted from serialization,
which keeps legacy payload bytes stable.

Partial provider output uses `GatewayEvent.raw_arguments_delta`. A delta can split at any byte or
token boundary and is not parsed in isolation. The complete call is validated only at its terminal
contract boundary.

## Request and attempt boundaries

Authorization freezes key-derived organization and identity, alias revision, target, API surface,
request hash, optional hashed caller-operation identity, catalog snapshot, and one monotonic deadline
before learned selection. Raw idempotency and client request values are not persisted in the
snapshot. A project target does not yet know its exact model, pool, or deployments at that point.
Selection and direct-target resolution produce a separate route-bound execution snapshot.

The attempt ledger records acceptance before selection. It records a provider attempt only
immediately before that provider dispatch. This separates requests that fail or crash before any
upstream work from attempts that may have incurred provider usage.

## Locked repository contracts

This foundation updates the following tracked locks:

- `AGENTS.md` names gateway package ownership and freezes `ModelCapabilities` digest placement.
- `docs/README.md` indexes this verified reference page.
- `wmo/common/models/model_test.py` proves legacy tool-call serialization is unchanged.
- `wmo/common/models/catalog_test.py` proves connections-only catalogs and deployment-local gateway
  metadata round trip.
- `wmo/repo_import_boundaries_test.py` already applies to the new runtime package and requires no
  allowlist widening.

The exact root CLI commands, config subcommands, release scope, dependency list, installed-wheel
driver, and `wmo run PROJECT` behavior are unchanged. Their executable locks therefore require no
edit in this foundation.

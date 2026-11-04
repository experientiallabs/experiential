# Immutable real-trace RAG contract

Experiential stores retrieval demonstrations as an immutable, versioned artifact derived only from verified
real imported traces. The artifact records its exact source manifests, stable trace, conversation,
transition, and leakage-lineage identities, the key schema, the embedder model and capability
digest, all vectors, vector dimensions, and the complete fit-lineage set.

Retrieval is read-only. It removes excluded query lineages before scoring, then ranks by cosine
similarity with transition ID as the stable tie break. The default local embedder is deterministic
and provider-free. A semantic embedder is allowed only through an explicit client and exact
`ModelSnapshot` that can be supplied again when the artifact is loaded.

Only a real action with a subsequently observed user or environment response becomes a retrieval
transition. A terminal assistant response has no following observation and is excluded. Generated
world-model predictions, simulator rollouts, teacher data, judgments, evaluations, and manual
examples cannot enter this index.

An empty tool reply is still an observation. Extraction preserves whitespace in textual tool
replies and retains explicit JSON null, false and zero results. A result with a matching call ID
at the call's end time is retained even when both spans have the same start timestamp. Missing
result fields remain missing evidence; they do not become empty replies. Saved indexes and
evaluation receipts remain immutable; corrected extraction applies when new evidence is built.

Trace count is not a validation boundary. A corpus with any positive number of valid traces is
accepted. The 100 to 1,000 range is only a common happy path for useful coverage.

## Scoped fit-only indexes

A build's fit-only index holds every fit transition, and `persist_fit_only_world_model` requires
exactly that complete fit subset of the world model's serving index. Studies that keep part of
the fit evidence unread declare a `CorpusScope` instead: a name, a `lineage` or `trace` kind, the
sorted allowed identities with their SHA-256, and the identity of the plan that chose them (an
artifact pointer or a digest).

`persist_trace_rag(..., corpus_scope=scope)` builds a fit-only index holding exactly the fit
transitions the scope admits. It refuses a scope that names an identity the lineage bindings do
not carry or that admits a held-out trace, and refuses `held_out` in `included_partitions`. The
scope is recorded in the index envelope and its content identity; an unscoped index records
nothing, so its envelope bytes and identity are unchanged. The loader refuses a scoped index
whose transitions fall outside its scope.

`persist_scoped_world_model(store, world_model, scoped_rag, retrieval_scope=...,
construction_scope=...)` persists the world model over a scoped index. The index must declare
exactly `retrieval_scope`, name only the serving index's fit lineages, and hold exactly the
serving fit transitions the scope admits. When `construction_scope` is given, every trace in the
world model's recorded construction knowledge must lie inside it and none may be held out. The
two scopes are independent and may overlap. The unscoped `persist_fit_only_world_model` and
`bind_fit_grounded_world_model` refuse a scoped index.

The world-model artifact records its construction scope as `construction_scope`. A world model
without one serializes no such field, so its envelope bytes and content identity are exactly
those it had before scopes existed. A recorded scope joins the identity, is re-derived on every
load, and requires construction knowledge; a recorded trace scope must contain every
construction trace. `persist_scoped_world_model` records the construction scope it verified, and
reuses an already recorded one when none is passed; a world model that records a scope accepts
only that exact scope. `persist_fit_only_world_model` and `persist_memory_first_world_model`
carry a recorded scope unchanged.

## Constructing from a declared scope

`construct_world_model(..., construction_scope=scope)` and
`ground_world_model_from_traces(..., construction_scope=scope)` construct from a declared corpus
instead of the index's whole corpus. The supplied traces must be exactly the in-scope fit traces
of the index's source datasets, with their exact recorded contents. Scope identities the corpus
does not carry, and scopes admitting a held-out trace, are refused. A trace's lineage and
partition come from the index transitions, a lineage being fit when the index lists it among its
fit lineages, or from `lineage_bindings`, the build's complete assignment, which must cover the
corpus exactly and agree with every index transition. A scope that depends on a trace neither
places is refused, so a retrieval-scoped index needs the bindings for traces outside it. The
index itself may expose any episodes, so construction and retrieval corpora may differ and
overlap; the direct-grounding tool inventory comes only from in-scope transitions. The scope is
recorded on the persisted world model. Without a scope, construction still requires the traces
to equal the index's complete corpus and its transition episodes, and `lineage_bindings` is
refused.

For variance studies, `exp.optimize.evaluation.variance.exposure.world_learning_scopes(plan,
construction=..., retrieval=...)` derives both scopes from an `ExposurePlan`. Each covers every
`world_learning` trace, or the traces of an optional tag selection checked with
`plan.restrict(selection, "world_learning")`, so mining-probe and final-evaluation traces can
never enter either corpus.

## Tag-aware retrieval

Tags are used twice. A tag selection decides which world-learning traces enter a scoped corpus
(above). Recorded tags can also shape what one query retrieves from it.

`persist_trace_rag(..., trace_tags=record)` records an optional `RAGIndexTags` on the index. It
holds at most one `RAGTagView` per view (`task`, `observations`), each with the bank ID and
revision, every question with its family, the fire threshold, the SHA-256 of the tag matrix and,
for each indexed trace, the sorted propositions whose probability was strictly above the
threshold. The build restricts each view to exactly the traces the index's transitions come from
and refuses a view that misses one; the loader refuses a record that covers any other set. The
record joins the envelope and content identity. An index built without it serializes no
`trace_tags` key, so its bytes and identity are unchanged.

A query brings `TaskQueryTags` for the current task, computed with the same bank and threshold,
and a `TagRetrievalPolicy` (bank ID, revision, threshold, `weight`, `mode` and `families`).
`TraceRAGRetriever.retrieve(query, tag_policy=..., query_tags=...)` scores each eligible
candidate as cosine similarity plus `weight` times the Jaccard overlap between the query's fired
tags and the candidate trace's fired task-view tags (0 when neither fired anything). In
`require_shared_family` mode, candidates sharing no fired tag from one of `families` with the
query are dropped first. Equal scores still break by transition ID. Each match records its
`tag_overlap`. The policy, query and index must agree on bank, revision and threshold, and the
query and families must name recorded columns; otherwise retrieval refuses.

Only task-view tags may inform a query. `TaskQueryTags` refuses any other view, because the
observations view describes what the episode revealed and would leak the outcome being
predicted. Observations-view tags may annotate indexed traces but ranking never reads them, and
an index with only observations tags cannot serve a tag policy.

Without a policy and query tags, retrieval is cosine ranking exactly as before and matches carry
no `tag_overlap` field.

A grounded session takes the policy through `GroundedWorldModel(tag_retrieval=...)` or
`load_grounded_world_model(..., tag_retrieval=...)`: a `TagRetrievalBinding` of the policy and
query tags keyed by task ID. The binding is checked against the index at construction.
`prepare_turn` ranks with the current task's tags when the binding lists them and records
`TagRetrievalEvidence` (the policy, its SHA-256, the query tags and the index matrix digest) on
the prepared call and in the session's prediction evidence. A task without listed tags retrieves
by cosine alone. The session representation identity includes the digest of the policy and the
task's own tags only when a binding is present, so unbound sessions keep their identity.

For variance studies, `exp.optimize.evaluation.variance.tag_retrieval.index_trace_tags(plan,
bank=..., task=..., observations=...)` builds the record from tag matrices, keeping only
`world_learning` traces, and `query_task_tags(matrix, trace_id, index_tags=..., plan=...)` reads a
probe task's query tags with the index's threshold. Both refuse matrices that mix banks,
revisions, columns or views, and query tags must come from a task-view matrix.

## Query and world-prompt capacity

New evaluations and router optimizations derive an omitted `maximum_retrieval_query_tokens`
from the world model's available input capacity after its declared output reservation. This
aggregate limit is separate from the embedding model's per-input context window: complete query
components are losslessly split into bounded UTF-8 chunks, and each dispatched batch retains its
actual input-byte accounting. An explicit query limit is still enforced exactly. Router spend
authorization includes the full resolved query allowance. Evaluation reports separately estimate
expected usage from the captured task evidence.

The `text-world-model-v3` prompt represents grounding with a content-addressed context table.
Each example's `context_ref` identifies its complete observed task and initial context in
`grounding_contexts`. Equal contexts appear once, while every ordered example retains its own
transition ID, full action and full observation. Resolving the references recovers exactly the
inline evidence; this format does not summarize or trim it. Packing removes a context only when
no retained example references it, and checks reference integrity even when the full request fits.

Prompt identity is immutable. An older grounded-world artifact is rejected with an instruction
to prepare a new one; its saved evaluations and receipts are not rewritten. For a plain evaluation
project with `system`, `build_spend_ledger`, `hosted_judge`, `router_policy` and `router_report`
all unset, Python applications can call `persist_grounded_world_model` with the unchanged serving
RAG pointer, then
`select_completed_build` with the same trace, task, fit/serving RAG and review pointers plus the new
world-model manifest. Pass `trace_import_id=project.load_project().trace_import_id` to retain the
project's existing import association; omitting it clears that association. This materializes the
current prompt without re-embedding source evidence.
Prepare a new evaluation afterward; never resume a frozen preparation under a different prompt.

Do not use this generic selector to refresh a hosted optimizer project. Its frozen build-spend
ledger and downstream judge, policy and report selections bind the original world-model artifact;
replacing only the build pointer breaks that graph. Create a fresh hosted setup through the hosted
workflow instead, leaving the original project, stage ledgers and selected results intact. This
release provides no in-place hosted-project prompt migration.

Before a text-world request is dispatched, optional retrieved examples are packed in priority
order as whole examples within the model context and frozen input reservation. Required task
instructions, tool definitions, conversation, environment state, system prompt and output controls
are retained. Protocol retries use the same rule when adding correction feedback. Saved grounding
IDs identify only the examples actually included in each request. Public world-model sessions use
the same fitting rule with their resolved model capacities. If required input cannot fit, dispatch
fails instead of truncating it. For models without a published output maximum, the existing
remaining-context output binding happens after optional examples yield to the requested output. The
100-step and 1,000,000-total-output-token rollout defaults are independent of this packing.

Required prompt framing and output capacity are checked before query embedding or retrieval cost
admission, then checked again before completion dispatch. Public `WorldModel.step` sessions report
these local capacity failures as `WorldModelSessionError`; low-level grounded runtimes use
`ValueError`. Provider failures retain their original exception types.

Low-level `GroundedWorldModel` construction, `load_grounded_world_model`, and
`bind_fit_grounded_world_model` require `capabilities=resolved_model.capabilities`. The metadata
must match the model identity frozen into the build artifact before either serving or fit-only
execution can proceed. A matching explicit `ModelCapabilities()` remains valid for a model with
unknown capacities; omitting known metadata cannot disable capacity admission.

## Historical provenance

This design restores useful behavior from the last coherent pre-refactor implementation:

- `e7aad17b:exp/simulation/retrieval/retriever.py::EmbeddingRetriever.index/topk/save/load`
  established offline embedding, cosine retrieval, and reload without re-embedding.
- `e7aad17b:exp/simulation/model/world_model.py::WorldModel.load/new_session/step` established the
  intended built-artifact to stateful simulation flow and the default top-k of five.
- `e7aad17b:exp/simulation/model/loader.py::load_world_model` established one shared artifact load
  path for Python and serving callers.
- The related historical retriever and world-model tests established persistence and top-k behavior.

The current contracts intentionally do not restore the historical provider types, GEPA, object
identity leak guards, or `EmbeddingRetriever.add`. In particular, the old `WorldModel.step` path
could add a generated prediction to the shared replay buffer. That behavior is forbidden now.

The current retrieval package provides the immutable artifact, loader, and retriever seam. A
dependent integration can wire canonical build output and the current simulator to it without
importing the deleted public types or provider stack.

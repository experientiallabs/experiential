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

Trace count is not a validation boundary. A corpus with any positive number of valid traces is
accepted. The 100 to 1,000 range is only a common happy path for useful coverage.

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

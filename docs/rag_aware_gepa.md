# Design note: RAG-aware GEPA (why, and how leakage is avoided)

## The problem

GEPA evolves the **env prompt template**, and that template has a "similar past examples" slot the
serving world model fills via DreamGym top-k retrieval (`WorldModel.step` →
`retriever.topk` → `build_env_prompt(..., demos=...)`).

Originally GEPA evaluated every held-out step with `demos=[]` — zero-shot. So it tuned the prompt
for a condition the prompt is *never deployed under*. A prompt that's great with retrieved exemplars
can be mediocre without them and vice-versa; optimizing one and serving the other is a train/serve
mismatch.

## The fix

GEPA now evaluates each step with the **same demos the serving model would retrieve for it**
(`GEPAOptimizer(provider, judge, retriever=...)`). The prompt is optimized under serving conditions.

Key realization that keeps this cheap and correct: retrieval depends only on `(state, action)`,
not on the candidate prompt. So a step's demos are identical across every GEPA candidate. We
therefore compute demos **once** (`GEPAOptimizer._eval_steps`) and bundle them with the step into the
`_EvalStep` that GEPA carries as its DataInst — rather than re-retrieving inside each candidate
evaluation.

This is *not* "RAG as an optimization mechanism." Retrieval isn't part of the search; it's part of
faithfully reproducing the serving environment the prompt is scored against.

## Avoiding leakage

A held-out step's ground-truth observation must never be visible to the model predicting it. Two
rules enforce this:

1. **Retrieve from the train corpus only.** The GEPA retriever is re-indexed over the *train*
   traces, never the held-out/val steps. Val steps come from disjoint test traces, so they can never
   retrieve themselves.
2. **Exclude the query's own trace.** Even within train (when GEPA's minibatches sample train
   steps), a step never retrieves a demo from its *own* trace — which would surface the exact
   observation we're asking it to predict, or an adjacent step that gives it away. We over-fetch
   `top_k + slack`, drop same-trace demos, then take `top_k`.

The serving retriever (used at `wmh serve` time) is separate and indexes the **full** corpus —
there's no held-out set at serve time, so retrieving from everything is correct there.

## Configuration

- `wmh/engine/build.py` passes a train-only GEPA retriever (sharing the stateless embedder with the
  serving retriever) so a real build is RAG-aware by default.
- `GEPAOptimizer(provider, judge)` with no `retriever` falls back to the original zero-shot behavior
  (used by unit tests and any caller that doesn't want retrieval).

## Tested

- `gepa_test.py::test_eval_steps_retrieves_demos_without_same_trace_leakage` — a step's demos never
  come from its own trace.
- `gepa_test.py::test_eval_steps_zero_shot_without_retriever` — no retriever ⇒ empty demos.

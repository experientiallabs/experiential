# Memory exposure and fixed-question decision measurements

The default simulator continues to use its existing memory and selection policies. Passive
receipts record evidence without adding instructions to the predictor or publishing a new memory
version. Council overviews are pinned before use, so background compression cannot change the
context of a saved evaluation. Generic recall retains per-action selection receipts and its
prepared-context digest; these are preparation evidence, not proof of final delivery.

## Inspect saved calls

Use the existing continuation CLI with an explicit inventory of canonical call artifact references:

```json
{
  "action": "inspect-memory",
  "memory": {
    "intents": [
      {"artifact_id": "<structured-intent-id>", "sha256": "<manifest-digest>"}
    ]
  }
}
```

Use the complete `ArtifactInput` emitted by the artifact store when constructing the JSON; it
includes the schema's required reference fields. Run:

```bash
exp eval PROJECT --root ROOT --continuations inspect.json --non-interactive
```

This is an offline read. It needs no provider credentials and writes no memory or report artifact.
The JSON output can be saved as a diagnostic snapshot. It includes every supplied call, including
calls with unknown outcomes. The inventory is caller-declared; it does not certify that every call
in an evaluation was supplied.

For automatic discovery, replace `intents` with `roots`, containing exact saved result references.
The reader traverses every declared manifest dependency, including ancestral learning and source
calls. It recognizes structured calls and continuation actor requests, and joins their physical
receipts. `maximum_artifacts` defaults to 10,000; exceeding it refuses rather than truncating.
The report retains the inventory, verified graph size and missing joins. This covers the recorded
dependency graph, not undeclared execution or cognitive use. No runtime or provider is loaded.

Each call records the final canonical request digest, execution-owner consumer, message/tool
locators, and linked physical attempts. Simulator research channels have separate locators inside
the framed input. A canonical intent does not prove that the provider received the request. A
saved HTTP response does not prove successful parsing or task correctness. Existing calls without
explicit role or physical joins remain unknown; they are not reconstructed from model aliases.
The current join covers `StructuredCompletion` and continuation actors through `AdmittedJsonTransport`. Other execution
paths and older receipts may lack that join.
Built-in executable grounded worlds also use this boundary, including their optional evidence
planner. Native source-reader requests and answers retain separate JEV receipts.

Reasoning diagnostics distinguish exposed text, provider summaries, opaque carriers, token usage,
missing capture and incomplete output. Reports retain locators and digests rather than copying
reasoning text. No hidden reasoning is requested. Missing mention does not mean ignored memory.
`validate_disposition` checks exact emitted spans and consumer-local input/output citations. It
checks citation integrity, not the truth of an analyst's interpretation or causal benefit.

## Current-delivery research intervention

The built-in `grounded_runtime` configuration accepts `research_history: false`. This changes its
frozen runtime identity. It removes constructor target/library investigations and memory session
and prior-trajectory hypotheses from predictor delivery. The memory selector still runs over the
same eligible catalog. Original observations, behavioral knowledge, current state and visible
conversation remain available. No memory record is retired or republished.

Delivered mixed overview summaries are opened into eligible original observation lines. Previously
omitted ranges remain omitted. Retired or withheld leaves cannot reappear. Expanded required input
is subject to the existing capacity refusal; it is not silently recompressed into a new summary.

This is a **current-delivery intervention**, not removal of a source's historical influence. It
cannot undo prior learning, supersession, selector exposure or construction effects. It also does
not establish a causal improvement by itself. A valid comparison must freeze task/state, memory,
models, budgets and source selection, retain failures in both arms, and inspect actual delivered
inputs. Changed packing of optional retrieval examples can confound that comparison. The default
remains `research_history: true`.

## Measure a fixed decision policy

`DecisionCalibration` accepts exact saved questions, ordered candidates, permitted state, student
and teacher distributions, and separately sourced labels. The distribution must cover the
candidates plus `OTHER`. JEV ordinal relevance scores are not correctness probabilities.
Candidate availability, classification, Brier score, accepted errors and missing predictions have
separate denominators. Teacher or student output, including a derived child artifact, cannot act
as the independent label.

Before confirmation, save `registration.json` as an immutable artifact. Its object contains
`definition`, `threshold`, `minimum_support`, `maximum_error`, and `inventory`. Each inventory row
contains `id`, `partition`, and `leakage_groups`. The definition binds student/readout, teacher,
fallback, question, renderer, candidate builder, tools, knowledge and hard policy. Confirmation
questions must postdate this registration. Their saved values include `question`, `candidates`,
`permitted_state`, `definition_sha256`, and `hard_allowed`. Predictions are exact saved
`{"selected": ..., "probabilities": ...}` values; label locators resolve to the target identity.

Submit the typed study through the same CLI:

```json
{"action": "measure-decisions", "decisions": "<DecisionCalibration object>"}
```

The string above is a schema placeholder, not an executable study. Generate the complete payload
with `ContinuationCommand(action="measure-decisions", decisions=study).model_dump_json()`.
Changing the policy, threshold or planned inventory invalidates registration. Declared source,
task, entity and dependency groups cannot cross partitions. Correlated confirmation rows count
as one connected lineage component for minimum support and the one-sided Wilson risk bound; any
accepted error makes that component a failure. The bound is a diagnostic under the declared
sampling assumptions, not a distribution-free guarantee.

These are fixed-question measurements only. Label authority, complete lineage declarations and
representative sampling need independent evidence. The code neither activates a specialist nor
trains a model, tunes a threshold on confirmation, replaces the ordinary judge, or establishes
carried-workflow success. Protected analyst calibration, paired carried experiments and any
subsequent specialist training remain separate qualification work. Synthetic contract tests are
not expert calibration or proof of training lift.

## Optional memory analyst

`analyse-memory` is an opt-in continuation action with a `memory_analyst` payload conforming to
`MemoryAnalystSpec`. It binds the inspection inventory, its `exposure_sha256`, the analyst model
and catalog, a deadline, whole-input byte limit, output limit and physical-attempt ceiling. Use
`--dry-run` to validate the inputs without calling a model. Execution uses the existing spend
consent, structured completion, dispatch accounting and trace-analysis owners. Completed results
replay without provider credentials. This action does not replace or rerun the ordinary task judge.

The analyst can read exact retained input fragments and explicitly exposed output text. Opaque
carriers and usage-only fields remain availability metadata. Findings need supplied citations;
interpretations need input and exposed output from one known consumer call. Their basis remains
hypothesis and attribution remains undetermined. The CLI labels the result **unqualified
diagnostic**. These checks constrain evidence claims; they do not establish the semantic accuracy
of an LLM interpretation. Independently labelled, protected examples are still required to qualify
an analyst rubric before using its findings as a learning signal.

## Optional reading before prediction

The existing continuation CLI accepts this addition to a built-in `GroundedRuntimeDefinition`
(or the `simulated` definition inside a mixed runtime):

```json
{
  "prediction_reading": {
    "reader_alias": "clef-flash",
    "policy": {
      "rounds": 2, "batch_size": 3, "workers": 3,
      "source_bytes": 12000, "packet_bytes": 48000
    }
  }
}
```

The alias must name a configured native decision model. Omit `prediction_reading` to retain
the existing selection path. Reading changes the frozen world identity. The simulator model
plans an evidence objective; independent source readers locate support and contradiction in
already eligible original memory observations and current RAG matches. A second bounded batch
can revise the objective. There is no background discovery, external search, evaluator feedback,
bank mutation or training. Current-instance facts still require current-instance evidence.

The final request contains a frozen `prediction_evidence` packet with exact originals, JSON
locations, digests, inspected/unread IDs, skipped sources and stop reason. Reader judgments remain
fallible. A stop is not proof of adequate evidence. The packet is required input: if it exceeds
capacity, prediction refuses rather than silently dropping it. Existing optional RAG examples may
still be packed out; inspect delivered requests when comparing treatments.

All planner, source-reader, recall, embedding and prediction calls share the runtime's physical
attempt allowance and deadline. A reader failure prevents prediction and leaves the world state
unchanged; receipts remain available. Restoring a world does not replay a future model draw or
buy a fresh call budget. This is an opt-in executable-world path, not automatic activation in
ordinary build learning, fidelity or council sessions. Quality improvement requires a separately
controlled comparison; successful protocol tests do not establish it.

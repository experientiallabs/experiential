# HarnessDelta: the update-representation interface for `wmh harness`

This is the interface `wmh optimize` mutates through and the archive records: a typed,
precondition-guarded, gate-audited delta object, replacing "a harness update is a file edit".
Getting this right matters more than any implementation detail downstream: **the update
representation IS the search space**, and everything the meta-agent can learn about *how to
improve harnesses* is bounded by what the update object can express.

Implementation: `wmh/harness/doc.py` (the document), `wmh/harness/delta.py` (the delta),
`wmh/harness/mutate.py` (delta parsing), `wmh/harness/proposer.py` (proposal runtimes), and
`wmh/harness/create.py` (clustering, gate, archive).

---

## 1. Why file edits are the wrong interface

A file-edit update — `{motivation, edits: [{path, content}], deletes}` over a harness directory —
inherits five structural blindnesses from the file/diff abstraction, two of which surfaced as
review-verified bugs in an earlier draft of this search loop:

| # | Failure mode | Bit us? |
|---|---|---|
| 1 | **Untyped target** — an edit changes bytes at a path; it cannot declare "this changes the *tool policy*" vs "the *recovery section of the prompt*". Search cannot reason over change *type*. | — |
| 2 | **No preconditions** — an edit proposed against one generation silently misapplies to another. Nothing asserts "the thing I edited still looks like what I saw". | — |
| 3 | **No element identity** — "which skill is this?" was inferred from filename vs frontmatter. | ✅ silent drop/clobber, order-dependent |
| 4 | **Unknown targets silently vanish** — an edit to a path the format doesn't know about round-trips to nothing. | ✅ a full eval spent on a no-op child with a false motivation |
| 5 | **Detached rationale** — *why* lives in one motivation string for the whole change, unbound from the field it justifies, so the archive can never answer "which kinds of edits work on which failure classes?" | — |

Patching these piecemeal (path allowlists, filename/frontmatter consistency checks, delete-target
validation) treats symptoms. The interface below makes each failure mode impossible by
construction.

One deliberate non-goal: the update *payload* stays a free-form string. Structure lives in the
envelope — identity, kind, hashes, preconditions, rationale, verdict — not inside the content. A
whole-component rewrite is easier for a model to produce correctly than a line diff, and richer
surface kinds can be added later without changing how updates work.

## 2. The interface

### 2.1 `HarnessDoc` — what a harness IS

A harness is a typed document of identity-keyed **surfaces**. Files (`SYSTEM.md`, `config.toml`,
`skills/*.md`) are a *render target* the store exports for running the harness elsewhere — they
stop being the interface.

```python
class SurfaceKind(StrEnum):
    PROMPT = "prompt"            # a section of the system prompt (joined in id order)
    SKILL = "skill"              # one skill: frontmatter (name, description) + body
    TOOL_POLICY = "tool_policy"  # the tool list, one tool name per line
    PARAM = "param"              # a scalar loop knob, serialized as its string form
    CODE = "code"                # the agent loop itself: a module defining run(kit)

class Surface(BaseModel):
    id: str                      # identity key: "prompt:core", "skill:count-words", "param:max-turns"
    kind: SurfaceKind
    content: str                 # the payload — deliberately unstructured
    budget: int | None = None    # size budget (chars), validated at construction, not advisory
    # content_hash: computed — blake2b(content)

class HarnessDoc(BaseModel):
    name: str
    version: int                 # immutable, assigned by the store on save; 0 = unsaved
    surfaces: list[Surface]      # unique ids; canonical order
    # doc_hash: computed over sorted (id, content_hash) pairs. "The score of harness X" is
    # well-defined because X is this hash.
```

A document validates as a whole at construction — tools resolve, `submit` present, params in
range, skill frontmatter names match their slugs, budgets respected — so an invalid harness cannot
exist as a value and nothing downstream re-checks.

The store (`.wmh/harnesses/<name>/`) keeps append-only version directories (`v1/`, `v2/`, …) with
`doc.json` authoritative plus rendered exports, and movable aliases in `aliases.toml`
(`champion = 3`). Rollback = re-point. Eval results key to immutable versions (or doc hashes),
never to "whatever the harness currently is".

### 2.2 `HarnessDelta` — what an update IS

```python
class FailureSignature(BaseModel):
    """Machine-clustered trigger: WHY this delta exists, queryably."""
    mechanism: str               # e.g. the shared unmet assertion, or "none: all tasks pass"
    task_ids: list[str]          # the cluster of failing tasks exhibiting it
    unmet_assertions: list[str]  # deduped gold assertions the mechanism explains

class SurfaceOp(BaseModel):
    op: Literal["add", "replace", "remove"]
    surface_id: str              # identity, never position or filename
    kind: SurfaceKind | None     # required on add; must match the existing kind on replace
    content: str | None          # the FULL new content (component rewrite, not a line diff)
    budget: int | None           # on replace, None inherits the existing budget
    rationale: str               # WHY THIS OP — bound to the op, not the delta (fixes blindness #5)

class GateRecord(BaseModel):
    """Filled at evaluation time; the delta carries its own verdict."""
    suite_delta: float           # regression suite: child − champion   (tier 1)
    suite_fraction_delta: float  # assertion credit when suite success ties
    full_delta: float            # full split: child − best-seen        (tier 2)
    full_fraction_delta: float   # assertion credit when full success ties
    holdout_delta: float | None  # held-out split: child − champion     (tier 3; None = no holdout)
    holdout_fraction_delta: float | None
    accepted: bool
    reason: str                  # accept/reject reasoning, incl. the expected-effect audit

class HarnessDelta(BaseModel):
    delta_id: str                  # content-addressed: blake2b(parent hash + ops)
    parent_doc_hash: str           # lineage by content, not name
    trigger: FailureSignature      # built by deterministic clustering, never free-typed
    preconditions: dict[str, str]  # surface_id -> expected content_hash of the PARENT surface the
                                   # meta-agent actually read. ANY mismatch rejects the WHOLE delta
                                   # atomically (fixes blindness #2). Every replace/remove target
                                   # MUST have one.
    ops: list[SurfaceOp]           # unknown surface_id on replace/remove = reject (fixes #4)
    expected_effect: str           # falsifiable prediction: "tasks in the trigger cluster flip"
    child_doc_hash: str | None     # recorded by application
    verdict: GateRecord | None     # None until evaluated; the archive stores deltas, not snapshots
```

### 2.3 Semantics

- **Application is atomic** (`apply_delta`): lineage hash, every precondition, and every op are
  validated against the parent doc — and the child re-validates as a whole `HarnessDoc` — before a
  token of eval budget is spent. Path safety, unknown-target rejection, skill-name consistency,
  and code compilation are impossible-by-construction rather than checked piecemeal.
- **The strongest lever is code** (`code:runtime`, `wmh/harness/code_runtime.py`): live search
  campaigns showed correct failure diagnoses that prompt- and skill-level edits could not fix —
  loop structure, retries, verification passes, context compaction, and token budgets are
  programs, not wording. The code surface holds a module defining `run(kit)`; the `RuntimeKit` is
  budgeted (hard caps on LLM calls and env actions), kit-recorded (the judged transcript is
  written by the kit, so code cannot claim work it did not do), and crash-isolated (an exception
  fails the episode, not the eval).
- **Verification is staged by cost**: before a full-split eval, a child is screened on its own
  trigger cluster — the failing tasks its delta claims to fix. The screen compares full-task
  success first and assertion-level partial credit second, so a partial fix is not flattened into
  a binary tie. A delta that improves neither signal is rejected (and archived) for a fraction of
  the price. The authoritative full gate repeats the same lexicographic contract across the whole
  split: binary success remains primary, and assertion credit cannot regress when success ties.
  Every judged delta is fed back to the proposer as trace-level history, so the search iterates
  instead of re-proposing rejected ideas.
- **Search breadth is independent from evaluation depth**: `proposal_batch_size` asks the
  proposer for sibling deltas against one selected parent before evaluating any sibling. `k`
  remains the number of rollout passes used to score each scenario. Project-backed proposers keep
  all parent source, failure traces, proposals, and candidate evaluations in one persistent agent
  project while every turn runs through the same ordinary agent-session runtime.
- **Acceptance is the gate, not "applied cleanly"** (`gate_delta`): regression-suite
  non-regression vs the champion → full-split never-worse-than-best → held-out non-regression vs
  the champion (when a holdout split is given). Binary ties consult assertion-level partial credit;
  a stepping stone may advance on dense signal but cannot hide a global dense regression. On
  accept, newly-passing tasks promote into the regression suite, so wins are locked in and later
  deltas cannot quietly trade them away. The verdict is written onto the delta.
- **The trigger is machine-made** (`cluster_failures`): failing tasks group by shared unmet gold
  assertions (connected components; the most common unmet assertion labels the mechanism),
  deterministically — no LLM, no entropy — so mechanism labels are comparable across deltas and
  runs. Selection weights cluster size but discounts rounds already spent on the same cluster and
  parent, so one environment-limited singleton cannot absorb the whole search. An all-pass parent
  gets an explicit generalization/economization trigger instead of a fabricated failure.
- **The archive is a lineage of audited deltas** (`DeltaArchive`): a root snapshot plus every
  proposed delta — accepted, gate-rejected, or invalid-before-eval — with its verdict. Docs are
  reconstructable by folding accepted deltas from the seed; snapshots are caches, not the record.
  This makes the meta-question queryable: *"across the archive, which (trigger.mechanism ×
  op.kind) pairs have positive mean gate deltas?"* — the beginning of the meta-agent learning
  which kinds of edits work. A pile of file snapshots cannot answer that; this can.
- **`expected_effect` makes every delta a tested prediction.** At gate time the trigger cluster is
  re-checked ("trigger cluster: 2/3 tasks now pass") and the result lands in `verdict.reason`.
  Over time this measures the proposer's *calibration*, not just its win rate.

### 2.4 What the meta-agent emits

The proposal prompt (`MUTATE_SYSTEM`) asks for exactly `{expected_effect, preconditions, ops}` as
JSON — trigger, lineage, and identity are filled by the caller from ground truth, never trusted
from the model. The prompt prints every parent surface with its id, kind, and content hash, so
preconditions are copy-not-guess. An unparseable or shape-invalid reply is a counted skip; a delta
that fails atomic application is a counted skip that is still archived with its rejection verdict.

## 3. Open questions

1. **Surface granularity of the prompt.** One `prompt:core` surface (whole-component rewrite) vs
   sectioned surfaces (`prompt:role`, `prompt:verification`, `prompt:recovery`)? Sections give
   finer credit assignment and smaller merges; they also impose our taxonomy on the meta-agent.
   Current position: start with `prompt:core` and let the meta-agent *split* a surface via
   `add` + `replace` — the taxonomy then emerges from search instead of from us.
2. **Merge.** Identity-keyed surfaces + content lineage make a surface-keyed three-way merge of
   two accepted lineages nearly free. Deferred to a follow-up.
3. **Gate resolution.** With k=3 and small suites, binary per-task deltas are coarse (0, ⅓, ⅔, 1).
   Assertion-level fractions break otherwise-flat ties; if flappy accepts show up in practice,
   raise k on gate evals rather than adding arbitrary thresholds.
4. **Suite demotion.** Newly-passing tasks promote into the regression suite; nothing ever leaves
   it. A permanently-flaky task could wedge the gate. Punted until observed.
5. **Sandboxing the `CODE` surface.** The kit is an interface contract, not a security boundary:
   searched code runs in-process and is only exercised against the world model during search.
   Running a searched harness against a real environment is a deployment decision that belongs
   behind a sandbox.

# Trajectory counts, summary forgetting and judge challenges

Three `exp eval PROJECT --continuations operation.json` actions. Each accepts `--dry-run`, which
prints the same result without saving anything; saved results replay without provider calls.

## How many trajectories: `sample-count`

Which memory checkpoint first exceeds an accuracy threshold is a search, so it is answered as a
registered procedure in three resolutions, run in order:

```json
{"action": "sample-count",
 "sample_count": {"resolution": "register", "grid": [4, 8, 16], "threshold": 0.75,
                  "criterion": "point_estimate", "development_lineages": ["LINEAGE"]}}
```

- `register` freezes the grid (strictly increasing whole-trajectory counts of the fit-only
  memory), the exact-reply metric, the threshold, the criterion (`point_estimate` or
  `wilson_lower_bound`) and the development lineages, before any measurement is read. The
  development lineages must be held out, and at least one held-out lineage must stay untouched
  for confirmation. Registering the same procedure again returns the first registration, so its
  place in the receipt order cannot move past earlier results.
- `select` takes the printed `registration` reference and `development`: one audit receipt ID
  per measured grid count (`exp eval PROJECT --audit N`, after the memory has learned that many
  fit traces). Each receipt is measured from its saved results. The grid is walked in order and
  the first count that meets the criterion is frozen; a missing or partly failed count stops the
  walk. A registration selects one count; a different selection is refused.
- `confirm` takes the printed `selection` reference and `confirmation`: the receipt of an
  independent audit opened after selection, over lineages the development audits never used,
  reading the frozen checkpoint with the same simulator, retrieval, judge and code revision. It
  reports the trajectory count, the exact-reply estimate with its Wilson interval at the lineage
  count, coverage, failures and whether the criterion holds. A development receipt or lineage, a
  re-audit, an invalidated receipt or a changed configuration is refused, and a selection is
  confirmed on one receipt only.
- Order is the audit receipt sequence, never a clock. Receipts are numbered while the root
  exposure ledger is locked; the registration and the saved selection are written under the same
  lock and record the latest receipt number, so only receipts numbered after them count as
  development or confirmation evidence. A receipt minted before the selection is refused even
  if its host's clock claims a later time. The selection cites the registration and every
  development receipt, and the confirmation cites the selection and its receipt.

## Forgetting a stale summary: `forget-summary`

```json
{"action": "forget-summary",
 "forget": {"lineage": "LINEAGE", "block": "#0-3", "reason": "why the summary is stale"}}
```

`block` is the summary ID memory views print. `lineage` may be omitted while only one recorded
learning plan's memory has published; once both the all-traces and the fit-only memory have,
name it, since nothing orders the two lineages. The forget is eligible only when:

- the block is a complete aligned block of the head version's log and its current summary
  generation is built (an unbuilt or already forgotten block has nothing to forget);
- the same block was not forgotten against the head for another reason;
- no registered `sample-count` study still has to learn checkpoints of this memory, since those
  must come from learning alone (forget once learning passes the registered grid).

The record binds the head version, the exact summary generation and the reason. Published
checkpoints never change: the forget applies from the next version, which raises the generation
of the block and of every complete ancestor; the next maintenance rebuilds them from the block's
lines, still excluding retired hypotheses. Memory lines, source evidence and retrieval are
untouched, and audit receipts keep reading their pinned versions. No provider call is made.

## Retiring checkpoints and reclaiming session state: `retire-checkpoint`, `collect-garbage`

Capturing a world checkpoint pins the source session's revisions under the checkpoint's
artifact ID, so that history stays in the session database. A pin is released only when the
operator retires the checkpoint:

```json
{"action": "retire-checkpoint",
 "retirement": {"checkpoint": {"artifact_id": "CHECKPOINT", "sha256": "MANIFEST_SHA256"},
                "reason": "why the checkpoint is no longer needed"}}
```

Retirement is refused while any saved artifact (a study, campaign, task, package, fork or other
result) names the checkpoint in its inputs or its data; the refusal lists the referrers. The
retirement record is saved first and then every pin the checkpoint owns is released, so an
interrupted retirement completes when run again. The checkpoint artifact itself stays loadable.

```json
{"action": "collect-garbage", "collection": {"batch_size": 500, "vacuum": false}}
```

`collect-garbage` reclaims, in every session database of the project, the state nodes no
session head and no pin reaches, and saves the inventory as a report. With `--dry-run` it
writes nothing and prints what would be reclaimed and every pin: its owner, session and
revision range, the nodes only it keeps, and whether its owner is `referenced`, `retirable`,
`retired` (run the retirement again to release a pin registered after it) or
`not_a_checkpoint` (never released by these actions). Event logs are never reclaimed, so every
revision stays replayable. Freed pages are reused by later writes; `vacuum: true` also rewrites
each database that reclaimed nodes so its file shrinks (the rewrite waits for, then excludes,
concurrent writers).

## Judge challenge studies: `challenge-judges`

`challenge: ChallengeRunSpec` holds a registered `JudgeChallengeStudy` over a calibration pack
of saved, independently labeled outputs, one pinned catalog model per judge arm
(`arm_id`, `alias`, `model`), the `catalog_sha256`, an output allowance,
`attempts_per_judgment` and a dispatch `deadline`.

- `invariances` declare irrelevant factors (`actor_identity`, `candidate_order`,
  `explanation_length`, `action_order`) with the JSON pointers each may vary.
- `defects` declare the study's own defect kinds: a `kind` name, a `check` and the JSON
  pointers where it lands. `changed_value` replaces values there with different ones;
  `defended_change` also replaces prose defending the change; `added_entry` appends or adds an
  entry there (an unauthorized write or effect); `changed_json_type` changes a value's type.
- `perturbations` edit saved outputs; each names an irrelevant factor or a declared kind and
  must satisfy its declaration.

Outputs are judged as saved `WorkflowEvidence` through the maintained dual evaluator; normalized
JSON outcomes are refused before any call. The pack's task must be the project's saved
evaluation task with the same candidate verifier. The run shows its known-rate estimate and
needs spend consent; each judgment's intent, response and report are saved, so an interrupted
run resumes and a recorded study replays offline. The report gives, per arm and panel, base
label agreement, stability per declared factor and judge sensitivity per declared defect kind,
with intervals at the count of independent base lineages. Labels are never rewritten.

Before dispatch, the run saves its complete model snapshots, catalog identity, allowances and
deadline. Each judgment and the completed result belong to that full run identity. Changing
any of those settings creates a separate run; only the unchanged registration replays without
credentials or calls. A study-only result does not establish a product run's configuration.

The memory-first simulator profile (no recorded-evidence retrieval) is not an audit or council
option: its memory has a different knowledge identity, so it would need its own learning lineage.

## Reconstruction study coverage

The six reconstruction questions are trace-only construction, rules versus original-evidence
ablations, retrieval selection, research history, carried-state continuity and partial-world
inspection. They are not all wired as executable product studies. The actions above provide
trajectory-count and judge-challenge procedures; continuation campaigns, selection, failure
research and memory inspection provide additional typed operations. These operations do not
by themselves establish a registered intervention and complete comparison for each of the six
questions. Each remaining study needs its own typed configuration, executed arms, saved results
and complete planned denominator before it can be reported as implemented or measured.

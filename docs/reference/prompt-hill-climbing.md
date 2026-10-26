# Prompt hill climbing

A manual hill climb improves an agent's system prompt one measured edit at a time. Each round
evaluates exactly one edit, written by a person or picked from an analyst's proposals, and keeps
it only when a paired held-out gain clears both zero and an A/A noise floor. Nothing searches,
mutates or optimizes prompts automatically.

All three actions run through the existing continuation command and accept `--dry-run`:

```bash
exp eval PROJECT --root ROOT --continuations round.json --dry-run
exp eval PROJECT --root ROOT --continuations round.json --yes --non-interactive
```

Paid actions show the credential-free actor (and maintained simulator) cost inventory and ask
for spend consent before any provider client is built. Each prints a short summary and saves the
full result to `<project>/runtime/hill-climb/<identity>/report.txt`, named on a `Report:` line.

## The climb design

Every payload carries the same `climb: ClimbSpec`:

- `cases`: one `ContinuationJob` per independent task lineage, each at an initial checkpoint
  (captured before the actor's first call), all with the same actor alias, model and system
  prompt. That shared prompt is the baseline. The runtime may be native, mixed or a grounded
  simulator; any runtime a continuation job accepts works.
- `test_fraction` (default 0.5) and `split_salt`: lineages are ranked by the shared salted
  held-out rule (`exp/common/evaluations/splits.py`) into train and test. Changing the salt draws
  a fresh split and therefore a fresh climb.
- `minimum_lineages` (default 4): fewer lineages on either side is refused before any call as
  underpowered.
- `screen_repeats`, `confirm_repeats`, `final_repeats` (even; defaults 4, 4, 8): fresh
  continuations per case and arm. Each case's `maximum_physical_attempts` must cover the largest
  repeat count times `maximum_model_calls`.
- `interval`: `case_bootstrap` (default; whole lineages resampled, seeded) or `lineage_wilson`
  (the Wilson bound on paired lineage differences, wider with few lineages).

Case deadlines are not part of the climb identity. Each round and report sets its own `deadline`.

## A round

```json
{"action": "climb-round",
 "climb_round": {"climb": {...}, "round": 1, "deadline": "2026-11-01T00:00:00Z",
                 "edit": {"kind": "append", "text": "Re-read the request before editing.",
                          "change_note": "Failures delete more rows than asked."}}}
```

`edit.kind` is `append` (one instruction block after the current prompt) or `replace` (the whole
prompt). Instead of `edit`, `choice: {"proposals": REFERENCE, "index": N}` picks an analyst edit
made after the previous round.

1. Round `n` needs rounds `1..n-1` decided. Its incumbent is the last kept candidate, or the
   baseline. An edit that reproduces any prompt already measured in this climb is refused.
2. Before the first dispatch the incumbent and candidate prompt versions are saved
   (`agent-prompt-version`, each naming its parent, edit and change note) and registered under
   the round number with the round's `deadline` and code revision, which fix both stage panels.
   An interrupted round resumes the same panels, reusing their completed and unknown cells; a
   different edit, deadline or code revision for a registered round is refused, since it would
   redraw the stages.
3. Screening runs an actor panel (`compare-actors` machinery) of the incumbent against the
   candidate on the train lineages. A prompt is part of the chat actor's saved context, so the
   candidate arm starts from each case's initial checkpoint re-instructed under its prompt
   (`instructed_checkpoint`). If the incumbent passes, or fails, every known train outcome, the
   round is refused as saturated: no edit can show a gain there.
4. An edit passes screening when its train gain estimate exceeds the train A/A floor.
5. Confirmation runs the same comparison on the test lineages with fresh repeats. The edit is
   kept only when every planned test outcome is known, the paired interval of the gain excludes
   zero and the gain estimate exceeds the test A/A floor.

The gain is the mean over lineages of each lineage's candidate-minus-incumbent repeat mean; a
lineage is one independent unit. The A/A floor comes from the same panel: the incumbent's repeats
split into first and second halves, and the floor is the largest absolute endpoint of that A/A
interval. Each half has half an arm's repeats, so the floor is conservative. Rejected edits stay
saved with their measurements; a resumed or repeated request replays the saved decision without
calls and re-derives it from the saved panels.

## Analyst proposals

```json
{"action": "propose-edits",
 "edit_proposals": {"climb": {...}, "after_round": 1, "maximum_edits": 3,
                    "analyst_alias": "ALIAS", "analyst_model": {...},
                    "catalog_sha256": "...", "deadline": "..."}}
```

The analyst reads the failed train branches of the prompt the next round will edit (from the
round's screening panel), localized by the existing failure analysis
(`exp/optimize/evaluation/analysis/failures.py`) and grouped by observed symptoms. It never sees
test traces, test scores, gains, floors or pass rates; earlier edits reach it only with a kept or
reverted bit. It names root-cause buckets citing failure refs and proposes up to `maximum_edits`
one-change edits. The answer is schema-checked; cited refs must come from the supplied inventory.
Proposals are never applied: the output lists them by index (with any edit whose prompt was
already measured marked) and prints the proposal set reference for `choice`. A request whose
inventory exceeds `maximum_input_bytes` is refused rather than truncated; long actions, replies
and final messages are cut to `maximum_reply_bytes` with a label.

## The final line

```json
{"action": "climb-report",
 "climb_report": {"climb": {...}, "rounds": 3, "deadline": "..."}}
```

The report lists every round, kept and reverted, with its change note and reason. When an edit
was kept, baseline and best are compared on the test lineages with `final_repeats` fresh repeats
under the keep rule. The verdict is `merge`, `within noise, do not merge` or `no edit kept`;
the second is a valid result, not a failure.

## Limits

- The test lineages are reused by every confirmation. Only keep bits flow back to the proposer,
  and the final comparison uses fresh repeats, but the same lineages; for an independent claim,
  audit the best prompt on lineages the climb never touched.
- Unknown outcomes never enter an interval; a confirmation or final comparison with any unknown
  outcome does not keep.
- A case bootstrap over few lineages is narrow; read it with the printed lineage count, or select
  `lineage_wilson`.
- Each comparison is a development measurement on the climb's cases, not a population claim.

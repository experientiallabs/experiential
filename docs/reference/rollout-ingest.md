# Rollout ingest and the evolving rubric

A decision-variance task trains against a two-part rubric: the frozen LLM episode judge (the
outcome reward) and the frozen decision-model dense tags run as hybrid dense (the dense
reward). A policy optimized against a frozen rubric learns whatever it rewards, including ways
to satisfy it without doing the task. The evolving rubric keeps both parts ahead of that: the
rubric is frozen within a training epoch and revised only at the boundary between epochs, from
the rollouts the trainer actually sampled.

This page is the contract between an external GRPO trainer and the boundary. The trainer writes
one export per epoch; everything else happens on the host.

## The export (`rollout-ingest-v1`)

One JSON document per epoch, `RolloutExport` in
`exp/optimize/evaluation/variance/rollout_ingest.py`. Write it after the epoch's last update,
as `<exports>/epoch-<n>.json`. A document is accepted whole or refused whole.

```json
{
  "contract": "rollout-ingest-v1",
  "trainer_run": "grpo-run-1",
  "epoch": 1,
  "rubric_version": 1,
  "rubric_sha256": "<the version identity the epoch trained against>",
  "policy_checkpoint": "policy-epoch-1",
  "rollouts": [
    {
      "rollout_id": "r-000117",
      "task_id": "parse-date",
      "lineage_id": "prompt-42",
      "group_id": "grpo-group-9",
      "instruction": "Fix parse_date so invalid dates return None.",
      "in_loop_reward": 1.0,
      "reward_parts": {"outcome": 1.0, "dense": 0.05},
      "fidelity": "full",
      "turns": [
        {"turn_index": 0,
         "calls": [{"call_id": "c1", "name": "edit_file",
                    "arguments": {"path": "src/parse.py", "patch": "..."},
                    "result": "ok", "is_error": false,
                    "tokens": {"token_ids": [17, 4], "token_texts": ["{\"patch\"", ": ...}"]}}]},
        {"turn_index": 1,
         "calls": [{"call_id": "c2", "name": "run_tests", "arguments": {}, "result": "3 passed"}]},
        {"turn_index": 2, "content": "Fixed parse_date; all tests pass.", "final": true,
         "content_tokens": {"token_ids": [1, 2, 3],
                            "token_texts": ["Fixed", " parse_date;", " all tests pass."]}}
      ]
    }
  ]
}
```

Fields the boundary relies on:

- `epoch` and `rubric_version` / `rubric_sha256`: the epoch that just ended and the frozen
  rubric version it trained against. An export for any other epoch, or naming any other version
  than the one frozen for its epoch, is refused, so epochs never mix.
- `lineage_id`: rollouts that share a prompt, a seed group or a parent lineage must share one.
  A rubric item is admitted only from two or more distinct lineages, so near-duplicates count
  once. Lineages of the held-out known-hack set are excluded from every step.
- `in_loop_reward`: the scalar the trainer optimized. `reward_parts` are recorded only.
- `task_id`: a task registered with the run (its frozen judge and dense definition).

Two fidelities, per rollout:

- `full` (preferred): the actor's own turns in order, each with its visible `content` and its
  tool `calls`; every call carries its `call_id` (unique in the rollout), `arguments`, the
  observation `result` and `is_error`. When the trainer has them, `tokens` (per call) and
  `content_tokens` (per message) carry the sampled token ids and decoded texts. Their
  concatenation must reproduce the part's text (message content exactly; a call's arguments as
  `json.dumps(arguments, sort_keys=True)`); then dense findings land on the exact sampled tokens
  (`sampled_tokens` granularity). Otherwise they land on whole actions (`action` granularity).
  The final answer is the turn with `"final": true`.
- `transcript` (fallback): `transcript` is a list of `{"role": "user" | "assistant" | "tool",
  "content": "..."}` turns plus the scalar reward. Tool calls are not separable, so findings
  land on whole assistant messages; at least one assistant turn is required.

Turn indexes are `0..n-1`. Bounds (`maximum_rollouts`, `maximum_turns`) are part of the
registration; an export over them is refused because the boundary was priced for them.

## What a boundary does

`run_boundary` (`rubric_boundary.py`) walks one explicit state machine
(`rubric_evo_states.py`, `rubric-evolution-boundary-v1`); every move is checked against its
table and journaled, and a finished boundary replays with no call:

1. ingest the export (or refuse it);
2. the decision model tags every actor step with the version's step bank (base behavior bank,
   error bank, the active items' questions);
3. the five-actor open-weight council audits a bounded sample of high-reward rollouts with
   cited evidence; a strict-majority fail flags a rollout. The epoch's mean in-loop reward and
   audit pass rate are recorded; when reward rises by `trigger.reward_rise` or more while the
   pass rate moves by `trigger.audit_bound` or less, the next boundary is forced (larger
   sample, and it cannot end without a diff: a forced boundary that flags nothing escalates);
4. flagged rollouts are diffed against genuine passes by tag signature; when the vocabulary
   cannot see the hack, the open-weight author proposes probe questions from the cited audit
   evidence, and only probes that pass the tag banks' discrimination rules are kept;
5. each adversarial signature is admitted by the rule admission (two or more independent
   lineages, no confident contradiction), compiled into a dense question and an outcome-judge
   criterion, and consolidated against the active items;
6. active items that stop discriminating for `retirement.retire_after` boundaries retire;
7. the held-out known-hack set is graded under the candidate with both parts; a candidate that
   lowers either recall is retried without its retirements, else the current version is kept;
8. the candidate is published as the next version and frozen for the next epoch.

Read the version the next epoch must train against with `current_version`; bind it into a
package with `bind_version`, which records it in the package's mining provenance. Labels made
under an earlier version are refused (`require_label`, and the judges' own identity checks).

## Spend

`boundary_costs` prices one boundary's upper bound (every decision, council, author and judge
request). The `rubric-evolution` front-door slot prices opening the evolution and every epoch
boundary at the forced bound in the dry-run plan, and the front door reserves them against the
single cumulative ledger before it runs. Outside the front door, pass a `PhaseLedger` to
`open_evolution` and `run_boundary`.

## Limits

- The trainer's own reward computation is not checked; only the rollouts and rewards it exports.
- The audit sample is bounded; a hack that appears only outside it is seen at a later boundary
  or through the trigger.
- Probes and items are authored by a model; the host checks shape and citations, the decision
  model and the admission rule decide what is kept.

# Task hardening and task validity

The development task factory can publish scenario variants that every strong actor solves. Two
`exp eval --continuations` actions address that without changing any verifier:

- `check-task-validity` screens one qualified scenario variant for answer leakage, closed-book
  solvability and control drift before it is kept.
- `harden-tasks` runs a registered strong-actor panel on a qualified world and, when the panel
  saturates, asks the world author for harder worlds of the same case, level by level.

Both work for any runtime that the factory can already search: the checks read saved native
witnesses and actor-visible context, and the author binds new worlds through the runtime's
trusted `bind_construction` hook (the workbook runtime ships one; a record-world runtime can
implement the same hook). Neither action trains a model or grants release qualification.

```bash
exp eval PROJECT --root ROOT --continuations validity.json --dry-run
exp eval PROJECT --root ROOT --continuations validity.json --yes --non-interactive
```

Each action shows its cost inventory and asks for spend consent before the first paid draw. A
saved report replays without consent or calls. The terminal shows a short summary; the full
report is saved under `<project>/runtime/task-validity/<id>/report.txt` or
`<project>/runtime/task-hardening/<id>/report.txt`, followed by a `Report:` line.

## check-task-validity

`action: "check-task-validity"`, `task_validity: TaskValiditySpec`:

| Field | Meaning |
| --- | --- |
| `search` | Saved `scenario-search-result` artifact that published the variant. |
| `cell` | Search cell whose variant is checked; it must have passed every control. |
| `leak_judge` | `AuthorSettings` for the model that judges derivability from public context. |
| `closed_book` | `AuthorSettings` for the model that attempts the task without the world. |
| `closed_book_attempts` | Closed-book draws (1 to 8, default 3); any solving draw rejects. |
| `rule` | `ValidityRule`: answer-value thresholds, `specified_values` and `value_recall`. |

Checks run in order. A free check that rejects stops the later paid checks (`not_run`).

1. `controls`: every saved native control still carries its expected independent label, and the
   variant's verifier still grades it the same way. Re-derived from saved witnesses only.
2. `mechanical_leak`: the produced answer values must not appear, as whole tokens, in anything
   the actor reads before acting (messages and tool schemas).
3. `model_leak`: the judge sees the public context, the produced values and the verifier. A
   `derivable` verdict rejects only when it quotes the public context exactly; an unquoted claim
   leaves the task undetermined.
4. `closed_book`: the model receives the instruction-only actor messages, no tools and no world.
   It rejects the task as memorizable if the verifier accepts its answer over the unchanged
   initial state, or if it reproduces at least `value_recall` of the produced values.

Produced answer values are the substantive scalars of the passing solution's final state and
answer, plus final-state equality literals of the verifier, minus values that the initial world
already contains, values every failing control also contains (an output file name, a status
every attempt writes), opaque digests, strings shorter than `minimum_text_characters`, numbers
with fewer than `minimum_number_digits` digits and `specified_values`. Declare in
`specified_values` the values the instruction legitimately asks for, such as a requested new
address; they are reported, never inferred.

The report status is `valid`, `rejected` (some check rejected) or `undetermined` (some check is
unknown, for example an unknown dispatch outcome, which is never redrawn). Only `valid` tasks may
be kept. These checks do not prove that no paraphrased leak exists or that no other model could
solve the task without its world.

## harden-tasks

`action: "harden-tasks"`, `hardening: TaskHardeningSpec`:

| Field | Meaning |
| --- | --- |
| `seed` | Saved `world-proposals` artifact (the factory's `propose-worlds` harbor action). |
| `candidate` | Proposal index of the qualified seed world in that experiment. |
| `author` | `AuthorSettings` for the world author of harder levels. |
| `actors` | Strong panel, two to ten frozen `PanelActor` models. |
| `repeats` | Attempts per actor per candidate (default 4). |
| `saturation` | `SaturationRule`: `minimum_pass_fraction` (default 1.0), `maximum_unknown_fraction` (default 0), `minimum_known_per_actor` (default 1). |
| `band` | `PassBand` for reporting (default 0.2 to 0.8). It is never a keep threshold. |
| `validity` | `ValiditySettings`, the `check-task-validity` models and rule for every candidate. |
| `maximum_levels` | Harder levels after the seed (1 to 5, default 2). |

Level 0 examines the seed candidate. Every level then, in proposal order:

1. takes each proposed world through the existing scenario-search controls (the author's
   verifier and controls, the seed's fixed independent acceptance);
2. runs `check-task-validity` on each world whose controls passed;
3. runs a one-case `compare-actors` panel on each valid world;
4. classifies the panel: `saturated` (every actor's known pass fraction reaches the rule),
   `separating`, `all_fail` or `undetermined` (too many unknown attempts).

A separating, valid world is `kept`, and the experiment stops after that level. Otherwise, when a
world saturated and levels remain, the first saturated world in proposal order is the source of
the next level: the author receives that world's own construction plan and the panel's counts
(never trajectories, verifier text or acceptance) with instructions to add state, plausible
distractors and multi-step dependencies by changing the world, never the wording. The instruction,
tools, permissions and acceptance stay the seed's. Report status is `kept`,
`saturated_at_limit`, `exhausted` (no saturated source remained) or `operationally_incomplete`.

Every level's proposals, scenario searches, validity reports and panels are retained, with each
candidate's disposition: `unqualified` (binding or control refusals), `invalid` or `undetermined`
(validity `rejections`), `saturated`, `all_fail`, `kept` or `operational_failure`. Hardened
worlds are scenario variations of the seed's base task: they keep its lineage, and the report
counts one family however many variants it holds. Unknown author or actor outcomes are never
redrawn.

### Pass-rate bands

Each panelled candidate reports `difficulty`: per-actor counts, the pass rate as the equal-weight
mean of per-actor known pass fractions, a two-sided 95 percent Wilson interval at the actor
count (repeats of one case are not independent units), missing-outcome bounds, and its position
against `band`. The band describes the variant; it is not used to choose, tune or replace a
verifier. With few actors the interval is wide by construction.

# Decision cascade

A decision cascade lets a cheap decision model take an actor's next-action decisions when it is
confident enough, and keeps the main model responsible for every other decision. The confidence
threshold is never a default: it is fitted on the agent's own development trajectories by a
registered objective, confirmed in episodes on held-out lineages, and only then deployed as an
explicit, versioned runtime option. Without that option there is no cascade.

The workflow has three steps, two of them `exp eval --continuations` actions:

1. `fit-decision-cascade` labels reached states, asks the decider once per state and selects a
   threshold.
2. `confirm-decision-cascade` runs the main model alone and the fitted cascade in episodes on
   held-out lineages and judges non-inferiority.
3. A caller deploys the confirmed `CascadePolicy` by wrapping its actor model client.

```bash
exp eval PROJECT --root ROOT --continuations fit.json --dry-run
exp eval PROJECT --root ROOT --continuations fit.json --yes --non-interactive
exp eval PROJECT --root ROOT --continuations confirm.json --dry-run
exp eval PROJECT --root ROOT --continuations confirm.json --yes --non-interactive
```

`--dry-run` validates every input, the labels and the decider without a provider call. Both
actions are paid: they show the decider (and, for confirmation, actor) cost inventory and ask
for spend consent before credentials are read. Each prints a short summary and a `Report:` path
to the full saved result under the project's runtime directory. A completed fit or confirmation
replays without credentials, runtime code or provider calls.

## What is decided

At each actor model call the decision is the next action: one visible tool, by name, or
`respond` (a final answer without another tool call). `OTHER` lets the decider abstain. A tool
named `respond` or `OTHER` collides with these options; the cascade then leaves the main model
responsible and records why.

Two deciders are maintained (`DeciderDefinition.kind`):

- `systemone` asks one native System One `choice` question over the options, through the
  TypeSafe decision wire. It ranks options and never writes arguments. When it is accepted, the
  main model is called once with the tool choice forced to the selected tool (or to no tool for
  `respond`) and fills in the arguments: the decision is delegated, the call is not avoided.
- `sampled_drafts` asks a small chat model for `samples` complete drafts (at least two). Its
  confidence is the share of drafts making the modal decision, a readout of agreement on the
  decision rather than on the arguments. When it is accepted, the first modal draft is executed
  as is, if every call names a visible tool with schema-valid arguments, and the main call is
  avoided.

Both read only the actor-visible messages and tool schemas. A long conversation keeps its first
message and latest turns within the decision wire's input bound and counts what was omitted.

The acceptance rule is shared by deployment, fitting and fixed-question measurement
(`delegates` in `exp/runtime/agents/cascade.py`): the decider is responsible only when its
choice is not `OTHER`, the policy's `never_delegate` restrictions allow it, and its probability
is at least the threshold. Restrictions apply before confidence, so a high probability never
overrides them. A decider failure, an abstention, a restricted or below-threshold choice and an
invalid draft all leave the main model responsible for that step.

## Fit

`action: "fit-decision-cascade"`, `cascade_fit: CascadeFitSpec`:

```json
{"action": "fit-decision-cascade",
 "cascade_fit": {
   "sources": [{"artifact_id": "continuation-study-...-result", "sha256": "..."}],
   "main_model": {"...": "catalog snapshot of the main actor"},
   "development_lineages": ["lineage-a", "lineage-b"],
   "decider": {"kind": "systemone", "alias": "jev", "model": {"...": "snapshot"}},
   "never_delegate": ["refund_ticket"],
   "grid": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
   "objective": {"success_margin": 0.05, "compliance_margin": 0.0,
                 "criterion": "wilson_lower_bound", "minimum_lineages": 20},
   "catalog_sha256": "...", "maximum_physical_attempts": 500,
   "deadline": "2026-11-01T00:00:00Z"}}
```

- **Sources** are complete saved continuation studies (`run`, panels, selections) or actor-panel
  reports of the main model. Panel cells of other models and cascade episodes are skipped; a
  directly named study of another model or with a cascade is refused. `development_lineages`
  must equal exactly the lineages the sources cover.
- **Labels on reached states.** Every saved main-model call is a decision point: the request is
  the reached state, the returned action is the main model's choice, and the branch's verifier
  grade is the verified outcome. Mistakes stay in as failed outcomes; unknown outcomes stay
  counted as unknown. Calls made under a forced tool choice are not the main model's free choice
  and are excluded. Branches from one checkpoint reach shared states, so each state and decision
  is labelled `verified_correct` (some trajectory taking it from there passed),
  `verified_error` (every assessed one failed) or `unverified`.
- **Shadow pass.** The decider is asked once per distinct reached state. Each answer is a durable
  receipt written after its intent; an interrupted dispatch stays unknown and is never redrawn,
  and an unknown answer is never delegated, as at runtime.
- **Estimate.** For every grid value a trajectory keeps its verified outcome unless the cascade
  would have accepted a decision different from the main model's (for a draft, a different
  complete action, ignoring call IDs). From that state the trajectories that took the decider's
  decision from the same state stand in for what would have followed; with none observed, the
  trajectory counts as a task failure and a policy violation. Task success and policy
  compliance are compared per lineage as the paired difference cascade minus main.
- **Selection** follows the registered objective `max_delegation_noninferior`: the largest
  delegated share whose success and compliance differences meet their margins under the
  registered criterion (`point_estimate`, or the lower end of the two-sided 95 percent Wilson
  interval at the lineage count), with at least `minimum_lineages` paired lineages. Ties go to
  the higher threshold. Status `selected` carries the deployable `CascadePolicy`;
  `no_feasible_threshold` and `no_delegation` carry none.

The report keeps every row: delegated decisions, avoided main calls, agreement with the main
model, verified-correct and verified-error delegations, affected trajectories and both paired
differences. It is development evidence only.

## Confirmation

`action: "confirm-decision-cascade"`, `cascade_confirmation: CascadeConfirmationSpec`:

```json
{"action": "confirm-decision-cascade",
 "cascade_confirmation": {
   "fit": {"artifact_id": "decision-cascade-fit-...-result", "sha256": "..."},
   "cases": [{"...": "ContinuationJob of the main model on a held-out lineage"}]}}
```

Each case must be a main-model-only `ContinuationJob` of the fitted model on a lineage, and
source traces, disjoint from every fit source; all cases share one main-model alias. The cases
run through the maintained actor panel twice, on identical checkpoints, worlds, verifiers and
allowances: once with the main model alone and once with the fitted policy. The cascade decides
in the loop, so the episodes reach states the fit never saw.

Confirmation compares task success and policy compliance per lineage with
`unit_difference_summary` (two-sided 95 percent Wilson interval at the lineage count). Unknown
outcomes are imputed against the cascade, as failures in its arm and successes in the main arm;
the known-only differences are reported beside them. The cascade is confirmed only when both
lower interval ends are within the fit's registered margins and at least `minimum_lineages`
lineages are paired. The report also measures delegation in the episodes (delegated share,
avoided calls, forced choices the main model did not follow, the reasons the main model stayed
responsible) beside the fit's offline share, as the size of the reached-state shift.

The paired interval bounds each lineage's difference by its full range, so it is wide with few
lineages: a zero difference over 20 lineages still spans about plus or minus 0.4. Plan the
number of held-out lineages for the margin you register.

Continuation execution consumes only development (`fit` partition) tasks. Held-out here means
lineages disjoint from the fit; the project's reserved held-out tasks stay closed.

## Deployment

A `CascadePolicy` (`version: "decision-cascade-v1"`) binds the exact decider definition, the
threshold and the hard restrictions. It is an explicit option wherever an actor runs:

- `ContinuationStudySpec.cascade` and `PanelActor.cascade` run continuation episodes with the
  cascade; `ContinuationJob` resolves the decider through the same catalog, admitted transport,
  attempt ledger and deadline as the actor. Absent, the study and its identity are unchanged.
- Any actor runtime that receives an injected model client can wrap it:

```python
from exp.runtime.agents.cascade import CascadeModelClient
from exp.runtime.agents.cascade_deciders import SystemOneDecider

cascade = CascadeModelClient(
    main=main_client,
    decider=SystemOneDecider(policy.decider, decide),
    policy=policy,
    record=records.append,
)
episode = runtime.run(task, model=cascade, environment=session)
```

Every step yields a `CascadeDecisionRecord`: the offered options, the decider's choice and
confidence, the threshold, whether the decider or the main model decided, the responsible
model's snapshot, whether the main model was called, whether it followed a forced choice, and
why the main model stayed responsible. Continuation branches keep these records in
`cascade_decisions`, outside the graded outcome, and every decider answer is also a durable
`cascade-decision` receipt. Changing the decider, its model, the threshold or the restrictions
is a different policy that needs its own fit and confirmation.

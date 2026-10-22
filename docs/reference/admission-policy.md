# Trajectory admission policy

Generated trajectories are admitted dimension by dimension under a policy registered before the
results it judges. No dimension is folded into one success flag, and a dimension that does not
apply is recorded as `unassessable`, never passed.

The owner is `exp/optimize/evaluation/equivalence/policy.py` (`AdmissionPolicy`,
`register_admission_policy`), with mechanical observation in `dimensions.py`, margin evaluation
in `calibration.py` and per-trajectory verdicts in `verdicts.py` (`admit_trajectory`).

## Dimensions and applicability

| Dimension | Applies when | Observed from |
|---|---|---|
| `goal_success` | the verifier has an outcome predicate beyond shape and tool policy | the verifier's goal grade |
| `policy_compliance` | the verifier declares `allowed_tools` or `argument_rules` | the verifier's policy grade, protected state excluded |
| `forbidden_effects` | the verifier declares `preserve_state` or `preserved_paths` | the verifier's policy grade, tool policy excluded |
| `state_fidelity` | at least one tool call was executed and observed | world receipts: all native admits; a simulated reply needs calibration; a reply without a receipt for its exact action is unassessable |
| `tool_validity` | at least one tool call was attempted | undeclared, blocked or rolled-back calls refuse; unknown effects or interrupted execution stay unassessable; error replies go to `recovery` |
| `recovery` | a call failed or returned an error reply | admitted when a successful call follows the last error and the goal passes |

Each verdict is `admitted`, `refused` or `unassessable`, with its reason and applicability.

## Writing a policy

Each `DimensionRule` declares:

- `gate`: `required` (must be admitted, so an inapplicable dimension blocks admission),
  `when_applicable` (gates only when its applicability rule holds) or `recorded` (kept in
  provenance, never gating).
- `simulated`: `calibrated` admits a simulated or unproven observation only under an in-scope
  calibration that admits that dimension; `observed` trusts the observation under an
  authored-world contract. `state_fidelity` is always `calibrated`; an authored-world policy
  gates it as `recorded` instead of claiming deployment equivalence.
- `margin` and optional `false_pass_margin`: the largest tolerated simulator disagreement and
  simulator-only admission rate, read at the 95 percent bound.
- `unit` (`task_family` or `lineage`) and `minimum_units`: the independent unit of uncertainty
  and the fewest units a conclusive calibration needs.

`AdmissionPolicy.families` maps source lineages to a declared task family. Lineages in one family
are one independent unit; cases, actors and repeats inside a unit are averaged first.

The trajectory verdict is `refused` when any gating dimension is refused, `unassessable` when any
is unassessable, and `admitted` only when every gating dimension is admitted. Calibration can
withhold an admission but never rescues a refusal.

## Registration and replay

```json
{"action": "register-admission-policy", "admission_policy": {
  "description": "Ticket-world deployment-equivalence claim",
  "families": {"ticket-lineage-0": "tickets", "ticket-lineage-1": "tickets"},
  "dimensions": [
    {"dimension": "goal_success", "gate": "required", "margin": 0.2, "false_pass_margin": 0.05},
    {"dimension": "policy_compliance", "gate": "when_applicable", "margin": 0.2},
    {"dimension": "forbidden_effects", "gate": "when_applicable", "margin": 0.1},
    {"dimension": "state_fidelity", "gate": "required", "margin": 0.2},
    {"dimension": "tool_validity", "gate": "required", "margin": 0.2},
    {"dimension": "recovery", "gate": "when_applicable", "margin": 0.3}
  ]}}
```

```bash
exp eval PROJECT --root ROOT --continuations policy.json --dry-run   # validates, prints identity
exp eval PROJECT --root ROOT --continuations policy.json             # prints the reference
```

The identity is the policy content and `ADMISSION_POLICY_PROTOCOL`; re-registering identical
content replays the original artifact and its original time. Every consumer refuses a policy
registered after the study or comparison registration it judges, so register the policy first
and generate results under it. A policy saved under another protocol fails closed and must be
registered again.

## Consumers

- `compare-environments`: set `equivalence.policy` to the reference. The policy is pinned in the
  comparison registration, and the saved report gains `calibration`, one entry per dimension
  with the candidate's unit-mean disagreement, the repeated-reference control, the
  candidate-minus-control difference and the false-pass rate, each with Wilson intervals at the
  unit count. Admission needs the candidate's own disagreement upper bound within the margin, so
  ordinary actor variability counts against the simulator. Refusal needs the lower bound of the
  excess over the repeated reference to exceed the margin. Unknown pairs, fewer units than
  `minimum_units`, a reference that admits the dimension in too few units (both-fail panels) or
  an interval crossing the margin leave the dimension `unassessable`.
- `export`: `export.policy` is required. Each branch keeps its per-dimension verdicts in the
  `continuation-admission` report, and a branch is admitted only when the independent
  specification and every gating dimension pass. A simulated branch uses the calibration of the
  comparison pinned in its training evidence when that comparison was registered under the same
  policy.
- `export-traces`: `traces.policy` is required and `traces.calibrations` names comparison reports
  registered under the same policy. Every branch records its verdicts in its
  occurrence under `exp.trace.metadata.occurrences[].admission`, so identical trajectories still
  collapse into one trace; the command prints admitted, refused and unassessable branch counts.
  Refused and unassessable branches are still exported, so the evidence denominator stays
  complete.

A calibration applies only to the units it measured and to the same candidate world identity and
actor model. New families, worlds and actors start outside that scope and stay unassessable.

## Limits

Calibration verdicts describe the measured panel under the declared probes and verifier. They are
not native truth, population fidelity outside the measured units, or evidence of training
benefit. Margins in examples and tests are illustrative, not recommendations.

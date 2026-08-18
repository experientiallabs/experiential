# Router optimization contracts

Router optimization consumes one combined plan containing only fit and held-out cells. Fidelity
cells and fidelity reports are not router inputs. The supported sequence is:

1. Complete fit evidence.
2. Fit and lock the bank and policy.
3. Open held-out evidence and write the report.

`wmo optimize router PROJECT --root ROOT` assembles this sequence from the completed project build,
confirmed candidates, approved manual judge calibration, and one bounded provider budget. It does
not run world-model fidelity evaluation before fitting.

Hosted applications use `wmo.run_hosted_router_workflow` instead. That noninteractive service
starts from a restored Project with prepared trace/task evidence, applies one late secret-free
`HostedRouterWorkflowSetup`, constructs the grounded build, records automatic judge setup only as
`provisional` machine evidence, and runs the same automatic router composition without a human or
fidelity gate. Provider clients remain transient injected dependencies; each newly completed build,
policy, and report selection returns a typed stage event and verified Project bundle, while the
terminal result includes a versioned component spend ledger under one finite ceiling.

The hosted setup fixes these values before the first provider dispatch:

- a `builtin_chat` system with a required trimmed `system_prompt` of 1–20,000 characters and
  `maximum_model_calls` from 1–64 (default 8);
- a secret-free catalog plus the world-model, judge, embedder, incumbent, and at least two unique
  candidate aliases;
- retrieval `top_k`; and
- one positive `numeric(20,6)` provider ceiling represented as exact `Decimal` text, never binary
  floating-point authorization.

Every current `ModelRecord` requires `billing_source`, and the resolved immutable `ModelSnapshot`
preserves it. The only values are `host_managed` and `customer_managed`; the source belongs to the
model binding, so two aliases using one provider connection may have different values. Schema-v1
local catalogs and trace datasets decode through narrow legacy migrations as `customer_managed`,
while current payloads that omit the field fail validation.

Each `ProviderSpendEntry` carries the source for its exact component operation, including observed,
locally priced, reserved, ambiguous, and explicit not-incurred evidence. Entries from unlike
sources remain separate. Online `RoutedCompletionEconomics` likewise preserves every alias-free
provider operation with its component, billing source, disposition, economics, and
`operation_count`. Its convenience `router_embedding`, `selected_candidate`,
`by_billing_source`, and overall totals reconcile exactly with those operations, including prior
reserved-ambiguous retries and definitely-not-incurred predispatch failures; it does not expose the
selected catalog alias or model ID.

Applications inject a `HostedAttemptAuthorityStore` that durably binds one random write-once
authority to the Project, attempt, and exact ceiling. It records every paid-operation reservation
before dispatch and keeps ambiguous spend closed across worker loss. A completed provider stage is
visible only after the caller atomically acknowledges the exact verified bundle digest, selected
`ProviderSpendLedger`, and exact ledger total; the workflow emits the completed stage event only
after that acknowledgment. Restarts use `restore_hosted_project_bundle` with the externally
committed digest and can resume only from that latest committed provider stage.

Python applications can use `wmo.compose_router` to run the same sequence with injected review and
setup suppliers, simulator, judge, runtime catalog, and finite budgets. Provider-free callers with
already completed evidence can use `fit_router`, `report_router`, or `optimize_router` directly.

## Router-only plan

Create the plan with `build_evaluation_plan`. Its public signature has no fidelity measurement
protocol or report argument. The resulting `EvaluationPlan` has:

- fit and held-out cells only;
- `fidelity_protocol_sha256=None`.

The router fitter rejects any plan containing a fidelity cell. This keeps world-model quality
measurement outside the policy authorization chain.

## Completed-evidence configuration

`RouterOptimizationConfig` is the provider-free Python contract for already completed evidence:

| Field | Required content |
|---|---|
| `fit` | `EvaluationInputs` containing fit cells only. |
| `held_out` | `EvaluationInputs` containing held-out cells only and naming the same plan. |
| `embedding_set_id` | Frozen embedding-set artifact covering the plan tasks. |
| `incumbent_alias` | Optional selected candidate used as the conservative baseline. |
| `pricing_snapshot_id` | Pricing artifact covering every plan candidate. |
| `guard` | Complete `KnnGuard` thresholds. |
| `judgment_status` | Exactly `provisional` or `human_calibrated`. |
| `created_at` | Timezone-aware artifact timestamp. |
| `code_revision` | Exact source revision producing the artifacts. |

Each `EvaluationInputs` object contains only:

| Field | Required content |
|---|---|
| `evaluation_plan_id` | Router-only combined plan ID. |
| `rollout_set_ids` | Completed rollout-set IDs consumed by this partition. |
| `protocols` | Serialized `EvaluationProtocol` values referenced by the cells. |
| `cell_evidence` | Serialized `EvaluationCellEvidence` values for the named partition. |

There is no `fidelity_report_ids` field. Generate the exact schema from the installed package:

```bash
uv run python -c 'import json; from wmo.optimize.router import RouterOptimizationConfig; print(json.dumps(RouterOptimizationConfig.model_json_schema(), indent=2))'
```

The direct Python boundary is:

```python
from wmo import fit_router, report_router

fit = fit_router(store, fit_config)
result = report_router(store, fit, report_config)
```

`fit_router` locks the policy before `report_router` reads held-out evidence. `optimize_router`
performs both calls in that order for a complete `RouterOptimizationConfig`.

## Separate world-model fidelity measurement

Fidelity testing is an explicit common-evaluation workflow. Call
`build_fidelity_evaluation_plan` with observed comparison cells, a positive `overlap_count`, and
the exact world-model protocol digest. Execute its fidelity cells, then call
`build_fidelity_report` to measure agreement.

```python
from wmo.common.evaluations import (
    build_fidelity_evaluation_plan,
    build_fidelity_report,
)
```

The resulting report contains overlap counts, pair failures, pair scores, and score MAE. It has no
gate, threshold, report-level decision status, or approval timestamp. Fidelity artifacts are not
accepted by `EvaluationInputs`, are not loaded by the router fitter or runtime, and cannot alter
policy fitting.

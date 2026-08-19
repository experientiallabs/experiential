# Router optimization contracts

Router optimization consumes one combined plan containing only fit and held-out cells. Fidelity
cells and fidelity reports are not router inputs. The supported sequence is:

1. Complete fit evidence.
2. Fit and lock the bank and policy.
3. Open held-out evidence and write the report.

`wmo optimize router PROJECT --root ROOT` assembles this sequence from the completed project build,
confirmed candidates, approved manual judge calibration, and one bounded provider budget. It does
not run world-model fidelity evaluation before fitting.

Every current `ModelRecord` requires `billing_source`, and the resolved immutable `ModelSnapshot`
preserves it. The only values are `host_managed` and `customer_managed`; the source belongs to the
model binding, so two aliases using one provider connection may have different values. Schema-v1
local catalogs and trace datasets decode through narrow legacy migrations as `customer_managed`,
while current payloads that omit the field fail validation.

Online `RoutedCompletionEconomics` preserves every alias-free provider operation with its
component, billing source, disposition, economics, and `operation_count`. Its convenience
`router_embedding`, `selected_candidate`, `by_billing_source`, and overall totals reconcile exactly
with those operations, including prior reserved-ambiguous retries and definitely-not-incurred
predispatch failures; it does not expose the selected catalog alias or model ID.

Python applications can use `wmo.compose_router` to run the same sequence with injected review and
setup suppliers, simulator, judge, runtime catalog, and finite budgets. Provider-free callers with
already completed evidence can use `fit_router` and `report_router` directly.

## Router-only plan

Create the plan with `build_evaluation_plan`. Its public signature has no fidelity measurement
protocol or report argument. The resulting `EvaluationPlan` has:

- fit and held-out cells only;
- `fidelity_protocol_sha256=None`.

The router fitter rejects any plan containing a fidelity cell. This keeps world-model quality
measurement outside the policy authorization chain.

## Completed-evidence configuration

`RouterFitConfig` is the provider-free Python contract for already completed fit evidence:

| Field | Required content |
|---|---|
| `fit` | `EvaluationInputs` containing fit cells only. |
| `embedding_set_id` | Frozen embedding-set artifact covering the plan tasks. |
| `incumbent_alias` | Optional selected candidate used as the conservative baseline. |
| `pricing_snapshot_id` | Pricing artifact covering every plan candidate. |
| `guard` | Complete `KnnGuard` thresholds. |
| `judgment_status` | Exactly `provisional` or `human_calibrated`. |
| `created_at` | Timezone-aware artifact timestamp. |
| `code_revision` | Exact source revision producing the artifacts. |

`RouterReportConfig` opens held-out evidence for one already frozen policy:

| Field | Required content |
|---|---|
| `held_out` | `EvaluationInputs` containing held-out cells only and naming the same plan. |
| `embedding_set_id` | Frozen embedding-set artifact covering the plan tasks. |
| `created_at` | Timezone-aware artifact timestamp. |
| `code_revision` | Exact source revision producing the artifacts. |

Each `EvaluationInputs` object contains only:

| Field | Required content |
|---|---|
| `evaluation_plan_id` | Router-only combined plan ID. |
| `rollout_set_ids` | Completed rollout-set IDs consumed by this partition. |
| `protocols` | Serialized `EvaluationProtocol` values referenced by the cells. |
| `cell_evidence` | Serialized `EvaluationCellEvidence` values for the named partition. |

There is no `fidelity_report_ids` field. Generate the exact schemas from the installed package:

```bash
uv run python -c 'import json; from wmo.optimize.router import RouterFitConfig; print(json.dumps(RouterFitConfig.model_json_schema(), indent=2))'
uv run python -c 'import json; from wmo.optimize.router import RouterReportConfig; print(json.dumps(RouterReportConfig.model_json_schema(), indent=2))'
```

The direct Python boundary is:

```python
from wmo import fit_router, report_router

fit = fit_router(store, fit_config)
result = report_router(store, fit, report_config)
```

`fit_router` locks the policy before `report_router` reads held-out evidence.

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

# Router optimization configuration

`wmo build` stops with `proposals_pending`. It does not create simulation, judgment, fidelity,
embedding, or pricing artifacts. Before optimization, an external or provider-authorized workflow
must complete and persist all of these inputs with explicit user consent and budget:

- One combined evaluation plan containing fit, fidelity, and held-out cells.
- Completed rollout sets and judgments for every referenced cell.
- Completed fidelity reports that satisfy the plan-bound fidelity gate.
- One frozen embedding set covering every task used for fitting and reporting.
- One pricing snapshot covering every candidate in the evaluation plan.

Review those artifacts before creating the config. `wmo optimize router` verifies them but does not
produce or repair them.

## Exact fields

The JSON root is `RouterOptimizationConfig`:

| Field | Required content |
|---|---|
| `fit` | `EvaluationInputs` for fit and fidelity cells only. |
| `held_out` | `EvaluationInputs` for held-out cells only, naming the same plan as `fit`. |
| `embedding_set_id` | Artifact ID of the completed frozen embedding set. |
| `incumbent_alias` | Optional candidate alias to use as the conservative baseline. |
| `pricing_snapshot_id` | Artifact ID of the completed pricing snapshot. |
| `guard` | All five `KnnGuard` thresholds: `maximum_neighbors`, `minimum_paired_observations`, `relative_similarity_threshold`, `uncertainty_multiplier`, and `quality_tolerance`. |
| `judgment_status` | Exactly `provisional` or `human_calibrated`. |
| `created_at` | Timezone-aware timestamp used for newly frozen router artifacts. |
| `code_revision` | Exact source revision producing those artifacts. |

Each `EvaluationInputs` object contains:

| Field | Required content |
|---|---|
| `evaluation_plan_id` | The same completed combined plan ID in both partitions. |
| `rollout_set_ids` | Completed rollout-set artifact IDs consumed by that partition. |
| `protocols` | Complete serialized `EvaluationProtocol` contracts referenced by its cells. |
| `cell_evidence` | Complete serialized `EvaluationCellEvidence` contracts for its allowed cells. |
| `fidelity_report_ids` | Completed fidelity-report IDs required by the fit-side gate, or an empty list when none are required by the plan. |

Generate the exact JSON Schema from the installed package when integrating another system:

```bash
uv run python -c 'import json; from wmo.optimize.router import RouterOptimizationConfig; print(json.dumps(RouterOptimizationConfig.model_json_schema(), indent=2))' > router-optimization.schema.json
```

## Create the config from completed typed outputs

The following function writes the accepted file. Every argument comes from a completed artifact or
reviewed evaluation result produced by the separately authorized workflow. No placeholder IDs or
live clients are involved.

```python
from datetime import datetime
from pathlib import Path
from typing import Literal

from wmo.common.evaluations import EvaluationCellEvidence, EvaluationProtocol
from wmo.common.routing import KnnGuard
from wmo.optimize.router import EvaluationInputs, RouterOptimizationConfig


def write_router_config(
    *,
    destination: Path,
    evaluation_plan_id: str,
    fit_rollout_set_ids: tuple[str, ...],
    fit_protocols: tuple[EvaluationProtocol, ...],
    fit_cell_evidence: tuple[EvaluationCellEvidence, ...],
    fidelity_report_ids: tuple[str, ...],
    held_out_rollout_set_ids: tuple[str, ...],
    held_out_protocols: tuple[EvaluationProtocol, ...],
    held_out_cell_evidence: tuple[EvaluationCellEvidence, ...],
    embedding_set_id: str,
    pricing_snapshot_id: str,
    guard: KnnGuard,
    judgment_status: Literal["provisional", "human_calibrated"],
    created_at: datetime,
    code_revision: str,
    incumbent_alias: str | None = None,
) -> None:
    """Write one validated router workflow config from completed evidence."""
    config = RouterOptimizationConfig(
        fit=EvaluationInputs(
            evaluation_plan_id=evaluation_plan_id,
            rollout_set_ids=fit_rollout_set_ids,
            protocols=fit_protocols,
            cell_evidence=fit_cell_evidence,
            fidelity_report_ids=fidelity_report_ids,
        ),
        held_out=EvaluationInputs(
            evaluation_plan_id=evaluation_plan_id,
            rollout_set_ids=held_out_rollout_set_ids,
            protocols=held_out_protocols,
            cell_evidence=held_out_cell_evidence,
        ),
        embedding_set_id=embedding_set_id,
        incumbent_alias=incumbent_alias,
        pricing_snapshot_id=pricing_snapshot_id,
        guard=guard,
        judgment_status=judgment_status,
        created_at=created_at,
        code_revision=code_revision,
    )
    destination.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
```

Run optimization only after this function validates and writes the reviewed completed inputs:

```bash
wmo optimize router support-agent --config router-optimization.json --root .wmo
```

The command opens fit and fidelity evidence first, freezes the bank and policy, and only then opens
held-out evidence. It makes no provider, simulator, environment, or judge calls.

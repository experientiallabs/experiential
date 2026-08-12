# World Model Optimizer

WMO turns local OpenTelemetry or PostHog trace exports into immutable task evidence, fits one
conservative offline router from completed evaluations, and runs the frozen router locally.

## Customer workflow

Install the package, then build a project from one explicit local export:

```bash
pip install world-model-optimizer
wmo build traces.otel.jsonl --source otlp --project support-agent --root .wmo
```

Build accepts 100 through 1000 valid normalized traces. It writes a manifest-bound
`TraceDataset`, deterministic `TaskSet`, and local review handoff whose status is
`proposals_pending`. Build makes zero model, provider, or judge paid calls. Anonymous aggregate
PostHog product telemetry may use the network after artifact persistence unless telemetry is
disabled. It never contains trace content.

The provider-free CLI build stops at review readiness. For CLI optimization, produce approved
rubric, calibration, simulation, judgment, fidelity, frozen embedding, and pricing artifacts under
explicit consent and budget, then create the single config defined in
[Router optimization configuration](docs/reference/router_optimization_config.md). Python callers
can instead use `wmo.compose_router` below. It composes those WMO stages from explicit injected
services and finite budgets, without hidden provider calls.

After the combined fit, fidelity, and held-out evidence is complete, freeze and report one router:

```bash
wmo optimize router support-agent --config router-optimization.json --root .wmo
```

The single JSON config names completed simulation, judgment, and fidelity evidence, a frozen
embedding set, pricing, guard settings, and an exact timestamp and code revision. The workflow
materializes fit-only evidence and freezes the bank and policy before it opens held-out evidence.
Repeating the command with the same config verifies and reuses the same immutable artifacts. It
never calls a provider.

Run the frozen router through the development-only loopback adapter:

```bash
wmo run support-agent --root .wmo --port 8000
```

The command can bind only to `127.0.0.1`. Every completion request must provide a caller-owned
`X-WMO-Episode-ID`, which keeps the selected candidate sticky for that episode. Startup loads and
verifies the policy, bank, pricing, feature contract, model aliases, and connection identities.
Selection happens online, but the policy never learns online.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-WMO-Episode-ID: customer-conversation-42' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

To train a model from an already persisted, project-bound SFT dataset, use
`wmo optimize model PROJECT`. The command validates the complete local artifact graph and a finite
cost bound before requesting consent for managed Tinker execution.

## Python composition

Python callers can compose the full artifact chain with explicit dependencies. WMO never resolves
a model, simulator, agent, judge, credential, consent, or budget implicitly:

```python
from datetime import UTC, datetime
from pathlib import Path

from wmo import (
    LocalTraceSource,
    RouterCompositionBudget,
    RouterWorkflowServices,
    compose_router,
)
from wmo.common.models import ModelMessage, ModelRequest
from wmo.common.project import ProjectConfig, ProjectStore

root = Path(".wmo")
project = ProjectStore(root, "support-agent")
project.initialize(ProjectConfig(project_id="support-agent"))

services = RouterWorkflowServices(
    # Persists an approved Rubric and JudgeCalibration under application consent.
    review_supplier=approved_review_supplier,
    # Supplies reviewed production overlaps, candidates, embeddings, pricing, and guards.
    setup_supplier=reviewed_evaluation_setup_supplier,
    # Binds WorldModelSimulator to WMO's plan with explicit model clients and AgentRuntime.
    simulator_factory=world_model_simulator_factory,
    judge=judge_service,
    fidelity_approval=fidelity_approval_service,
    runtime_catalog=runtime_catalog,
)
result = compose_router(
    project,
    LocalTraceSource(Path("traces.otel.jsonl"), source="otlp"),
    services=services,
    budget=RouterCompositionBudget(
        maximum_simulation_cost_usd=25.0,
        maximum_judgments=100,
    ),
    created_at=datetime.now(UTC),
    code_revision="your-exact-revision",
)

# This is the explicit online model-call boundary.
response = result.runtime.complete(
    ModelRequest(messages=(ModelMessage(role="user", content="Help me"),)),
    episode_id="customer-conversation-42",
)
print(response.decision.selected_alias, response.response.output)
```

`compose_router` creates the plan and finite-cost `SimulationSpec`, executes the injected
simulator, invokes the injected judge only for missing judgments, builds and explicitly approves
fidelity, freezes fit artifacts, opens held-out evidence only after policy lock, reports, and
returns the verified W11 `RouterRuntime`. Exact replay does not repeat completed simulation or
judgment calls. Callable contracts and `RouterEvaluationSetup` fields live in
`wmo.workflow.router`.

## Telemetry

Anonymous aggregate PostHog product telemetry is enabled by default. It never includes prompts,
traces, actions, observations, paths, model names, credentials, or raw customer content.

```bash
wmo config telemetry status
wmo config telemetry disable
wmo config telemetry enable
```

The preference is stored locally in `.wmo/settings.toml`.

## Development

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest -q
```

Repository and documentation conventions live in [AGENTS.md](./AGENTS.md).

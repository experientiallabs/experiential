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

WMO stops at review readiness. Rubric review, simulation, judgment, fidelity validation, frozen
embeddings, and pricing are required completed inputs to optimization. Produce them through an
explicit external or provider-authorized workflow with its own consent and budget. Then create the
single configuration using the exact typed recipe and field definitions in
[Router optimization configuration](docs/reference/router_optimization_config.md).

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

The CLI calls the same domain services available to Python callers:

```python
from datetime import UTC, datetime
from pathlib import Path

from wmo import build_project, load_project_router, optimize_router
from wmo.common.models import ModelMessage, ModelRequest
from wmo.common.project import ProjectConfig, ProjectStore
from wmo.optimize.router import RouterOptimizationConfig
from wmo.simulation.ingest.otlp import load_otlp_file

root = Path(".wmo")
project = ProjectStore(root, "support-agent")
project.initialize(ProjectConfig(project_id="support-agent"))

built = build_project(
    load_otlp_file(Path("traces.otel.jsonl")),
    project,
    created_at=datetime.now(UTC),
    code_revision="your-exact-revision",
)
assert built.review.status == "proposals_pending"

# Stop here until an explicitly authorized external workflow has persisted the completed
# evaluation plan, rollout sets, judgments, fidelity reports, frozen embeddings, and pricing.
# Create this file from those typed outputs using docs/reference/router_optimization_config.md.
router_config = RouterOptimizationConfig.model_validate_json(
    Path("router-optimization.json").read_bytes()
)
optimized = optimize_router(project.artifacts, router_config)

# This is the explicit online model-call boundary.
runtime = load_project_router(
    "support-agent",
    root,
    policy_id=optimized.optimization.policy.policy_id,
)
response = runtime.complete(
    ModelRequest(messages=(ModelMessage(role="user", content="Help me"),)),
    episode_id="customer-conversation-42",
)
print(response.decision.selected_alias, response.response.output)
```

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

# Frozen router cookbook

This walkthrough uses only local files and already completed evidence until the final explicit
online request.

## 1. Build deterministic task evidence

```bash
wmo build support-agent --traces traces.otel.jsonl --source otlp --root .wmo
```

The command requires 100 through 1000 valid normalized traces. It reads the raw file once,
persists normalized traces and representative fit and held-out tasks, then shows the selected
models, split, conservative embedding cost, and strict ceiling. Under that ceiling it builds the
serving and fit-only RAG indexes without a confirmation prompt and binds the grounded world model.
It does not call the selected world model or judge. Use `--dry-run` for the same preflight with no
provider call or completed-build selection. Anonymous aggregate PostHog product telemetry may send
after persistence unless disabled. Repeating the command with identical source content and code
revision verifies and reuses the exact artifact manifests and payloads without provider calls.

## 2. Complete review and evaluation explicitly

The build ceiling authorizes only its bounded embedding work, so the next input remains a reviewed
completed evidence config. Python applications can use `wmo.compose_router` for actual WMO composition. They
inject an approved-review supplier, reviewed setup supplier, plan-bound simulator factory, judge,
runtime catalog, finite simulation-dollar ceiling, and finite judgment-call ceiling. WMO creates
the plan and phase-scoped simulation specs, runs only fidelity and fit cells, persists judgments
and an immutable approval receipt, fits and verifies the policy lock, then opens held-out cells,
reports, and returns `RouterRuntime`. A complete replay reuses those verified artifacts without
calling the simulator, judge, approval callback, or fit workflow again. The finite simulation
ceiling is shared across both phases: held-out receives only the remainder after verified
candidate plus simulator or environment spend, and unknown spend blocks the phase transition.
After the workflow completes, one anonymous metadata-only simulation event may send unless
telemetry is disabled.

The setup supplier provides the application-owned facts WMO cannot invent: approved rubric and
calibration, candidate snapshots, reviewed production overlap rollouts, exact protocols, frozen
embeddings, pricing, and guard thresholds. The simulator factory binds explicit candidate and
world-model clients plus the customer `AgentRuntime` to WMO's frozen plan. See the executable
public example in the root README and exact contracts in `wmo.optimize.router.composition`.

Create `router-optimization.json` from those exact typed outputs using
[the configuration recipe](../reference/router_optimization_config.md). The recipe names and
explains every field. The file separates fit plus fidelity evidence from held-out evidence under
one plan. It also fixes the embedding and pricing IDs, guard, judgment status, timestamp, and code
revision.

## 3. Freeze and report

```bash
wmo optimize router support-agent \
  --config router-optimization.json \
  --root .wmo
```

The command verifies completed rollout-set membership and the plan-bound fidelity gate. It opens
fit evidence, writes the bank and policy, then and only then opens held-out evidence and writes the
weighted report. It has no model, environment, judge, E2B, or network client.

Run the same command again to verify idempotent resume. Changed content under an existing artifact
identity fails instead of overwriting evidence.

## 4. Run locally

```bash
wmo run support-agent --root .wmo --port 8000
```

This development adapter binds only to `127.0.0.1`. Send a standard OpenAI request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

Chat requests are stateless. Responses continuations preserve routing affinity through the official
`previous_response_id` field. Request-time embedding failure falls back to the frozen conservative
baseline. Neither path updates the policy or evidence bank.

By default, completed traffic enters the durable project journal and can later feed runtime RAG and
SFT preparation. Use ghost mode when traffic must not be saved:

```bash
wmo run support-agent --root .wmo --ghost
```

Python applications use the same boundary with `wmo.load_router("support-agent", ghost=True)`.
Ghost mode still performs routed provider calls, but it writes no interaction, replay, RAG, or SFT
state. Caller idempotency keys are accepted but cannot replay, so a retry may dispatch again.

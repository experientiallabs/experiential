# Frozen router cookbook

This walkthrough uses only local files and already completed evidence until the final explicit
online request.

## 1. Build deterministic task evidence

```bash
wmo build traces.otel.jsonl --source otlp --project support-agent --root .wmo
```

The command reads the raw file once. It persists normalized traces and representative fit and
held-out tasks, then records that rubric proposals are pending. It makes zero paid calls. Repeating
the command with identical source content and code revision reuses the same artifact identities.

## 2. Complete review and evaluation explicitly

Use the rubric, simulation, judgment, and evaluation services to produce a combined plan and its
completed evidence. Provider-backed work must have explicit consent and a budget outside `build`.
Freeze the resulting IDs in one `router-optimization.json` file. That file contains separate fit
plus fidelity evidence and held-out evidence under one plan, along with the frozen embedding and
pricing IDs, guard, judgment status, timestamp, and code revision.

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

This development adapter binds only to `127.0.0.1`. Send a caller-owned episode ID with every
request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-WMO-Episode-ID: customer-conversation-42' \
  -d '{"model":"support-agent","messages":[{"role":"user","content":"Help me"}]}'
```

The first decision is sticky for that episode. Request-time embedding failure falls back to the
frozen conservative baseline. Neither path updates the policy or evidence bank.

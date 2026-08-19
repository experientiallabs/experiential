---
name: test-build
description: Run the live WMO build-to-serve path as an unattended automation. Fits the pinned terminal-tasks trace export through the CLI, calibrates the judge non-interactively, fits a router, verifies exact replay, and serves the frozen policy on loopback. Use when asked to test the build path, verify cloud setup, or attach this skill to an automation.
---

# Test the build path

Execute the current CLI path on one isolated local root. This is a live provider run, not the
deterministic fixture test. It proves the commands complete, that the fitted router replays
exactly, and that routed loopback traffic journals durably. It does not claim router quality.

Every step is a `wmo` or standard shell command. Do not edit `models.toml`, project files, or
artifacts by hand. Do not call internal Python APIs. If a step cannot be done through the CLI,
stop and report the UX gap.

Do not use OpenRouter. Do not run `wmo optimize model`. Do not start E2B sandboxes. Do not commit
`.wmo/`, traces, or any generated artifact.

Time every numbered step and report the table at the end.

## Fit dataset

This skill fits the pinned public terminal-tasks export by default. A caller may override any
variable below; when overriding, the caller owns the digest. Record the effective values in the
final report so the fitted dataset is unambiguous.

- `WMO_TEST_BUILD_PROJECT` (default `terminal-tasks`): local project ID passed to every command
- `WMO_TEST_BUILD_TRACES` (default `$HOME/wmo-fixtures/terminal-tasks-traces.otel.jsonl`):
  local trace-export file
- `WMO_TEST_BUILD_TRACES_SHA256` (default
  `21c62cba7e3372cbf03df051dc2408699fbf8ea3561ba661b599e4949f0e5d42`): lowercase hex digest
- `WMO_TEST_BUILD_SOURCE` (default `otlp`): trace source format

When the default file is absent, download the pinned export once with `HF_TOKEN`:

```
https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl
```

Do not skip the digest check. If the file is absent and cannot be downloaded, or the digest does
not match, stop.

Optional work root: `WMO_TEST_BUILD_ROOT` (default `/tmp/wmo-test-build`). Keep the trace file
outside that directory. The skill deletes the work root at the start of the run.

## Secrets

Fail before any write if a required variable is missing or empty. Never print a secret value.

Required from the first step:

- `OPENAI_API_KEY`: embeddings, judge, and the OpenAI router candidate
- `EXP_WM_ENDPOINT`: hosted OpenAI-compatible base URL, used exactly as provided. The service
  serves `/chat/completions` at that base, so never append a `/v1` suffix.
- `EXP_WM_API_KEY`: credential for that endpoint

Required only when the pinned export must be downloaded:

- `HF_TOKEN`

Present but unused by this skill:

- `TINKER_API_KEY`
- `E2B_API_KEY`

## Pins

Keep these exact. If a pin is unavailable, stop and report which one failed.

- World model, first router candidate, and incumbent: hosted OpenAI-compatible
  `deepseek-v4-flash` as alias `dsflash`
- Second hosted router candidate: `deepseek-ai/DeepSeek-V4-Flash-0731` as alias `dsflash0731`,
  served by the same hosted endpoint, so both hosted model IDs exercise the same connection.
  The endpoint reports `deepseek-v4-flash` as the served identity for this dated ID, so the
  alias pins `served_model_id` to `deepseek-v4-flash` and identity validation stays strict
- Judge and third router candidate: OpenAI `gpt-5.6-luna` as alias `luna`. It rejects an
  explicit temperature, so its capabilities pin `supports_temperature` false and
  `reasoning_effort` `xhigh`. Its `maximum_output_tokens` is `32000` because the judge
  preflight requires an output budget of at least `16384` and fails closed below it.
- Embedder: OpenAI `text-embedding-3-small` as alias `embed`
- Shared command budget: `$75` via `wmo config budget`
- Build ceiling: `$5`
- Judge calibration sample: `10` lineages, labeled non-interactively in two passes (see step 5)
- Router ceilings: `$60` provider cost, `210` judgments, `10` model calls. Three candidates
  over 50 fit plus 20 held-out tasks schedule 210 judgment cells, and preflight reserves the
  complete schedule conservatively, so lower ceilings fail closed before any call
- Loopback: `127.0.0.1:8000` with durable journaling (do not pass `--ghost`)

The uniform label of `1` is a path-exercise input on the default 0-1 task-success axis. It is
not a human quality review. Every label carries a paired `--judgment` rationale so a label that
corrects the judge proposal is complete on the same pass.

Retry-exhausted simulation cells and oversized or empty judge cells become durable per-cell
exclusions instead of aborting the run; report them if they appear, they are not a failure.
Stochastic world-model protocol failures are retryable: resume supersedes them with a fresh
attempt, and a named incumbent may fit with partial coverage as long as at least one fit task
is covered, with the uncovered task IDs surfaced in the frozen policy.

## Timing

Keep the timing log outside the wiped work root. Wrap every numbered command with this helper
except the long-running `wmo run` process, which is timed as `run-ready` until the listen line
appears. Do not use Python.

```bash
WORK="${WMO_TEST_BUILD_ROOT:-/tmp/wmo-test-build}"
BENCH=/tmp/wmo-test-build-bench.tsv
: > "$BENCH"
bench() {
  local name="$1"
  shift
  local start end status
  start=$(date +%s.%N)
  "$@"
  status=$?
  end=$(date +%s.%N)
  awk -v n="$name" -v s="$start" -v e="$end" -v st="$status" \
    'BEGIN { printf "%s\t%.3f\t%d\n", n, e - s, st }' >> "$BENCH"
  return "$status"
}
```

## 1. Preconditions

```bash
export PATH="$HOME/.local/bin:$PATH"
test -n "$OPENAI_API_KEY"
test -n "$EXP_WM_ENDPOINT"
test -n "$EXP_WM_API_KEY"
PROJECT="${WMO_TEST_BUILD_PROJECT:-terminal-tasks}"
TRACES="${WMO_TEST_BUILD_TRACES:-$HOME/wmo-fixtures/terminal-tasks-traces.otel.jsonl}"
TRACES_SHA256="${WMO_TEST_BUILD_TRACES_SHA256:-21c62cba7e3372cbf03df051dc2408699fbf8ea3561ba661b599e4949f0e5d42}"
SOURCE="${WMO_TEST_BUILD_SOURCE:-otlp}"
if [ ! -f "$TRACES" ]; then
  test -n "$HF_TOKEN"
  mkdir -p "$(dirname "$TRACES")"
  curl -sSL --fail -H "Authorization: Bearer $HF_TOKEN" \
    -o "$TRACES" \
    "https://huggingface.co/datasets/experiential-labs/wmo-terminal-tasks-traces/resolve/540883e451dc13d34fb50fdd36b143cb0f1fb0db/traces.otel.jsonl"
fi
printf '%s  %s\n' "$TRACES_SHA256" "$TRACES" > /tmp/wmo-test-build-traces.sha256
bench verify-traces sha256sum -c /tmp/wmo-test-build-traces.sha256
bench preconditions uv run wmo --help
```

## 2. Isolated work root

```bash
WORK="${WMO_TEST_BUILD_ROOT:-/tmp/wmo-test-build}"
rm -rf "$WORK"
mkdir -p "$WORK"
ROOT="$WORK/.wmo"
test -f "$TRACES"
```

## 3. Configure providers

Write only secret-free catalog names. Expand the OpenAI-compatible base URL from the environment
so the catalog stores the URL, not a credential. `openai-compatible` requires `base_url` on
`--connection-json`. Fail if the base URL contains a double quote.

```bash
case "$EXP_WM_ENDPOINT" in *\"*) echo "EXP_WM_ENDPOINT cannot contain double quotes" >&2; exit 1 ;; esac
EXPWM_JSON="{\"name\":\"expwm\",\"provider\":\"openai-compatible\",\"api_key_env\":\"EXP_WM_API_KEY\",\"base_url\":\"$EXP_WM_ENDPOINT\"}"
bench config-providers uv run wmo config providers \
  --root "$ROOT" \
  --non-interactive \
  --connection-json '{"name":"openai","provider":"openai","api_key_env":"OPENAI_API_KEY"}' \
  --connection-json "$EXPWM_JSON" \
  --model-json '{"alias":"dsflash","connection":"expwm","model":"deepseek-v4-flash","capabilities":{"supports_completions":true,"supports_tools":true,"supports_structured_output":true,"supports_temperature":true,"supports_embeddings":false,"maximum_output_tokens":16000,"context_window_tokens":1048576,"input_cost_per_million_tokens_usd":0,"output_cost_per_million_tokens_usd":0,"cached_input_cost_per_million_tokens_usd":0,"cache_write_cost_per_million_tokens_usd":0}}' \
  --model-json '{"alias":"dsflash0731","connection":"expwm","model":"deepseek-ai/DeepSeek-V4-Flash-0731","served_model_id":"deepseek-v4-flash","capabilities":{"supports_completions":true,"supports_tools":true,"supports_structured_output":true,"supports_temperature":true,"supports_embeddings":false,"maximum_output_tokens":16000,"context_window_tokens":1048576,"input_cost_per_million_tokens_usd":0,"output_cost_per_million_tokens_usd":0,"cached_input_cost_per_million_tokens_usd":0,"cache_write_cost_per_million_tokens_usd":0}}' \
  --model-json '{"alias":"luna","connection":"openai","model":"gpt-5.6-luna","capabilities":{"supports_completions":true,"supports_temperature":false,"reasoning_effort":"xhigh","supports_tools":true,"supports_structured_output":true,"supports_embeddings":false,"context_window_tokens":1050000,"maximum_output_tokens":32000,"input_cost_per_million_tokens_usd":0.2,"output_cost_per_million_tokens_usd":1.2,"cached_input_cost_per_million_tokens_usd":0.02,"cache_write_cost_per_million_tokens_usd":0.25}}' \
  --model-json '{"alias":"embed","connection":"openai","model":"text-embedding-3-small","capabilities":{"supports_completions":false,"supports_embeddings":true,"input_cost_per_million_tokens_usd":0.02,"context_window_tokens":8192}}' \
  --world-model dsflash \
  --judge luna \
  --embedder embed
```

Confirm with `grep`, not by editing the file: `models.toml` names the four aliases, points
`world_model` at `dsflash`, and contains no credential values. Zero token prices on the hosted
world-model alias mean self-hosted spend, not a missing price.

```bash
bench config-budget uv run wmo config budget 75 --root "$ROOT"
```

The default per-command budget is `$10`. The router step freezes a `$60` provider ceiling and
reserves the complete conservative schedule up front, so the shared ceiling must be raised
before that command or it fails closed.

## 4. Build

```bash
bench build uv run wmo build "$PROJECT" "$TRACES" \
  --source "$SOURCE" \
  --root "$ROOT" \
  --yes \
  --non-interactive \
  --max-build-cost-usd 5
```

Success prints a completed build and an embedding spend ceiling. Record the serving RAG, fit RAG,
and world-model IDs from that output. Fail if the command asks for a prompt or exits nonzero.

## 5. Judge setup and calibration

```bash
bench judge-setup uv run wmo config judge setup "$PROJECT" \
  --root "$ROOT" \
  --approve \
  --non-interactive
```

Calibration is judge-first and fully non-interactive in two passes. The first pass pays for the
judge proposals, then fails closed with a paste-ready `--label TRACE_ID:task-success=SCORE`
expression for every sampled lineage. That exit is expected; do not treat it as a run failure.

```bash
CAL_LOG="$WORK/calibrate-propose.log"
bench judge-calibrate-propose bash -c \
  'uv run wmo config judge calibrate "$0" --root "$1" --sample-size 10 --yes --non-interactive > "$2" 2>&1; test $? -ne 0' \
  "$PROJECT" "$ROOT" "$CAL_LOG"
grep -q -- '--label' "$CAL_LOG"
```

Extract every printed label key and rerun with the uniform score of `1` plus a paired
`--judgment` rationale, so a label that corrects the judge proposal is complete on the same
pass. The second pass replays the already-paid proposals, applies the labels, and approves.
The error renders in a wrapped panel that can split `--label` from its key, so match the
`KEY=SCORE` token directly instead of the flag prefix.

```bash
CAL_ARGS=()
while IFS= read -r key; do
  CAL_ARGS+=(--label "${key}=1")
  CAL_ARGS+=(--judgment "${key}=Uniform path-exercise label pinned to the top score.")
done < <(grep -oE '[A-Za-z0-9_:-]+=SCORE' "$CAL_LOG" | sed 's/=SCORE$//' | sort -u)
test "${#CAL_ARGS[@]}" -ge 2
bench judge-calibrate uv run wmo config judge calibrate "$PROJECT" \
  --root "$ROOT" \
  --sample-size 10 \
  --yes \
  --approve \
  --non-interactive \
  "${CAL_ARGS[@]}"
```

Catalog prices supply the judge cost. Success prints an approved calibration artifact ID.

## 6. Optimize the router

At least two distinct candidates are required; `dsflash`, `dsflash0731`, and `luna` are all
already configured, so no extra provider setup happens here. The two hosted candidates share
one endpoint and exercise both of its served model IDs.

```bash
bench optimize uv run wmo optimize router "$PROJECT" \
  --root "$ROOT" \
  --candidate dsflash \
  --candidate dsflash0731 \
  --candidate luna \
  --incumbent dsflash \
  --maximum-provider-cost-usd 60 \
  --maximum-judgments 210 \
  --maximum-model-calls 10 \
  --yes \
  --non-interactive
bench optimize-replay uv run wmo optimize router "$PROJECT" \
  --root "$ROOT" \
  --candidate dsflash \
  --candidate dsflash0731 \
  --candidate luna \
  --incumbent dsflash \
  --maximum-provider-cost-usd 60 \
  --maximum-judgments 210 \
  --maximum-model-calls 10 \
  --yes \
  --non-interactive
```

Success prints a policy ID and a report ID. The second run must print
`replay: verified completed optimization` and make no new provider calls. Per-cell exclusions
for retry-exhausted or unusable judge cells are durable evidence, not failures; record any that
appear.

A malformed world-model transition (`TextWorldModelProtocolError`, failure phase
`world_model_protocol`) is a stochastic, retryable failure: resume supersedes it with a fresh
attempt, and one optimize invocation re-executes only the superseded cells until they converge
or the per-cell attempt cap is exhausted, at which point the cell becomes durable excluded
evidence. A named incumbent fits with partial coverage; the frozen policy lists any uncovered
fit task IDs. Report retry counts and coverage gaps if they appear, they are not a failure.

## 7. Serve the router and send official traffic

Start durable journaling in the background. Do not pass `--ghost`. The gateway is
authenticated: startup prints `Gateway key file: <path>` under `$ROOT/gateway/`, and every
request must carry that key as a bearer token. Time readiness separately from the Chat
Completions request. The repeated request with the same `Idempotency-Key` must return the
journaled completion with the same response ID, which proves durable journaling without any
new provider call.

```bash
SERVE_LOG=/tmp/wmo-test-build-serve.log
uv run wmo run "$PROJECT" --root "$ROOT" --port 8000 > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
bench run-key bash -c "until grep -q 'Gateway key file:' $SERVE_LOG; do sleep 0.2; done"
GATEWAY_KEY="$(tr -d '\n' < "$(sed -n 's/^Gateway key file: //p' "$SERVE_LOG" | head -1)")"
bench run-ready bash -c "until curl -sS --fail -H 'Authorization: Bearer $GATEWAY_KEY' http://127.0.0.1:8000/v1/models >/dev/null; do sleep 0.2; done"
SMOKE_KEY="test-build-$(date +%s)"
bench chat-smoke curl -sS --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $SMOKE_KEY" \
  -o /tmp/wmo-test-build-chat-first.json \
  -d "{\"model\":\"$PROJECT\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}]}"
bench chat-replay curl -sS --fail http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_KEY" \
  -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $SMOKE_KEY" \
  -o /tmp/wmo-test-build-chat-replay.json \
  -d "{\"model\":\"$PROJECT\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word ready.\"}]}"
kill "$SERVER_PID"
wait "$SERVER_PID" || true
grep -o '"id":"[^"]*"' /tmp/wmo-test-build-chat-first.json | head -1
grep -o '"id":"[^"]*"' /tmp/wmo-test-build-chat-replay.json | head -1
```

Both responses must be HTTP 200 JSON with a completion choice, and the two response IDs must be
identical. The listen line `Uvicorn running on http://127.0.0.1:8000` must appear in the serve
log before the first request. The key stays in the local root and never enters the report.

The locked `wmo config` surface has no retrieval-refresh command, so the durable journal
produced here is the closed-loop evidence this skill verifies. Ingesting journaled traffic into
a new retrieval index is out of scope for this skill.

## Pass or fail

Pass only when every step above succeeded, the effective digest matched, both optimize runs
agreed on the same policy ID, the loopback request returned 200, and the idempotent replay
returned the same response ID.

In the final report include:

- Fit dataset: project ID, source, trace path, and SHA-256
- Build, calibration, policy, and report IDs
- Observed spend ceilings printed by build, calibrate, and optimize
- Whether optimize replayed on the second run
- The HTTP status of both loopback requests and whether the response IDs matched
- Any durable per-cell exclusions reported by optimize
- A timing table with every `bench` row: step name, seconds, exit status
- Any command that prompted, retried, or used a model other than the pins

If any step fails (other than the expected first calibration pass), stop. Do not substitute a
different trace export, loosen a digest, invent a project ID, substitute OpenRouter, skip
calibration approval, raise a spend ceiling, edit files by hand, or run Tinker SFT to force a
green result.

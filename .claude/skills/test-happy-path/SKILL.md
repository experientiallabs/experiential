---
name: test-happy-path
description: Run the live WMO happy path as an unattended Cursor automation. Fits one caller-supplied trace export through the CLI, calibrates the judge, fits a router, serves it, and refreshes retrieval from the new journal. Use when asked to test the happy path, verify cloud setup, or attach this skill to a Cursor automation.
---

# Test the happy path

Execute the current CLI path on one isolated local root. This is a live provider run, not the
deterministic fixture test. It proves the commands complete and that new routed traffic can enter
a new retrieval index. It does not claim router quality.

Every step is a `wmo` or standard shell command. Do not edit `models.toml`, project files, or
artifacts by hand. Do not call internal Python APIs. If a step cannot be done through the CLI,
stop and report the UX gap.

Do not use OpenRouter. Do not run `wmo optimize model`. Do not start E2B sandboxes. Do not commit
`.wmo/`, traces, or any generated artifact.

Time every numbered step and report the table at the end.

## Fit dataset

This skill fits exactly one caller-supplied trace export. It does not choose a corpus, download a
named public dataset, or fall back to a fixture.

The caller must set every variable below before the first write. Fail if any is missing, empty,
or unusable. Record the values in the final report so the fitted dataset is unambiguous.

- `WMO_HAPPY_PATH_PROJECT`: local project ID passed to every `wmo` command
- `WMO_HAPPY_PATH_TRACES`: existing local trace-export file (OTLP or PostHog)
- `WMO_HAPPY_PATH_TRACES_SHA256`: lowercase hex digest of that file
- `WMO_HAPPY_PATH_SOURCE`: `otlp` or `posthog`, matching the file

Do not invent a project ID. Do not fetch a replacement file. Do not skip the digest check. If the
file is absent or the digest does not match, stop.

Optional work root: `WMO_HAPPY_PATH_ROOT` (default `/tmp/wmo-happy-path`). Keep the trace file
outside that directory. The skill deletes the work root at the start of the run.

## Secrets

Fail before any write if a required variable is missing or empty. Never print a secret value.

Required from the first step:

- `OPENAI_API_KEY`: embeddings, judge, and the first router candidate
- `WMO_ENDPOINT_BASE_URL`: hosted OpenAI-compatible origin, including the `/v1` suffix
- `WMO_ENDPOINT_API_KEY`: credential for that origin

Required only when optimize needs a second candidate:

- `ANTHROPIC_API_KEY`: Haiku 4.5, added immediately before `wmo optimize router`

Present but unused by this skill:

- `TINKER_API_KEY`
- `E2B_API_KEY`

## Pins

Keep these exact. If a pin is unavailable, stop and report which one failed.

- World model: hosted OpenAI-compatible `deepseek-v4-flash` as alias `deepseek-v4-flash`
- Judge, OpenAI candidate, incumbent: `gpt-5.6-luna` as alias `gpt-5-6-luna`
- Embedder: `text-embedding-3-small`
- Second optimize candidate, only because the router CLI requires two: `claude-haiku-4-5`
  (`claude-haiku-4-5-20251001`)
- Shared command budget: `$25` via `wmo config budget` (covers the optimize estimate)
- Build ceiling: `$5`
- Judge calibration sample: `10` labels via `--label-all 1`
- Router ceiling: `$25`
- Refresh ceiling: `$5`
- Loopback: `127.0.0.1:8000` with durable journaling (do not pass `--ghost`)

The uniform label of `1` is a path-exercise input on the default 0-1 task-success axis. It is
not a human quality review.

Do not add Anthropic or Haiku during build setup. The second candidate exists only for
`wmo optimize router`.

## Timing

Keep the timing log outside the wiped work root. Wrap every numbered command with this helper
except the long-running `wmo run` process, which is timed as `run-ready` until the listen line
appears. Do not use Python.

```bash
WORK="${WMO_HAPPY_PATH_ROOT:-/tmp/wmo-happy-path}"
BENCH=/tmp/wmo-happy-path-bench.tsv
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
test -n "$WMO_ENDPOINT_BASE_URL"
test -n "$WMO_ENDPOINT_API_KEY"
test -n "$WMO_HAPPY_PATH_PROJECT"
test -n "$WMO_HAPPY_PATH_TRACES"
test -n "$WMO_HAPPY_PATH_TRACES_SHA256"
test -n "$WMO_HAPPY_PATH_SOURCE"
test -f "$WMO_HAPPY_PATH_TRACES"
case "$WMO_HAPPY_PATH_SOURCE" in
  otlp|posthog) ;;
  *) echo "WMO_HAPPY_PATH_SOURCE must be otlp or posthog" >&2; exit 1 ;;
esac
printf '%s  %s\n' "$WMO_HAPPY_PATH_TRACES_SHA256" "$WMO_HAPPY_PATH_TRACES" \
  > /tmp/wmo-happy-path-traces.sha256
bench verify-traces sha256sum -c /tmp/wmo-happy-path-traces.sha256
bench preconditions uv run wmo --help
```

## 2. Isolated work root

```bash
WORK="${WMO_HAPPY_PATH_ROOT:-/tmp/wmo-happy-path}"
rm -rf "$WORK"
mkdir -p "$WORK"
ROOT="$WORK/.wmo"
test -f "$WMO_HAPPY_PATH_TRACES"
```

## 3. Configure build providers

Write only secret-free catalog names. Expand the OpenAI-compatible origin from the environment so
the catalog stores the URL, not a credential.

```bash
bench config-providers uv run wmo config providers \
  --root "$ROOT" \
  --non-interactive \
  --connection-json '{"name":"openai","provider":"openai","api_key_env":"OPENAI_API_KEY"}' \
  --connection-json '{"name":"hosted-wm","provider":"openai-compatible","api_key_env":"WMO_ENDPOINT_API_KEY","base_url_env":"WMO_ENDPOINT_BASE_URL"}' \
  --model-json '{"alias":"deepseek-v4-flash","connection":"hosted-wm","model":"deepseek-v4-flash","capabilities":{"supports_completions":true,"supports_tools":true,"supports_structured_output":false,"supports_embeddings":false,"input_cost_per_million_tokens_usd":0,"output_cost_per_million_tokens_usd":0,"cached_input_cost_per_million_tokens_usd":0,"cache_write_cost_per_million_tokens_usd":0}}' \
  --model-json '{"alias":"gpt-5-6-luna","connection":"openai","model":"gpt-5.6-luna","capabilities":{"supports_completions":true,"supports_tools":true,"supports_structured_output":true,"supports_embeddings":false,"context_window_tokens":1050000,"maximum_output_tokens":128000,"input_cost_per_million_tokens_usd":1.0,"output_cost_per_million_tokens_usd":6.0,"cached_input_cost_per_million_tokens_usd":0.1,"cache_write_cost_per_million_tokens_usd":1.25}}' \
  --model-json '{"alias":"text-embedding-3-small","connection":"openai","model":"text-embedding-3-small","capabilities":{"supports_completions":false,"supports_embeddings":true,"input_cost_per_million_tokens_usd":0.02,"context_window_tokens":8192}}' \
  --world-model deepseek-v4-flash \
  --judge gpt-5-6-luna \
  --embedder text-embedding-3-small
```

Confirm with `grep`, not by editing the file: `models.toml` names the three build aliases, points
`world_model` at `deepseek-v4-flash`, does not mention Anthropic or Haiku, and contains no
credential values. Zero token prices on the hosted world-model alias mean self-hosted spend, not a
missing price.

```bash
bench config-budget uv run wmo config budget 25 --root "$ROOT"
```

The default per-command budget is `$10`. Optimize reports a `$25` conservative estimate, so the
shared ceiling must be raised before that command or it fails closed.

## 4. Build

```bash
bench build uv run wmo build "$WMO_HAPPY_PATH_PROJECT" "$WMO_HAPPY_PATH_TRACES" \
  --source "$WMO_HAPPY_PATH_SOURCE" \
  --root "$ROOT" \
  --yes \
  --no-interactive \
  --max-build-cost-usd 5
```

Success prints a completed build and an embedding spend ceiling. Record the serving RAG, fit RAG,
and world-model IDs from that output. Fail if the command asks for a prompt or exits nonzero.

## 5. Judge setup and calibration

```bash
bench judge-setup uv run wmo config judge setup "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --approve \
  --non-interactive
bench judge-calibrate uv run wmo config judge calibrate "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --sample-size 10 \
  --label-all 1 \
  --yes \
  --approve \
  --non-interactive
```

Catalog prices supply the judge cost. Success prints an approved calibration artifact ID.

## 6. Add the second optimize candidate

Optimize requires two completion candidates. Add Haiku only here.

```bash
test -n "$ANTHROPIC_API_KEY"
bench config-haiku uv run wmo config providers \
  --root "$ROOT" \
  --non-interactive \
  --connection-json '{"name":"anthropic","provider":"anthropic","api_key_env":"ANTHROPIC_API_KEY"}' \
  --model-json '{"alias":"claude-haiku-4-5","connection":"anthropic","model":"claude-haiku-4-5-20251001","capabilities":{"supports_completions":true,"supports_tools":true,"supports_structured_output":true,"supports_embeddings":false,"context_window_tokens":200000,"maximum_output_tokens":64000,"input_cost_per_million_tokens_usd":1.0,"output_cost_per_million_tokens_usd":5.0,"cached_input_cost_per_million_tokens_usd":0.1,"cache_write_cost_per_million_tokens_usd":1.25}}'
```

## 7. Optimize the router

```bash
bench optimize uv run wmo optimize router "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --candidate gpt-5-6-luna \
  --candidate claude-haiku-4-5 \
  --incumbent gpt-5-6-luna \
  --yes \
  --approve-fidelity \
  --non-interactive \
  --maximum-provider-cost-usd 25
bench optimize-replay uv run wmo optimize router "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --candidate gpt-5-6-luna \
  --candidate claude-haiku-4-5 \
  --incumbent gpt-5-6-luna \
  --yes \
  --approve-fidelity \
  --non-interactive \
  --maximum-provider-cost-usd 25
```

Success prints a policy ID and a report ID. The second run must print
`replay: verified completed optimization` and make no new provider calls.

## 8. Serve the router and send official traffic

Start durable journaling in the background. Do not pass `--ghost`. Time readiness separately
from the Chat Completions request.

```bash
uv run wmo run "$WMO_HAPPY_PATH_PROJECT" --root "$ROOT" --port 8000 &
SERVER_PID=$!
bench run-ready bash -c 'until curl -sS --fail http://127.0.0.1:8000/v1/models >/dev/null; do sleep 0.2; done'
bench chat-smoke curl -sS --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$WMO_HAPPY_PATH_PROJECT\",\"messages\":[{\"role\":\"user\",\"content\":\"Help me\"}]}"
kill "$SERVER_PID"
wait "$SERVER_PID" || true
```

The response must be HTTP 200 JSON with a completion choice. The listen line
`OpenAI API router at http://127.0.0.1:8000/v1` must appear before the request.

## 9. Closed-loop retrieval refresh

```bash
bench rag-refresh uv run wmo config rag refresh "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --yes \
  --non-interactive \
  --maximum-cost-usd 5
bench rag-refresh-replay uv run wmo config rag refresh "$WMO_HAPPY_PATH_PROJECT" \
  --root "$ROOT" \
  --yes \
  --non-interactive \
  --maximum-cost-usd 5
```

Pass only when the first refresh prints:

- a refresh ID
- `completed_targets` of at least 1
- a combined-trace dataset ID
- a retrieval index ID different from the build serving and fit RAG IDs
- `completed build serving RAG ... and fit RAG ... are unchanged` with the IDs from step 4

The second refresh must reprint the same refresh ID and make no new embedding calls.

This writes a new retrieval index beside the frozen build. It does not mutate the completed-build
world model.

## Pass or fail

Pass only when every step above succeeded, the caller-supplied digest matched, both optimize runs
agreed on the same policy ID, the loopback request returned 200, and both refresh runs agreed on
the same refresh ID.

In the final report include:

- Fit dataset: project ID, source, trace path, and SHA-256
- Build, calibration, policy, report, refresh, snapshot, combined-dataset, and new retrieval IDs
- Observed spend ceilings printed by build, calibrate, optimize, and refresh
- Whether optimize and refresh replayed on the second run
- The HTTP status of the loopback smoke
- A timing table with every `bench` row: step name, seconds, exit status
- Any command that prompted, retried, or used a model other than the pins

If any step fails, stop. Do not substitute a different trace export, loosen a digest, invent a
project ID, substitute OpenRouter, skip calibration approval, raise a spend ceiling, edit files
by hand, or run Tinker SFT to force a green result.

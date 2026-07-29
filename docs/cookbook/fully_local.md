# Fully local: a routed endpoint with no cloud provider at all

Three commands, one machine, $0.0000:

```
wmo build      # traces -> a simulation to optimize against
wmo optimize   # fit the router over your candidate models
wmo serve      # an OpenAI-compatible endpoint that routes per request
```

This walk runs every part of that pipeline on models served by
[Ollama](https://ollama.com): the candidates the router chooses between, the model that
serves the simulation, and the judge that scores episodes. No provider account is
involved; the only spend is your own hardware. Everything below is a real transcript.

Two notes before the numbers mean anything:

- **This is a plumbing walk, not a benchmark.** Two small local models over a
  12-trace corpus proves the pipeline end to end; it does not measure routing lift.
  For an honest evaluation, use a real corpus (see the tau-bench cookbook) and read
  `wmo optimize route report`.
- **Local means free, and the pool records that honestly.** Locally hosted candidates
  register with explicit `$0/Mtok` prices, so every downstream cost number (request
  log, savings, the sweep's spend line) is a declared zero, never an accident.

## Step 0: two local models

```bash
brew install ollama          # or the installer for your platform
ollama serve &               # serves OpenAI-compatible at http://localhost:11434/v1
ollama pull qwen3:4b
ollama pull llama3.2:1b
```

Any OpenAI-compatible server works the same way (vLLM, llama.cpp, LM Studio); only the
URL changes.

## Step 1: register the candidates

```bash
uv run wmo providers set --provider openai --model "qwen3:4b" \
  --endpoint http://localhost:11434/v1 \
  --pool-model "qwen3:4b" --pool-model "llama3.2:1b" --tier open
```

```
  locally hosted endpoint http://localhost:11434/v1: pricing at $0/Mtok
  ✓ added qwen3-4b (openai qwen3:4b, $0/$0 per Mtok)
  ✓ added llama3.2-1b (openai llama3.2:1b, $0/$0 per Mtok)
```

One command did two jobs: the local **worker** role landed in `.wmo/settings.toml`, and
both models joined `.wmo/pool.toml` as routing candidates (each live-pinged first). Run
it interactively (no flags) to pick models off the server's own list instead. Every
pool entry participates in routing by default; set `enabled = false` on an entry to
take it out of selection without deleting it.

## Step 2: build the simulation (`wmo build`)

The build needs a trace corpus and two model roles: the **serve provider** (plays the
environment) and the **judge** (scores episodes). Both go local through the OpenAI
SDK's standard environment variables, which the `openai` provider kind reads:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama   # any non-empty value; a local server ignores it

uv run wmo build --name support --file traces.otel.jsonl \
  --provider openai --model "qwen3:4b" --judge-model "qwen3:4b" \
  --train-split 0.5
```

```
✓ ingested 12 traces → normalized 12 steps
✓ split 6 train / 1 val / 5 test traces
✓ indexed 12 steps into the replay buffer
╭──────────────── world model ready: support ────────────────╮
│      serve provider  openai (qwen3:4b)                      │
╰─────────────────────────────────────────────────────────────╯
run 93028edb: 0.0s, 0 tokens, $0.0000 (0 calls)
```

Where do traces come from? Two answers:

- **Your own agent**: anything your stack already records. `--source` accepts plain
  chat logs (`chat-json`) and the major observability stacks (Phoenix, Langfuse,
  LangSmith, Braintrust, PostHog, Mastra); `wmo ingest` normalizes them standalone.
- **An existing benchmark**: `uv run wmo download tau-bench` fetches a published
  trace corpus captured from real benchmark runs, and `wmo build --file
  packages/environment-capture/tau-bench/traces.otel.jsonl --name tau-bench` builds
  from it. Ten benchmarks are published; `wmo download` with no arguments lists them.

(`--train-split 0.5` only matters for a corpus this tiny: the router is measured on
the TEST band of the build's three-way split, and 12 traces at the default split leave
it empty. A real corpus keeps the default.)

## Step 3: fit the router (`wmo optimize`)

Measure both candidates closed-loop against the simulation, then fit:

```bash
uv run wmo optimize route sweep support --traces traces.otel.jsonl \
  --scenarios 5 --episodes 1 --max-steps 4 --out matrix.json --yes
uv run wmo optimize route fit matrix.json --kind knn --embedder hashing \
  --min-pairs 2 --rag-num 3
```

```
Scored coverage per candidate (`fit` SKIPS unscored cells)
┃ Candidate   ┃ Scored ┃ Unscored ┃
│ qwen3-4b    │      5 │        0 │
│ llama3.2-1b │      5 │        0 │
  next: wmo optimize route fit matrix.json --kind knn

✓ fitted knn policy over 4 scenarios -> models/support/policy.json
  bank models/support/policy.json.bank.npz, fallback qwen3-4b, z=0.5
  routed away from the fallback 0.0% of the time; cost/scenario $0.00000
  fit-set accuracy 1.0000 is IN-SAMPLE (every request retrieves its own row);
  measure on held-out scenarios with `wmo optimize route report`
```

(A transient episode error is normal on tiny local models: the paid cells stay
on disk and re-running the sweep measures only what is missing. The fit's own
output says the honest thing twice: it never left the fallback on this corpus,
and its accuracy line is in-sample, not a claim.)

Every `(candidate, scenario)` cell ran one full episode against the simulation, judged
by the local judge, at `$0.0000`. `--embedder hashing` keeps query embedding offline
too; the two small-number knobs are corpus-size scaling for a 5-scenario bank (a real
corpus uses the defaults, which are the validated champion). `wmo optimize model
support` is the staged one-command version of this step: preflight, sweep, fit, tune,
report in one go, resumable, with one cost confirmation up front.

## Step 4: serve it (`wmo serve`)

```bash
uv run wmo serve --name support --port 8000
```

One command, one OpenAI-compatible endpoint. Point any OpenAI client at it:

```bash
curl -D - http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"support","messages":[{"role":"user","content":"hi"}]}'
```

```
< HTTP/1.1 200 OK
< x-wmo-routed-model: qwen3-4b
{"model": "support", "choices": [{"message": {"role": "assistant",
  "content": "That's a great question..."}}], ...}
```

And the request log's decision row for that call:

```json
{"model": "qwen3-4b", "provider_model": "qwen3:4b",
 "routing_reason": "knn: baseline qwen3-4b leads 3 neighbors (profile 1.000)",
 "cost_usd": 0.0}
```

The response body names the ENDPOINT; the routed candidate rides the
`x-wmo-routed-model` debug header, and the request log
(`.wmo/serving/requests.jsonl`) records the decision, its evidence, and the `$0.00`
effective cost per call. `GET /v1/endpoints/support/config` exposes the cost/quality
dial; `GET /v1/endpoints/support/savings` stays honest about what a free pool can save
(nothing, and it says so).

As more traces accumulate, re-run `wmo build` (same `--name` rebuilds in place) and
`wmo optimize model` to refresh the simulation and the router; new pool entries join
the next sweep automatically, and `enabled = false` retires a candidate from future
fits without touching policies that already measured it.

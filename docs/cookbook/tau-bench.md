# Cookbook: tau-bench, end to end

One pass through the whole pipeline on one benchmark. Every step is a single real command plus
the artifact it leaves on disk, in the order you would actually run them, ending in an
OpenAI-compatible endpoint whose router was fitted on measured evidence.

tau-bench is the example because its trace corpus is public and its tasks are scored, so every
number below has a provenance you can check. Nothing here is specific to it: swap the corpus and
the name and the same six commands apply.

| Step | Command | Artifact |
|---|---|---|
| 0 | `just setup` | `.env`, a synced dev environment |
| 1 | `wmo build --file traces.otel.jsonl --name tau-bench` | `.wmo/models/tau-bench/` (config, splits, index, prompts) |
| 2 | `wmo providers set --pool-model ...` | `.wmo/pool.toml`, the routing candidate roster |
| 3 | `wmo optimize model tau-bench` | `optimize/matrix.json`, `policy.json` + bank, `optimize/report.json` |
| 4 (optional) | `wmo optimize distill run --config run.toml` | a run dir, a gate verdict, an adapter |
| 5 (optional) | `wmo optimize model --compressor ...` (or the `route sweep`/`route fit` pair) | per-arm matrices, a policy stamped with its fit geometry |
| 6 | `wmo serve --name tau-bench` | a live endpoint, plus its savings and dial routes |

## How to read the numbers in this walk

Three rules hold everywhere below, and they are the difference between a measurement and a
marketing figure.

**World-model simulated is not real-episode.** Steps 3 through 5 measure candidates against the
world model, which is a learned simulator of the environment, not the environment. A quality
delta from those steps is a simulated delta. It is useful because the ranking it produces has
been checked against real episodes on this benchmark, not because the point estimate transfers.
Real-episode validation is a separate protocol with a separate cost.

**The cost metric is cache-adjusted effective cost per completed task.** Per-token prices and
per-call costs are inputs to it, never the headline: a router that halves the price per call and
doubles the calls has saved nothing. A served endpoint reports this metric (its request log
records cached tokens and the compressor's own bill). The sweep's report is a stricter,
pre-cache view: single-shot list prices with cache effects not modeled, so read it as a lower
bound on savings and an upper bound on cost.

**Savings are estimates.** Every savings figure compares what the endpoint did against what a
named anchor model would have been billed for the same work. The anchor's counterfactual was not
run for every request, so the comparison is an estimate by construction. The endpoint always
names its anchor; quote the two together or neither.

## Step 0: setup

```bash
just setup
```

Copies `.env.example` to `.env` if you have no `.env` yet (an existing one is never touched) and
runs `uv sync --extra dev`. Then fill in the credentials you actually have: `.env.example`
documents each block inline, including which ones are optional and which single feature needs
them. Bedrock or Anthropic direct is enough to get through steps 1 through 3; `TINKER_API_KEY` is
step 4 only, and the `WMO_COMPRESSOR_*` block is step 5 only.

There is nothing to verify yet, and that is fine. Credentials are exercised at first use, and the
two places that spend money check theirs before they spend it. Step 2 live-pings every candidate
it registers over that candidate's own route, and refuses to write a roster it could not call, so
a wrong key surfaces there rather than inside a paid sweep. Step 3's preflight then re-resolves
every candidate's backend as far as it can without making a request, before it asks you to confirm
any spend.

(`wmo providers verify` is worth knowing about from step 1 onward: it reads the providers a
**built** world model recorded and pings them on both the completion and the embedding path. On a
fresh project it has nothing to read, so run it after a build, not before one.)

## Step 1: build the world model

Get the corpus, then build:

```bash
uv run wmo download tau-bench
uv run wmo build \
  --file packages/environment-capture/tau-bench/traces.otel.jsonl \
  --name tau-bench --fidelity low --embed-provider hashing
```

`wmo download` fetches the published data bundle (trace corpus plus task data) into
`packages/environment-capture/tau-bench/`; run it with no arguments for a picker over everything
published. `--source` defaults to `otel-genai`, which is what that corpus is. Traces from an
observability stack (Phoenix, Langfuse, LangSmith, Braintrust, PostHog, Mastra) go through the
same command with `--source <name>`, or through `wmo ingest` first.

The two flags above are the free configuration: `--fidelity low` takes the estimated-best config
with no search, and `--embed-provider hashing` indexes with an offline embedder. That is what
produced the transcript below, at a measured `$0.0000`. `low` is also the default, so a plain
`wmo build` does not spend on search. Opt into `--fidelity medium` for light prompt optimization
and a cheap-lever search, or `high`/`max` to search harder; those tiers cost real money. Every
searching tier is floored at low's estimate, so more effort never ships a worse config than low.

```
✓ ingested 1033 traces → normalized 5289 steps
✓ split 822 train / 105 val / 106 test traces
✓ indexed 5289 steps into the replay buffer
✓ GEPA done: val 0.000 (selection sample), 0 frontier candidates, 0 rollouts used
╭───────────── world model ready: tau-bench ─────────────╮
│                name  tau-bench                         │
│            artifact  .wmo/models/tau-bench             │
│      serve provider  bedrock (claude-opus-4-8)         │
│   held-out accuracy  0.000                             │
│       rollouts used  0                                 │
│ frontier candidates  0                                 │
╰────────────────────────────────────────────────────────╯
run 45165221: 1.0s, 0 tokens, $0.0000 (0 calls)
```

The three zeros on that card are honest: `--fidelity low` runs no GEPA rollouts, so there is no
held-out accuracy to report and no frontier. A `medium` or higher build fills them in. Your trace
and step counts will differ with your corpus.

### What landed

```
.wmo/models/tau-bench/
├── config.toml                 the build's resolved configuration
├── card.json                   corpus counts, provider, model id, build timestamp
├── auto_fidelity.json          the config that serves under --max-fidelity
├── metrics.json                the build's own quality measurements
├── index/
│   ├── steps.jsonl             every normalized step, the serving retrieval corpus
│   ├── embeddings.npy          their phi vectors
│   └── meta.json               index shape and embedder identity
├── prompts/
│   ├── base.txt                the seed prompt
│   ├── optimized.txt           what GEPA selected (identical to base at --fidelity low)
│   └── frontier.json           the candidates it kept
└── runs/                       one record per build or serve run: time, tokens, cost
```

`config.toml` is the reproduction record. It pins the serve provider, the embedder and its
dimension, the retrieval depth, the split ratio, the GEPA budget, the judge model, and the trace
adapter, which is every input that decides what this artifact is.

The split line matters more than it looks. The build cuts a deterministic three-way split by
trace id: prompt optimization and knowledge extraction see the **train** band only, `val` is
GEPA's internal selection set, and the **test** band stays untouched so it can be the evidence
later steps are measured on. Step 3's scenarios come from that test band. Two honest limits: the
serving index covers the full corpus, so the simulator can retrieve a held-out trace's own
recorded steps as demonstrations when it simulates that scenario, and bands are cut per trace id
rather than per task text, so a task repeated across traces can land on both sides.

Look at the thing before optimizing it:

```bash
uv run wmo play --name tau-bench      # step in yourself
uv run wmo demo --name tau-bench      # replay a recorded scenario, open loop
```

## Step 2: register the routing candidates

The router picks between the models in a pool file. Write it with:

```bash
uv run wmo providers set \
  --provider bedrock --model claude-opus-4-8 \
  --pool-model claude-opus-4-8 --pool-model claude-haiku-4-5 --tier frontier
```

Two jobs in one command. `--provider`/`--model` set the local worker model in
`.wmo/settings.toml`, and each `--pool-model` registers a routing candidate in `.wmo/pool.toml`.
Run it bare at a terminal for a guided version that searches each backend's catalog.

```
set local worker provider to bedrock (claude-opus-4-8) in .wmo/settings.toml
  ✓ added claude-opus-4-8 (bedrock us.anthropic.claude-opus-4-8, $5/$25 per Mtok)
  ✓ added claude-haiku-4-5 (bedrock us.anthropic.claude-haiku-4-5-20251001-v1:0, $1/$5 per Mtok)
```

```toml
# .wmo/pool.toml
[[model]]
name = "claude-opus-4-8"
kind = "bedrock"
model = "us.anthropic.claude-opus-4-8"
model_type = "claude-opus-4-8"

[[model]]
name = "claude-haiku-4-5"
kind = "bedrock"
model = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
model_type = "claude-haiku-4-5"
```

Five things to know about the roster:

- **Every candidate is proved before anything is written.** Each one is live-pinged over its own
  route, not the worker's: a candidate can differ from the verified worker in model, deployment,
  endpoint, and credential, so the worker's ping proves nothing about it. The whole batch is built
  and called first, so a bad third id cannot leave half a roster behind, and a candidate that
  cannot be called is a loud failure here instead of a 401 inside a paid sweep.
- **One provider kind per invocation.** Pool entries inherit the `--provider` of the call that
  registered them. Re-run the command per backend; later runs add entries beside the existing
  ones rather than replacing them.
- **Price every candidate.** Prices come from the published catalog when there is one (which is
  why they are not written into the TOML above). A model with no published price needs
  `--input-per-mtok` and `--output-per-mtok`, both or neither. An unpriced candidate reports $0,
  and a cost-aware policy routes everything to it.
- **In a script, `--provider` and `--model` are required**, even when all you want is
  `--pool-model`. Without them the command exits with a usage error rather than guessing.
- **A pool of one is a legitimate stopping point.** If you only want a single model served
  through the endpoint, skip to `wmo optimize route pin` in step 3's stage-by-stage section.

## Step 3: optimize the endpoint

This is the whole routing workflow, and it has one question in it.

```bash
uv run wmo optimize model tau-bench \
  --traces packages/environment-capture/tau-bench/traces.otel.jsonl \
  --scenarios 8
```

Pass `--traces` because a build keeps no copy of the corpus it read. `--scenarios` caps the
held-out scenarios measured: more scenarios is better evidence and more spend, linearly.

One plan table prints before anything spends, and one confirmation covers the run:

```
optimize model: tau-bench    pool: 2 candidate(s) (.wmo/pool.toml)

stage        plan                                       est. cost    status
preflight    resolve 2 backend(s), check prices              free    ok
sweep        2 candidate(s) x 8 scenario(s) x 1            ~$3.12    will run (never completed here)
             episode(s)
fit          knn (guarded, fallback best single on           free    will run (runs after sweep, which
             the sweep)                                              will change its input)
tune         cost_quality 0.25 (Balanced (default))          free    will run (runs after sweep, which
                                                                     will change its input)
report       3-objective headline vs the fitted              free    will run (runs after sweep, which
             fallback                                                will change its input)

  estimated candidate spend ~$3.12 (a projection: 2,000 input + 250 assumed output token(s) per
call, times the real cell and call counts, at each candidate's own pool price)
  the world model's own serve and judge calls are NOT in that figure and are not projectable
before this model's first sweep: nothing predicts the simulator's and the judge's token use per
episode in advance. It is not a rounding error either, measuring 7.0x the candidate side on one
real tau corpus, so treat the number above as a lower bound.
```

That last paragraph is the one to read twice. The `~$3.12` is the candidate half of the bill,
projected from assumed tokens per call. The simulator's own serve and judge calls are real money
and are not in it, and on a real tau corpus they measured seven times the candidate side. Both
halves are measured and printed when the sweep finishes, and the next run forecasts the second
one from what this one observed.

### Running it unattended

Three flags, three different jobs. They compose, and none substitutes for another.

```bash
uv run wmo optimize model tau-bench --traces ... --dry-run             # preview
uv run wmo optimize model tau-bench --traces ... --yes --max-usd 25    # consent, bounded
```

**`--dry-run` previews.** It prints the same plan table and exits 0 having spent nothing, run no
episode, and written no artifact, not even resume state. It is the honest way to see what a run
would cost, in a script or at a terminal, and it prints the table even when `--max-usd` would have
refused the real run.

**`--yes` consents.** Consent has to be said, never inferred from the absence of a terminal: a
non-interactive session (a pipe, a CI job) that would buy a sweep and was not told `--yes` exits 2
with `cannot ask for spend consent`, naming both honest ways forward. Nothing is bought. A run
with no sweep left to buy needs no consent at all, so a resume down to fit, tune, and report
proceeds unattended.

**`--max-usd` bounds.** It is a cap, not consent, and the two are independent. It stops before any
paid stage whose projection would carry the run past that total, counting what earlier runs already
spent and counting the candidate and world-model sides both. The run stays resumable and prints how
to continue it.

### The four stages

**sweep** is the only paid stage. Every (candidate, scenario, episode) cell runs one full episode
against the world model, which scores it: a matrix without verified rewards is not evidence.
Progress prints per cell, which is also your live cost meter:

```
sweep (never completed here)
  [1/16] claude-opus-4-8 0007cd4198dd207a37423204296b117b ep0: reward=0.05 $0.24856 steps=20
  [2/16] claude-opus-4-8 00c3937622c3e9f29b43064203256ff7 ep0: reward=0.00 $0.25851 steps=20
  [3/16] claude-opus-4-8 030730a96c210d8090a0fe4e64e5c488 ep0: reward=0.85 $0.27644 steps=17
```

Those per-cell dollar figures are candidate-side only. A cell can come back `reward=unscored`
when its episode or its scoring failed, and both fitters skip unscored rows.

Fit-readiness is a coverage contract, not a nonzero count: every candidate must have the same
number of scored episodes on the same scenarios, or the policy ends up decided by whichever cells
each candidate happened to lose. Per-candidate scored counts always print. When the evidence
differs the matrix is still written (those cells were paid for, and their `error` fields are the
diagnosis) but the handoff to `fit` is withheld and the command exits non-zero, naming each
candidate and the scenarios it is missing. `--allow-uneven-coverage` is the opt-out for an
operator who knows the bias and wants the partial data anyway.

**fit** fits the guarded nearest-neighbor policy: a pinned fallback serves every request unless
paired neighbor evidence clears a confidence bar, with a novelty floor that abstains to the
fallback when a request is unlike anything in the evidence bank. It is instant, and free with the
offline embedder (a fit that resolves to a hosted embedding model pays that model to embed every
scenario, and the plan table says so instead of printing "free").

**tune** sets the cost/quality dial in place, with no refit. `0.25` is the shipped default.
See [the dial reference](../reference/cost_quality_dial.md) for the measured anchors and their
limits.

**report** builds the three-objective comparison against the anchor.

### The payoff

The run closes on what the endpoint now is, what it bought, and how to serve it:

```
  policy: knn (guarded, fallback claude-haiku-4-5)   dial: 0.25 balanced (default)
  quality  +X.Xpt vs claude-opus-4-8   (world-model simulated, N held-out scenario(s) scored on both sides)
  cost     -XX% per episode   (measured candidate-side at list prices, single-shot; cache effects not modeled)
  latency  p50 -X.XXs   (wall time per policy call during the sweep, env time excluded)

  serve it:   wmo serve --name tau-bench
  endpoint:   POST /v1/chat/completions  (model="tau-bench")
```

**Your numbers will differ**: the placeholders above are shape, not results. This walk's own
sweep was not carried to completion, and quoting someone else's deltas as yours is exactly the
error the provenance labels exist to prevent.

Read each line with the parenthetical attached to it. Quality is simulated, over the scenarios
that had a scored episode on *both* sides (a comparison over different scenarios is not a
comparison, so unmatched scenarios are excluded and counted). Cost is candidate-side list price,
single-shot, cache effects not modeled, which is the conservative view: the served endpoint's
cache-adjusted effective cost is lower. Latency is per policy call, excluding environment time.
Any of the three prints its reason instead of a number when it cannot honestly be computed, for
instance a cost delta against an anchor whose episodes measured $0.

### Where the artifacts land

```
.wmo/models/tau-bench/
├── policy.json                 what `wmo serve` reads
├── policy.json.bank.npz        the fitted evidence bank (the policy's sidecar)
├── policy.base.json            the un-tuned fit, written by the first tune
└── optimize/
    ├── matrix.json             the OutcomeMatrix the sweep bought
    ├── report.json             the improvement report
    └── optimize-run.json       the resume manifest
```

Serving artifacts stay where serving already looks for them. `optimize/` holds only the staged
run's own bookkeeping and is disposable: deleting it resets resume and breaks no serving path.

### Re-running, and dropping to a stage

Re-running is cheap and safe. A stage is skipped when its recorded input fingerprints still match
and its artifact is unchanged on disk, and the reason prints either way, so a run that stopped
halfway resumes at the stage that stopped it.

```bash
uv run wmo optimize model tau-bench                    # resumes; skipped stages say why
uv run wmo optimize model tau-bench --force-from sweep  # buy fresh cells anyway
```

`--force-from` takes `sweep | fit | tune | report` and redoes that stage and everything after it.

### Stage by stage

The one command above calls the same library functions these four commands call, so you can drop
to any of them and the next `optimize model` run notices and resumes around it.

```bash
# 1. buy the evidence (the only paid step)
uv run wmo optimize route sweep tau-bench \
  --traces packages/environment-capture/tau-bench/traces.otel.jsonl \
  --pool .wmo/pool.toml --scenarios 8 --out matrix.json

# 2. fit the policy
uv run wmo optimize route fit matrix.json --kind knn \
  --out .wmo/models/tau-bench/policy.json

# 3. set the dial
uv run wmo optimize route tune .wmo/models/tau-bench/policy.json --cost-quality 0.6

# 4. build the report
uv run wmo optimize route report matrix.json .wmo/models/tau-bench/policy.json \
  --baseline claude-opus-4-8 --endpoint tau-bench --out report.json
```

One difference worth knowing before scripting these: `route sweep` does **not** share the staged
command's consent refusal. On a non-interactive session it prints that it is proceeding without
confirmation and buys the sweep. So pass `--yes` there because you mean it, and treat an unattended
`route sweep` as spending by default; `--dry-run` is the staged command's flag, not this one's.

Reach for these when you want a knob the staged command does not expose: `--kind rank` for
Avengers cluster ranks instead of kNN evidence, `--z` for a stricter or looser confidence bar,
`--rag-num` for the neighbor budget (it must scale with the bank, or routing collapses to the
fallback), `--floor-q` for the novelty floor, `--embedder` to choose the geometry. Note that
`--embedder auto` resolves to a paid Azure embedding API when the Azure variables are set;
`--embedder hashing` keeps the fit offline and free, and either way it prints which one it picked.

Two further route commands:

```bash
uv run wmo optimize route pin tau-bench --model claude-haiku-4-5   # serve one model, no matrix, no fit
uv run wmo optimize route student <run-dir> --input-per-mtok 0.1 --output-per-mtok 0.4
```

`pin` writes a `kind="static"` policy: every request goes to one pool model. It is the honest
"before" state, and a static endpoint's savings route will say it has saved nothing. `student` is
step 4's keystone.

## Step 4 (optional): train the agent model

The first three optimizers change the harness around a model. This one changes the model.
`wmo optimize distill run` does on-policy distillation of a Tinker LoRA student from rollouts of
harbor's own `terminus_2` agent on harbor tasks: the student samples, a larger teacher scores the
exact tokens the student sampled, and each step nudges the student toward the teacher under a
per-token reverse-KL objective.

```bash
uv run wmo optimize distill run --config run.toml --run-dir runs/d1 \
  --task-ids train.json --holdout-task-ids holdout.json --backend e2b --yes
```

Everything durable lands in `--run-dir`: the config snapshot, metrics, checkpoints, evals,
rollout artifacts, and the gate verdict. Read a finished run back for free with
`wmo optimize distill report --run-dir runs/d1`, which prints the gate verdict and the held-out
before/after table.

Promotion is gated on held-out solve rates. The student-after must reach a configured fraction of
the teacher's solve rate and must not regress against student-before; only then does the adapter
version land with the champion alias. The harness is pinned for the whole run and never edited:
this command measures model quality, not scaffold quality.

See [the distill reference](../reference/distill.md) for the run TOML's sections, the cost and
budget model, resume, and troubleshooting.

Then make the student routable, which is the join back to step 3:

```bash
uv run wmo optimize route student runs/d1 --input-per-mtok 0.1 --output-per-mtok 0.4
uv run wmo optimize model tau-bench --traces ... --force-from sweep
```

The first command turns the `tinker://` adapter into a priced `[[model]]` entry in the pool, with
no hand-edited TOML in between. Both prices are required: an unpriced candidate reports $0 and a
cost-aware policy would route everything to it. The re-run then re-sweeps the enlarged pool and
refits, and traffic shifts to the student only on the cells where it earned them. A student that
earns nothing changes nothing, which is the point of measuring rather than assuming.

## Step 5 (optional): compress the prompt

Compression is a stage in front of routing: request, then compress, then route, then call. It is
**default-off**, and it should stay off in production until its accuracy gate passes. What
follows is how to measure it, not a recommendation to ship it.

Measure a compressed arm, then fit in the same geometry:

```bash
uv run wmo optimize route sweep tau-bench \
  --traces packages/environment-capture/tau-bench/traces.otel.jsonl \
  --compressor llmlingua2-endpoint --aggressiveness 0.4 --out matrix-c04.json

uv run wmo optimize route fit matrix-c04.json --kind knn \
  --compressor llmlingua2-endpoint --aggressiveness 0.4 \
  --out .wmo/models/tau-bench/policy.json
```

`wmo optimize model tau-bench --compressor llmlingua2-endpoint --aggressiveness 0.4` runs that same
pair end to end (the arm configures its sweep and its fit, and the compact row in the plan table
names it), which is the one-command way to measure an arm once you know which one you want.

One sweep per arm, one matrix per arm. `--aggressiveness` is a dial in `[0, 1]` where 0.0 is a
no-op and higher never removes less, but it is not an exact removal fraction: the achieved ratio
is recorded per episode. Available compressors are `identity`, `truncate`, and
`llmlingua2-endpoint`; the last is a learned 177M-parameter classifier served from a GPU box and
needs the `WMO_COMPRESSOR_*` variables from `.env.example`. It fails closed on an unreachable
endpoint or a missing certificate rather than silently serving uncompressed text, because a
compressor that degrades quietly makes every cost and accuracy result depend on the health of a
box nobody was watching.

**The arm must match the fit.** A fitted policy records the compression config its evidence was
fitted under (`fit_compression`) alongside the config it would serve, and an endpoint whose two
disagree does not mount at all. This is not pedantry: a bank, its centroids, and its novelty
floor are geometry in the space of the text the fit embedded, and serving a different
representation against them was measured to trip the novelty floor 10 to 13 times more often,
collapse route-away, and raise cost 11 to 41 percent while accuracy sat flat to negative. That is
a broken policy, not a degraded one. Refit under the serving config, or serve the config the
artifact was fitted under. Static policies embed nothing and are exempt.

Only user-message content is compressed. System prompts, the model's own prior replies, tool
calls, and tool results pass through verbatim. And a compressor's own inference cost counts:
effective cost per completed task includes it, so an arm that saves prompt tokens and pays more
for the compressor than it saved has not saved anything.

## Step 6: serve it

```bash
uv run wmo serve --name tau-bench
```

The endpoint is OpenAI-compatible. `model` in the request names the endpoint, and the fitted
policy picks which pool model actually serves each call:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "tau-bench", "messages": [{"role": "user", "content": "cancel my flight"}]}'
```

Responses stay OpenAI-pure and name only the endpoint. Which model actually answered goes to the
request log and the `x-wmo-routed-model` debug header, never into customer-facing copy. Tool
calls, `tool_choice`, and parallel tool calls all round-trip, and conversation affinity keeps a
multi-turn exchange on its incumbent model so the provider's prompt cache stays warm.

Two operator routes sit deliberately outside the OpenAI surface, so a customer's OpenAI client
sees exactly what it saw before:

```bash
curl http://localhost:8000/v1/endpoints/tau-bench/savings
curl http://localhost:8000/v1/endpoints/tau-bench/config
curl -X PUT http://localhost:8000/v1/endpoints/tau-bench/config \
  -H 'content-type: application/json' -d '{"cost_quality": 0.6}'
```

`savings` totals what the endpoint has actually saved out of its request log, and takes a
`?window=` period. It survives a restart because it is derived from the log rather than from
memory, and it starts at an honest empty state, which on a fresh endpoint reads:

```json
{"requests_served": 0, "cost_saved_usd": 0.0, "cost_saved_pct": 0.0,
 "estimate_basis": ["This endpoint has not served any requests yet, so there is nothing to compare.",
                    "Your invoices remain the record of what you were charged."],
 "window": "all_time", "actual_cost_usd": 0.0, "baseline_cost_estimate_usd": 0.0}
```

Every savings response carries its `estimate_basis` for exactly the reason in that second line.
The comparison is a modelled counterfactual; your invoices are the record.

`GET .../config` reports the dial, and `PUT .../config` moves it on the live runtime with no
restart and no refit, the same dial `wmo optimize route tune` writes to disk and `endpoint.toml`
can carry. It also reports whether the endpoint is dialable at all, along with the measured
anchors:

```json
{"endpoint": "tau-bench", "dialable": false, "cost_quality": null, "named_point": "as-fitted",
 "knobs": null, "anchors": [{"s": 0.0, "label": "Quality max", ...}]}
```

That is a pinned static endpoint: there is nothing to trade, so it is not dialable, and its
`cost_quality` is `null` rather than a number it could not honestly report. A fitted kNN policy is
dialable and reports its position. The `anchors` array is the same measured frontier the dial
reference documents, returned so a UI never has to hardcode it.

## What this walk does not prove

Worth stating plainly, because the numbers are persuasive and the caveats are not.

- Every quality delta in steps 3 through 5 is measured against the world model. That is a
  simulation of the environment, and its per-point estimates are not real-episode results.
- The sweep's cost deltas are single-shot list prices with cache effects unmodeled. They are
  neither the effective cost per completed task nor the price you will be billed under caching.
- The evidence is the corpus's test band, held out from prompt optimization and knowledge
  extraction but not from the serving index, and cut per trace id rather than per task text.
- Savings are estimates against a named anchor that was not itself run on your traffic.

The reproducible part is the protocol: the exact commands above, the pins in
`.wmo/models/<name>/config.toml`, and the matrix the sweep wrote. Re-measure on your own corpus
before repeating any figure as yours.

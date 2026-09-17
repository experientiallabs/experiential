# CLI usage

The root surface is deliberately small:

| Command | Purpose | Local result |
|---|---|---|
| `exp` | Open the branded home screen. `Run Gateway` is the first option and runs setup when needed. | Interactive gateway menu, or the default gateway in a non-interactive terminal. |
| `exp login [--root ROOT]` | Sign in to Experiential Cloud through the Platform browser approval flow, save the returned organization key, and synchronize the authenticated account's model identities. | User-local credential plus secret-free hosted provider/model records in `.exp/models.toml`. |
| `exp run [PROJECT] [--root ROOT] [--check]` | Start the local gateway directly, optionally with one project-backed alias. | OpenAI-compatible endpoint, readiness routes, and content-free usage view. |
| `exp build PROJECT [-t PATH] --source SOURCE --root ROOT [--provider NAME ...]` | Launch the guided end-to-end build when traces are omitted, or use one explicit local source for automation. | Simulation, serving RAG, fit RAG, syllabus, evaluation evidence, and a runnable automatic router. |
| `exp optimize router PROJECT --root ROOT [--yes]` | Complete bounded simulation and judgment, fit a frozen router, then verify held-out evidence. | Fit evaluation, policy, held-out evaluation, and router report. |
| `exp optimize model PROJECT --root ROOT [--yes]` | Verify one project-bound W12 dataset and conservatively preflight bounded managed Tinker SFT. | Completed W13 result and registered frozen alias, or a fail-closed preflight with no paid dispatch. |
| `exp --root ROOT [--check]` | Validate or start the initialized authenticated default gateway on loopback; the native data plane serves every route, including Chat Completions, Responses, and Anthropic Messages. | OpenAI-compatible and Anthropic Messages endpoints, readiness routes, and content-free usage view. |
| `exp --project PROJECT --root ROOT [--ghost]` | Activate a frozen policy as one project-backed alias and launch the normal gateway. | The same authenticated OpenAI endpoint and SQLite accounting as the default gateway. |
| `exp config gateway ...` | Author provider references, identities, virtual keys, grants, aliases, certified exact-model pools, monthly limits, status, and usage without optimizer roles. | Private SQLite authority, immutable catalog snapshots, and versioned receipts. |
| `exp config gateway call ALIAS PROMPT [--json]` | Send one chat completion to a live gateway as a caller, streaming text to stdout. | One HTTP request against the running gateway; no local state. |
| `exp config gateway models [--json]` | List the aliases a live gateway grants to the presented key (caller view of `GET /v1/models`). | One HTTP request against the running gateway; no local state. |
| `exp config gateway key check [--json]` | Validate one raw virtual key against a live gateway and print its granted aliases without storing the key. | One HTTP request against the running gateway; no local state. |
| `exp config providers [--provider NAME ...]` | Collect secret-free provider connections, model aliases, and build roles. `experiential-cloud` points at the hosted Platform gateway and reuses the credential from `exp login`; login already performs its provider/model synchronization. Setup also persists, replaces, or removes user-local provider keys. | Local `.exp/models.toml` plus optional records in the user-data credential file. |
| `exp config budget [USD] --root ROOT` | Read or set the maximum conservative estimate allowed for one paid command (default `$50.00`). | Local `.exp/settings.toml`. |
| `exp config telemetry status\|enable\|disable` | Read or update aggregate product telemetry preference. | Local `.exp/settings.toml`. |

`build`, judge calibration, `optimize router`, and `optimize model` use the same cost authorization
policy. An estimate at or below 50% of the budget runs automatically. A higher estimate
up to the budget requires a clear terminal confirmation or `--yes`; an estimate above the budget
warns and requires an explicit interactive override that defaults to no, and fails closed before
credentials or provider clients when no terminal is available. Set the deterministic ceiling with
`exp config budget USD --root ROOT`. `--yes` confirms only an in-budget invocation and never raises
the ceiling. Exact completed replays report a zero-dollar estimate and do not prompt.

Successful build, router, simulation, and SFT operations preserve anonymous aggregate PostHog
product telemetry, which may send unless disabled. Gateway startup makes no provider call.
`build --dry-run` and exact completed-build replay make zero paid provider calls. A new grounded
build calls only the configured embedder; automatic router optimization separately executes the
bounded candidate, world-model, and judge schedule shown in its cost preflight.
An authenticated gateway request is the explicit online model-call boundary. Project selectors
remain frozen for the process lifetime and return only an exact model pool. `--ghost` remains a
compatibility flag for project-journal behavior; gateway authentication, replay, attempts, and
usage accounting stay enabled.

The default and project gateway forms use one gateway lifecycle. It binds only `127.0.0.1`, starts with no
provider call, and requires an explicit provider environment reference, exact model alias, identity,
grant, and a virtual key. Interactive first-run setup can persist multiple provider connections and
creates one initial gateway alias; additional deployments and certified pools remain explicit
gateway configuration. From the interactive home screen, `Setup Gateway` also offers a confirmation-gated
reconfiguration path for an initialized gateway: it replaces the selected provider and alias revisions
while preserving existing identities, keys, grants, usage, and history. `exp --non-interactive --json` returns `gateway_not_initialized` plus exact
next commands on an empty root. `exp --check` validates local readiness without binding.
First-run setup prints `EXP_GATEWAY_URL` and the newly issued `EXP_GATEWAY_KEY` before readiness is
checked, so the credentials remain available even when a provider route is not ready. The gateway-
specific variables avoid overwriting an upstream provider's `OPENAI_API_KEY`. The error names the
unavailable alias and provider configuration; fix that configuration and rerun `exp`. If the
one-time key was not saved, issue a replacement with
`exp config gateway key issue IDENTITY --key-id KEY --json`.
The gateway writes no prompts, responses, tool arguments, raw keys, or provider secrets to SQLite.
`GET /usage` and `GET /usage.json` expose the same schema-v2 content-free overall and per-identity
counts, token usage, latency, terminal states, and attributed estimated cost. Their attempt-only
`by_billing_source` buckets conserve attempts, tokens, known cost, unknown-cost attempts, and
terminal states across `host_managed` and `customer_managed`; logical request counts are not
partitioned. Estimated cost is not provider invoice cost.

One-time virtual-key material appears only in the successful key-issue receipt or a newly created
mode-`0600` output file. Human key issuance on a non-terminal requires `--json` or `--output`.
Provider catalogs and gateway SQLite store an environment-variable name, never a raw credential
value. Interactive `exp config providers` persists a pasted key in the platform user-data file
(`~/.local/share/exp/auth.json` on Linux) and can replace or remove that stored key when the
same provider is edited again. Runtime commands never prompt. They resolve an explicit
caller-supplied environment mapping first, then a non-empty process environment value, then
the stored key for that connection ID. Environment values override the store without rewriting
it. Missing credentials fail with the environment name and a recovery that points at
`exp config providers`. Bedrock stays on the AWS credential chain. Current provider revisions
live in SQLite; immutable serving snapshots bind exact revisions while build and evaluation
artifacts remain in the project artifact store.

To add ordered failover, first author each deployment as a direct alias with the same
`--exact-model`. Then certify their equivalence and order with:

```console
exp config gateway pool certify PUBLIC_ALIAS \
  --deployment-alias PRIMARY --deployment-alias SECONDARY \
  --exact-model EXACT_MODEL --certification-id CERTIFICATION \
  --provenance PROVENANCE --evidence-sha256 SHA256 \
  --certified-at TIMESTAMP --expected-catalog-sha256 CATALOG_SHA256 \
  --revision REVISION --root ROOT --non-interactive --json
```

`--expected-catalog-sha256` prevents stale authoring from activating a different catalog. Every
pool member must resolve to the same exact model identity. Retryable transport, availability, rate,
and malformed precommit failures may advance through the certified order. Refusal fallback requires
the explicit `--refusal-failover` option and is persisted on that alias revision. No failure can
switch providers after outward text, refusal, or tool-call output commits the response.

Direct deployments declare `--billing-source host_managed` or `customer_managed`. The selected
value is frozen on each physical attempt before dispatch and remains unchanged across catalog
replacement and restart. Usage JSON and HTML expose content-free physical-attempt buckets by source;
they do not partition logical request counts. Legacy schema-v1/v2 attempts migrate explicitly as
`customer_managed`.

Monthly serving limits are separate from the one-shot `exp config budget` command ceiling. They use
integer nano-USD (a billionth of a dollar; `20000000000000` is $20,000) and explicit immutable UTC
periods. The local team, an identity, a total alias
pool, and each provider deployment can have overlapping hard limits:

```console
exp config gateway budget set --period 2026-08 --scope identity \
  --identity TEAM_MEMBER --limit-nano-usd 20000000000000 \
  --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment AZURE_DEPLOYMENT \
  --limit-nano-usd 10000000000000 --root ROOT --non-interactive --json
exp config gateway budget set --period 2026-08 --scope deployment \
  --alias PUBLIC_ALIAS --pool EXACT_POOL --deployment BEDROCK_DEPLOYMENT \
  --limit-nano-usd 10000000000000 --root ROOT --non-interactive --json
exp config gateway budget remaining --period 2026-08 --root ROOT --json
```

Omit required values in an interactive terminal to receive prompts. Pass `--non-interactive` to
fail immediately instead. `--replace` changes a configured limit without deleting the month or its
spend. Exhausting one deployment continues to the next certified exact-model route. Exhausting all
applicable shared capacity returns OpenAI `insufficient_quota`. By default an unpriced attempt is
admitted and recorded as unknown cost; `budget remaining` reports the unknown-cost attempts and
their observed token volume. `--strict-unknown-cost` on `budget set` opts one limit into failing
closed instead: unpriced attempts are rejected, recorded unknown-cost attempts block the limit even
after `--replace` raises it, and `exp config gateway budget reconcile --period 2026-08 --scope
team --assigned-cost-nano-usd COST --root ROOT --non-interactive` settles each unknown-cost
attempt at an explicit assigned cost and restores service with exact per-attempt attribution.
There is no budget reset job and no budgets dashboard.

Official OpenAI SDK clients use the issued virtual key and loopback base URL:

```python
from openai import OpenAI

with OpenAI(api_key=VIRTUAL_KEY, base_url="http://127.0.0.1:8000/v1") as client:
    response = client.responses.create(model="PUBLIC_ALIAS", input="hello")
```

Release evidence fixes the SDK at OpenAI `3.0.0` and covers `OpenAI` plus `AsyncOpenAI`, Chat
Completions plus Responses, and stream plus non-stream calls. Provider protocol fixtures are
deterministic. Hosted-provider runs require credentials and are reported separately in
[`release-scope.md`](release-scope.md); fixture success is not presented as a live-provider result.

The guided build uses one bounded consent to create simulation evidence, separate serving and fit
RAG indexes, a judge syllabus, closed-loop candidate evaluations, and a runnable router. Human
judge calibration is recommended but optional: provisional judgment provenance remains visible,
and later approval plus another build creates an immutable human-calibrated successor. Running the
endpoint records traffic by default so a later optimization can use newly attributed outcomes.
Explicit trace automation can still stop after the grounded build, then use
[`router-optimization.json`](reference/router_optimization_config.md) with `exp optimize router`.
Router fitting never invokes world-model fidelity testing. Applications that need a world-model
quality measurement can call the separate `build_fidelity_evaluation_plan` and
`build_fidelity_report` APIs; those results never enter router fitting or activation. Fidelity
reports contain measurements only, never an approval or denial. See the
[router contracts](reference/router_optimization_config.md).

## Independent continual learning

`exp optimize claas burst RUN.json` drains a frozen snapshot of exact, training-ready experience
and writes `run-report.json` plus resumable checkpoints. `exp optimize claas serve RUN.json`
keeps one learner and its HTTP generation endpoint alive for a finite full run. Both commands
accept `--modal MODAL.json` to host the same process in a temporary Modal Sandbox. The foreground
command waits for completion; interruption requests a checkpointed shutdown.

Install `experiential[claas-verl]` for CUDA burst training, or
`experiential[claas-rollout]` for a Linux CUDA run with generation. Add `claas-modal` to the launch
machine for remote hosting. Use Python 3.12 for the pinned training dependency stack. The public
backend pins veRL 0.9.0 and vLLM 0.22.0. A run owns exactly one visible BF16-capable GPU.
Set `CUDA_VISIBLE_DEVICES=0` inside the worker process. Full runs
retain both engine lifetimes but alternate their GPU memory ownership through upstream sleep,
wake, and LoRA synchronization. CUDA graph startup happens once per run; a new burst or full run
still has a cold start. A burst only needs the training engine.

The run file is the JSON representation of `RunLaunchConfiguration`. This example requires your
immutable model/tokenizer commit and paths; the model must support the selected LoRA modules:

```json
{
  "directory": "/absolute/path/to/learning-state",
  "compute_reservation_usd": 3.0,
  "spec": {
    "scope": {"user_id": "local", "application_id": "my-agent"},
    "adapter_id": "my-agent-adapter",
    "base_model": "MODEL_REPOSITORY",
    "model_revision": "REPLACE_WITH_40_CHARACTER_MODEL_COMMIT",
    "tokenizer_id": "MODEL_REPOSITORY",
    "tokenizer_revision": "REPLACE_WITH_40_CHARACTER_TOKENIZER_COMMIT",
    "initial_policy_revision": "initial",
    "objective": "sdpo",
    "target_modules": ["q_proj", "v_proj"],
    "max_sequence_tokens": 4096,
    "max_batch_examples": 4
  },
  "run": {
    "mode": "run",
    "maximum_updates": 20,
    "maximum_run_seconds": 1200,
    "minimum_ready_examples": 4
  },
  "runtime": {
    "checkpoint_root": "/absolute/path/to/learning-state/checkpoints",
    "decoder": "qwen35",
    "maximum_output_tokens": 512
  }
}
```

Choose `text`, `hermes`, or `qwen35` decoding to match the student's native output format.
`compute_reservation_usd` is the operator's conservative full-run estimate used by shared CLI
spend consent. It is not a provider invoice meter. Include GPU, CPU, RAM, startup, cleanup, and
storage charges in that estimate. Runtime/update limits bound the run, while a Modal Sandbox
also has a hard lifetime limit. The launcher never schedules future runs automatically.
An explicit zero estimate declares that existing local hardware incurs no billable infrastructure
cost for this run. The local adapter starts a process on the current host and never rents a GPU.
Use a positive estimate when that host is metered; Modal always requires a positive estimate.

Set `EXPERIENTIAL_CLAAS_TOKEN` to a strong secret before serving; only its environment-variable
name appears in persisted configuration. The endpoint is independent of the production gateway:

```console
exp optimize claas serve RUN.json --root ROOT --yes
```

```python
import os
from exp.runtime.claas.client import LearningClient

with LearningClient(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["EXPERIENTIAL_CLAAS_TOKEN"],
    model="my-agent-adapter",
) as learner:
    response = learner.sdk.responses.create(
        model=learner.model, input="Choose the next tool action.", max_output_tokens=128
    )
    learner.submit_feedback(response.id, text="Check the tool result before answering.")
    learner.trigger_train()
```

The learner supports nonstreaming text and function tools through `/v1/responses`,
`/v1/chat/completions`, and `/v1/completions`. Responses requests supply complete history.
Unsupported sampling controls, media, streaming, and server-side continuations fail explicitly.
Use `Idempotency-Key` to replay an admitted generation without resampling. Feedback targets the
standard returned response ID. `/v1/train` schedules an update and returns immediately;
`/v1/status` reports queue and update state, while `/v1/drain` waits for a bounded ready snapshot.
All routes require the bearer credential.

Feedback may arrive later as `success=True/False`, scalar `reward` in [-1, 1], or `text`.
SDPO waits for text, REINFORCE waits for a scalar, and `hybrid` waits for both. Sparse missing
signals remain pending; absence is never converted into a zero reward. Exact duplicate feedback
is idempotent; conflicting changes require a new experience. Only complete ready examples train.
The durable buffer retains original student token IDs, sampled log probabilities, tokenizer and
policy identities. Arbitrary provider traces cannot be imported as if they were exact RL samples.

For a later training-only burst, use `"mode": "burst"` with the same immutable recipe/state,
then import `TrainingExample` JSONL if needed:

```console
exp optimize claas burst BURST.json --import exact-experience.jsonl --root ROOT --yes
```

A burst opens the trainer once, performs multiple bounded updates, persists optimizer/teacher/
adapter state before acknowledging consumed records, and closes. An empty burst skips trainer
initialization. A selected Modal host still reserves its configured Sandbox resources until exit.
Pending, rejected, and ready records left by a limit remain visible in the report. Policy-lag and
exact-token checks can reject stale data instead of silently treating it as current-policy data.
One application/user scope owns each directory and adapter; use separate directories for others.
Consumed evidence is retained and counts toward finite storage limits. A full queue rejects new
work; it does not silently delete evidence or replay old training steps. There is no automatic
retention/compaction policy.

Scaffolds and worlds remain outside CLaaS. Reuse the existing `AgentRuntime` and
`EnvironmentRuntime` interfaces through `run_learning_episode`:

```python
from exp.optimize.workflows.learning.scaffold import run_learning_episode

result = run_learning_episode(client=learner, agent=agent, environment=world, task=task)
# The application evaluates the episode and chooses which response gets feedback.
learner.submit_feedback(result.response_ids[-1], text=evaluation_feedback)
```

`world` can call a hosted world model or implement a real tool environment. The workflow does not
mine failures, assign rewards, or copy episode-level feedback onto every action. Applications own
those decisions, traffic import, scenario generation, and held-out evaluation. A completed run
produces artifacts only; it does not hot-swap or deploy a production model.

For Modal, provision an App, a version-2 Volume, and an immutable image containing the same
Experiential build with its selected CUDA dependencies and GNU `/usr/bin/sync`. Reference them with
`ModalLaunch` JSON: `app_name`, `environment_name`, `volume_name`, `image_id`, `gpu`, resource
limits, and `timeout_seconds`. Full runs additionally reference an existing Modal Secret using
`authentication_secret_name`; it must provide the configured authentication environment key.
Never put the token itself in JSON. Use state/checkpoint paths under `/state`,
`"persistence": "modal-volume"`, and `"host": "0.0.0.0"` for a remote full run. The Sandbox
lifetime must exceed startup + run + cleanup by more than 30 seconds.

```console
exp optimize claas serve REMOTE_RUN.json --modal MODAL.json --run-id my-run --root ROOT --yes
exp optimize claas burst REMOTE_BURST.json --modal MODAL.json --run-id my-burst \
  --import exact-experience.jsonl --root ROOT --yes
```

The adapter binds the Volume to one App and permits one live writer. Resume that remote Volume
in place. JSONL is transported before launch; copying a local SQLite queue or checkpoint directory
onto the remote mount is unsupported. Remote durability uses explicit Volume commits before API
acknowledgements. A failed or forced stop is reported as failure and requires inspecting retained
state, not assuming that every leased batch finished. Embedded in-process GPU operations retain
ownership while cleanup joins them, so their Python timeouts are soft if native work stalls.
The local CLI supervises the entire learner POSIX session, including Ray workers in separate
process groups, and escalates from graceful termination to forced cleanup after the configured bounds. Modal supplies the remote hard lifetime limit.
Neither host acknowledges an interrupted update as completed; inspect the retained queue,
checkpoint, and host/run receipts before resuming.

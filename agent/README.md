# exp-code

exp-code is Experiential's open-source coding and cowork agent. Run it in any folder and it
reads your files, writes and runs code, searches the web, and hands back finished work: code,
reports, spreadsheets, decks and PDFs. It is the same agent that powers Chat on the
Experiential platform.

- **Every model on the Experiential gateway.** Models come from the gateway's
  `GET /v1/models`; every call goes through your local `exp` gateway or the hosted one, with
  its cost reported per turn.
- **Cowork tools.** A visible task list (`todo_write`), clarifying questions (`ask_user`), web
  search and page reading (`web_search`, `web_fetch`), and document skills for Word, Excel,
  PowerPoint, PDF, charts, and data analysis.
- **A full terminal agent.** An interactive terminal UI, sessions you can continue, automatic
  context compaction, `/model` to switch models, `/usage` for the session's tokens and cost,
  your own extensions, skills and MCP servers, and print, JSON and RPC modes for scripts.
- **A headless runner.** A file-protocol runner that executes one agent run inside a sandbox,
  for products that host agent sessions.

## Log in

```bash
exp-code login        # opens your browser: sign in, pick an organization, approve
exp-code              # start working in the current folder
```

`exp-code login` starts a one-time listener on `127.0.0.1`, opens the Experiential platform's
approval page, and receives an organization API key named `exp-code (<hostname>)` once you
approve. The key is checked against the gateway (`GET /v1/models`) and saved in
`~/.exp-code/agent/credentials.json`, readable only by you.

| Command | What it does |
|---|---|
| `exp-code login --no-browser` | Prints the approval URL to open on another device and reads the key it shows as a hidden paste (needs an interactive terminal). |
| `exp-code login --status` (or `exp-code whoami`) | Shows which key is in use and where it comes from, masked. |
| `exp-code logout` | Removes the saved key from this computer. The key stays valid until you revoke it on your organization's API keys page. |

Prefer an environment variable? Set `EXP_API_KEY=xpl_...` instead; `EXP_API_KEY` and
`EXP_GATEWAY_KEY` always take precedence over the saved login. Starting the interactive UI with
no key anywhere offers to log in first. `EXP_PLATFORM_URL` points login at another platform origin.

## Install

exp-code needs Node.js 22.19 or newer.

```bash
exp code                       # through the Experiential CLI (recommended)
npx @experiential-labs/exp-code   # or directly
npm install -g @experiential-labs/exp-code   # or installed globally
```

`exp code` runs the exp-code release pinned by your installed `experiential` package (through
`npx --yes @experiential-labs/exp-code@<version>`), points it at the right gateway, and passes every argument on.
For development, `EXP_CODE_DIR=/path/to/agent exp code` runs a built checkout instead.

## Gateway setup

| Gateway | How exp-code finds it |
|---|---|
| Hosted (`https://api.experientiallabs.ai/v1`) | `exp-code login` once, or `EXP_API_KEY=xpl_...`, or `exp login` once and launch with `exp code` |
| Local `exp` gateway (`http://127.0.0.1:8000/v1`) | `EXP_GATEWAY_KEY` set to the virtual key the gateway issued |

`exp code` prefers a local gateway that is running and accepts `EXP_GATEWAY_KEY`, and otherwise
uses the hosted gateway with `EXP_API_KEY`, `EXPLABS_API_KEY`, or your `exp login` credential.
It never starts a gateway: start one with `exp` in another terminal (its first run prints the
key). Run directly, exp-code reads these variables, and falls back to the key `exp-code login`
saved when neither key variable is set:

| Variable | Meaning |
|---|---|
| `EXP_API_KEY` | Hosted gateway key. Selects the hosted gateway. |
| `EXP_GATEWAY_KEY` | Local gateway key. Selects the local gateway when no hosted key is set. |
| `EXP_CODE_GATEWAY` | `hosted` or `local`, to choose explicitly. |
| `EXP_CODE_BASE_URL` | Another base URL for the chosen gateway. |
| `EXP_CODE_MODEL` | Default model alias (otherwise `experiential-large-model` when listed, else the first alias the gateway lists). |
| `EXP_CODE_WIRE` | Force one wire API: `completions`, `responses`, or `messages`. |
| `EXP_CODE_HOME` | Configuration directory (default `~/.exp-code`). |
| `EXP_CODE_PLATFORM_TOOLS` | `0` (or `false`, `off`, `no`) turns off the tools that read your organization's Experiential data. See [Working with your Experiential data](#working-with-your-experiential-data). |
| `EXP_PLATFORM_URL` | Platform origin `exp-code login` opens (default `https://platform.experientiallabs.ai`). |

The model listing is cached for five minutes in `~/.exp-code/cache/` (keyed by a digest of the
endpoint and key; the key is never written). Each alias uses Chat Completions by default; OpenAI
reasoning families (`gpt-5*`, `gpt-6*`, `o*`, `codex*`) use the Responses API, and `claude*`
aliases on the hosted gateway use the Anthropic Messages API. The gateway reports each call's
settled cost, which exp-code shows as the cost of the turn.

## Configuration directory

exp-code keeps its own configuration in `~/.exp-code/agent` (settings, saved credentials,
sessions, extensions, skills). In a project it reads `AGENTS.md` for instructions and `.pi/`
for project settings and resources. Telemetry and the update check are off. Before anything
loads, exp-code points the harness at this directory (`PI_CODING_AGENT_DIR=~/.exp-code/agent`),
so it never shares state with another agent install, and it leaves `PI_TELEMETRY` and
`PI_SKIP_VERSION_CHECK` off unless you set them.

## Modes

```bash
exp-code                                  # interactive terminal UI
exp-code -p "Fix the failing test"        # print the answer and exit
exp-code --mode json "List the TODOs"     # one JSON event per line
exp-code --mode rpc                       # JSON-RPC over stdin/stdout
exp-code --model experiential/<alias>     # pick a model
exp-code -c                               # continue the last session
```

`exp-code --help` lists every flag. The JSON and RPC formats are described in the reference
docs linked under [Acknowledgements](#acknowledgements).

New sessions start on `experiential-large-model` when the gateway lists it (otherwise the first
alias it lists). `EXP_CODE_MODEL`, `--model`, and a model you chose earlier with `/model` all
take precedence. When a session's model is not a gateway model, `web_search` runs on
`experiential-small-model` if the gateway lists it.

The first interactive start installs the two file-search helpers (`fd` and `rg`) into
`~/.exp-code/agent/bin` when they are not on your `PATH`, with one line of progress.

## Session usage

`/usage` (or `/cost`) shows what the current session has used so far, in one block:

```text
Usage · session 01a21fa5, started 2026-11-28 00:54Z
  Time          24.5s since start · model 814ms
  Activity      2 prompts · 2 model calls
  Tokens        7,300 input (3,800 cached) · 420 output (180 reasoning) · 7,720 total
  Cost          $0.0533 charged · $0.000912 free (list value, not charged)
  Context       4,380 / 128,000 tokens (3%) on fast-model

  Model                     Calls  Input  Cached  Output     Cost  Out tok/s
  experiential-large-model      1  3,100       0     240  $0.0533      586.8
  fast-model                    1  4,200   3,800     180     free      444.4

  Organization  Acme Labs · 1,734.57 credits left ($17.35) · today (UTC) $1.235 charged over 42 requests
  History and per-request detail: https://platform.experientiallabs.ai/spend
```

| Line | Where it comes from |
|---|---|
| Time | Wall time since the session was created, plus the summed response time of every model call (`model`) and tool run (`tools`). A resumed session also says how long ago it was opened here and how many of its calls came before. |
| Activity | Your prompts, and every model call the session file records on any branch, including failed calls, compaction summaries and cache warming. |
| Tokens | Each call's reported usage: input is the whole prompt, with the cached part and any cache writes; output includes reasoning tokens when the model reports them. |
| Cost | The settled cost the gateway reported on each call (`usage.cost`), summed exactly in nano-dollars. Calls on your own provider key and free or discounted calls show their list value separately; neither is charged. A call with no reported price is counted as unpriced, never estimated. |
| Context | The same figure as the footer: tokens in the current context over the model's window. |
| Model table | One row per model used (switch with `/model`); `Out tok/s` is output tokens over response time. |
| Organization | With a hosted key: the credit balance (`GET /api/v1/credits`) and today's UTC spend (`GET /api/gateway/usage/daily`), read with the same key. It waits at most 3 seconds and turns into a one-line note when the read fails or the gateway is a local `exp` gateway. |

A response still streaming is added when it finishes. A `web_search` makes its own gateway call,
which the dashboard shows but `/usage` does not yet count. In the terminal UI the block is not sent
to the model; in RPC mode it arrives as a notice. `/session` still shows the session file and
message counts.

## The extension

| Piece | What it does |
|---|---|
| `todo_write` | A visible task list. The whole list is sent on each call. |
| `ask_user` | Asks one question. In the terminal UI, and in RPC mode through `extension_ui_request`, it shows a choice with a free-text option and returns the answer. Without a UI (print, JSON, the runner) it ends the turn and the next message is the answer. |
| `web_search` | One Chat Completions call with `web_search_options` through the gateway, returning ranked results with source URLs. |
| `web_fetch` | Reads a page's main text with Python `requests` and BeautifulSoup, run through the same shell as `bash`. |
| Cowork prompt | A `<cowork>` section: produce deliverables as files, verify them, explain results plainly. |
| Skills | `docx`, `xlsx` (with live formulas), `pptx`, `pdf`, `charts`, and `data-analysis`, in the [Agent Skills](https://agentskills.io) format under `skills/`. The model loads one when a task matches. |

The extension is also a library export (`createExpCodeExtension`) for SDK hosts. It registers the
platform tools below only when given `platformTools: true`.

## Working with your Experiential data

With a hosted gateway key, exp-code can read the organization the key belongs to: its gateway
request log (traces), usage and spend, models and prices, and evaluations. Ask things like "find
the failed requests from the last hour and tell me why", "why did this request cost $2", "what
did we spend on each model this week", or "which model is cheapest with a 200k context". The
tools only read; they never change anything.

| Tool | What it reads |
|---|---|
| `experiential_search_requests` | The request log (`GET /api/orgs/{org}/usage/requests`): filters for time (`since` such as `1h`, or `start`/`end`), model, status (`error` means anything not completed), error class, conversation, API key, lane (`platform` or `byok`), end-user label, and cache hits, sorted newest first or by cost, latency or tokens. One compact row per request (time, model, outcome, tokens, cost, latency, conversation, request id), at most 50 per call, with an opaque cursor for the next page. Default: the last 24 hours. |
| `experiential_get_request` | One request (`.../usage/requests/{id}?include_body=true` and `.../attempts`): the cost breakdown, every provider attempt with its error, and the stored prompt and response, cut to `max_chars` (default 12,000) with a visible marker. When the organization does not store prompt content, it says so. |
| `experiential_usage` | Spend and usage over 24h, 7d or 30d (`.../usage/timeseries`, `by-provider`, `by-label`, `by-key`, `latency`), grouped by model, provider, API key, end-user label or time, with p50/p95 latency and time to first token; `group_by=day` reads daily history (`GET /api/gateway/usage/daily`). |
| `experiential_models` | The models this key can call (`GET /v1/models`, including `experiential-small-model`, `experiential-medium-model` and `experiential-large-model`) joined with the catalog (`GET /api/models`): default provider, context window and limits, and list prices per million tokens. Search by words, filter by provider, context or input price, sort by price or context. |
| `experiential_evals` | Imported eval results and A/B tests (`.../evaluations/results`, `.../evaluations/ab-tests`). |

The org is found once per session with `GET /api/whoami`. A bundled `experiential-traces` skill
tells the model how to debug with these tools; it is offered only when they are on. Refusals come
back as plain messages: a key the API does not accept (401), a read the key may not make (403;
per-key usage and evaluations need the organization's provisioning key), an unknown request
(404), a busy or rate-limited API (429, with the retry delay), or a server problem (5xx). The key
is sent only as the `Authorization` header and never appears in tool output.

Safety defaults. These tools can read your organization's stored prompts and responses, so a
prompt injection in something the agent reads (a web page, a file in the project) could try to
make it fetch traces and copy them elsewhere. Tool output that carries stored content is marked
as data, not instructions. The tools are on by default in the interactive CLI, where you watch the
session and own the key; set `EXP_CODE_PLATFORM_TOOLS=0` to turn them off. They are off in the
headless runner unless its host passes `--platform-tools`, and they are not registered for a
local `exp` gateway, which has no platform API.

## Headless runner

`npm run bundle` writes `dist/runner.mjs`, one self-contained file (the skills are embedded) to
copy into a sandbox:

```bash
AGENT_GATEWAY_KEY=xpl_... AGENT_CALLBACK_TOKEN=... node runner.mjs <runId> \
  [--session-dir /opt/agent/session] [--work-dir /home/user/work] [--user user] [--platform-tools]
```

`--platform-tools` gives the model the read-only [Experiential data tools](#working-with-your-experiential-data)
for the run's key. They are off by default: a run reads pages and files nobody reviewed, and a
prompt injection there could otherwise read the organization's stored prompts and leak them
through another tool (`web_fetch`, `bash`). Enable them only where the key's reach and the
sandbox's network make that acceptable.

The runner reads the two secrets once and deletes them from its environment. When it runs as
root, every tool the model uses runs as the unprivileged `--user` account: its shell commands
(with a minimal environment) and its file reads, writes and edits too, so the kernel decides what
the model may touch and the private session directory (which holds the run's key) is out of its
reach. The bundled skills are then written to `/usr/local/share/exp-code`, which must be owned by
root and writable by nobody else (the runner refuses otherwise): the model can read them but
never plant one. Extensions, prompt templates, and themes are never loaded from the working
directory, which the model can write.

The protocol is files under the session directory (types in `src/runner/protocol.ts`, exported
as `@experiential-labs/exp-code/protocol`):

| File | Direction | Contents |
|---|---|---|
| `runs/<runId>/run.json` | in | The `RunSpec`: model, prompt, attachments, gateway URL, turn and time limits, callback and event-push URLs. |
| `runs/<runId>/events.jsonl` | out | One `RunEvent` per line, `seq` from 1, append-only: `run_start`, `user_message`, `text`, `thinking`, `tool_start`, `tool_update`, `tool_end`, `todo`, `question`, `turn`, `files`, `error`, `run_end`. |
| `runs/<runId>/inbox.jsonl` | in | `{"type":"steer","text":...}` or `{"type":"stop"}`, each with an optional `requestId`; a repeated `requestId` is ignored. |
| `runs/<runId>/status.json` | out | `RunStatus` with a heartbeat every 5 seconds. |
| `session.jsonl` | both | The session file: the transcript, with compaction, carried across runs. |
| `runs/<runId>/runner.json` | internal | Recovery state: restart count, interrupted calls, inbox position. |

Behavior worth knowing:

- Every event is appended to `events.jsonl` before anything else sees it.
- A stopped run leaves a note in the session, so the next message does not resume the stopped
  work. Errors carry codes such as `model_unavailable`, `insufficient_credits`, `rate_limited`,
  `run_limit`, `restart_limit`, and `interrupted`.
- When `eventsUrl` is set, events are also POSTed as `{"runId", "events": [...]}` with
  `authorization: Bearer <AGENT_CALLBACK_TOKEN>`, every ~150 ms or sooner at 200 events or
  ~256 KB. Any 2xx accepts a batch; a failed batch is resent unchanged with backoff (250 ms
  doubling to 5 s), and the receiver deduplicates by `(runId, seq)`. A 404 or 410 ends pushing for
  the run. The final batch, with `run_end`, is awaited for up to 5 seconds before exit. The file
  log never depends on the push.
- A runner started again on a run that never reached `run_end` continues it with the same log:
  tool calls the crash interrupted are re-run when the tool is replay-safe (read-only or
  idempotent) and otherwise reported to the model as interrupted; a model call the crash cut off
  counts as a retry (at most 3); after 5 restarts the run ends with `restart_limit`.

## Development

```bash
npm ci
npm run typecheck
npm test                       # no keys needed; a local fake gateway stands in
npm run build                  # dist/ for the CLI and library
npm run bundle                 # dist/runner.mjs
EXP_API_KEY=xpl_... npm run test:live   # optional live smoke against the hosted gateway
```

Releases are tagged `exp-code-v<version>`. The version in `package.json` must equal
`EXP_CODE_VERSION` in `src/version.ts` and in `exp/cli/code/launcher.py`; tests check both.
Pushing the tag runs `.github/workflows/agent-release.yml`: it tests and bundles the package and
attaches `exp-code-runner.mjs` (the self-contained sandbox runner), its `.sha256`, and
`exp-code-protocol.ts` (the runner protocol) to a GitHub release, so a host can pin a version
and verify the file it runs. With an `NPM_TOKEN` secret configured it also publishes to npm.

## Acknowledgements

exp-code is a fork of [Pi's coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)
(MIT), which it uses as an exact-pinned dependency; its [CLI](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/cli.md),
[JSON](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/json.md), and
[RPC](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/rpc.md) docs describe the formats exp-code keeps.

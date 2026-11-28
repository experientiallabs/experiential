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

### Updates

Once a day, the interactive terminal UI asks the npm registry for the latest release in the
background (a 3 second limit, silent on any failure) and, when it is newer than yours, shows one
line such as:

```
exp-code 0.1.10 is available (you have 0.1.9). Update: npm install -g @experiential-labs/exp-code@latest
```

The answer is cached in `~/.exp-code/cache/update-check.json`. There is no check in print, JSON
or RPC mode, in the headless runner, without a terminal, in CI (`CI` set), offline, or with
`EXP_CODE_NO_UPDATE_CHECK=1`. `exp-code update` updates exp-code the way it was installed:

| Installed with | `exp-code update` |
|---|---|
| `npm install -g` (or pnpm, bun) | Runs `npm install -g @experiential-labs/exp-code@latest` (or `pnpm add -g`, `bun add -g`). |
| `exp code` | Runs `uv tool upgrade experiential` or `pipx upgrade experiential`, or prints `pip install -U experiential`: the `experiential` package pins the exp-code release `exp code` runs. |
| `npx` | Prints `npx @experiential-labs/exp-code@latest`. |
| A source checkout | Prints `git pull && npm ci && npm run build`. |

`exp-code update --extensions`, `--models`, `--extension <source>` and `<source>` update installed
extensions and model catalogs; `--all` updates exp-code and then the extensions.

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
| `EXP_CODE_NO_UPDATE_CHECK` | `1` turns off the daily check for a newer release. |
| `EXP_CODE_NPM_REGISTRY` | Registry the update check reads (default `https://registry.npmjs.org`). |

Every model runs through the gateway, which does the routing and fallback. exp-code does not
load direct vendor providers, so vendor keys in your environment (`GEMINI_API_KEY`,
`OPENAI_API_KEY`, ...) add no models, and `/model`, `--model` and `--list-models` show only the
gateway's aliases. A model saved as a direct provider's (such as `google/gemini-4-neon`) or named
that way with `--model` moves to the gateway alias of the same id, or to the default model when the
gateway does not list it.

The model listing is cached for five minutes in `~/.exp-code/cache/` (keyed by a digest of the
endpoint and key; the key is never written). Each alias uses Chat Completions by default; OpenAI
reasoning families (`gpt-5*`, `gpt-6*`, `o*`, `codex*`) use the Responses API, and `claude*`
aliases on the hosted gateway use the Anthropic Messages API. The gateway reports each call's
settled cost, which exp-code shows as the cost of the turn.

## Configuration directory

exp-code keeps its own configuration in `~/.exp-code/agent` (settings, saved credentials,
sessions, extensions, skills). In a project it reads `AGENTS.md` for instructions and `.exp-code/`
for project settings and resources (`.exp-code/settings.json`, `.exp-code/extensions/`,
`.exp-code/skills/`, `.exp-code/prompts/`). Install telemetry is off.

These variables tune the terminal agent:

| Variable | Meaning |
|---|---|
| `EXP_CODE_SESSION_DIR` | Session storage directory (overridden by `--session-dir`). |
| `EXP_CODE_OFFLINE` | `1` disables startup network operations, as `--offline` does. |
| `EXP_CODE_CACHE_RETENTION` | `long` asks for long prompt-cache retention where the model supports it. |
| `EXP_CODE_EXPERIMENTAL` | `1` turns on experimental features. |
| `EXP_CODE_CODEMODE_MAX_WORKERS` | Maximum parallel codemode script workers. |
| `EXP_CODE_CLIPBOARD_OSC52` | `1` also copies through the terminal (OSC 52), as over SSH. |
| `EXP_CODE_HARDWARE_CURSOR` | `1` uses the terminal cursor instead of the drawn one. |
| `EXP_CODE_CLEAR_ON_SHRINK` | `1` clears the screen when the UI shrinks. |
| `EXP_CODE_TRUE_COLOR` | `1` or `0` overrides true-color detection. |
| `EXP_CODE_HYPERLINKS` | `1` or `0` overrides terminal hyperlink detection. |
| `EXP_CODE_IMAGE_PROTOCOL` | Inline image protocol (`kitty`, `iterm2`, or `none`). |
| `EXP_CODE_PROGRAM_STATUS` | `1` or `0` states whether the terminal supports program status (OSC 7501) instead of asking it. |
| `EXP_CODE_TUI_ESC_TIMEOUT` | Milliseconds to wait after Esc for an escape sequence. |
| `EXP_CODE_TIMING` | `1` prints startup timings (debugging). |
| `EXP_CODE_STARTUP_BENCHMARK` | `1` measures interactive startup and exits (debugging). |
| `EXP_CODE_TUI_DEBUG` | `1` writes terminal UI debug logs. |
| `EXP_CODE_TUI_DEBUG_REDRAW` | `1` logs full redraws (debugging). |
| `EXP_CODE_TUI_WRITE_LOG` | File that receives every terminal write (debugging). |

Shell commands the agent runs see `EXP_CODE_AGENT=true`, `EXP_CODE_SESSION_ID`,
`EXP_CODE_SESSION_FILE`, `EXP_CODE_SESSION_PROVIDER`, `EXP_CODE_SESSION_MODEL` and
`EXP_CODE_SESSION_REASONING_LEVEL`.

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

exp-code remembers your choices. A model you pick (with `/model`, `/model <name>` or Ctrl+P) is
the one the next session starts on, and a thinking level you pick (Shift+Tab or `/thinking`) is
kept for that model and used again whenever you select it. Settings changed in `/settings` are
saved as you change them. `--model` and `EXP_CODE_MODEL` choose the model for one run and are
never saved. With no saved choice, new sessions start on `experiential-large-model` when the
gateway lists it (otherwise the first alias it lists). When the gateway no longer lists your saved
model, the session starts on that default with a one-line note, and your choice is used again once
the model is listed.

Each model's context window and output limit come from the gateway listing
(`context_window_tokens`, `maximum_output_tokens`), and `/thinking` and Shift+Tab offer only the
reasoning levels the model accepts (`supported_reasoning_efforts`). The footer shows the whole
window (for example `1.05M`). That window includes the model's maximum output, which the provider
reserves on every request, so automatic compaction runs before the prompt reaches the window less
that output (about 906K tokens of prompt on a 1.05M window with 128K output). When a session's model is not a gateway model, `web_search` runs on
`experiential-small-model` if the gateway lists it.

Every model that reasons starts at `medium` thinking, or at the closest level its route accepts
(the next one up, else the next one down: a route offering `none`, `low` and `high` starts at
`high`), unless you chose a level for that model or saved a default level. The level is sent with
every request (`reasoning_effort` on Chat Completions, `reasoning.effort` on Responses), so the
footer always names what the model was asked for. `thinking off` exists only on a route that
accepts the `none` effort; a model that always reasons offers its lowest level instead. A model
without reasoning shows no level.

Compaction asks the model for a summary of the conversation and replaces the older part of the
conversation with it, so exp-code checks the summary first. The request asks for at most 32K
tokens of output (or the model's maximum, when smaller) at no more than `low` reasoning. A summary
cut off at the output limit, empty, or implausibly short for the conversation is asked for once
more, with the model's whole output limit and its lowest reasoning level. When that also fails,
the conversation is kept as it was and exp-code says why; automatic compaction then pauses for that
model in that session, so it is not retried before every call. Run `/compact` to try again, `/model`
to switch to another model, or `/new` to start a fresh session. The same check guards a split
turn's summary and the branch summaries `/tree` writes.

In the terminal UI, `/bug` opens a new GitHub issue for exp-code prefilled with your exp-code
version, operating system, Node.js version and model (nothing is sent until you submit it), and
`/changelog` links exp-code's release notes. There is no `/share`: it uploaded sessions to a
service outside Experiential; `/export` saves a session as HTML or JSONL.

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

## Stash a prompt (Ctrl+S)

Halfway through writing a prompt and need to ask something else first? Press **Ctrl+S**: the
prompt is set aside and the input is cleared. Send your other message, and the stashed prompt comes
back into the input by itself. It works the same as Claude Code's Ctrl+S.

| You do | What happens |
|---|---|
| Ctrl+S with text in the input | The text is stashed exactly (new lines, spaces, emoji, large pastes, image and file paths, the cursor position) and the input is cleared. A line below the input shows what is stashed. |
| Ctrl+S again with more text | It is stashed too. Stashed prompts stack, newest on top; nothing is replaced. |
| Ctrl+S on an empty input | The newest stashed prompt comes back. Two quick presses put the text back where it was. |
| Send a message (also a steer while the agent works, a follow-up, or a `!` command) | The newest stashed prompt comes back, but only into an empty input: if you have started typing again, it stays stashed and a hint says so. |
| Built-in commands such as `/model` or `/settings` | Not a send: the stash stays. |
| Ctrl+C, Escape, Up and Down | The stash stays. Stashing a message you recalled with Up brings back the draft it was hiding. |
| `/new`, `/resume`, `/fork` | The stash belongs to the folder, not the session, so it stays. |
| Exit (Ctrl+D, `/quit`) or a crash | The stash is already on disk (`~/.exp-code/agent/prompt-stash/`, readable only by you). The next exp-code started in that folder puts it back in the input and removes the file. |

Input that is only spaces counts as empty. Undo after a stash cannot bring the text back a second
time, and undo after a restore continues the prompt's own edits. Two exp-code windows in the same
folder each keep their own stash (one file per running exp-code, written atomically); a stash is
picked up by a new start only once the exp-code that made it has exited. A stash file that cannot
be read is set aside (renamed `.corrupt`) and ignored.

Ctrl+S reaches exp-code even in terminals that use it to pause output (XON/XOFF flow control):
exp-code puts the terminal in raw mode, which turns that off while it runs and restores it on exit.
To use another key, or none, set `exp-code.prompt.stash` in `~/.exp-code/agent/keybindings.json`,
for example `{ "exp-code.prompt.stash": "alt+s" }` (a list binds several keys; `[]` binds none).
Without a key, `/stash` does the same:

| Command | What it does |
|---|---|
| `/stash <text>` | Stashes the text. |
| `/stash` or `/stash pop` | Brings the newest stashed prompt back into the empty input. |
| `/stash list` | Shows what is stashed, newest first. |
| `/stash drop` | Discards the newest stashed prompt. |
| `/stash help` | Explains the above. |

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

exp-code runs its harness as an exact-pinned dependency and changes a few things the
dependency has no option for (its name and wording, the `.exp-code/` project folder, the direct
vendor providers, and commands that talk to the harness's own services) through a table of exact
source edits in `src/cli/pi-patches.ts`. The CLI applies them with a Node.js module load hook
before the harness loads, and `npm run bundle` applies the same table while bundling the runner.
An edit that no longer matches fails the tests (`src/cli/pi-patches.test.ts`), so a dependency
upgrade cannot ship unpatched text silently.

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

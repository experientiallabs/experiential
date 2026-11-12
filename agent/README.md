# exp-code

exp-code is Experiential's open-source coding and cowork agent. It is
[Pi's coding agent](https://github.com/earendil-works/pi/tree/main/packages/coding-agent)
(`@earendil-works/pi-coding-agent`, an exact-pinned dependency, not a fork) with three things
preloaded:

- **The Experiential gateway as a model provider.** Models come from the gateway's
  `GET /v1/models`; every call goes through your local `exp` gateway or the hosted one.
- **The exp-code extension.** Cowork tools (`todo_write`, `ask_user`, `web_search`,
  `web_fetch`), a cowork section in the system prompt, and document skills for Word, Excel,
  PowerPoint, PDF, charts, and data analysis.
- **A headless runner.** A file-protocol runner that executes one agent run inside a sandbox,
  for products that host agent sessions.

Everything else is Pi: the terminal UI, sessions, compaction, `/model`, extensions, skills,
MCP, print mode, JSON mode, and RPC mode all work as Pi documents them.

## Install

exp-code needs Node.js 22.19 or newer.

```bash
exp code                       # through the Experiential CLI (recommended)
npx exp-code                   # or directly
npm install -g exp-code        # or installed globally
```

`exp code` runs the exp-code release pinned by your installed `experiential` package (through
`npx --yes exp-code@<version>`), points it at the right gateway, and passes every argument on.
For development, `EXP_CODE_DIR=/path/to/agent exp code` runs a built checkout instead.

## Gateway setup

| Gateway | How exp-code finds it |
|---|---|
| Hosted (`https://api.experientiallabs.ai/v1`) | `EXP_API_KEY=xpl_...`, or `exp login` once and launch with `exp code` |
| Local `exp` gateway (`http://127.0.0.1:8000/v1`) | `EXP_GATEWAY_KEY` set to the virtual key the gateway issued |

`exp code` prefers a local gateway that is running and accepts `EXP_GATEWAY_KEY`, and otherwise
uses the hosted gateway with `EXP_API_KEY`, `EXPLABS_API_KEY`, or your `exp login` credential.
It never starts a gateway: start one with `exp` in another terminal (its first run prints the
key). Run directly, exp-code reads these variables:

| Variable | Meaning |
|---|---|
| `EXP_API_KEY` | Hosted gateway key. Selects the hosted gateway. |
| `EXP_GATEWAY_KEY` | Local gateway key. Selects the local gateway when no hosted key is set. |
| `EXP_CODE_GATEWAY` | `hosted` or `local`, to choose explicitly. |
| `EXP_CODE_BASE_URL` | Another base URL for the chosen gateway. |
| `EXP_CODE_MODEL` | Default model alias (otherwise the first alias the gateway lists). |
| `EXP_CODE_WIRE` | Force one wire API: `completions`, `responses`, or `messages`. |
| `EXP_CODE_HOME` | Configuration directory (default `~/.exp-code`). |

The model listing is cached for five minutes in `~/.exp-code/cache/` (keyed by a digest of the
endpoint and key; the key is never written). Each alias uses Chat Completions by default; OpenAI
reasoning families (`gpt-5*`, `gpt-6*`, `o*`, `codex*`) use the Responses API, and `claude*`
aliases on the hosted gateway use the Anthropic Messages API. The gateway reports each call's
settled cost, which exp-code shows as the cost of the turn.

## Configuration directory

exp-code keeps its own configuration in `~/.exp-code/agent` (settings, saved credentials,
sessions, extensions, skills), separate from any Pi installation, by setting Pi's
`PI_CODING_AGENT_DIR` before Pi loads. Project files keep Pi's conventions (`AGENTS.md`, and
`.pi/` for project settings and resources), so a project can share them with Pi. Telemetry
(`PI_TELEMETRY`) and the version check (`PI_SKIP_VERSION_CHECK`) are off unless you set them.

## Modes

```bash
exp-code                                  # interactive terminal UI
exp-code -p "Fix the failing test"        # print the answer and exit
exp-code --mode json "List the TODOs"     # one JSON event per line
exp-code --mode rpc                       # JSON-RPC over stdin/stdout
exp-code --model experiential/<alias>     # pick a model
exp-code -c                               # continue the last session
```

See Pi's [CLI](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/cli.md),
[JSON](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/json.md), and
[RPC](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/rpc.md) docs.

## The extension

| Piece | What it does |
|---|---|
| `todo_write` | A visible task list. The whole list is sent on each call. |
| `ask_user` | Asks one question. In the terminal UI, and in RPC mode through `extension_ui_request`, it shows a choice with a free-text option and returns the answer. Without a UI (print, JSON, the runner) it ends the turn and the next message is the answer. |
| `web_search` | One Chat Completions call with `web_search_options` through the gateway, returning ranked results with source URLs. |
| `web_fetch` | Reads a page's main text with Python `requests` and BeautifulSoup, run through the same shell as `bash`. |
| Cowork prompt | A `<cowork>` section: produce deliverables as files, verify them, explain results plainly. |
| Skills | `docx`, `xlsx` (with live formulas), `pptx`, `pdf`, `charts`, and `data-analysis`, in the [Agent Skills](https://agentskills.io) format under `skills/`. The model loads one when a task matches. |

The extension is also a library export (`createExpCodeExtension`) for SDK hosts.

## Headless runner

`npm run bundle` writes `dist/runner.mjs`, one self-contained file (the skills are embedded) to
copy into a sandbox:

```bash
AGENT_GATEWAY_KEY=xpl_... AGENT_CALLBACK_TOKEN=... node runner.mjs <runId> \
  [--session-dir /opt/agent/session] [--work-dir /home/user/work] [--user user]
```

The runner reads the two secrets once and deletes them from its environment. When it runs as
root, every tool the model uses runs as the unprivileged `--user` account: its shell commands
(with a minimal environment) and its file reads, writes and edits too, so the kernel decides what
the model may touch and the private session directory (which holds the run's key) is out of its
reach. The bundled skills are then written to `/usr/local/share/exp-code`, which must be owned by
root and writable by nobody else (the runner refuses otherwise): the model can read them but
never plant one. Extensions, prompt templates, and themes are never loaded from the working
directory, which the model can write.

The protocol is files under the session directory (types in `src/runner/protocol.ts`, exported
as `exp-code/protocol`):

| File | Direction | Contents |
|---|---|---|
| `runs/<runId>/run.json` | in | The `RunSpec`: model, prompt, attachments, gateway URL, turn and time limits, callback and event-push URLs. |
| `runs/<runId>/events.jsonl` | out | One `RunEvent` per line, `seq` from 1, append-only: `run_start`, `user_message`, `text`, `thinking`, `tool_start`, `tool_update`, `tool_end`, `todo`, `question`, `turn`, `files`, `error`, `run_end`. |
| `runs/<runId>/inbox.jsonl` | in | `{"type":"steer","text":...}` or `{"type":"stop"}`, each with an optional `requestId`; a repeated `requestId` is ignored. |
| `runs/<runId>/status.json` | out | `RunStatus` with a heartbeat every 5 seconds. |
| `session.jsonl` | both | Pi's session file: the transcript, with compaction, carried across runs. |
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

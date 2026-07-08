# pi-swe — a coding-agent corpus with harness capture

Four small SWE episodes of the [pi coding agent](https://www.npmjs.com/package/@earendil-works/pi-coding-agent)
working in `/workspace`, captured as OTel GenAI JSONL. What makes this corpus different from the
benchmark corpora under `packages/environment-capture/` is that it records the **agent-side
harness**:

- `gen_ai.system_instructions` — the system prompt pi actually assembles (tools list,
  guidelines, cwd, date), rendered by the real `@earendil-works/pi-coding-agent@0.74.0`
  package's own `buildSystemPrompt`;
- `gen_ai.tool.definitions` — pi's real `read` / `bash` / `edit` / `write` JSON schemas from
  `createAllToolDefinitions`;
- `wmh.state.structured` — `{"cwd": "/workspace", "harness": "pi"}` on every action span.

A world model built from it therefore knows the harness contract it is simulating (tool argument
schemas, validation-error shapes, output conventions), and `wmh scenarios create` can assemble a
token-realistic scenario — the exact system prompt + tools + verbatim user message — for any new
task.

## Provenance

The episodes are hand-authored demonstrations (not captures of a live run), but every harness
artifact and observation format is taken from the real pi package: `write` returns
`Successfully wrote <n> bytes to <path>`, `edit` returns
`Successfully replaced <n> block(s) in <path>.`, `read` returns raw file contents, `bash` returns
raw stdout — all probed against pi v0.74.0's actual tools. Regenerate with:

```bash
uv run python .agents/scripts/make_pi_swe_corpus.py
```

## Walkthrough: task description -> scenario -> play

```bash
# 1. Build a world model of the pi SWE environment (the harness is persisted with the model):
uv run wmh build --name pi-swe --file examples/pi-swe/traces.otel.jsonl

# 2. Create a token-realistic scenario from a plain task description:
uv run wmh scenarios create --task "build a python airbnb clone" --name pi-swe
#    prints the exact system prompt + tools + user message the pi harness would send,
#    and writes scenario.json (seed state + judgeable checklist synthesized by an LLM).

# 3. Step it against the world model, turn by turn:
uv run wmh play --name pi-swe --task "build a python airbnb clone"
```

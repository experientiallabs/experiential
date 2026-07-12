# kimi-gui-control

Computer-use agent runs that drive macOS GUI apps (Safari, Chrome, Notes, Finder, Calculator, …)
through the macOS Accessibility API plus a shell. Each trajectory is a task like *"browse the latest
cs.CL listings on arXiv, open the top paper, and report the title, author count, and abstract"*: the
agent reads the accessibility tree, takes a single targeted action, and re-reads the tree to confirm.

## Contents

- `traces.otel.jsonl` - the trace corpus (**Hub-hosted, not committed**; see § Data & license):
  ~60 trajectories, one Step per agent tool call, enough for the 30 train / 8 val / 8 test benchmark
  split. Materialize it with `uv run wmh download kimi-gui-control` (or
  `uv run python -m environment_capture.hub fetch kimi-gui-control`).
- `convert_to_wmh.py` - the converter that produced the corpus (see § Regenerate).
- `evals/default.toml` - fidelity suite; run with
  `uv run wmh eval run kimi-gui-control/default --examples-root packages/environment-capture`.

## What the converter produces

`convert_to_wmh.py` reads the source JSONL **streaming** (the raw dump is ~9 GB / 1000 trajectories,
so it never loads the file into memory) and emits one trace per trajectory, one Step per agent
**tool call**:

- `action` - the real tool call (`name` + `arguments`, e.g. `read`, `bash`, GUI actions).
- `observation` - the **real recorded tool output** the agent saw (`gen_ai.tool.message`), error
  flag from the recorded `isError`.
- `Trace.metadata` - `benchmark`, `task_category`, `task_url`, `model`, `provider`, `returncode`.

`state_before` is intentionally **empty**: the real GUI/OS state (full accessibility tree, open
windows, filesystem) isn't captured as a compact snapshot. Open-loop replay reconstructs the
environment from the action + retrieved similar past steps + the teacher-forced session history.

Trajectories with **zero tool calls** are skipped: open-loop replay scores predicted observations
for `(state, action)`, and a chat-only turn has no environment observation to score. The output is
OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.

## Data & license

- **Harness:** GUI-control agent trajectories captured with the [screenpipe](https://github.com/mediar-ai/screenpipe)
  `gui-control` stack (MIT).
- **Trajectories:** produced by us running **Kimi-K2.6 via Azure AI Foundry** driving the harness on
  real macOS apps; the observations are the verbatim tool outputs the agent saw. These are our own
  captures, published under MIT alongside the harness attribution above.
- **Payload is not committed.** Per this package's `.gitignore`, `traces.otel.jsonl` is not tracked
  in git; it lives in the public dataset `experiential-labs/wmh-kimi-gui-control-traces` on the
  Hugging Face Hub and is fetched on demand (see § Contents). This keeps the repo free of large
  binary blobs and follows the same contract as every other corpus here (`../README.md`).

## Regenerate

From the raw screenpipe dump (path is machine-local - the ~9 GB source is not redistributed):

```bash
cd packages/environment-capture/kimi-gui-control
python convert_to_wmh.py <raw_screenpipe_dump>.jsonl --out traces.otel.jsonl --limit 60
```

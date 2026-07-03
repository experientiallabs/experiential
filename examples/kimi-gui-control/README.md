# kimi-gui-control GUI-control trace corpus

This directory converts a **screenpipe `gui-control` agent** trace dump into the world-model-harness
trace corpus (`examples/kimi-gui-control/traces.otel.jsonl`).

The dataset is a set of agent trajectories that drive macOS GUI apps (Safari, Chrome, Notes, Finder,
Calculator, …) through the macOS Accessibility API plus a shell, produced by **Kimi-K2.6 via
`azure-foundry`**. Each trajectory is a task like *"browse the latest cs.CL listings on arXiv, open
the top paper, and report the title, author count, and abstract"*: the agent reads the accessibility
tree, takes a single targeted action, and re-reads the tree to confirm.

Like the tau-bench example, this folder does **not** make `wmh` depend on the capture stack — only
the converter and the produced `traces.otel.jsonl` are tracked. `examples/` is excluded from the
`wmh` lint/type gate.

## What the converter produces

`convert_to_wmh.py` reads the source JSONL **streaming** (the source is ~9GB / 1000 trajectories, so
it never loads the file into memory) and emits one trace per trajectory, one Step per agent **tool
call**:

- `action` — the real tool call (`name` + `arguments`, e.g. `read`, `bash`, GUI actions).
- `observation` — the **real recorded tool output** the agent saw (`gen_ai.tool.message`), error
  flag from the recorded `isError`.
- `Trace.metadata` — `benchmark`, `task_category`, `task_url`, `model`, `provider`, `returncode`.

`state_before` is intentionally **empty**: the real GUI/OS state (full accessibility tree, open
windows, filesystem) isn't captured as a compact snapshot. Open-loop replay reconstructs the
environment from the action + retrieved similar past steps + the teacher-forced session history.

Trajectories with **zero tool calls** are skipped: open-loop replay scores predicted observations
for `(state, action)`, and a chat-only turn has no environment observation to score.

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.

## Regenerate

The committed corpus is limited to ~60 trajectories (enough for the 30 train / 8 val / 8 test
benchmark split):

```bash
cd examples/kimi-gui-control
python convert_to_wmh.py /path/to/traces_kimi_k26_1000.jsonl --out traces.otel.jsonl --limit 60
```

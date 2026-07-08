"""Generate `examples/pi-swe/traces.otel.jsonl`: pi coding-agent SWE episodes with harness capture.

The harness data under `.agents/scripts/pi_harness/` (system prompt + tool JSON schemas) was
rendered by the REAL `@earendil-works/pi-coding-agent` package (v0.74.0) via its own
`buildSystemPrompt` / `createAllToolDefinitions` with cwd=/workspace and the default tool set —
see the node one-liner in this dir's README. The episodes themselves are hand-written but use
pi's actual observation formats (probed against the real tools):

  - write -> "Successfully wrote <n> bytes to <path>"
  - edit  -> "Successfully replaced <n> block(s) in <path>."
  - read  -> the raw file contents
  - bash  -> raw stdout/stderr

Every trace's first LLM span carries `gen_ai.system_instructions` + `gen_ai.tool.definitions`
(the harness capture this corpus exists to demonstrate) and every LLM span carries the
`wmh.state.structured` snapshot `{"cwd": "/workspace", "harness": "pi"}`.

Run from the repo root:  uv run python .agents/scripts/make_pi_swe_corpus.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
OUT = REPO / "examples" / "pi-swe" / "traces.otel.jsonl"

SYSTEM_PROMPT = (HERE / "pi_harness" / "pi_system_prompt.txt").read_text(encoding="utf-8")
TOOLS = json.loads((HERE / "pi_harness" / "pi_tools.json").read_text(encoding="utf-8"))

STATE = {"cwd": "/workspace", "harness": "pi"}


def wrote(path: str, content: str) -> str:
    return f"Successfully wrote {len(content.encode('utf-8'))} bytes to {path}"


FLASK_APP = """from flask import Flask, render_template, request, redirect

app = Flask(__name__)
todos: list[str] = []


@app.route("/")
def index():
    return render_template("index.html", todos=todos)


@app.route("/add", methods=["POST"])
def add():
    text = request.form.get("text", "").strip()
    if text:
        todos.append(text)
    return redirect("/")


@app.route("/delete/<int:i>", methods=["POST"])
def delete(i: int):
    if 0 <= i < len(todos):
        todos.pop(i)
    return redirect("/")


if __name__ == "__main__":
    app.run(debug=True)
"""

FLASK_TEMPLATE = """<!doctype html>
<html>
  <body>
    <h1>Todos</h1>
    <form action="/add" method="post"><input name="text"><button>Add</button></form>
    <ul>
      {% for todo in todos %}
      <li>{{ todo }}
        <form action="/delete/{{ loop.index0 }}" method="post"><button>x</button></form>
      </li>
      {% endfor %}
    </ul>
  </body>
</html>
"""

CSV_SCRIPT = """import csv
import json
import sys


def convert(csv_path: str, json_path: str) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
"""

MATHLIB_BROKEN = """def add(a, b):
    return a - b


def mul(a, b):
    return a * b
"""

LANDING_PAGE = """<!doctype html>
<html>
  <head><title>Driftwood Coffee</title></head>
  <body>
    <h1>Driftwood Coffee</h1>
    <p>Small-batch roasts on the waterfront. Open 7-3 daily.</p>
  </body>
</html>
"""


def tool_step(name: str, args: dict, obs: str, *, is_error: bool = False) -> dict:
    return {"tool": name, "args": args, "obs": obs, "is_error": is_error}


def message_step(text: str) -> dict:
    return {"message": text}


TRACES: list[dict] = [
    {
        "trace_id": "9f1e6a2c47d3000000000000000000a1",
        "task": "Make me a simple flask todo app with add and delete",
        "reward": 1.0,
        "steps": [
            tool_step("bash", {"command": "python3 --version && pip show flask | head -2"},
                      "Python 3.11.9\nName: Flask\nVersion: 3.0.3\n"),
            tool_step("bash", {"command": "mkdir -p /workspace/todo/templates"}, ""),
            tool_step("write", {"path": "/workspace/todo/app.py", "content": FLASK_APP},
                      wrote("/workspace/todo/app.py", FLASK_APP)),
            tool_step("write",
                      {"path": "/workspace/todo/templates/index.html", "content": FLASK_TEMPLATE},
                      wrote("/workspace/todo/templates/index.html", FLASK_TEMPLATE)),
            tool_step("bash", {"command": "cd /workspace/todo && python3 -c 'import app; print(\"ok\")'"},
                      "ok\n"),
            message_step("Created a Flask todo app in /workspace/todo: app.py serves an index "
                         "with an add form and per-item delete buttons backed by an in-memory "
                         "list. Run it with `python3 app.py`."),
        ],
    },
    {
        "trace_id": "9f1e6a2c47d3000000000000000000a2",
        "task": "Write a python script that converts a csv file to json",
        "reward": 1.0,
        "steps": [
            tool_step("write", {"path": "/workspace/convert.py", "content": CSV_SCRIPT},
                      wrote("/workspace/convert.py", CSV_SCRIPT)),
            tool_step("bash",
                      {"command": "printf 'name,age\\nada,36\\nalan,41\\n' > /workspace/people.csv"},
                      ""),
            tool_step("bash",
                      {"command": "cd /workspace && python3 convert.py people.csv people.json "
                                  "&& cat people.json"},
                      '[\n  {\n    "name": "ada",\n    "age": "36"\n  },\n  {\n    "name": "alan",'
                      '\n    "age": "41"\n  }\n]'),
            message_step("Wrote /workspace/convert.py: `python3 convert.py <in.csv> <out.json>` "
                         "reads the CSV with DictReader and dumps a JSON array of row objects. "
                         "Verified on a sample file."),
        ],
    },
    {
        "trace_id": "9f1e6a2c47d3000000000000000000a3",
        "task": "The tests in /workspace/mathlib are failing, find the bug and fix it",
        "reward": 1.0,
        "steps": [
            tool_step("bash", {"command": "cd /workspace/mathlib && python3 -m pytest -q"},
                      "..F\n=== FAILURES ===\ntest_core.py::test_add - assert add(2, 2) == 4\n"
                      "  where -0 = add(2, 2)\n1 failed, 2 passed in 0.04s\n",
                      is_error=True),
            tool_step("read", {"path": "/workspace/mathlib/core.py"}, MATHLIB_BROKEN),
            tool_step("edit",
                      {"path": "/workspace/mathlib/core.py",
                       "edits": [{"oldText": "return a - b", "newText": "return a + b"}]},
                      "Successfully replaced 1 block(s) in /workspace/mathlib/core.py."),
            tool_step("bash", {"command": "cd /workspace/mathlib && python3 -m pytest -q"},
                      "...\n3 passed in 0.03s\n"),
            message_step("Fixed: core.py's add() subtracted instead of adding. All 3 tests pass."),
        ],
    },
    {
        "trace_id": "9f1e6a2c47d3000000000000000000a4",
        "task": "Make me a landing page for my coffee shop",
        "reward": 1.0,
        "steps": [
            # The empty-arguments validation error the real pi corpus is full of.
            tool_step("write", {},
                      'Validation failed for tool "write":\n'
                      "  - path: must have required properties path, content\n\n"
                      "Received arguments:\n{}",
                      is_error=True),
            tool_step("write", {"path": "/workspace/site/index.html", "content": LANDING_PAGE},
                      wrote("/workspace/site/index.html", LANDING_PAGE)),
            tool_step("bash", {"command": "ls /workspace/site"}, "index.html\n"),
            message_step("Created /workspace/site/index.html — a minimal landing page with the "
                         "shop name, tagline, and hours. Open it in a browser to preview."),
        ],
    },
]


def _attr(key: str, value: str) -> dict:
    return {"key": key, "value": {"stringValue": value}}


def build_spans(trace: dict) -> list[dict]:
    spans: list[dict] = []
    t = 0

    def span_id() -> str:
        return f"{trace['trace_id'][:12]}{len(spans):03d}a"

    for i, step in enumerate(trace["steps"]):
        t += 1
        attrs = [_attr("gen_ai.operation.name", "chat")]
        if i == 0:
            attrs.append(_attr("gen_ai.system_instructions", SYSTEM_PROMPT))
            attrs.append(_attr("gen_ai.tool.definitions", json.dumps(TOOLS)))
            attrs.append(
                _attr("wmh.trace.metadata",
                      json.dumps({"benchmark": "pi-swe", "reward": trace["reward"]}))
            )
        attrs.append(_attr("gen_ai.prompt", trace["task"]))
        attrs.append(_attr("wmh.state.structured", json.dumps(STATE)))
        if "message" in step:
            attrs.append(_attr("gen_ai.completion", step["message"]))
            spans.append({
                "traceId": trace["trace_id"], "spanId": span_id(), "parentSpanId": "",
                "name": "chat pi", "startTimeUnixNano": t, "endTimeUnixNano": t + 1,
                "status": {"code": "STATUS_CODE_OK"}, "attributes": attrs,
            })
            continue
        attrs.append(_attr("gen_ai.tool.name", step["tool"]))
        attrs.append(_attr("gen_ai.tool.call.arguments", json.dumps(step["args"])))
        spans.append({
            "traceId": trace["trace_id"], "spanId": span_id(), "parentSpanId": "",
            "name": "chat pi", "startTimeUnixNano": t, "endTimeUnixNano": t + 1,
            "status": {"code": "STATUS_CODE_OK"}, "attributes": attrs,
        })
        t += 1
        spans.append({
            "traceId": trace["trace_id"], "spanId": span_id(), "parentSpanId": "",
            "name": f"execute_tool {step['tool']}", "startTimeUnixNano": t,
            "endTimeUnixNano": t + 1,
            "status": {"code": "STATUS_CODE_ERROR" if step["is_error"] else "STATUS_CODE_OK"},
            "attributes": [
                _attr("gen_ai.operation.name", "execute_tool"),
                _attr("gen_ai.tool.name", step["tool"]),
                _attr("gen_ai.tool.message", step["obs"]),
            ],
        })
    return spans


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for trace in TRACES:
            for span in build_spans(trace):
                fh.write(json.dumps(span, ensure_ascii=False) + "\n")
    count = sum(1 for _ in OUT.open())
    print(f"wrote {count} spans across {len(TRACES)} traces -> {OUT}")  # noqa: T201


if __name__ == "__main__":
    main()

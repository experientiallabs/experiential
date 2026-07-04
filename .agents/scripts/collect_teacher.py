"""Privileged-teacher trajectory collection in the world model (charter stage 2).

A Gemini 2.5 Pro teacher rolls every TRAIN-pool scenario against the frozen tau world model,
conditioned on the scenario's SOURCE trace digest as a reference walkthrough (arXiv 2606.12072's
privileged Demonstrator). Each turn the teacher must produce explicit reasoning ("think") plus one
tool call — BENCH-B2 showed SFT without deliberation stops checking policy constraints. Episodes
are graded by a Gemini Flash checklist judge; only passing episodes are kept (rejection sampling),
capped per scenario, and exported as Qwen-convention chat messages (<think>, <tool_call>,
<tool_response>) ready for LoRA SFT.

Outputs (under .agents/docs/research/distill/):
    teacher_episodes.jsonl   one record per KEPT episode (messages + provenance + judge verdict)
    teacher_report.json      per-scenario keep/attempt stats

Usage (from the repo root):
    uv run python .agents/scripts/collect_teacher.py [--samples 4] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / ".agents" / "scripts"))

from run_scenario_e2e import NOVA_LITE, TRACES, WM_DIR, RetryProvider, bedrock  # noqa: E402

import os  # noqa: E402

from wmh.core.parsing import extract_json_object  # noqa: E402
from wmh.core.types import ActionKind, Trace  # noqa: E402
from wmh.engine.world_model import WorldModel  # noqa: E402
from wmh.env.base import WorldModelEnv  # noqa: E402
from wmh.ingest import get_adapter  # noqa: E402
from wmh.providers import get_provider  # noqa: E402
from wmh.providers.base import Message, Provider, ProviderConfig, ProviderKind  # noqa: E402
from wmh.scenarios import ChecklistJudge, ScenarioSet, trace_digest  # noqa: E402
from wmh.scenarios.synthesis import EvalScenario  # noqa: E402

DISTILL = REPO / ".agents" / "docs" / "research" / "distill"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/"
TEACHER_MODEL = "gemini-2.5-pro"
JUDGE_MODEL = "gemini-2.5-flash"
MAX_STEPS = 10
KEEP_PER_SCENARIO = 2
WORKERS = 6

TEACHER_SYSTEM = """You are an expert agent completing a task in a tool environment. You are
given a REFERENCE WALKTHROUGH from a prior successful episode of a similar task. Use it as a
guide to which tools exist and what a good strategy looks like — but the environment may respond
differently this time; adapt to what it actually returns instead of copying blindly.

Each turn, respond with ONLY a JSON object, no prose around it — one of:
{"think": "<2-4 sentences: what you know, what policy constraints apply, why this next call>",
 "tool": "<tool name>", "arguments": {...}}
{"think": "<why the task is complete or impossible>", "done": true,
 "summary": "<what you achieved, including anything you must communicate to the user>"}

Rules:
- ALWAYS check policy constraints (eligibility windows, fees, membership rules) BEFORE acting on
  anything irreversible; verify state with a read call when unsure.
- Work efficiently: no redundant calls; finish as soon as the task is done."""


def _read_env_value(path, name):  # noqa: ANN001, ANN202
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return None


def _load_gemini_key() -> None:
    """Load provider keys from the experientiallabs .env.local files (never echoed).

    - AZURE_FOUNDRY_API_KEY -> WMH_ENDPOINT_API_KEY (custom-endpoint auth: DeepSeek-V4-Pro)
    - OPENAI_API_KEY -> OPENAI_API_KEY (gpt-5 family, direct)
    Falls back to GEMINI_API_KEY for WMH_ENDPOINT_API_KEY only if Foundry's key is absent.
    """
    labs = REPO.parent
    if not os.environ.get("WMH_ENDPOINT_API_KEY"):
        value = _read_env_value(labs / "world-model-harness" / ".env.local", "AZURE_FOUNDRY_API_KEY")
        if value is None:
            value = _read_env_value(labs / "platform" / ".env.local", "GEMINI_API_KEY")
        if value is None:
            raise RuntimeError("no AZURE_FOUNDRY_API_KEY or GEMINI_API_KEY found")
        os.environ["WMH_ENDPOINT_API_KEY"] = value
    if not os.environ.get("OPENAI_API_KEY"):
        value = _read_env_value(labs / "world-models" / ".env.local", "OPENAI_API_KEY")
        if value is not None:
            os.environ["OPENAI_API_KEY"] = value


FOUNDRY_ENDPOINT = "https://silen-resource.services.ai.azure.com/openai/v1/"


def foundry(model: str) -> Provider:
    """A model on the user's Azure AI Foundry resource (OpenAI-compatible)."""
    _load_gemini_key()
    return RetryProvider(
        get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=model, endpoint=FOUNDRY_ENDPOINT))
    )


OPUS_JUDGE_REGION = "us-east-2"  # Opus 4.8 is enabled here under AWS_PROFILE=claas-bedrock


def opus_judge() -> Provider:
    """Claude Opus 4.8 on Bedrock (claas-bedrock profile, us-east-2) — the checklist judge."""
    os.environ.setdefault("AWS_PROFILE", "claas-bedrock")
    return bedrock("us.anthropic.claude-opus-4-8", region=OPUS_JUDGE_REGION)


def openai_direct(model: str) -> Provider:
    """A model on the OpenAI API directly (reads OPENAI_API_KEY)."""
    _load_gemini_key()
    return RetryProvider(get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=model)))


def gemini(model: str) -> Provider:
    return RetryProvider(
        get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=model, endpoint=GEMINI_ENDPOINT))
    )


class TeacherTurn:
    """One teacher step: reasoning, the action taken, and the env's reply."""

    def __init__(self, think: str, tool: str | None, arguments: dict, observation: str) -> None:
        self.think = think
        self.tool = tool
        self.arguments = arguments
        self.observation = observation


def run_teacher_episode(
    world_model: WorldModel,
    teacher: Provider,
    scenario: EvalScenario,
    reference: str,
    *,
    temperature: float,
) -> tuple[list[TeacherTurn], str, str]:
    """Roll one privileged-teacher episode. Returns (turns, final_summary, stop_reason)."""
    env = WorldModelEnv(world_model)
    env.reset(task=scenario.task, seed_state=scenario.seed_state)
    turns: list[TeacherTurn] = []
    summary = ""
    stop = "max_steps"
    try:
        for _ in range(MAX_STEPS):
            prompt = _render_teacher_turn(scenario, reference, turns)
            reply = teacher.complete(
                TEACHER_SYSTEM,
                [Message(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=4096,
            )
            raw = extract_json_object(reply.text)
            data = json.loads(raw) if raw else {}
            think = str(data.get("think", "")).strip()
            if data.get("done") or not data.get("tool"):
                summary = str(data.get("summary", "")).strip()
                stop = "done"
                break
            from wmh.core.types import Action  # local to keep module imports tidy

            action = Action(
                kind=ActionKind.TOOL_CALL,
                name=str(data["tool"]),
                arguments=data.get("arguments") or {},
            )
            observation = env.step(action)
            turns.append(
                TeacherTurn(think, action.name, action.arguments, observation.content)
            )
    finally:
        env.close()
    return turns, summary, stop


def _render_teacher_turn(
    scenario: EvalScenario, reference: str, turns: list[TeacherTurn]
) -> str:
    lines = [f"TASK: {scenario.task}"]
    if scenario.seed_state.scratchpad:
        lines.append(f"ENVIRONMENT NOTES: {scenario.seed_state.scratchpad}")
    lines.append(f"\nREFERENCE WALKTHROUGH (prior similar episode):\n{reference}\n")
    if turns:
        lines.append("YOUR EPISODE SO FAR:")
        for i, turn in enumerate(turns):
            lines.append(
                f"{i}. {turn.tool}({json.dumps(turn.arguments, default=str)}) "
                f"-> {turn.observation[:600]}"
            )
    lines.append("Your next move (JSON only):")
    return "\n".join(lines)


def to_qwen_messages(
    scenario: EvalScenario, turns: list[TeacherTurn], summary: str, tool_hint: str
) -> list[dict[str, str]]:
    """Render an episode as Qwen-convention chat messages (the student's native format)."""
    system = (
        "You are a customer-service agent operating in a tool environment.\n"
        f"Available tools (from prior episodes in this domain): {tool_hint}\n"
        "Think before every action inside <think></think>, then either emit exactly one\n"
        '<tool_call>{"name": ..., "arguments": {...}}</tool_call>\n'
        "or reply to the user in plain text when the task is complete. Always check policy "
        "constraints before irreversible actions."
    )
    user = scenario.task
    if scenario.seed_state.scratchpad:
        user += f"\n\nEnvironment notes: {scenario.seed_state.scratchpad}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for turn in turns:
        call = json.dumps({"name": turn.tool, "arguments": turn.arguments}, default=str)
        messages.append(
            {
                "role": "assistant",
                "content": f"<think>{turn.think}</think>\n<tool_call>{call}</tool_call>",
            }
        )
        messages.append({"role": "user", "content": f"<tool_response>{turn.observation}</tool_response>"})
    messages.append({"role": "assistant", "content": summary or "Task complete."})
    return messages


def domain_tool_hint(traces: list[Trace], provenance: str) -> str:
    """Comma-separated tool names seen in the source trace's domain (best-effort schema)."""
    source = next((t for t in traces if t.trace_id == provenance), None)
    domain = source.metadata.get("domain") if source else None
    names: set[str] = set()
    for trace in traces:
        if domain is not None and trace.metadata.get("domain") != domain:
            continue
        for step in trace.steps:
            if step.action.kind is ActionKind.TOOL_CALL and step.action.name:
                names.add(step.action.name)
    return ", ".join(sorted(names)[:40])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only first N scenarios (smoke)")
    parser.add_argument(
        "--teacher",
        default="nova-pro",
        choices=["nova-pro", "gemini-pro"],
        help="Teacher backend: Nova Pro (Bedrock, no daily quota) or Gemini Pro (1000 req/day).",
    )
    parser.add_argument("--resume", action="store_true", help="skip scenarios already in output")
    args = parser.parse_args()

    _load_gemini_key()
    DISTILL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pool = ScenarioSet.load(DISTILL / "train_pool.json")
    scenarios = pool.scenarios[: args.limit] if args.limit else pool.scenarios
    existing_records: list[dict] = []
    out_path = DISTILL / "teacher_episodes.jsonl"
    if args.resume and out_path.exists():
        existing_records = [
            json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines() if line
        ]
        done_ids = {r["scenario_id"] for r in existing_records}
        scenarios = [s for s in scenarios if s.scenario_id not in done_ids]
        print(f"resume: {len(existing_records)} episodes kept already; {len(scenarios)} scenarios to go")
    traces = get_adapter("otel-genai").from_file(str(TRACES))
    traces_by_id = {t.trace_id: t for t in traces}

    teacher = (
        bedrock("us.amazon.nova-pro-v1:0") if args.teacher == "nova-pro" else gemini(TEACHER_MODEL)
    )
    judge = ChecklistJudge(gemini(JUDGE_MODEL))
    world_model = WorldModel.load(str(WM_DIR), bedrock(NOVA_LITE), telemetry_root=str(REPO / ".wmh"))

    kept_records: list[dict] = []
    stats: dict[str, dict[str, float]] = defaultdict(lambda: {"attempts": 0, "kept": 0})

    def collect_for(scenario: EvalScenario) -> list[dict]:
        source = traces_by_id.get(scenario.provenance[0])
        reference = trace_digest(source) if source else "(no reference available)"
        tool_hint = domain_tool_hint(traces, scenario.provenance[0])
        candidates = []
        for sample_index in range(args.samples):
            temperature = 0.3 if sample_index == 0 else 0.8
            try:
                turns, summary, stop = run_teacher_episode(
                    world_model, teacher, scenario, reference, temperature=temperature
                )
            except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the batch
                print(f"  episode error {scenario.scenario_id}: {exc}", flush=True)
                continue
            stats[scenario.scenario_id]["attempts"] += 1
            if not turns:
                continue
            from wmh.core.types import Action, Observation, Step  # judge needs Step objects

            steps = [
                Step(
                    action=Action(kind=ActionKind.TOOL_CALL, name=t.tool, arguments=t.arguments),
                    observation=Observation(content=t.observation),
                )
                for t in turns
            ]
            # The final user-facing message is where "inform/refuse/communicate" checklist
            # items are satisfied — the judge MUST see it, not just the tool calls.
            if summary:
                steps.append(
                    Step(
                        action=Action(kind=ActionKind.MESSAGE, content=summary),
                        observation=Observation(content=""),
                    )
                )
            verdict = judge.score(scenario.task, scenario.checklist, steps)
            if verdict.success:
                candidates.append(
                    {
                        "scenario_id": scenario.scenario_id,
                        "source_trace": scenario.provenance[0],
                        "cluster": scenario.cluster_name,
                        "pass_rate": verdict.pass_rate,
                        "n_turns": len(turns),
                        "stop": stop,
                        "temperature": temperature,
                        "messages": to_qwen_messages(scenario, turns, summary, tool_hint),
                    }
                )
        # Keep the best few: highest checklist pass rate, then fewest turns (efficiency).
        candidates.sort(key=lambda r: (-r["pass_rate"], r["n_turns"]))
        kept = candidates[:KEEP_PER_SCENARIO]
        stats[scenario.scenario_id]["kept"] = len(kept)
        print(
            f"  {scenario.scenario_id}: kept {len(kept)}/{args.samples} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        return kept

    with world_model.frozen(), ThreadPoolExecutor(max_workers=WORKERS) as pool_executor:
        for records in pool_executor.map(collect_for, scenarios):
            kept_records.extend(records)

    out = DISTILL / "teacher_episodes.jsonl"
    kept_records = existing_records + kept_records
    with out.open("w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    report = {
        "scenarios": len(scenarios),
        "episodes_kept": len(kept_records),
        "keep_rate_by_scenario": {k: v for k, v in sorted(stats.items())},
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    (DISTILL / "teacher_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(
        f"kept {len(kept_records)} episodes from {len(scenarios)} scenarios "
        f"in {time.time() - t0:.0f}s -> {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()

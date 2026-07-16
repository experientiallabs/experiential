"""The delta proposer: the meta-agent reads a harness document and emits one `HarnessDelta`.

The proposer is shown the parent's surfaces — each with its id, kind, content hash, and content —
plus the failure evidence for one clustered mechanism, and replies with the delta's ops,
preconditions, and expected effect as JSON. The trigger, lineage, and identity fields are filled by
the caller from ground truth (the cluster and the parent's hashes), never trusted from the model.

Preconditions are copy-not-guess: the prompt prints every surface's current hash, and the proposer
must echo the hash of each surface it replaces or removes. `apply_delta` then rejects atomically on
any mismatch, so a proposal is only ever applied to exactly the document it was drafted against.

An unusable reply (no JSON, wrong shape) returns None — the search counts it as a skipped
iteration; a flaky meta-model costs budget, not the run.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.text import normalize_durable_text
from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence
from wmh.evals.gold import GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness.delta import FailureSignature, HarnessDelta, SurfaceOp, compute_delta_id
from wmh.harness.doc import HarnessDoc
from wmh.harness.tools import SUBMIT, TOOL_REGISTRY
from wmh.providers.base import Message, Provider

_TRACE_CHARS_PER_ATTEMPT = 12_000

MUTATE_SYSTEM = f"""You are a meta-agent improving an agent harness — a document of named \
SURFACES that configure an agent. You do NOT solve the agent's tasks yourself. You are shown the \
harness's current surfaces (each with its id, kind, and content hash) and evidence of ONE failure \
mechanism its agent exhibits. Propose exactly ONE focused change, grounded in that evidence. Do \
not shotgun many unrelated changes at once.

Choose the lever that can actually EXPRESS the fix:
- Structural/behavioral mechanisms (the agent claims without acting, loses context, wastes turns,
  never verifies, mishandles errors) -> edit the agent's CODE. For an in-process harness that is
  the singleton `code:runtime` program; for a real multi-file harness (e.g. the vendored pi
  agent) it is the pathful `code:<...>` surface that owns the behavior — the agent's actual
  source, shown to you above with its path. Edit the file where the mechanism lives (the turn
  loop, context compaction, tool dispatch, answer checking). This is the strongest lever:
  loops, verification passes, retries, observation truncation, and compaction are all code.
- A missing technique on specific tasks -> ADD a `skill:<slug>` teaching it.
- The agent misusing, missing, or not needing a tool -> edit `tool_policy:main`.
- Erratic sampling or turn caps -> adjust a `param:*`.
- Replace a `prompt:*` surface only for wording-level problems; prompt rewrites are the weakest
  lever and most often regress tasks that currently pass.

The `code:runtime` contract — a Python module defining `run(kit) -> str` (the final answer):
- `kit.instruction` (the task), `kit.system_prompt` (the assembled prompt), `kit.task_id`.
- `kit.complete(system, messages, temperature=..., max_tokens=...) -> str` — one LLM call;
  messages are `("user"|"assistant", content)` tuples. BUDGETED (default 40 calls/episode).
- `kit.execute(tool, arguments) -> Observation` (`.content`, `.is_error`) — one environment
  action, validated against the tool policy. BUDGETED (default 40/episode). Every call is
  recorded to the transcript the judge scores; work not done through `kit.execute` does not
  exist as far as grading is concerned.
- `kit.parse_tool_call(text) -> ToolCall | None` (`.tool`, `.arguments`), `kit.tools_text()`,
  `kit.skills_index()`, `kit.read_skill(name)`.
- Exceptions and exhausted budgets fail the episode (partial transcript kept). The module must
  be self-contained (stdlib imports only) and deterministic apart from LLM calls.

Surface kinds:
- prompt — a section of the agent's system prompt (all prompt surfaces are joined in id order).
- tool_policy — the tool list, one tool name per line. Valid names: \
{", ".join(sorted(TOOL_REGISTRY))}. The `{SUBMIT.name}` tool is REQUIRED (without it a run \
cannot end).
- param — a scalar loop knob: `param:max-turns` (int >= 1), `param:max-output-tokens` for a
  `pi-node` harness (int >= 1), or `param:temperature` (float in [0, 2]).
- code — the agent's source. Either the singleton in-process `code:runtime` (contract above; must
  compile and define `run`), OR the pathful `code:<...>` files of a real multi-file harness (the
  vendored pi agent's own source). Editing a pathful `code:` surface REPLACES that file's whole
  content; keep it a valid module and change only what the fix needs — you are editing the real
  harness that runs, not a description of it.
- skill — one reusable technique, shaped as:
  ---
  name: <kebab-slug matching the surface id>
  description: <one-line trigger description>
  ---
  <skill body markdown>

Reply with ONLY a JSON object, no prose:
{{"expected_effect": "<falsifiable prediction: what should change if this works>",
 "preconditions": {{"<surface_id>": "<that surface's content hash, copied from above>"}},
 "ops": [{{"op": "add" | "replace" | "remove",
          "surface_id": "<kind>:<kebab-slug>",
          "kind": "<required on add>",
          "content": "<the FULL new content (omit on remove)>",
          "rationale": "<why this op should help>"}}]}}

Rules:
- Every surface you replace or remove MUST appear in `preconditions` with its hash copied verbatim.
- `content` is the complete new surface content, not a diff.
- An `add` uses a fresh surface id of the right kind (e.g. a new `skill:<slug>` or a new
  `prompt:<slug>` section)."""


class _RawDelta(BaseModel):
    """What the meta-agent is trusted to provide — everything else is filled from ground truth."""

    expected_effect: str
    preconditions: dict[str, str] = Field(default_factory=dict)
    ops: list[SurfaceOp] = Field(min_length=1)


def propose_delta(
    parent: HarnessDoc,
    trigger: FailureSignature,
    evidence: str,
    provider: Provider,
    *,
    history: list[HarnessDelta] | None = None,
) -> HarnessDelta | None:
    """Ask the meta-agent for one delta against `parent`, or None if the reply is unusable.

    `history` is the run's previously judged deltas (most recent last): their ops and gate
    verdicts are shown to the proposer so it iterates instead of re-proposing rejected ideas.
    """
    user = _build_prompt(parent, trigger, evidence, history or [])
    # The reply must hold a COMPLETE replacement surface (ops carry full content, not diffs).
    # The largest vendored pi source file is ~36 KB (~10k tokens before JSON escaping), so 4k
    # silently truncated every real code-surface proposal into an unusable reply; the search
    # "ran" its iterations but skipped them all. 16k fits any single-surface rewrite with room
    # for preconditions/rationale, and stays under common provider output caps.
    completion = provider.complete(
        MUTATE_SYSTEM,
        [Message(role="user", content=user)],
        temperature=0.9,
        max_tokens=16384,
    )
    return parse_delta(parent, trigger, completion.text)


def parse_delta(
    parent: HarnessDoc,
    trigger: FailureSignature,
    text: str,
) -> HarnessDelta | None:
    """Parse one full-content or compact-edit proposal against ``parent``."""
    raw = extract_json_object(text)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None
        value["ops"] = _expand_compact_ops(parent, value.get("ops"))
        proposed = _RawDelta.model_validate(value)
        return HarnessDelta(
            delta_id=compute_delta_id(parent.doc_hash, proposed.ops),
            parent_doc_hash=parent.doc_hash,
            trigger=trigger,
            preconditions=proposed.preconditions,
            ops=proposed.ops,
            expected_effect=proposed.expected_effect,
        )
    except (TypeError, ValueError, ValidationError):
        return None


def _expand_compact_ops(parent: HarnessDoc, value: object) -> list[dict[str, object]]:
    """Expand exact replacement hunks into ordinary full-content surface ops."""
    if not isinstance(value, list):
        raise ValueError("ops must be an array")
    expanded: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each op must be an object")
        op = dict(item)
        edits = op.pop("edits", None)
        if op.get("op") == "replace" and "content" not in op and edits is not None:
            surface_id = op.get("surface_id")
            if not isinstance(surface_id, str) or (surface := parent.surface(surface_id)) is None:
                raise ValueError("compact replace targets an unknown surface")
            op["content"] = _apply_edits(surface.content, edits)
        elif edits is not None:
            raise ValueError("edits are only valid on content-less replace ops")
        expanded.append(op)
    return expanded


def _apply_edits(content: str, value: object) -> str:
    """Apply ordered exact edits, rejecting missing or ambiguous anchors."""
    if not isinstance(value, list) or not value:
        raise ValueError("edits must be a non-empty array")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each edit must be an object")
        old = item.get("old")
        new = item.get("new")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("edits need non-empty old and string new values")
        if content.count(old) != 1:
            raise ValueError("edit old text must occur exactly once")
        content = content.replace(old, new, 1)
    return content


def render_evidence(
    trigger: FailureSignature, report: ClosedLoopReport, tasks: list[TaskSpec]
) -> str:
    """Render one failure cluster (instructions + unmet assertions) as reflection fuel.

    A trigger with no failing tasks (the all-pass case) gets an explicit "nothing failed" prompt
    asking for a generalization/efficiency improvement — not a fake failure section that would
    send the meta-agent chasing nonexistent problems.
    """
    if not trigger.task_ids:
        return (
            "The harness passed every task on every pass. There are no failures to fix. "
            "Propose a change that should GENERALIZE or ECONOMIZE: a tighter, more transferable "
            "prompt surface; a lower param:max-turns if runs finish early; or a reusable skill "
            "distilled from what worked."
        )
    by_id = {task.task_id: task for task in tasks}
    selected = set(trigger.task_ids)
    scorecard = [
        "## Evaluation scorecard",
        "The selected failure is marked TARGET; preserve behavior on the other tasks.",
    ]
    for task in tasks:
        outcome = report.per_task.get(task.task_id)
        success = outcome.success_rate if outcome is not None else 0.0
        fraction = outcome.mean_fraction if outcome is not None else 0.0
        instruction = " ".join(normalize_durable_text(task.instruction).split())
        if len(instruction) > 240:
            instruction = f"{instruction[:237]}..."
        marker = "TARGET" if task.task_id in selected else "other"
        scorecard.append(
            f"- [{marker}] {task.task_id}: success={success:.2f}, "
            f"assertion_fraction={fraction:.2f} — {instruction}"
        )
    sections = [
        "\n".join(scorecard),
        f"## Selected failure\n\nFailure mechanism: {trigger.mechanism}",
    ]
    for task_id in trigger.task_ids:
        task = by_id.get(task_id)
        outcome = report.per_task.get(task_id)
        instruction = (
            normalize_durable_text(task.instruction) if task is not None else "(unknown task)"
        )
        rate = f"{outcome.success_rate:.2f} over {outcome.passes} passes" if outcome else "?"
        fraction = f", assertion_fraction={outcome.mean_fraction:.2f}" if outcome else ""
        task_section = [
            f"### Task {task_id} (success_rate={rate}{fraction})",
            f"Instruction: {instruction}",
        ]
        if outcome is not None:
            for index, verdict in enumerate(outcome.verdicts, 1):
                attempt = outcome.attempts[index - 1] if index <= len(outcome.attempts) else None
                task_section.append(_render_attempt(index=index, attempt=attempt, verdict=verdict))
        sections.append("\n\n".join(task_section))
    unmet = "\n".join(f"- {a}" for a in trigger.unmet_assertions)
    sections.append(
        "Original trigger assertions from the parent (current attempt verdicts above are "
        f"authoritative):\n{unmet or '- (none recorded)'}"
    )
    return "\n\n".join(sections)


def _render_attempt(*, index: int, attempt: RolloutEvidence | None, verdict: GoldVerdict) -> str:
    """Render one rollout plus its judge feedback without unbounded prompt growth."""
    # Keep this helper tolerant of old/deserialized reports that predate RolloutEvidence. The
    # verdict fields remain useful even when no trace was retained.
    lines = [
        f"#### Attempt {index} (passed={str(verdict.passed).lower()}, "
        f"assertion_fraction={verdict.fraction:.2f})"
    ]
    if attempt is not None:
        lines.append(
            f"Stop: {attempt.stop_reason.value}; turns={attempt.turns}\n"
            f"Final answer:\n{normalize_durable_text(attempt.answer) or '(none)'}\n\n"
            f"Execution transcript:\n"
            f"{_bounded_trace(normalize_durable_text(attempt.transcript))}"
        )
    if verdict.assertions:
        judged = "\n".join(
            f"- {'PASS' if assertion.passed else 'FAIL'}: "
            f"{normalize_durable_text(assertion.assertion)}"
            f" — {normalize_durable_text(assertion.why) if assertion.why else '(no judge reason)'}"
            for assertion in verdict.assertions
        )
        lines.append(f"Judge feedback:\n{judged}")
    else:
        rationale = normalize_durable_text(verdict.rationale) if verdict.rationale else ""
        lines.append(f"Judge feedback: {rationale or '(no per-assertion feedback)'}")
    return "\n\n".join(lines)


def _bounded_trace(trace: str, limit: int = _TRACE_CHARS_PER_ATTEMPT) -> str:
    """Keep both the setup and terminal behavior when a long run exceeds the evidence budget."""
    if not trace:
        return "(empty)"
    if len(trace) <= limit:
        return trace
    head = limit // 2
    tail = limit - head
    omitted = len(trace) - limit
    return f"{trace[:head]}\n... ({omitted} trace characters omitted) ...\n{trace[-tail:]}"


def render_history(history: list[HarnessDelta], limit: int = 5) -> str:
    """The last `limit` judged deltas as lessons: what was tried, and exactly how it fared."""
    lines: list[str] = []
    for delta in history[-limit:]:
        ops = ", ".join(f"{op.op} {op.surface_id}" for op in delta.ops)
        verdict = delta.verdict.reason if delta.verdict is not None else "(never evaluated)"
        rationale = delta.ops[0].rationale if delta.ops else ""
        lines.append(f"- [{ops}] rationale: {rationale[:120]}\n  outcome: {verdict}")
    return "\n".join(lines)


def _build_prompt(
    parent: HarnessDoc, trigger: FailureSignature, evidence: str, history: list[HarnessDelta]
) -> str:
    """The meta-agent's user prompt: every parent surface with its identity, then the evidence."""
    blocks: list[str] = []
    for surface in parent.surfaces:
        budget = f", budget={surface.budget}" if surface.budget is not None else ""
        blocks.append(
            f"### {surface.id} (kind={surface.kind.value}, "
            f"hash={surface.content_hash}{budget})\n```\n{surface.content}\n```"
        )
    surfaces_block = "\n\n".join(blocks)
    history_block = (
        f"## Previous attempts this run (do NOT repeat failed ideas — iterate or change lever)"
        f"\n\n{render_history(history)}\n\n"
        if history
        else ""
    )
    return (
        f"## Current harness surfaces ({parent.name}, doc_hash={parent.doc_hash})\n\n"
        f"{surfaces_block}\n\n"
        f"## Failure evidence\n\n{evidence}\n\n"
        f"{history_block}"
        "Propose ONE focused change as the JSON object described in your instructions. "
        "Remember: copy the hash of every surface you replace or remove into `preconditions`."
    )

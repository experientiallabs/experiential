"""The checklist judge: grade one episode against a scenario's success criteria."""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import ActionKind, Step
from wmh.providers.base import Message, Provider

CHECKLIST_SYSTEM = """You grade one AI-agent episode against a checklist of success criteria.
You see the task, the checklist, and a digest of the episode (tool calls, observations, messages).

Respond with ONLY a JSON object, no prose around it:
{"passed": [<true/false per checklist item, in order>],
 "success": <true if the episode as a whole achieved the task>,
 "critique": "<one or two sentences: what was achieved, what was missed>"}

Judge OUTCOMES, not mechanics: a different-but-valid strategy that achieves a criterion passes it.
An episode that never addresses a criterion fails it."""

_MAX_JUDGE_STEP_CHARS = 400


class ChecklistResult(BaseModel):
    """The judge's verdict for one episode against one scenario checklist."""

    passed: list[bool] = Field(default_factory=list)
    success: bool = False
    critique: str = ""

    @property
    def pass_rate(self) -> float:
        return sum(self.passed) / len(self.passed) if self.passed else 0.0


class _RawChecklist(BaseModel):
    passed: list[bool] = Field(default_factory=list)
    success: bool = False
    critique: str = ""


class ChecklistJudge:
    """LLM judge grading a trajectory against a scenario's checklist."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, task: str, checklist: list[str], steps: list[Step]) -> ChecklistResult:
        prompt = _render_judge_prompt(task, checklist, steps)
        # Generous budget: reasoning judges spend completion tokens on thinking before the JSON;
        # a tight cap truncates the verdict mid-string and silently scores as failure.
        completion = self._provider.complete(
            CHECKLIST_SYSTEM,
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=8192,
        )
        raw = extract_json_object(completion.text)
        if raw is not None:
            try:
                parsed = _RawChecklist.model_validate_json(raw)
            except ValidationError:
                parsed = None
            if parsed is not None:
                # Normalize the verdict list to the checklist length: a judge that returned too
                # few items fails the unaddressed criteria, never silently passes them.
                passed = list(parsed.passed[: len(checklist)])
                passed += [False] * (len(checklist) - len(passed))
                return ChecklistResult(
                    passed=passed, success=parsed.success, critique=parsed.critique.strip()
                )
        return ChecklistResult(
            passed=[False] * len(checklist),
            success=False,
            critique=(
                f"Unparseable judge response; treated as failure. Raw: {completion.text[:200]}"
            ),
        )


def _render_judge_prompt(task: str, checklist: list[str], steps: list[Step]) -> str:
    lines = [f"TASK: {task}", "CHECKLIST:"]
    lines.extend(f"{index + 1}. {item}" for index, item in enumerate(checklist))
    lines.append("EPISODE:")
    if not steps:
        lines.append("(the agent took no actions)")
    for index, step in enumerate(steps):
        action = step.action
        if action.kind is ActionKind.TOOL_CALL:
            call = f"CALL {action.name}({action.arguments})"
        else:
            call = f"MSG {action.content}"
        observation = step.observation.content[:_MAX_JUDGE_STEP_CHARS]
        error_mark = " [ERROR]" if step.observation.is_error else ""
        lines.append(f"{index}. {call} -> {observation}{error_mark}")
    return "\n".join(lines)

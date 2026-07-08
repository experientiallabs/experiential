"""`CodeRuntime`: the agent loop itself as a searchable harness surface.

Two live search campaigns showed the same wall: the meta-agent diagnoses failure mechanisms
correctly, but prompt- and skill-level edits cannot express the fixes that actually make a
harness good — loop structure, retries, context compaction, observation truncation, token
budgets. Those are *code*. A `code:runtime` surface holds a Python module defining
`run(kit) -> str`; the search edits that program.

The contract is the `RuntimeKit`, and it carries three guarantees the search relies on:

- **Budgeted.** `kit.complete` and `kit.execute` enforce hard caps on LLM calls and environment
  actions. A runaway loop raises `BudgetExceeded` instead of wedging an eval; cost is bounded per
  episode by construction, not by hope.
- **Kit-recorded.** Every environment action is appended to the transcript by the kit itself, so
  the judge always scores ground truth: generated code cannot fake, omit, or reorder what it did.
- **Crash-isolated.** An exception inside `run` fails that episode (scored as a failure with the
  partial transcript), never the evaluation loop around it.

The kit is an interface contract, not a security boundary: harness code runs in-process and is
trusted to the same degree as the rest of the search (it is reviewed via the delta archive, and
only ever exercised against the world model during search). Running searched code against real
environments is a deployment decision that belongs behind a sandbox.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, Field

from wmh.core.types import Action, ActionKind, EnvState, JsonObject, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import RunResult, StopReason
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import ToolCall, ToolSpec, parse_tool_call, render_tools
from wmh.providers.base import Message, Provider

CODE_ENTRYPOINT = "run"

# Defaults sized so a reasonable loop never notices them: the fixed baseline loop uses at most
# `max_turns` (20) of each.
DEFAULT_MAX_LLM_CALLS = 40
DEFAULT_MAX_ENV_ACTIONS = 40


class RunBudget(BaseModel):
    """Hard per-episode caps enforced by the kit."""

    max_llm_calls: int = Field(default=DEFAULT_MAX_LLM_CALLS, ge=1)
    max_env_actions: int = Field(default=DEFAULT_MAX_ENV_ACTIONS, ge=1)


class BudgetExceeded(RuntimeError):
    """Raised by the kit when harness code exhausts an episode budget."""


class RuntimeKit:
    """Everything harness code may touch, and the recorder of what it actually did."""

    def __init__(
        self,
        *,
        task_id: str,
        instruction: str,
        environment: AgentEnvironment,
        provider: Provider,
        tools: list[ToolSpec],
        skills: SkillLibrary,
        temperature: float,
        budget: RunBudget,
        system_prompt: str = "",
    ) -> None:
        self.task_id = task_id
        self.instruction = instruction
        self.temperature = temperature
        self.tools = tools
        # The doc's assembled prompt (prompt surfaces + tools + skills index): prompt surfaces
        # stay meaningful alongside a code surface, and code may use or ignore this.
        self.system_prompt = system_prompt
        self._environment = environment
        self._provider = provider
        self._skills = skills
        self._budget = budget
        self._llm_calls = 0
        self._env_actions = 0
        self.steps: list[Step] = []

    # -- the two budgeted primitives -----------------------------------------------------------

    def complete(
        self,
        system: str,
        messages: list[Message] | list[tuple[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> str:
        """One LLM call. Accepts `Message`s or plain `(role, content)` tuples."""
        if self._llm_calls >= self._budget.max_llm_calls:
            raise BudgetExceeded(f"llm call budget exhausted ({self._budget.max_llm_calls})")
        self._llm_calls += 1
        normalized = [_to_message(m) for m in messages]
        completion = self._provider.complete(
            system,
            normalized,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        return completion.text

    def execute(self, tool: str, arguments: JsonObject) -> Observation:
        """One environment action, validated against the tool policy and recorded verbatim."""
        if self._env_actions >= self._budget.max_env_actions:
            raise BudgetExceeded(f"env action budget exhausted ({self._budget.max_env_actions})")
        action = Action(kind=ActionKind.TOOL_CALL, name=tool, arguments=arguments)
        if tool not in {t.name for t in self.tools} or not is_env_action(action):
            observation = Observation(content=f"tool {tool!r} not available", is_error=True)
        else:
            self._env_actions += 1
            observation = self._environment.execute(action)
        self.steps.append(
            Step(
                action=action,
                observation=observation,
                state_before=EnvState(),
                task=self.instruction,
            )
        )
        return observation

    # -- conveniences (unbudgeted, side-effect free) --------------------------------------------

    def parse_tool_call(self, text: str) -> ToolCall | None:
        return parse_tool_call(text)

    def tools_text(self) -> str:
        return render_tools(self.tools)

    def skills_index(self) -> str:
        return self._skills.render_index()

    def read_skill(self, name: str) -> str | None:
        skill = self._skills.get(name)
        return skill.body if skill is not None else None


def _to_message(m: Message | tuple[str, str]) -> Message:
    if isinstance(m, Message):
        return m
    role, content = m
    if role == "user":
        return Message(role="user", content=content)
    if role == "assistant":
        return Message(role="assistant", content=content)
    raise ValueError(f"message role must be 'user' or 'assistant', got {role!r}")


def compile_harness_code(code: str) -> None:
    """Front-loaded validation: the code must compile and define `run` at module scope.

    Raises `ValueError` so `HarnessDoc` construction rejects an unrunnable harness before any
    eval budget could be spent on it. Behavioral quality is the gate's job, not this one's.
    """
    try:
        compiled = compile(code, "<code:runtime>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"code:runtime does not compile: {exc}") from exc
    names = set(compiled.co_names)
    # A module that never binds `run` can't be dispatched. co_names covers references, so also
    # accept a top-level def by scanning consts for the code object.
    defines_run = (
        any(getattr(const, "co_name", None) == CODE_ENTRYPOINT for const in compiled.co_consts)
        or CODE_ENTRYPOINT in names
    )
    if not defines_run:
        raise ValueError(f"code:runtime must define `{CODE_ENTRYPOINT}(kit)` at module scope")


class CodeRuntime:
    """Drives episodes through a harness-defined `run(kit)` instead of the fixed loop."""

    def __init__(
        self,
        provider: Provider,
        *,
        code: str,
        tools: list[ToolSpec],
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
        budget: RunBudget | None = None,
        system_prompt: str = "",
    ) -> None:
        compile_harness_code(code)
        namespace: dict[str, object] = {"__name__": "wmh_harness_code"}
        exec(code, namespace)  # noqa: S102 - the code surface IS the artifact under search
        entry = namespace.get(CODE_ENTRYPOINT)
        if not callable(entry):
            raise ValueError(f"code:runtime `{CODE_ENTRYPOINT}` is not callable")
        self._run = cast("Callable[[RuntimeKit], object]", entry)
        self._provider = provider
        self._tools = tools
        self._temperature = temperature
        self._skills = skills if skills is not None else SkillLibrary()
        self._budget = budget if budget is not None else RunBudget()
        self._system_prompt = system_prompt

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        kit = RuntimeKit(
            task_id=task_id,
            instruction=instruction,
            environment=environment,
            provider=self._provider,
            tools=self._tools,
            skills=self._skills,
            temperature=self._temperature,
            budget=self._budget,
            system_prompt=self._system_prompt,
        )
        try:
            answer = self._run(kit)
        except BudgetExceeded as exc:
            return self._result(kit, StopReason.BUDGET, note=str(exc))
        except Exception as exc:  # noqa: BLE001 - crash-isolation is the contract
            return self._result(kit, StopReason.ERROR, note=f"{type(exc).__name__}: {exc}")
        return RunResult(
            task_id=kit.task_id,
            steps=kit.steps,
            stop_reason=StopReason.SUBMITTED,
            answer=answer if isinstance(answer, str) else "",
            turns=len(kit.steps),
        )

    def _result(self, kit: RuntimeKit, stop_reason: StopReason, *, note: str) -> RunResult:
        # The kit-recorded partial transcript survives, plus one error step so the judge (and the
        # failure clustering) can see WHY the episode ended.
        kit.steps.append(
            Step(
                action=Action(kind=ActionKind.MESSAGE, content="(harness runtime)"),
                observation=Observation(content=note, is_error=True),
                state_before=EnvState(),
                task=kit.instruction,
            )
        )
        return RunResult(
            task_id=kit.task_id,
            steps=kit.steps,
            stop_reason=stop_reason,
            answer="",
            turns=len(kit.steps),
        )


# The reference loop, as the seed content of a `code:runtime` surface: functionally the fixed
# `AgentRuntime` loop, expressed through the kit so the search can restructure it.
DEFAULT_RUNTIME_CODE = '''"""Baseline agent loop: one JSON tool call per turn.

Calling `submit` ends the episode."""


def run(kit):
    messages = [("user", "TASK: " + kit.instruction)]
    nudged = False
    for _ in range(20):
        reply = kit.complete(kit.system_prompt, messages).strip()
        call = kit.parse_tool_call(reply)
        if call is None:
            if nudged:
                return ""
            nudged = True
            messages.append(("assistant", reply))
            messages.append(("user", "[ERROR] reply with EXACTLY one JSON object: "
                             "{\\"tool\\": \\"<name>\\", \\"arguments\\": {...}}"))
            continue
        if call.tool == "submit":
            answer = call.arguments.get("answer")
            return answer if isinstance(answer, str) else ""
        if call.tool == "read_skill":
            name = call.arguments.get("name")
            body = kit.read_skill(name if isinstance(name, str) else "")
            text, is_error = (body, False) if body is not None else ("no such skill", True)
        else:
            observation = kit.execute(call.tool, call.arguments)
            text, is_error = observation.content, observation.is_error
        messages.append(("assistant", reply))
        messages.append(("user", ("[ERROR] " if is_error else "[OK] ") + text))
    return ""
'''

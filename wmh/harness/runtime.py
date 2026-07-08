"""`AgentRuntime`: the minimal agent loop that drives closed-loop rollouts.

A plain, owned while-loop: build the system prompt, ask the agent model for one action,
dispatch it to the environment, append the observation, repeat until `submit` or the turn cap. The
loop is deliberately fixed and small — closed-loop eval tests the *world model*, so the agent must
be a constant: any divergence is then attributable to the world model alone.

Every run yields a `RunResult` whose `steps` are `wmh.core.types.Step`s, so transcripts render with
the same types the rest of the harness uses.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.skills import SkillLibrary
from wmh.harness.tools import (
    DEFAULT_TOOLS,
    READ_SKILL,
    SUBMIT,
    ToolCall,
    parse_tool_call,
    render_tools,
    resolve_tools,
    to_action,
)
from wmh.providers.base import Message, Provider

DEFAULT_SYSTEM_PROMPT = """You are a capable command-line agent working inside a Linux environment.
You are given a task. Accomplish it by taking ONE action at a time.

Every reply MUST be a single JSON object and nothing else:
{"tool": "<tool name>", "arguments": {<the tool's arguments>}}

Work in small, verifiable steps: inspect state, act, check the result, then continue. When the
task is done, call `submit` with your answer. Prefer composing small bash commands over guessing."""

DEFAULT_MAX_TURNS = 20  # small shell tasks converge well before this; raise for longer horizons

# Per-observation cap in the judge-facing transcript. Generous rather than tight: gold evidence
# routinely lives deep in long outputs (`cat` of a produced file, `ls -R`), and truncating it away
# turns real successes into judged failures.
TRANSCRIPT_OBS_CHARS = 2000

_NUDGE = (
    "[ERROR] that reply was not a single valid JSON tool call. Reply with EXACTLY one JSON "
    'object: {"tool": "<tool name>", "arguments": {...}}'
)


class StopReason(StrEnum):
    SUBMITTED = "submitted"  # the agent called submit
    MAX_TURNS = "max_turns"  # hit the turn cap without submitting
    NO_ACTION = "no_action"  # the agent produced no parseable tool call


class RunResult(BaseModel):
    """The outcome of one rollout: the transcript, why it stopped, and any answer."""

    task_id: str
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason
    answer: str = ""
    turns: int = 0

    def transcript(self) -> str:
        """A compact judge-readable transcript of the run."""
        lines: list[str] = []
        for i, step in enumerate(self.steps, 1):
            act = step.action
            desc = act.name or (act.content or "")
            if act.kind == ActionKind.TOOL_CALL and act.arguments:
                desc = f"{act.name} {act.arguments}"
            lines.append(f"[{i}] {act.kind.value}: {desc}")
            lines.append(f"    -> {step.observation.content[:TRANSCRIPT_OBS_CHARS]}")
        return "\n".join(lines)


class AgentRuntime:
    """Drives the fixed agent loop against one `AgentEnvironment`.

    `provider` is the *agent* model — a separate role from the world model serving the simulated
    environment (they may be the same backend). The prompt/tools/limits are parameters so the
    harness layer can construct configured runtimes, but within a run they never change.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: list[str] | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        temperature: float = 0.7,
        skills: SkillLibrary | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._provider = provider
        self._system_prompt = system_prompt
        self._skills = skills if skills is not None else SkillLibrary()
        tool_names = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        # A skill-bearing harness needs read_skill for progressive disclosure; add it implicitly so
        # harness config files don't have to remember the plumbing tool.
        if len(self._skills) and READ_SKILL.name not in tool_names:
            tool_names.append(READ_SKILL.name)
        self._tools = resolve_tools(tool_names)
        self._max_turns = max_turns
        self._temperature = temperature

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        messages: list[Message] = [Message(role="user", content=f"TASK: {instruction}")]
        steps: list[Step] = []
        state = EnvState()
        nudged = False

        for turn in range(1, self._max_turns + 1):
            completion = self._provider.complete(
                self._full_system_prompt(), messages, temperature=self._temperature
            )
            reply = completion.text.strip()
            call = parse_tool_call(reply)
            if call is None:
                # One recovery nudge per run (symmetric with the unavailable-tool path, which also
                # feeds an error back): at nonzero temperature a single malformed reply is agent
                # noise, and aborting the rollout would charge it to the world model's score.
                if not nudged:
                    nudged = True
                    messages.append(Message(role="assistant", content=reply))
                    messages.append(Message(role="user", content=_NUDGE))
                    continue
                return self._result(task_id, steps, StopReason.NO_ACTION, turns=turn)

            if call.tool == SUBMIT.name:
                answer = _str_arg(call, "answer")
                steps.append(
                    _step(to_action(call), Observation(content=answer), state, instruction)
                )
                return self._result(task_id, steps, StopReason.SUBMITTED, answer=answer, turns=turn)

            action, observation = self._dispatch(call, environment)
            steps.append(_step(action, observation, state, instruction))
            state = _advance(state, observation)
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_observation_text(observation)))

        return self._result(task_id, steps, StopReason.MAX_TURNS, turns=self._max_turns)

    def _dispatch(
        self, call: ToolCall, environment: AgentEnvironment
    ) -> tuple[Action, Observation]:
        """Route one non-submit call: read_skill handled here, env tools to the environment."""
        action = to_action(call)
        if call.tool not in {t.name for t in self._tools}:
            return action, Observation(content=f"tool {call.tool!r} not available", is_error=True)
        if call.tool == READ_SKILL.name:
            name = _str_arg(call, "name")
            skill = self._skills.get(name)
            if skill is None:
                return action, Observation(content=f"no skill named {name!r}", is_error=True)
            return action, Observation(content=skill.body)
        if not is_env_action(action):
            return action, Observation(content=f"tool {call.tool!r} not available", is_error=True)
        return action, environment.execute(action)

    def _full_system_prompt(self) -> str:
        prompt = f"{self._system_prompt}\n\n## Tools\n{render_tools(self._tools)}"
        index = self._skills.render_index()
        if index:
            prompt += f"\n\n## Your skills (read a body with read_skill)\n{index}"
        return prompt

    def _result(
        self,
        task_id: str,
        steps: list[Step],
        stop_reason: StopReason,
        *,
        answer: str = "",
        turns: int,
    ) -> RunResult:
        return RunResult(
            task_id=task_id, steps=steps, stop_reason=stop_reason, answer=answer, turns=turns
        )


def _str_arg(call: ToolCall, key: str) -> str:
    value = call.arguments.get(key)
    return value if isinstance(value, str) else ""


def _step(action: Action, observation: Observation, state: EnvState, instruction: str) -> Step:
    return Step(action=action, observation=observation, state_before=state, task=instruction)


def _advance(state: EnvState, observation: Observation) -> EnvState:
    """Carry a one-line note forward into the next step's state (mirrors WorldModel scratchpad)."""
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note.strip():
        prefix = f"{state.scratchpad}\n" if state.scratchpad else ""
        return EnvState(structured=state.structured, scratchpad=f"{prefix}- {note.strip()}")
    return state


def _observation_text(observation: Observation) -> str:
    tag = "ERROR" if observation.is_error else "OK"
    return f"[{tag}] {observation.content}"

"""`AgentRuntime`: the pi-style agent loop that owns its own control flow (12-factor).

The loop is a plain, owned while-loop (Ralph/12-factor: "own your control flow"): build the prompt
from the `HarnessSpec` + skill index, ask the agent model for one action, dispatch it, append the
observation, repeat until `submit` or the turn cap. Harness tools (save_skill/read_skill/submit) are
handled here; env tools go to whichever `AgentEnvironment` is wired in. The runtime is *fixed* —
only the injected `HarnessSpec` changes between variants — so a score delta is attributable to the
spec, which is what makes the evolutionary search sound.

Every run yields a `RunResult` whose `steps` are `Step`s, so a real run drops into the trace
pipeline (`wmh.agent.capture`) and a simulated run drops into the gold judge.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from wmh.agent.environment import AgentEnvironment, is_env_action, message_observation
from wmh.agent.skills import SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tools import (
    READ_SKILL,
    SAVE_SKILL,
    SUBMIT,
    ToolCall,
    parse_tool_call,
    render_tools,
    resolve_tools,
    to_action,
)
from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.providers.base import Message, Provider


class StopReason(StrEnum):
    SUBMITTED = "submitted"  # the agent called submit
    MAX_TURNS = "max_turns"  # hit the turn cap without submitting
    NO_ACTION = "no_action"  # the agent produced no parseable tool call


class RunResult(BaseModel):
    """The outcome of one agent run: the transcript, why it stopped, and any answer."""

    task_id: str
    harness: str  # the HarnessSpec.name that produced this run
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason
    answer: str = ""
    turns: int = 0
    saved_skills: list[str] = Field(default_factory=list)  # skills saved during this run

    def transcript(self) -> str:
        """A compact human/judge-readable transcript of the run."""
        lines: list[str] = []
        for i, step in enumerate(self.steps, 1):
            act = step.action
            desc = act.name or (act.content or "")
            if act.kind == ActionKind.TOOL_CALL and act.arguments:
                desc = f"{act.name} {act.arguments}"
            lines.append(f"[{i}] {act.kind.value}: {desc}")
            lines.append(f"    -> {step.observation.content[:500]}")
        return "\n".join(lines)


class AgentRuntime:
    """Drives one `HarnessSpec` against one `AgentEnvironment`.

    `provider` is the *agent* model (distinct from the world model serving the simulated env).
    `library` is the persistent skill library; the spec's `seed_skills` are surfaced up front and
    any `save_skill` call writes through to it.
    """

    def __init__(
        self,
        spec: HarnessSpec,
        provider: Provider,
        library: SkillLibrary | None = None,
    ) -> None:
        self._spec = spec
        self._provider = provider
        # `library or SkillLibrary()` would be wrong: an empty SkillLibrary is falsy (it defines
        # __len__), so a caller's fresh library would be silently discarded. Check for None.
        self._library = library if library is not None else SkillLibrary()
        self._tools = resolve_tools(spec.tools)

    def run(self, task_id: str, instruction: str, environment: AgentEnvironment) -> RunResult:
        messages: list[Message] = [Message(role="user", content=f"TASK: {instruction}")]
        steps: list[Step] = []
        saved_skills: list[str] = []
        state = EnvState()

        for turn in range(1, self._spec.max_turns + 1):
            completion = self._provider.complete(
                self._system_prompt(),
                messages,
                temperature=self._spec.temperature,
            )
            reply = completion.text.strip()
            call = parse_tool_call(reply)
            if call is None:
                return RunResult(
                    task_id=task_id,
                    harness=self._spec.name,
                    steps=steps,
                    stop_reason=StopReason.NO_ACTION,
                    turns=turn,
                    saved_skills=saved_skills,
                )

            if call.tool == SUBMIT.name:
                answer = _str_arg(call, "answer")
                steps.append(
                    _step(to_action(call), Observation(content=answer), state, instruction)
                )
                return RunResult(
                    task_id=task_id,
                    harness=self._spec.name,
                    steps=steps,
                    stop_reason=StopReason.SUBMITTED,
                    answer=answer,
                    turns=turn,
                    saved_skills=saved_skills,
                )

            action, observation = self._dispatch(call, environment, saved_skills)
            step = _step(action, observation, state, instruction)
            steps.append(step)
            state = _advance(state, observation)
            messages.append(Message(role="assistant", content=reply))
            messages.append(Message(role="user", content=_observation_text(observation)))

        return RunResult(
            task_id=task_id,
            harness=self._spec.name,
            steps=steps,
            stop_reason=StopReason.MAX_TURNS,
            turns=self._spec.max_turns,
            saved_skills=saved_skills,
        )

    def _dispatch(
        self, call: ToolCall, environment: AgentEnvironment, saved_skills: list[str]
    ) -> tuple[Action, Observation]:
        """Route one non-submit call: harness tools handled here, env tools sent to the env."""
        if call.tool == SAVE_SKILL.name:
            skill = self._library.save(
                _str_arg(call, "name"), _str_arg(call, "description"), _str_arg(call, "body")
            )
            saved_skills.append(skill.name)
            return to_action(call), Observation(content=f"saved skill '{skill.name}'")
        if call.tool == READ_SKILL.name:
            skill = self._library.get(_str_arg(call, "name"))
            content = (
                skill.body if skill is not None else f"no skill named {_str_arg(call, 'name')!r}"
            )
            return to_action(call), Observation(content=content, is_error=skill is None)

        action = to_action(call)
        if call.tool not in {t.name for t in self._tools}:
            return action, Observation(content=f"tool {call.tool!r} not available", is_error=True)
        if not is_env_action(action):
            return action, message_observation(reply_summary(call))
        return action, environment.execute(action)

    def _system_prompt(self) -> str:
        """Assemble the variant's system prompt: its text + tool list + progressive skill index."""
        return (
            f"{self._spec.system_prompt}\n\n"
            f"## Tools\n{render_tools(self._tools)}\n\n"
            f"## Your skills (read a body with read_skill)\n{self._seed_index()}"
        )

    def _seed_index(self) -> str:
        """The skill index shown up front: the seed subset if the spec names one, else all."""
        library = (
            self._library.subset(self._spec.seed_skills)
            if self._spec.seed_skills
            else self._library
        )
        return library.render_index()


def reply_summary(call: ToolCall) -> str:
    """A short echo for a non-env tool call the environment can't run (defensive)."""
    return f"acknowledged {call.tool}"


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

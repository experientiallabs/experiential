"""The agent harness execution seam: run and score one agent episode.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment` - an interface, not a backend: closed-loop eval (`wmo.evals.closed_loop`)
binds it to the world model, and a real execution backend can bind the same loop to reality.
What the loop runs with is a `HarnessDoc` - a typed document of identity-keyed surfaces
(prompt sections, tool policy, loop params, skills) stored as immutable versions with movable
aliases (`wmo.harness.store`).

The machinery that IMPROVES harnesses (delta search, mutation, population search, live
sessions) lives in the agent-optimization repo and imports this seam; executing and scoring
episodes stays here because closed-loop eval and distillation depend on it.
"""

from wmo.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmo.harness.environment import AgentEnvironment, is_env_action
from wmo.harness.runtime import AgentRuntime, RunResult, StopReason
from wmo.harness.skills import Skill, SkillLibrary
from wmo.harness.store import HarnessStore
from wmo.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

__all__ = [
    "TOOL_REGISTRY",
    "AgentEnvironment",
    "AgentRuntime",
    "HarnessDoc",
    "HarnessStore",
    "RunResult",
    "Skill",
    "SkillLibrary",
    "StopReason",
    "Surface",
    "SurfaceKind",
    "SurfaceOp",
    "ToolCall",
    "is_env_action",
    "parse_tool_call",
]

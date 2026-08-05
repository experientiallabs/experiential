"""The harness optimization seam: represent, run, store, and score one agent harness.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment` - an interface, not a backend: closed-loop eval (`wmo.evals.closed_loop`)
binds it to the world model, and a real execution backend can bind the same loop to reality.
What the loop runs with is a `HarnessDoc` - a typed document of identity-keyed surfaces
(prompt sections, tool policy, loop params, skills) stored as immutable versions with movable
aliases (`wmo.optimize.harness.store`).

The machinery that improves harnesses (delta search, mutation, population search, live
sessions) lives in the agent-optimization repo and imports this seam. The shared artifact and
execution contracts stay under `wmo.optimize` because closed-loop eval and model optimization
also depend on them.
"""

from wmo.optimize.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmo.optimize.harness.environment import AgentEnvironment, is_env_action
from wmo.optimize.harness.runtime import AgentRuntime, RunResult, StopReason
from wmo.optimize.harness.skills import Skill, SkillLibrary
from wmo.optimize.harness.store import HarnessStore
from wmo.optimize.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

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
    "ToolCall",
    "is_env_action",
    "parse_tool_call",
]

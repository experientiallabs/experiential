"""Represent, run, store, and score one agent harness.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment`, an interface rather than a backend. Closed-loop evaluation in
`wmo.simulation.evaluation.closed_loop` binds it to the world model, and a real execution backend
can bind the same loop to reality.
What the loop runs with is a `HarnessDoc` - a typed document of identity-keyed surfaces
(prompt sections, tool policy, loop params, skills) stored as immutable versions with movable
aliases (`wmo.runtime.harness.store`).

The machinery that improves harnesses (delta search, mutation, population search) lives in the
agent-optimization repo and imports this runtime seam. The live-session host stays here because it
drives the retained local pi runner for `wmo run`. Closed-loop evaluation and model optimization
use the same execution contracts.
"""

from wmo.runtime.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmo.runtime.harness.environment import AgentEnvironment, is_env_action
from wmo.runtime.harness.runtime import AgentRuntime, RunResult, StopReason
from wmo.runtime.harness.skills import Skill, SkillLibrary
from wmo.runtime.harness.store import HarnessStore
from wmo.runtime.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

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

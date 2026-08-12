"""Represent, run, store, and score one agent harness.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment`, an interface rather than a backend. The retired closed-loop text evaluator
bound that seam to the former world model; a real backend can bind the same loop to reality.
What the loop runs with is a `HarnessDoc` - a typed document of identity-keyed surfaces
(prompt sections, tool policy, loop params, skills) stored as immutable versions with movable
aliases (`wmo.runtime.harness.store`).

The machinery that improves harnesses (delta search, mutation, population search) lives in the
agent-optimization repo and imports this runtime seam. Closed-loop evaluation and model
optimization use the same execution contracts. W14R owns any future CLI orchestration.
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

"""The agent harness: the scaffold a live agent runs with, and the machinery to improve it.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment` — an interface, not a backend: closed-loop eval (`wmh.evals.closed_loop`) binds
it to the world model, and a real execution backend can bind the same loop to reality. What the
loop runs with is a `HarnessDoc` — a typed document of identity-keyed surfaces (prompt sections,
tool policy, loop params, skills) stored as immutable versions with movable aliases
(`wmh.harness.store`) and updated through audited `HarnessDelta`s (`wmh.harness.delta`,
docs/harness_delta.md) proposed by a meta-agent and gated on non-regression (`wmh.harness.create`,
the `wmh harness create` search).

`create` and `mutate` are imported directly (not re-exported here): they depend on
`wmh.evals.closed_loop`, which itself binds to this package's runtime — re-exporting them would
make `import wmh.evals` observe a partially initialized module.
"""

from wmh.harness.delta import (
    FailureSignature,
    GateRecord,
    HarnessDelta,
    SurfaceOp,
    apply_delta,
)
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import AgentRuntime, RunResult, StopReason
from wmh.harness.skills import Skill, SkillLibrary
from wmh.harness.store import HarnessStore
from wmh.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

__all__ = [
    "TOOL_REGISTRY",
    "AgentEnvironment",
    "AgentRuntime",
    "FailureSignature",
    "GateRecord",
    "HarnessDelta",
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
    "apply_delta",
    "is_env_action",
    "parse_tool_call",
]

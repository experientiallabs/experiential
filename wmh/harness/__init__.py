"""The agent harness: the scaffold a live agent runs with.

A minimal, fixed agent loop (`AgentRuntime`) drives one action at a time against an
`AgentEnvironment` — an interface, not a backend: closed-loop eval (`wmh.evals.closed_loop`) binds
it to the world model, and a real execution backend can bind the same loop to reality. `tools`
defines the small tool registry the loop dispatches.

The loop is deliberately fixed and small: closed-loop eval tests the *world model*, so the agent
must be a constant — any divergence is then attributable to the environment alone.
"""

from wmh.harness.environment import AgentEnvironment, is_env_action
from wmh.harness.runtime import AgentRuntime, RunResult, StopReason
from wmh.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

__all__ = [
    "TOOL_REGISTRY",
    "AgentEnvironment",
    "AgentRuntime",
    "RunResult",
    "StopReason",
    "ToolCall",
    "is_env_action",
    "parse_tool_call",
]

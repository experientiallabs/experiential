"""Closed-loop evaluation: a live agent runs tasks against the world model as its environment.

Open-loop eval (`wmh eval <files>`) replays recorded steps teacher-forced and scores per-step
fidelity. This package is the closed-loop counterpart (docs/reference/closed_loop.md): a minimal
minimal agent loop (`AgentRuntime`) steps against `WorldModel.step` via the `AgentEnvironment`
seam, a gold-assertion judge scores *task success* on the resulting transcript, and
`compute_agreement` compares two closed-loop reports (e.g. simulated vs real) — the
outcome-agreement number docs name as the headline closed-loop validity metric.

`wmh harness create` (the harness search) builds on this scoring loop in a follow-up.
"""

from wmh.harness.agreement import AgreementReport, compute_agreement
from wmh.harness.closed_loop import (
    ClosedLoopReport,
    TaskOutcome,
    evaluate_closed_loop,
    evaluate_with_env,
)
from wmh.harness.environment import AgentEnvironment, WorldModelEnvironment
from wmh.harness.gold import GoldJudge, GoldVerdict
from wmh.harness.runtime import AgentRuntime, RunResult, StopReason
from wmh.harness.tasks import TaskSpec, load_tasks
from wmh.harness.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

__all__ = [
    "TOOL_REGISTRY",
    "AgentEnvironment",
    "AgentRuntime",
    "AgreementReport",
    "ClosedLoopReport",
    "GoldJudge",
    "GoldVerdict",
    "RunResult",
    "StopReason",
    "TaskOutcome",
    "TaskSpec",
    "ToolCall",
    "WorldModelEnvironment",
    "compute_agreement",
    "evaluate_closed_loop",
    "evaluate_with_env",
    "load_tasks",
    "parse_tool_call",
]

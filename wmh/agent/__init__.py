"""Managed agent harness + runtime (the build agent path).

`wmh.agent` closes the loop the rest of the harness leaves open: it *produces* the traces `wmh
build` consumes, and it *consumes* the world model those traces build. A pi-style minimal agent
(four env tools + a skill library it writes itself) runs tasks either in a real E2B sandbox (trace
collection) or against the world model (closed-loop evaluation), and an evolutionary manager uses
the closed-loop score deltas to steer a population of harness variants (see docs/autoharness.md
for the design and its prior art: ADAS, DGM, AlphaEvolve/OpenEvolve, GEPA, Voyager, pi).
"""

from wmh.agent.agreement import AgreementReport, compute_agreement, sim_real_agreement
from wmh.agent.capture import run_to_trace, write_otel_traces
from wmh.agent.closed_loop import ClosedLoopReport, evaluate_closed_loop, evaluate_with_env
from wmh.agent.collect import collect_traces
from wmh.agent.environment import AgentEnvironment, E2BEnvironment, WorldModelEnvironment
from wmh.agent.evolve import ArchiveEntry, HarnessArchive, evolve, mutate
from wmh.agent.gold import GoldJudge, GoldVerdict
from wmh.agent.real_loop import evaluate_real
from wmh.agent.runtime import AgentRuntime, RunResult, StopReason
from wmh.agent.skills import Skill, SkillLibrary
from wmh.agent.spec import HarnessSpec
from wmh.agent.tasks import TaskSpec, load_tasks
from wmh.agent.tools import TOOL_REGISTRY, ToolCall, parse_tool_call

__all__ = [
    "TOOL_REGISTRY",
    "AgentEnvironment",
    "AgentRuntime",
    "AgreementReport",
    "ArchiveEntry",
    "ClosedLoopReport",
    "E2BEnvironment",
    "GoldJudge",
    "GoldVerdict",
    "HarnessArchive",
    "HarnessSpec",
    "RunResult",
    "Skill",
    "SkillLibrary",
    "StopReason",
    "TaskSpec",
    "ToolCall",
    "WorldModelEnvironment",
    "collect_traces",
    "compute_agreement",
    "evaluate_closed_loop",
    "evaluate_real",
    "evaluate_with_env",
    "evolve",
    "load_tasks",
    "mutate",
    "parse_tool_call",
    "run_to_trace",
    "sim_real_agreement",
    "write_otel_traces",
]

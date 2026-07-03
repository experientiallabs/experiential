"""Run benchmarks for real and record agent-environment transitions as OTel GenAI JSONL."""

from environment_capture.adapter import (
    AgentRun,
    BenchmarkAdapter,
    CaptureAgent,
    CaptureResult,
    CommandEnv,
    ExecResult,
    TaskFailure,
    run_capture,
)
from environment_capture.baseline_cache import load_baseline_cache
from environment_capture.hygiene import (
    HygieneFinding,
    host_escape_findings,
    partition_contained,
    scan_spans_jsonl,
)
from environment_capture.otel import trace_id_for, trajectory_to_spans, write_spans_jsonl
from environment_capture.split_expansion import (
    CandidateTask,
    PlannedTask,
    plan_appended_tasks,
)
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall, Trajectory

__all__ = [
    "AgentRun",
    "BenchmarkAdapter",
    "CandidateTask",
    "CaptureAgent",
    "CaptureResult",
    "CommandEnv",
    "ExecResult",
    "HygieneFinding",
    "JsonValue",
    "PlannedTask",
    "StepRecord",
    "Task",
    "TaskFailure",
    "ToolCall",
    "Trajectory",
    "host_escape_findings",
    "load_baseline_cache",
    "partition_contained",
    "plan_appended_tasks",
    "run_capture",
    "scan_spans_jsonl",
    "trace_id_for",
    "trajectory_to_spans",
    "write_spans_jsonl",
]

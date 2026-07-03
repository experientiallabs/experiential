"""Load trajectories from a baseline-cache directory of frozen REAL benchmark runs.

The cache layout: ``manifest.json`` (per-task reward + the model that produced the runs),
``tasks/<task_id>.json`` (the agent-visible task), and ``traces/<task_id>.json`` (the native
trajectory: a ``messages`` list where each assistant turn carries exactly one fenced bash command
block and the following user turn carries the real ``<returncode>``/``<output>`` the environment
returned). This module parses that DATA format into Trajectories; the runs themselves were
executed for real elsewhere, so nothing here synthesizes an observation.

Some harnesses reply to a malformed assistant turn (e.g. two commands at once) with a short
free-text correction instead of executing anything. Such a turn carries no observation markers
at all, so the command never ran and is not a transition — it is skipped. A follow-up turn that
*does* look like an observation but is missing a marker is real format drift and still raises.

One normalization is applied to the loaded text: the recording harness's submission sentinel (an
ALLCAPS ``*_SUBMIT`` protocol keyword echoed into the final command and its output) becomes the
neutral ``SUBMIT``. The sentinel belongs to the recording apparatus, not the environment being
modeled — no result, path, or number is altered — and normalizing keeps corpora harness-agnostic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall, Trajectory

_FENCE_RE = re.compile(r"```\w*bash\s*\n(.*?)```", re.DOTALL)
_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
_OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>\s*\Z", re.DOTALL)
# No leading \b: the sentinel is often glued to prior output (e.g. a cat without a trailing
# newline directly followed by the sentinel echo), which a word boundary would miss.
_SENTINEL_RE = re.compile(r"[A-Z][A-Z0-9]{1,7}_SUBMIT\b")


def _normalize_sentinel(text: str) -> str:
    """Rewrite the recording harness's ``*_SUBMIT`` sentinel to the neutral ``SUBMIT``."""
    return _SENTINEL_RE.sub("SUBMIT", text)


def _is_observation(content: str) -> bool:
    """Whether a follow-up user turn is an observation attempt at all (vs. a free-text nudge)."""
    return "<returncode>" in content or "<output>" in content


def _parse_observation(content: str, *, task_id: str) -> tuple[str, int]:
    rc_match = _RETURNCODE_RE.search(content)
    out_match = _OUTPUT_RE.search(content)
    if rc_match is None or out_match is None:
        raise ValueError(
            f"{task_id}: expected <returncode>/<output> markers in observation, got: "
            f"{content[:120]!r}. The cache trace format may have changed — update "
            f"baseline_cache.py."
        )
    return out_match.group(1), int(rc_match.group(1))


def _steps_from_messages(messages: list[dict[str, str]], *, task_id: str) -> list[StepRecord]:
    steps: list[StepRecord] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        fence = _FENCE_RE.search(message.get("content", ""))
        if fence is None:
            continue  # a final free-text turn issues no command -> no environment transition
        command = fence.group(1).strip()
        follow = messages[index + 1] if index + 1 < len(messages) else None
        if follow is None or follow.get("role") != "user":
            continue  # command with no recorded observation (run cut off) -> not a transition
        follow_content = follow.get("content", "")
        if not _is_observation(follow_content):
            continue  # harness rejected the command with a free-text nudge -> command never ran
        output, returncode = _parse_observation(follow_content, task_id=task_id)
        steps.append(
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": _normalize_sentinel(command)}),
                output=_normalize_sentinel(output),
                is_error=returncode != 0,
            )
        )
    return steps


def load_baseline_cache(cache_dir: Path) -> list[Trajectory]:
    """Parse every task in a baseline-cache directory into Trajectories, in manifest order."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    model = str(manifest.get("model", ""))
    split = str(manifest.get("split", ""))

    trajectories: list[Trajectory] = []
    for entry in manifest["tasks"]:
        task_id = str(entry["task_id"])
        task_raw = json.loads((cache_dir / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        trace_raw = json.loads(
            (cache_dir / "traces" / f"{task_id}.json").read_text(encoding="utf-8")
        )
        metadata: dict[str, JsonValue] = {
            "source_format": "baseline-cache-v1",
            "passed": bool(entry.get("passed", False)),
            "exit_status": str(trace_raw.get("exit_status", "")),
        }
        trajectories.append(
            Trajectory(
                task=Task(
                    task_id=task_id,
                    prompt=str(task_raw.get("prompt", "")),
                    data=task_raw.get("data", {}),
                ),
                steps=_steps_from_messages(trace_raw.get("messages", []), task_id=task_id),
                final_answer=_normalize_sentinel(str(trace_raw.get("submission", ""))),
                reward=float(entry["reward"]) if entry.get("reward") is not None else None,
                model=model,
                split=split,
                metadata=metadata,
            )
        )
    return trajectories

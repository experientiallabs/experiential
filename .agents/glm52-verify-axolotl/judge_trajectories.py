"""Classify collected terminal-task trajectories with an auditable Bedrock judge."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, Field, ValidationError


LOGGER = logging.getLogger(__name__)
PROMPT_VERSION = "terminal-trajectory-quality-v3"
TOOL_NAME = "record_trajectory_quality"

SYSTEM_PROMPT = """You are a strict quality judge for terminal-agent training trajectories.

Your task is to decide whether the assistant trajectory provides strong transcript-grounded
evidence that it completed the user's terminal task correctly. This is a dataset filtering
judgment, not an official benchmark verifier result.

Rules:
1. Judge task completion, not writing style or apparent effort.
2. Use only observable evidence in the supplied task and transcript.
3. A model's assertion that it succeeded is not evidence.
4. Successful commands, tests, inspections, and final artifact checks are positive evidence.
5. Unresolved errors, abandoned requirements, malformed tool calls, context exhaustion, and
   contradictions are failure evidence.
6. PASS does not require an official hidden-verifier result. Strong direct tests, independent
   recomputation, and artifact inspection can establish success when they cover every explicit
   task requirement.
7. Use unverified_requirements only for an explicit material requirement in the user's task whose
   completion is not established by the transcript. Do not list hypothetical hidden-grader
   preferences, unspecified newline or formatting conventions, robustness beyond the requested
   inputs, or the generic absence of an official verifier.
8. PASS requires every explicit material requirement to be addressed, concrete success evidence,
   no unresolved failure signal, and an empty unverified_requirements list. If a material explicit
   requirement remains unverified, choose UNCERTAIN and name it precisely.
9. Set use_for_sft true only for a PASS with confidence at least 90 and concrete success evidence.
10. Keep the rationale concise. Do not reproduce hidden chain-of-thought. Cite concrete transcript
   evidence in short paraphrases.
"""

TOOL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "confidence",
        "use_for_sft",
        "task_completion_summary",
        "rationale",
        "success_evidence",
        "failure_evidence",
        "unverified_requirements",
        "failure_modes",
        "replay_risk",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNCERTAIN"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "use_for_sft": {"type": "boolean"},
        "task_completion_summary": {"type": "string"},
        "rationale": {"type": "string"},
        "success_evidence": {"type": "array", "items": {"type": "string"}},
        "failure_evidence": {"type": "array", "items": {"type": "string"}},
        "unverified_requirements": {
            "type": "array",
            "description": (
                "Only explicit material task requirements not established by the transcript. "
                "Never include hypothetical hidden-grader preferences or unspecified edge cases. "
                "Must be empty for PASS."
            ),
            "items": {"type": "string"},
        },
        "failure_modes": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "NONE",
                    "UNRESOLVED_COMMAND_ERROR",
                    "MISSING_REQUIREMENT",
                    "INCORRECT_RESULT",
                    "MALFORMED_TOOL_CALL",
                    "CONTEXT_EXHAUSTION",
                    "UNSUPPORTED_SUCCESS_CLAIM",
                    "INCOMPLETE_TRANSCRIPT",
                    "CONTRADICTORY_EVIDENCE",
                    "OTHER",
                ],
            },
        },
        "replay_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
    },
}


class JudgeDecision(BaseModel):
    """Validated structured decision returned by the judge."""

    verdict: str
    confidence: int = Field(ge=0, le=100)
    use_for_sft: bool
    task_completion_summary: str
    rationale: str
    success_evidence: list[str] = Field(default_factory=list)
    failure_evidence: list[str] = Field(default_factory=list)
    unverified_requirements: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    replay_risk: str = "HIGH"


@dataclass(frozen=True)
class SourceRow:
    """One source row with the minimum provenance needed by the judge."""

    row_index: int
    raw_line: str
    rollout_id: str
    task_id: str
    manifest_order: int
    replica: int
    message_log_json: str
    tools_json: str
    n_student_tokens: int
    n_supervised_tokens: int
    student_truncated_at_train_axis: bool


@dataclass(frozen=True)
class JudgeConfig:
    """Runtime configuration for one model pass."""

    model_id: str
    region: str
    corpus_revision: str
    source_sha256: str
    max_attempts: int


_THREAD_LOCAL = threading.local()


def sha256_text(value: str) -> str:
    """Return a hexadecimal SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def bedrock_runtime(region: str):
    """Return one thread-local Bedrock Runtime client."""
    client = getattr(_THREAD_LOCAL, "bedrock_runtime", None)
    if client is None:
        client = boto3.client("bedrock-runtime", region_name=region)
        _THREAD_LOCAL.bedrock_runtime = client
    return client


def build_user_prompt(row: SourceRow) -> str:
    """Build the deterministic judge input for one trajectory."""
    metadata = {
        "rollout_id": row.rollout_id,
        "task_id": row.task_id,
        "manifest_order": row.manifest_order,
        "replica": row.replica,
        "n_student_tokens": row.n_student_tokens,
        "n_supervised_tokens": row.n_supervised_tokens,
        "student_truncated_at_train_axis": row.student_truncated_at_train_axis,
    }
    return (
        "Evaluate the following collected terminal-agent trajectory.\n\n"
        f"METADATA\n{json.dumps(metadata, sort_keys=True)}\n\n"
        f"AVAILABLE TOOLS\n{row.tools_json}\n\n"
        f"MESSAGE TRANSCRIPT\n{row.message_log_json}\n"
    )


def normalize_decision(value: dict[str, object]) -> tuple[JudgeDecision, list[str]]:
    """Validate a raw decision and conservatively normalize semantic conflicts."""
    decision = JudgeDecision.model_validate(value)
    warnings: list[str] = []
    if decision.verdict == "PASS" and decision.unverified_requirements:
        warnings.append("downgraded PASS with unverified requirements to UNCERTAIN")
        decision = decision.model_copy(
            update={
                "verdict": "UNCERTAIN",
                "confidence": min(decision.confidence, 89),
                "use_for_sft": False,
            }
        )
    if decision.verdict == "PASS" and not decision.success_evidence:
        warnings.append("downgraded PASS without success evidence to UNCERTAIN")
        decision = decision.model_copy(
            update={
                "verdict": "UNCERTAIN",
                "confidence": min(decision.confidence, 89),
                "use_for_sft": False,
            }
        )
    should_use = decision.verdict == "PASS" and decision.confidence >= 90
    if decision.use_for_sft != should_use:
        warnings.append("normalized use_for_sft to match the strict PASS threshold")
        decision = decision.model_copy(update={"use_for_sft": should_use})
    return decision, warnings


def extract_tool_decision(
    response: dict[str, object],
) -> tuple[dict[str, object], JudgeDecision, list[str]]:
    """Extract and validate the forced tool-use decision from Converse output."""
    output = response.get("output")
    if not isinstance(output, dict):
        raise ValueError("Bedrock response has no output object")
    message = output.get("message")
    if not isinstance(message, dict):
        raise ValueError("Bedrock response has no output message")
    content = message.get("content")
    if not isinstance(content, list):
        raise ValueError("Bedrock response message has no content list")
    for block in content:
        if not isinstance(block, dict):
            continue
        tool_use = block.get("toolUse")
        if isinstance(tool_use, dict) and tool_use.get("name") == TOOL_NAME:
            tool_input = tool_use.get("input")
            if not isinstance(tool_input, dict):
                raise ValueError("Judge tool input is not an object")
            decision, warnings = normalize_decision(tool_input)
            return tool_input, decision, warnings
    raise ValueError("Bedrock response did not call the forced judge tool")


def classify(row: SourceRow, config: JudgeConfig) -> dict[str, object]:
    """Classify one trajectory with retry and full request provenance."""
    user_prompt = build_user_prompt(row)
    started = time.monotonic()
    requested_at = datetime.now(UTC).isoformat()
    last_error: Exception | None = None
    for attempt in range(1, config.max_attempts + 1):
        try:
            response = bedrock_runtime(config.region).converse(
                modelId=config.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user_prompt}],
                    }
                ],
                inferenceConfig={"maxTokens": 1400},
                toolConfig={
                    "tools": [
                        {
                            "toolSpec": {
                                "name": TOOL_NAME,
                                "description": "Record the strict trajectory-quality decision.",
                                "inputSchema": {"json": TOOL_SCHEMA},
                            }
                        }
                    ],
                    "toolChoice": {"tool": {"name": TOOL_NAME}},
                },
            )
            raw_decision, decision, normalization_warnings = extract_tool_decision(response)
            metadata = response.get("ResponseMetadata", {})
            result: dict[str, object] = {
                "schema": "glm52-terminal-trajectory-judgment-v1",
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": sha256_text(SYSTEM_PROMPT),
                "source_corpus_revision": config.corpus_revision,
                "source_corpus_sha256": config.source_sha256,
                "source_row_sha256": sha256_text(row.raw_line),
                "message_log_sha256": sha256_text(row.message_log_json),
                "row_index": row.row_index,
                "rollout_id": row.rollout_id,
                "task_id": row.task_id,
                "manifest_order": row.manifest_order,
                "replica": row.replica,
                "judge_provider": "bedrock",
                "judge_model_id": config.model_id,
                "judge_region": config.region,
                "requested_at": requested_at,
                "latency_seconds": round(time.monotonic() - started, 3),
                "attempts": attempt,
                "request_id": metadata.get("RequestId")
                if isinstance(metadata, dict)
                else None,
                "stop_reason": response.get("stopReason"),
                "usage": response.get("usage", {}),
                "metrics": response.get("metrics", {}),
                "raw_decision": raw_decision,
                "decision": decision.model_dump(),
                "normalization_warnings": normalization_warnings,
            }
            return result
        except (BotoCoreError, ClientError, ValidationError, ValueError) as error:
            last_error = error
            if attempt == config.max_attempts:
                break
            delay = min(30.0, (2 ** (attempt - 1)) + random.random())
            time.sleep(delay)
    return {
        "schema": "glm52-terminal-trajectory-judgment-error-v1",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "source_corpus_revision": config.corpus_revision,
        "source_corpus_sha256": config.source_sha256,
        "source_row_sha256": sha256_text(row.raw_line),
        "message_log_sha256": sha256_text(row.message_log_json),
        "row_index": row.row_index,
        "rollout_id": row.rollout_id,
        "task_id": row.task_id,
        "manifest_order": row.manifest_order,
        "replica": row.replica,
        "judge_provider": "bedrock",
        "judge_model_id": config.model_id,
        "judge_region": config.region,
        "requested_at": requested_at,
        "latency_seconds": round(time.monotonic() - started, 3),
        "attempts": config.max_attempts,
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": str(last_error) if last_error else "unknown error",
    }


def parse_source_row(row_index: int, raw_line: str) -> SourceRow:
    """Parse one corpus line into the compact source-row contract."""
    value = json.loads(raw_line)
    return SourceRow(
        row_index=row_index,
        raw_line=raw_line.rstrip("\n"),
        rollout_id=str(value["rollout_id"]),
        task_id=str(value["task_id"]),
        manifest_order=int(value["manifest_order"]),
        replica=int(value["replica"]),
        message_log_json=str(value["message_log_json"]),
        tools_json=str(value["tools_json"]),
        n_student_tokens=int(value["n_student_tokens"]),
        n_supervised_tokens=int(value["n_supervised_tokens"]),
        student_truncated_at_train_axis=bool(value["student_truncated_at_train_axis"]),
    )


def load_indices(path: Path | None) -> set[int] | None:
    """Load optional row indices from JSON, JSONL, or newline-delimited text."""
    if path is None:
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        return {int(value) for value in json.loads(text)}
    indices: set[int] = set()
    for line in text.splitlines():
        value = json.loads(line) if line.lstrip().startswith("{") else line
        indices.add(int(value["row_index"] if isinstance(value, dict) else value))
    return indices


def load_completed(path: Path) -> set[int]:
    """Return source row indices already durably present in an output JSONL."""
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open(encoding="utf-8") as output:
        for line in output:
            if line.strip():
                completed.add(int(json.loads(line)["row_index"]))
    return completed


def iter_rows(
    corpus_path: Path,
    selected_indices: set[int] | None,
    completed_indices: set[int],
    limit: int | None,
):
    """Yield selected, unfinished rows from a JSONL corpus."""
    yielded = 0
    with corpus_path.open(encoding="utf-8") as corpus:
        for row_index, raw_line in enumerate(corpus):
            if selected_indices is not None and row_index not in selected_indices:
                continue
            if row_index in completed_indices:
                continue
            if limit is not None and yielded >= limit:
                return
            yield parse_source_row(row_index, raw_line)
            yielded += 1


def run(args: argparse.Namespace) -> None:
    """Run concurrent classification with append-only durable output."""
    selected_indices = load_indices(args.indices)
    completed_indices = load_completed(args.output)
    config = JudgeConfig(
        model_id=args.model_id,
        region=args.region,
        corpus_revision=args.corpus_revision,
        source_sha256=args.source_sha256,
        max_attempts=args.max_attempts,
    )
    rows = iter_rows(args.corpus, selected_indices, completed_indices, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submitted = 0
    finished = 0
    verdict_counts: dict[str, int] = {}
    with args.output.open("a", encoding="utf-8") as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            pending: dict[concurrent.futures.Future[dict[str, object]], SourceRow] = {}
            for row in rows:
                future = executor.submit(classify, row, config)
                pending[future] = row
                submitted += 1
                if len(pending) >= args.concurrency * 2:
                    done, _ = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for completed_future in done:
                        result = completed_future.result()
                        output.write(json.dumps(result, sort_keys=True) + "\n")
                        output.flush()
                        pending.pop(completed_future)
                        finished += 1
                        decision = result.get("decision")
                        verdict = (
                            str(decision.get("verdict"))
                            if isinstance(decision, dict)
                            else "ERROR"
                        )
                        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                        LOGGER.info(
                            "finished=%d submitted=%d row=%s verdict=%s counts=%s",
                            finished,
                            submitted,
                            result["row_index"],
                            verdict,
                            verdict_counts,
                        )
            for completed_future in concurrent.futures.as_completed(pending):
                result = completed_future.result()
                output.write(json.dumps(result, sort_keys=True) + "\n")
                output.flush()
                finished += 1
                decision = result.get("decision")
                verdict = (
                    str(decision.get("verdict")) if isinstance(decision, dict) else "ERROR"
                )
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
                LOGGER.info(
                    "finished=%d submitted=%d row=%s verdict=%s counts=%s",
                    finished,
                    submitted,
                    result["row_index"],
                    verdict,
                    verdict_counts,
                )
    LOGGER.info("complete submitted=%d finished=%d counts=%s", submitted, finished, verdict_counts)


def main() -> None:
    """Parse command-line arguments and execute the judge pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--region", default="us-west-1")
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run(args)


if __name__ == "__main__":
    main()

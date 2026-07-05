"""A Bedrock-backed capture agent: bash tool-use loop over a CommandEnv.

The agent gets two tools — ``bash`` (executed for real in the environment; the REAL output is
returned as the tool result) and ``submit`` (ends the episode with a final answer). Every bash
call becomes one recorded transition. Throttling is retried with linear backoff; real errors
propagate (fail fast).
"""

from __future__ import annotations

import time
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectTimeoutError, ReadTimeoutError

from environment_capture.adapter import AgentRun, CommandEnv
from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall

DEFAULT_SYSTEM_PROMPT = """You are an autonomous analyst agent working in a Unix workspace.
Everything you need is INSIDE the current directory — start with `ls` and stay in the
workspace; commands that target host paths (absolute paths, ~, $HOME, cd ..) are blocked and
waste a step. Investigate before answering: list and read the workspace files, search with
grep, and compute with python3 when arithmetic is needed. Use the bash tool for every
investigation step — one focused command per call — and check intermediate results rather than
assuming them. When you are confident, call submit with your final answer (concise: the number
or short phrase the question asks for, with units)."""

_TOOL_CONFIG: dict[str, JsonValue] = {
    "tools": [
        {
            "toolSpec": {
                "name": "bash",
                "description": "Run one bash command in the task workspace; returns its real "
                "output and exit code.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "submit",
                "description": "Submit the final answer and end the task.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                },
            }
        },
    ]
}

_THROTTLE_CODES = {"ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException"}
_MAX_RETRIES = 6


class ConverseClient(Protocol):
    """The slice of the Bedrock runtime client the agent needs (stubbed in tests)."""

    def converse(
        self,
        *,
        modelId: str,
        messages: list[JsonValue],
        system: list[JsonValue],
        toolConfig: JsonValue,
        inferenceConfig: JsonValue,
    ) -> dict[str, JsonValue]: ...


def make_bedrock_client(region: str = "us-east-1") -> ConverseClient:
    """A real Bedrock runtime client with generous read timeouts for long completions."""
    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=300, retries={"max_attempts": 0}),
    )


class BedrockBashAgent:
    """CaptureAgent driving a CommandEnv through Bedrock converse tool-use."""

    def __init__(
        self,
        model_id: str,
        *,
        client: ConverseClient | None = None,
        region: str = "us-east-1",
        max_steps: int = 12,
        max_tokens: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        retry_backoff_s: float = 5.0,
    ) -> None:
        self.model_id = model_id
        self._client = client if client is not None else make_bedrock_client(region)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.retry_backoff_s = retry_backoff_s

    def _converse(self, messages: list[JsonValue]) -> dict[str, JsonValue]:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    system=[{"text": self.system_prompt}],
                    toolConfig=_TOOL_CONFIG,
                    inferenceConfig={"maxTokens": self.max_tokens},
                )
            except (ReadTimeoutError, ConnectTimeoutError):
                # A hung/slow call under load: transient, retry with linear backoff.
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "")
                if code not in _THROTTLE_CODES or attempt == _MAX_RETRIES:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
        raise RuntimeError("unreachable")

    def run(self, task: Task, env: CommandEnv) -> AgentRun:
        """Drive the environment until the agent submits, answers in text, or hits max_steps.

        A single turn may contain several tool-use blocks (Bedrock emits parallel tool calls);
        every one must be answered with a toolResult or the next Converse call is rejected. So all
        bash calls in a turn are executed in order, each recorded as its own transition, and their
        results returned together in one follow-up user message.
        """
        messages: list[JsonValue] = [{"role": "user", "content": [{"text": task.prompt}]}]
        steps: list[StepRecord] = []
        final_answer = ""

        while len(steps) < self.max_steps:
            response = self._converse(messages)
            output = response.get("output")
            assert isinstance(output, dict)
            message = output.get("message")
            assert isinstance(message, dict)
            messages.append(message)

            tool_uses = _tool_uses(message)
            if not tool_uses:
                final_answer = _text_content(message)
                break

            tool_results: list[JsonValue] = []
            submitted = False
            for tool_use_id, name, tool_input in tool_uses:
                if name == "submit":
                    answer = tool_input.get("answer", "")
                    final_answer = answer if isinstance(answer, str) else str(answer)
                    submitted = True
                    break

                command = tool_input.get("command", "")
                command_text = command if isinstance(command, str) else str(command)
                result = env.execute(command_text)
                steps.append(
                    StepRecord(
                        action=ToolCall(name="bash", arguments={"command": command_text}),
                        output=result.output,
                        is_error=result.returncode != 0,
                    )
                )
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "text": f"<returncode>{result.returncode}</returncode>\n"
                                    f"{result.output}"
                                }
                            ],
                            "status": "error" if result.returncode != 0 else "success",
                        }
                    }
                )
                if len(steps) >= self.max_steps:
                    break

            if submitted:
                break
            messages.append({"role": "user", "content": tool_results})

        return AgentRun(steps=steps, final_answer=final_answer, model=self.model_id)


def _tool_uses(message: dict[str, JsonValue]) -> list[tuple[str, str, dict[str, JsonValue]]]:
    """Every (id, name, input) tool-use block in the message, in order (Bedrock may emit many)."""
    content = message.get("content")
    if not isinstance(content, list):
        return []
    uses: list[tuple[str, str, dict[str, JsonValue]]] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
            tool_use = block["toolUse"]
            assert isinstance(tool_use, dict)
            tool_use_id = tool_use.get("toolUseId", "")
            name = tool_use.get("name", "")
            tool_input = tool_use.get("input", {})
            assert isinstance(tool_use_id, str) and isinstance(name, str)
            assert isinstance(tool_input, dict)
            uses.append((tool_use_id, name, tool_input))
    return uses


def _text_content(message: dict[str, JsonValue]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(str(p) for p in parts).strip()

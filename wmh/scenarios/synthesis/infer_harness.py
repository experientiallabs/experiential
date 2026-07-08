"""Predict an agent harness from sparse trace evidence, when the capture recorded none.

Most existing corpora predate harness capture: their spans carry tool calls, arguments,
observations, and errors — but no `gen_ai.system_instructions` or `gen_ai.tool.definitions`.
This module reconstructs the most plausible `HarnessContext` from what the traces *do* leak:

  - the tool inventory (every tool name the agent actually called);
  - argument shapes (JSON types observed across calls -> schema properties);
  - validation errors, which state required properties verbatim
    (`must have required properties path, content`);
  - task phrasing, structured state (e.g. a `harness` marker), and response formats.

Evidence extraction is deterministic; an LLM writes the system prompt and refines the tool
schemas from the evidence digest. The result is marked `HarnessSource.INFERRED` so downstream
consumers can label it, and it never overrides a captured harness — corpora that record the real
attributes take precedence wherever both exist. On an unparseable LLM reply the deterministic
schemas alone are returned (empty system prompt), which is still enough for schema-faithful
scenario rendering and agent rollouts.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.render import render_json
from wmh.core.types import (
    ActionKind,
    HarnessContext,
    HarnessSource,
    JsonObject,
    JsonValue,
    Step,
    ToolDefinition,
)
from wmh.providers.base import Message, Provider

INFERENCE_SYSTEM = """You reconstruct an AI agent's harness — the system prompt and tool
definitions its runtime most plausibly sends — from partial evidence: observed tool calls,
argument shapes, validation errors, environment responses, and tasks.

Respond with ONLY a JSON object, no prose around it:
{"system_prompt": "<the full system prompt this agent most plausibly runs under: role, the
available tools with one-line descriptions, response conventions, practical guidelines —
written the way real agent harnesses write them>",
 "tools": [{"name": "<tool>", "description": "<one line>", "parameters": {<JSON Schema>}}]}

Rules:
- Include EVERY tool observed in the evidence. NEVER include a tool the evidence doesn't show.
- Each tool's parameters must be a JSON Schema consistent with every observed call. Validation
  errors state required properties verbatim — honor them exactly.
- Ground the system prompt in the evidence (working directory, conventions, the harness name if
  the state reveals one). Do not invent capabilities, product names, or constraints the evidence
  doesn't support. Plain and plausible beats elaborate and speculative."""

# Bounds keeping the evidence digest prompt-sized on large corpora.
_EXAMPLES_PER_TOOL = 3
_ERRORS_PER_TOOL = 2
_TASK_SAMPLES = 3
_VALUE_CHARS = 200

# Validation errors of the common JSON-schema flavor name the missing properties verbatim.
_REQUIRED_RE = re.compile(r"must have required properties?\s+([\w\-, ]+)")


class _RawTool(BaseModel):
    name: str
    description: str = ""
    parameters: JsonObject = Field(default_factory=dict)


class _RawHarness(BaseModel):
    system_prompt: str = ""
    tools: list[_RawTool] = Field(default_factory=list)


class _ToolEvidence(BaseModel):
    """Everything the corpus reveals about one tool, accumulated deterministically."""

    name: str
    calls: int = 0
    # property name -> JSON-schema type name; first observed type wins (stable).
    property_types: dict[str, str] = Field(default_factory=dict)
    # property names present in EVERY non-empty call (candidate `required` set).
    always_present: set[str] | None = None
    example_calls: list[str] = Field(default_factory=list)
    error_messages: list[str] = Field(default_factory=list)
    required_from_errors: list[str] = Field(default_factory=list)


def _json_type(value: JsonValue) -> str:
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def _collect_evidence(steps: list[Step]) -> dict[str, _ToolEvidence]:
    evidence: dict[str, _ToolEvidence] = {}
    for step in steps:
        action = step.action
        if action.kind is not ActionKind.TOOL_CALL or not action.name:
            continue
        tool = evidence.setdefault(action.name, _ToolEvidence(name=action.name))
        tool.calls += 1
        if action.arguments:
            for key, value in action.arguments.items():
                tool.property_types.setdefault(key, _json_type(value))
            present = set(action.arguments)
            tool.always_present = (
                present if tool.always_present is None else tool.always_present & present
            )
        if len(tool.example_calls) < _EXAMPLES_PER_TOOL:
            rendered = render_json(action.arguments)[:_VALUE_CHARS]
            if rendered not in tool.example_calls:
                tool.example_calls.append(rendered)
        if step.observation.is_error and len(tool.error_messages) < _ERRORS_PER_TOOL:
            tool.error_messages.append(step.observation.content[:_VALUE_CHARS])
        match = _REQUIRED_RE.search(step.observation.content) if step.observation.is_error else None
        if match and not tool.required_from_errors:
            tool.required_from_errors = [
                name.strip() for name in match.group(1).split(",") if name.strip()
            ]
    return evidence


def _tool_definition(tool: _ToolEvidence) -> ToolDefinition:
    """The deterministic schema for one observed tool (used as fallback and backfill)."""
    properties: JsonObject = {
        key: {"type": type_name} for key, type_name in sorted(tool.property_types.items())
    }
    # Validation errors are authoritative; otherwise require what every call supplied.
    required = tool.required_from_errors or sorted(tool.always_present or set())
    # A property named only by a validation error was never observed in a call; type unknown.
    for name in required:
        properties.setdefault(name, {})
    parameters: JsonObject = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return ToolDefinition(name=tool.name, parameters=parameters)


def observed_tools(steps: list[Step]) -> list[ToolDefinition]:
    """Deterministic tool schemas from observed calls + validation errors, most-called first."""
    evidence = sorted(_collect_evidence(steps).values(), key=lambda t: (-t.calls, t.name))
    return [_tool_definition(tool) for tool in evidence]


def harness_evidence(steps: list[Step]) -> str:
    """The bounded, deterministic evidence digest the inference LLM reads."""
    lines: list[str] = []
    tasks: list[str] = []
    states: list[str] = []
    for step in steps:
        if step.task and step.task not in tasks and len(tasks) < _TASK_SAMPLES:
            tasks.append(step.task)
        if step.state_before.structured:
            rendered = render_json(step.state_before.structured)[:_VALUE_CHARS]
            if rendered not in states and len(states) < 2:
                states.append(rendered)
    lines.append("TASKS GIVEN TO THE AGENT (sample):")
    if tasks:
        lines.extend(f"- {task[:_VALUE_CHARS]}" for task in tasks)
    else:
        lines.append("- (none)")
    if states:
        lines.append("STRUCTURED ENV STATE (sample):")
        lines.extend(f"- {state}" for state in states)
    lines.append("OBSERVED TOOLS:")
    for tool in sorted(_collect_evidence(steps).values(), key=lambda t: (-t.calls, t.name)):
        lines.append(f"- {tool.name} ({tool.calls} calls)")
        lines.extend(f"    call: {example}" for example in tool.example_calls)
        for error in tool.error_messages:
            lines.append(f"    error: {error.replace(chr(10), ' | ')}")
    return "\n".join(lines)


def infer_harness(steps: list[Step], provider: Provider) -> HarnessContext:
    """Predict the harness behind `steps`. Marked `INFERRED`; never use over a captured one.

    The LLM writes the system prompt and refines schemas from the evidence digest; its reply is
    then reconciled against the observed inventory — invented tools are dropped, observed tools
    the reply missed are backfilled with their deterministic schemas. A garbage reply degrades
    to the deterministic schemas with an empty system prompt.
    """
    fallback = observed_tools(steps)
    completion = provider.complete(
        INFERENCE_SYSTEM,
        [Message(role="user", content=harness_evidence(steps))],
        temperature=0.0,
        max_tokens=4096,
    )
    raw = extract_json_object(completion.text)
    parsed: _RawHarness | None = None
    if raw is not None:
        try:
            parsed = _RawHarness.model_validate_json(raw)
        except ValidationError:
            parsed = None
    if parsed is None:
        return HarnessContext(tools=fallback, source=HarnessSource.INFERRED)
    observed_names = {tool.name for tool in fallback}
    tools = [
        ToolDefinition(name=t.name, description=t.description, parameters=t.parameters)
        for t in parsed.tools
        if t.name in observed_names  # evidence-free tools don't survive
    ]
    replied_names = {tool.name for tool in tools}
    tools.extend(tool for tool in fallback if tool.name not in replied_names)
    return HarnessContext(
        system_prompt=parsed.system_prompt.strip(),
        tools=tools,
        source=HarnessSource.INFERRED,
    )

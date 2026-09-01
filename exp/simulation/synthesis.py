"""Principled scenario synthesis beyond observed traffic.

Mined tasks describe what production traffic already shows. Synthesis is the
layer that generates scenarios past that evidence on purpose. Its first piece
is frontier probe generation: one model call reads a sample of observed
scenario tasks and proposes variants at the frontier of plausibility,
situations that are possible but unlikely and adjacent to the observed
traffic. Probes are triage material for a human reviewer; they carry no
partition or workload weight and never become selection evidence. The
Experiential Labs platform is the first consumer of this seam.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from pydantic import Field, ValidationError, field_validator

from exp.common.core.artifacts import ContractModel, JsonValue
from exp.common.models import (
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    StructuredReplyError,
    structured_reply_json,
)

DEFAULT_PROBE_COUNT = 5
# Reasoning models spend hidden reasoning tokens (hundreds per call on
# current frontier models) from the same output budget as the visible JSON,
# and an exhausted budget is rejected as a truncated reply. The default
# leaves room for both; a longer reply costs nothing unless produced.
DEFAULT_PROBE_OUTPUT_TOKENS = 4_000

FRONTIER_PROBE_PROMPT = (
    "You generate PROBE scenarios for an AI agent, given a sample of "
    "scenarios mined from its real traffic. A probe sits slightly PAST the "
    "frontier of what was observed: plausible for this agent's domain, but "
    "unlike anything in the sample, at the edge of the unknown unknowns. "
    "Rules:\n"
    "- Slightly unrealistic on purpose; never absurd or off-domain.\n"
    "- A good probe makes a human reviewer pause between 'unlikely' and "
    "'unrealistic'; that pause is the point.\n"
    "- One concrete situation per probe, written as a task the agent would "
    "face, in the sample's own style and length.\n"
    "Respond with ONLY a JSON array, no prose and no code fences: "
    '[{"task": "...", "rationale": "..."}, ...]. '
    "task is the probe scenario (at most 2000 characters); rationale is one "
    "line on why it sits at the frontier (at most 300 characters)."
)


class FrontierProbeError(ValueError):
    """The generator's reply broke the frontier-probe output contract."""


class FrontierProbe(ContractModel):
    """One generated frontier probe scenario.

    Args:
        task: The probe scenario, written as a task the agent would face.
        rationale: One line on why the probe sits at the frontier.
    """

    task: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=300)

    @field_validator("task", "rationale")
    @classmethod
    def _require_visible_text(cls, value: str) -> str:
        """Trim edge whitespace and reject blank-looking text."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("must contain non-whitespace text")
        return stripped

    @property
    def task_sha256(self) -> str:
        """Content-derived digest so regenerating an identical probe converges."""
        return hashlib.sha256(self.task.encode("utf-8")).hexdigest()


class FrontierProbeBatch(ContractModel):
    """Parsed probes plus the resolved generator identity for provenance.

    Args:
        probes: Contract-valid probes, at most the requested count.
        model: The generator model that produced the reply.
    """

    probes: tuple[FrontierProbe, ...]
    model: ModelSnapshot


def _parse_probes(raw: JsonValue) -> tuple[FrontierProbe, ...]:
    """Validate the generator's parsed JSON array, loudly refusing malformed output.

    Args:
        raw: Parsed JSON value from the generator's visible reply.

    Returns:
        Every probe in the array, in reply order.

    Raises:
        FrontierProbeError: The value is not a JSON array of contract-valid
            probes; rerun the generation or adjust the generator model.
    """
    if not isinstance(raw, list):
        msg = "the generator returned JSON that is not an array; rerun the generation"
        raise FrontierProbeError(msg)
    try:
        return tuple(FrontierProbe.model_validate(entry) for entry in raw)
    except ValidationError as error:
        msg = "the generator returned a probe outside the contracted shape; rerun the generation"
        raise FrontierProbeError(msg) from error


def generate_frontier_probes(
    client: ModelClient,
    observed_tasks: Sequence[str],
    *,
    probe_count: int = DEFAULT_PROBE_COUNT,
    maximum_output_tokens: int = DEFAULT_PROBE_OUTPUT_TOKENS,
) -> FrontierProbeBatch:
    """One model call proposing frontier probes from an observed sample.

    Args:
        client: Configured generator model client.
        observed_tasks: Sampled observed scenario task texts. The caller owns
            sampling and ordering; every entry is sent to the model verbatim.
        probe_count: Probes requested; a longer reply is truncated to this
            count and a shorter contract-valid reply is returned as is.
        maximum_output_tokens: Upper bound for the generator's reply.

    Returns:
        Parsed probes with the resolved generator identity.

    Raises:
        ValueError: ``observed_tasks`` is empty or contains an empty task,
            ``probe_count`` is not positive, or ``maximum_output_tokens`` is
            not positive.
        FrontierProbeError: The reply was truncated at the token limit,
            carried no text, or broke the JSON contract.
    """
    if not observed_tasks:
        raise ValueError("frontier probe generation needs at least one observed task")
    if any(not task.strip() for task in observed_tasks):
        raise ValueError("observed tasks must be non-empty strings")
    if probe_count <= 0:
        raise ValueError("probe_count must be positive")
    if maximum_output_tokens <= 0:
        raise ValueError("maximum_output_tokens must be positive")
    payload = json.dumps(
        {
            "observed_scenarios": [{"task": task} for task in observed_tasks],
            "probe_count": probe_count,
        },
        separators=(",", ":"),
    )
    response = client.complete(
        ModelRequest(
            messages=(
                ModelMessage(role="system", content=FRONTIER_PROBE_PROMPT),
                ModelMessage(role="user", content=payload),
            ),
            maximum_output_tokens=maximum_output_tokens,
        )
    )
    try:
        raw = structured_reply_json(response)
    except StructuredReplyError as error:
        raise FrontierProbeError(str(error)) from error
    probes = _parse_probes(raw)
    return FrontierProbeBatch(probes=probes[:probe_count], model=response.model)

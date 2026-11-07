"""Calibration inputs of one target: sampled from saved role draws, split by lineage.

:func:`sample_role_inputs` reads the forced draws a role already saved through the shared role
runner (artifact ids ``decision-<runner role>-...-transcript``). From each draw it takes the
exact request the role saw (system prompt, user content and answer tool). The lineage of a draw
is the set of saved artifacts its documents were rendered from, so the stances of one packet and
any redraw over the same sources share a lineage. Draws whose system prompt does not contain the
target text exactly once, whose answer tool differs, or whose sources or content name a
final-evaluation (held-out build) trace are skipped with the reason.

:func:`split` assigns whole lineages to the rewrite or the held-out side, in an order fixed by
the seed, so near-duplicates never straddle the split.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from exp.common.core.artifacts import sha256_json
from exp.common.models import ModelRequest
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.astra_prompt_contracts import (
    CalibrationExample,
    CalibrationSet,
    Split,
)
from exp.optimize.evaluation.variance.evidence_tools import TRANSCRIPT_TYPE, AnalysisTranscript
from exp.optimize.evaluation.variance.prompt_targets import PromptTarget


def sample_role_inputs(
    project: ProjectStore,
    target: PromptTarget,
    *,
    excluded_trace_ids: Sequence[str] = (),
    maximum: int = 48,
) -> tuple[CalibrationSet | None, dict[str, str]]:
    """Sample a target's calibration set from the role's saved forced draws.

    Args:
        project: Project holding the saved draws.
        target: A target whose role runs through the shared role runner.
        excluded_trace_ids: Final-evaluation traces; a draw naming one is skipped.
        maximum: Most examples kept, in artifact-id order.

    Returns:
        The set (None when fewer than two usable draws exist) and, per skipped draw, why.
    """
    if target.runner_role is None:
        return None, {"*": f"{target.target_id} has no runner role to sample saved draws from"}
    store = project.artifacts
    prefix = f"decision-{target.runner_role}-"
    excluded = set(excluded_trace_ids)
    examples: list[CalibrationExample] = []
    skipped: dict[str, str] = {}
    answer = None
    for artifact_id in sorted(store.list_ids(artifact_type=TRANSCRIPT_TYPE)):
        if not artifact_id.startswith(prefix) or len(examples) >= maximum:
            continue
        transcript = AnalysisTranscript.model_validate_json(
            store.read_bytes(artifact_id, "transcript.json")
        )
        if transcript.profile.mode != "forced_response" or not transcript.calls:
            skipped[artifact_id] = "not a forced draw; its request does not hold the documents"
            continue
        response = store.read(transcript.calls[0].response.artifact_id).manifest
        intent = response.inputs[0]
        request = ModelRequest.model_validate_json(
            store.read_bytes(intent.artifact_id, "request.json")
        )
        system, user = (str(message.content or "") for message in request.messages[:2])
        if system.count(target.text) != 1:
            skipped[artifact_id] = "the system prompt does not hold the target text exactly once"
            continue
        if len(request.tools) != 1 or (answer is not None and request.tools[0] != answer):
            skipped[artifact_id] = "the answer tool differs from the other draws"
            continue
        sources = tuple(ref.artifact_id for ref in store.read(intent.artifact_id).manifest.inputs)
        named = sorted(trace for trace in excluded if trace in sources or trace in user)
        if named:
            skipped[artifact_id] = f"names final-evaluation traces {named[:3]}"
            continue
        answer = request.tools[0]
        examples.append(
            CalibrationExample(
                example_id=artifact_id.removesuffix("-transcript")[-40:],
                lineage_id=sha256_json(sorted(sources)) if sources else artifact_id,
                trace_ids=(),
                source="pipeline-record",
                source_ref=artifact_id,
                system=system,
                user=user,
            )
        )
    if answer is None or len(examples) < 2:
        return None, skipped
    return CalibrationSet(
        target_id=target.target_id,
        base_sha256=target.sha256,
        answer=answer,
        examples=tuple(examples),
        excluded_trace_ids=tuple(sorted(excluded)),
    ), skipped


def split(calibration: CalibrationSet, fraction: float, seed: int) -> Split:
    """Assign whole lineages to the held-out side until it holds ``fraction`` of the examples.

    Lineages are ordered by a seeded digest; at least one lineage lands on each side.

    Raises:
        ValueError: Fewer than two lineages exist, so no split keeps lineages apart.
    """
    lineages: dict[str, list[str]] = {}
    for row in calibration.examples:
        lineages.setdefault(row.lineage_id, []).append(row.example_id)
    if len(lineages) < 2:
        raise ValueError("a lineage split needs at least two lineages")
    order = sorted(
        lineages,
        key=lambda name: hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest(),
    )
    want = fraction * len(calibration.examples)
    heldout: list[str] = []
    for name in order[:-1]:
        if heldout and sum(len(lineages[n]) for n in heldout) >= want:
            break
        heldout.append(name)
    rewrite = [name for name in order if name not in heldout]
    return Split(
        rewrite=tuple(e for name in rewrite for e in lineages[name]),
        heldout=tuple(e for name in heldout for e in lineages[name]),
        rewrite_lineages=tuple(sorted(rewrite)),
        heldout_lineages=tuple(sorted(heldout)),
    )


def calibration_from_json(text: str) -> CalibrationSet:
    """Read a calibration set file (for build-fit data assembled outside the role runner)."""
    return CalibrationSet.model_validate(json.loads(text))

"""Pure current-delivery research ablation; never retirement or removal from a memory bank."""

import json
from collections.abc import Callable

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest
from exp.common.project import ArtifactStore
from exp.simulation.engines.text.prompt import MemoryContext
from exp.simulation.world_model.memory_tree import AUTHORITY, MemoryLine, read_line

RESEARCH_HISTORY_ABLATION = "off-current-delivery-v1"
"""Identity mark for a world model whose current delivery withholds research history."""


def observation_overview(
    store: ArtifactStore,
    lineage: str,
    rows: tuple[JsonObject, ...],
    *,
    admits: Callable[[MemoryLine], bool],
) -> tuple[JsonObject, ...]:
    """Open only delivered summary spans into their original observation leaves.

    The caller supplies an already eligible, pinned overview. Its summaries can only cover
    eligible leaves. We preserve existing omission markers and never add omitted history.
    Expanded observations are required context: downstream capacity admission must refuse
    when they do not fit, rather than silently recompressing or discarding them.
    """
    result: list[JsonObject] = []
    for row in rows:
        kind = row.get("kind")
        if kind in {"observed", "omitted"}:
            result.append(dict(row))
        elif kind == "hypothesis":
            continue
        elif kind == "summary":
            identifier = row.get("id")
            if not isinstance(identifier, str) or not identifier.startswith("#"):
                raise ValueError("summary projection requires an exact log range")
            lo, hi = (int(part) for part in identifier[1:].split("-"))
            if lo < 0 or hi < lo:
                raise ValueError("summary projection has an invalid log range")
            for index in range(lo, hi + 1):
                line = read_line(store, lineage, index)
                if line.kind == "observed" and admits(line):
                    result.append(
                        {
                            "id": f"#{index}",
                            "kind": "observed",
                            "text": line.text,
                            "origin": line.origin,
                            "authority": AUTHORITY["observed"],
                        }
                    )
        else:
            raise ValueError("unknown overview authority cannot be projected safely")
    return tuple(result)


def without_research(memory: MemoryContext) -> MemoryContext:
    """Remove current hypothesis delivery only after every mixed summary was opened."""
    if any(row.get("kind") not in {"observed", "omitted"} for row in memory.overview):
        raise ValueError("research-off requires an observation-only expanded overview")
    return MemoryContext(observations=memory.observations, overview=memory.overview)


def project_research_request(request: ModelRequest) -> ModelRequest:
    """Remove constructor and session research from the final grounded input, preserving facts.

    The observation-only overview is retained. This leaves original prompt instructions in
    place so the only treatment is data delivery; it does not claim source-level removal.
    """
    messages = list(request.messages)
    found = False
    for index, message in enumerate(messages):
        if message.role != "user" or not message.content:
            continue
        try:
            body = json.loads(message.content)
        except ValueError:
            continue
        if not isinstance(body, dict) or "environment_state" not in body:
            continue
        if found:
            raise ValueError("research projection requires one grounded input frame")
        found = True
        research = body.get("research_history", {})
        if not isinstance(research, dict):
            raise ValueError("research history is not a framed object")
        overview = research.get("memory_overview", [])
        if not isinstance(overview, list) or any(
            not isinstance(row, dict) or row.get("kind") not in {"observed", "omitted"}
            for row in overview
        ):
            raise ValueError("research-off cannot retain unresolved or mixed overview content")
        body["research_history"] = {"memory_overview": overview} if overview else {}
        messages[index] = message.model_copy(
            update={
                "content": json.dumps(
                    body, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
            }
        )
    if not found:
        raise ValueError("research projection requires a grounded world-model request")
    return request.model_copy(update={"messages": tuple(messages)})

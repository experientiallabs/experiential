"""One memory version always delivers the same overview, even after later compression."""

import json
from pathlib import Path

from exp.optimize.evaluation.tests.memory_forget_test import LINEAGE, NOW, Compressor, Memory
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    memory_view,
    summary_generations,
)
from exp.simulation.world_model.memory_tree import overview
from exp.simulation.world_model.overview_pin import pinned_summaries

BUDGET = 32  # the default overview line budget


def rendered(memory: Memory, version: MemoryVersion, pinned: frozenset | None) -> str:
    """The overview a reader of ``version`` receives, as canonical text."""
    retired = set(memory_view(memory.store, version, Eligibility()).retired)
    rows = overview(
        memory.store,
        LINEAGE,
        version.log_length or 0,
        BUDGET,
        admits_line=lambda line: line.origin not in retired,
        generations=summary_generations(version),
        pinned=pinned,
    )
    return json.dumps(rows, sort_keys=True)


def test_compression_after_the_first_read_never_changes_a_pinned_version(
    tmp_path: Path,
) -> None:
    """The first read pins the built summaries; later compression stays invisible to it."""
    memory = Memory(tmp_path)
    for i in range(17):
        version = memory.learn(f"task-{i}", f"[unresolved] cue {i} means answer {i}")
    pin = pinned_summaries(memory.store, version, created_at=NOW, code_revision="fixture")
    before = rendered(memory, version, pin)
    assert '"omitted"' in before, "nothing is compressed yet, so older lines are omitted"

    assert memory.maintain(Compressor()), "compression completes after the first read"
    again = pinned_summaries(memory.store, version, created_at=NOW, code_revision="fixture")
    assert again == pin
    assert rendered(memory, version, again) == before
    assert rendered(memory, version, None) != before, "unpinned, the same version changes"

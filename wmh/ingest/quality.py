"""Corpus hygiene: filters for damaged traces that carry no learnable signal.

Discovered empirically (2026-07-01, swe-bench): an overnight capture left 168/255 traces as
single-step sessions whose only observation is EMPTY — failed capture runs, not real agent
sessions. Such traces poison everything downstream: retrieval serves them as demos, evals score
"predict the empty output of a broken capture," and fidelity numbers measure corpus damage
instead of world-model quality.

Filtering is EXPLICIT, never a silent ingest default — dropping traces changes corpora, splits,
and every downstream number, so callers opt in (e.g. `run_trace_scaling.py --drop-degenerate`)
and report what was dropped.
"""

from __future__ import annotations

from wmh.core.types import Trace


def drop_degenerate_traces(traces: list[Trace]) -> tuple[list[Trace], int]:
    """Drop traces whose EVERY observation is empty/whitespace; return (kept, dropped_count).

    A trace with no non-empty observation teaches the world model nothing (there is no
    environment behavior in it) and is the signature of a failed capture session. Traces with a
    MIX of empty and real observations are kept — genuinely-empty outputs (silent commands,
    grep misses) are real environment behavior the model must learn.
    """
    kept = [t for t in traces if any(s.observation.content.strip() for s in t.steps)]
    return kept, len(traces) - len(kept)

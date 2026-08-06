"""Leak-free demo retrieval shared by GEPA optimization and replay evaluation.

Both replay-style evaluations (`wmo.simulation.model.replay`) and GEPA's held-out scoring
(`wmo.optimize.gepa`) must retrieve the demos the serving world model would use, WITHOUT letting a
held-out step see its own answer. The rule is identical in both places, so it lives here once:

  - retrieve only from a TRAIN corpus (never the held-out/val steps);
  - never return a demo from the query step's OWN trace (which would surface the very observation we
    are asking the model to predict, or an adjacent step that gives it away).

Implementation note: same-trace exclusion keys on object identity (`id`). That is correct here
because the corpus `Step`s flow straight from `retriever.index(corpus)` into `retriever.topk(...)`
in-process — the exact same objects. Do NOT build a `DemoRetriever` over a retriever that was
reloaded from disk (`EmbeddingRetriever.load`), as that reconstructs fresh `Step`s whose identities
won't match the corpus and the exclusion would silently no-op.
"""

from __future__ import annotations

from wmo.common.core.types import Step, Trace
from wmo.simulation.retrieval.retriever import Retriever

# Over-fetch margin so that, after dropping same-trace demos, we can still fill top_k.
_SLACK = 5


class DemoRetriever:
    """Per-step leak-free top-k retrieval over an in-memory train corpus.

    Construct once over the train traces (it indexes them and records each step's origin trace),
    then call `demos_for(trace_id, step)` per held-out step. When `retriever` is None or the corpus
    is empty, it yields no demos (zero-shot) — callers get the same interface either way.
    """

    def __init__(self, retriever: Retriever | None, corpus: list[Trace], *, top_k: int = 5) -> None:
        self._retriever = retriever
        self._top_k = top_k
        self._enabled = retriever is not None and any(t.steps for t in corpus)
        if self._enabled:
            assert retriever is not None  # for type-checkers; guarded by _enabled
            retriever.index(corpus)
            # id(step) -> originating trace_id, to exclude a query's own trace from its demos.
            self._origin: dict[int, str] = {id(s): t.trace_id for t in corpus for s in t.steps}

    def demos_for(self, trace_id: str, step: Step) -> list[Step]:
        """Return up to top_k demos for `step`, excluding any from `trace_id`'s own trace."""
        if not self._enabled:
            return []
        assert self._retriever is not None  # guarded by _enabled
        # Over-fetch, drop same-trace demos, then take top_k.
        candidates = self._retriever.topk(step.state_before, step.action, self._top_k + _SLACK)
        return [d for d in candidates if self._origin.get(id(d)) != trace_id][: self._top_k]

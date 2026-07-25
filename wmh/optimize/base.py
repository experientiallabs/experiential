"""The pluggable optimizer interface: one Protocol, switchable implementations.

Every optimizer, whatever it produces, satisfies `Optimizer` and returns an `OptimizeResult`.
What differs between families is the ARTIFACT each one emits, carried as typed `ArtifactRef`s:

- prompt evolution (GEPA, `wmh.optimize.gepa`): the winning env prompt (also kept in
  `OptimizeResult.prompt`, the field the build pipeline serves);
- routing optimization: a `routing_policy` artifact (per-cluster model assignment served by the
  OpenAI-compatible endpoint);
- training-type optimizers (e.g. CoT distillation into a smaller model, future): `model_weights`
  and `adapter` references. The plumbing exists so they fit; nothing builds them yet.

Kept import-light (pydantic + core types only) so any subsystem can consume the interface
without pulling in an optimizer's machinery.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmh.core.types import JsonObject, Trace

# The artifact families the optimizer switch must carry. Adding a family here (and nowhere
# else) is the extension point for a new optimizer type.
ArtifactKind = Literal["prompt", "routing_policy", "model_weights", "adapter"]


class ArtifactRef(BaseModel):
    """A typed reference to something an optimizer produced.

    `path` is a filesystem path or URI to the persisted artifact; None for artifacts carried
    inline on the result (the GEPA prompt). `metadata` holds small artifact-specific facts
    (adapter rank, checkpoint step, policy version), not the artifact itself.
    """

    kind: ArtifactKind
    path: str | None = None
    metadata: JsonObject = Field(default_factory=dict)


class OptimizeMetrics(BaseModel):
    """Outcome metrics from an optimization run."""

    # Mean judge score on the SELECTION data, measured on the path that decided the run: GEPA's
    # search-time valset aggregate when base won the search; the fresh paired re-check mean (over
    # the valset, or the `recheck` set when supplied) when a non-base winner was accepted or
    # reverted; the HARD-subset mean under `select_on_hard` when base won. Comparable across runs
    # only at a fixed configuration - report a separate test-split score for cross-run numbers.
    held_out_accuracy: float = 0.0
    rollouts_used: int = 0  # search metric calls + the acceptance re-check's paired passes
    # True when the search's winning candidate LOST the fresh base-vs-winner acceptance re-check
    # and the base prompt was restored (see `GEPAOptimizer.optimize`). Surfaced so research runs
    # can report how often GEPA's search wins fail to survive an independent evaluation.
    reverted_to_base: bool = False
    # Reserved: judge self-consistency / human-agreement proxy. Populating it needs repeated or
    # independent judging (not yet implemented); `None` until then so it never reads as a real 0.0.
    judge_agreement: float | None = None


class OptimizeResult(BaseModel):
    prompt: str  # winning specialized env prompt ("" for optimizers that don't evolve prompts)
    frontier: list[str] = Field(default_factory=list)  # Pareto candidates
    metrics: OptimizeMetrics = Field(default_factory=OptimizeMetrics)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


@runtime_checkable
class Optimizer(Protocol):
    def optimize(
        self,
        train: list[Trace],
        test: list[Trace],
        base_prompt: str,
        budget: int,
        *,
        rag_corpus: list[Trace] | None = None,
    ) -> OptimizeResult: ...

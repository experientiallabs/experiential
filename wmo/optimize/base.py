"""The pluggable optimizer interface: one Protocol, switchable implementations.

Every optimizer, whatever it produces, satisfies `Optimizer` and returns an `OptimizeResult`.
What differs between families is the ARTIFACT each one emits, carried as typed `ArtifactRef`s:

- prompt evolution (GEPA, `wmo.optimize.gepa`): the winning env prompt (also kept in
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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wmo.core.types import JsonObject, Trace

# The artifact families the optimizer switch must carry. Adding a family here (and nowhere
# else) is the extension point for a new optimizer type.
ArtifactKind = Literal["prompt", "routing_policy", "model_weights", "adapter"]

# Ownership boundary (pricing/commercial contract, carried as type shape): artifacts trained
# on a customer's traces (prompts, checkpoints, adapters) are customer-owned and exportable;
# routing policies encode OUR serving intelligence and are never exportable, whatever a caller
# passes. Enforced by `ArtifactRef._enforce_exportability`.
_NEVER_EXPORTABLE: frozenset[str] = frozenset({"routing_policy"})


class ArtifactProvenance(BaseModel):
    """Where an artifact came from: enough to audit, reproduce, and hand over ownership.

    Required on exportable artifacts (a customer-owned checkpoint without provenance is not
    auditable); optional on internal ones. `trace_sha256` fingerprints the training corpus
    without carrying it; `config` holds the small reproducibility facts (seed, budget,
    hyperparameters), not datasets.
    """

    optimizer: str  # e.g. "gepa", "route-fit"
    base_model: str = ""  # the model the artifact was trained from/for ("" when N/A)
    trace_count: int = 0
    trace_sha256: str = ""  # sha256 over the ordered training trace ids ("" when N/A)
    created_at: str = ""  # ISO timestamp, stamped by the producing optimizer
    config: JsonObject = Field(default_factory=dict)


class ArtifactRef(BaseModel):
    """A typed reference to something an optimizer produced.

    `path` is a filesystem path or URI to the persisted artifact; None for artifacts carried
    inline on the result (the GEPA prompt). `metadata` holds small artifact-specific facts
    (adapter rank, checkpoint step, policy version), not the artifact itself. `exportable`
    marks whether the artifact may leave the platform (customer-owned checkpoints/adapters:
    yes, with `provenance`; routing policies: never - forced False regardless of input).

    `validate_assignment` re-runs the exportability validator on every field write, so
    `ref.exportable = True` cannot flip the boundary after construction. Two documented ways
    around it remain, both by pydantic design and neither reachable by accident:
    `model_copy(update=...)` writes fields without validating, and `model_construct` skips
    validation entirely. Anything crossing a serialization boundary is re-normalized, since
    `model_validate`/`model_validate_json` run the validator again.
    """

    model_config = ConfigDict(validate_assignment=True)

    kind: ArtifactKind
    path: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
    exportable: bool = True
    provenance: ArtifactProvenance | None = None

    @model_validator(mode="after")
    def _enforce_exportability(self) -> ArtifactRef:
        # The guard on `self.exportable` is what makes this terminate under
        # validate_assignment: the corrective write re-enters the validator exactly once, and
        # the second pass has nothing left to correct.
        if self.kind in _NEVER_EXPORTABLE and self.exportable:
            self.exportable = False
        return self


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
    # The base-vs-winner acceptance re-check's fresh paired scores (see `GEPAOptimizer.optimize`):
    # `base_fresh` is the base prompt's mean judge score and `best_fresh` the search's winning
    # candidate's mean judge score, both measured in the SAME pass over the SAME (ideally
    # selection-disjoint) sample - the exact comparison the revert decision is made from. This is
    # the answer to "did GEPA actually help", which the boolean `reverted_to_base` alone cannot
    # give: `reverted_to_base=False` is ambiguous between "the winner beat base and was kept" and
    # "GEPA's own search already picked base, nothing to re-check" and "GEPA never ran at all"
    # (`--fidelity low`, budget<=0). `base_fresh`/`best_fresh` stay `None` (never a misleading 0.0)
    # in the two cases where no fresh comparison happened - a `--fidelity low` build and a GEPA
    # pass whose search-time winner was already the base prompt - and are both populated (even
    # when they tie, `fresh_delta == 0.0`) whenever the acceptance re-check actually ran.
    base_fresh: float | None = None
    best_fresh: float | None = None
    # `best_fresh - base_fresh`: positive means the accepted/rejected winner beat base on the
    # fresh sample; `None` exactly when `base_fresh`/`best_fresh` are `None`.
    fresh_delta: float | None = None
    # Whether the fresh comparison above actually ran on data disjoint from GEPA's selection
    # valset (a caller-supplied `recheck` set that yielded real steps), or fell back to re-scoring
    # the selection valset itself (no `recheck` given, or it flattened to zero steps - e.g. the
    # validation split was too small to leave anything past the selection cap). A selection-set
    # comparison tests selection, not generalization: without this flag, a build whose corpus
    # forced the fallback would persist `base_fresh`/`best_fresh`/`fresh_delta` in the exact same
    # shape as a genuinely held-out measurement, so a reader auditing the artifact could not tell
    # the two apart. `None` exactly when `base_fresh`/`best_fresh` are `None` (no fresh comparison
    # happened at all); `True`/`False` whenever the acceptance re-check ran.
    fresh_recheck_disjoint: bool | None = None
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


@runtime_checkable
class ResumableOptimizer(Protocol):
    """Continual-learning hook: extend a prior artifact with newly ingested traces.

    Type shape only for now - no optimizer implements it yet. The contract: `prior` is an
    `ArtifactRef` a previous `optimize`/`resume` run produced (its provenance identifies the
    corpus already learned from), `new_traces` is the increment, and the result's artifacts
    supersede `prior`. Optimizers that cannot warm-start simply never claim this protocol,
    and callers fall back to a full `optimize`.
    """

    def resume(
        self,
        prior: ArtifactRef,
        new_traces: list[Trace],
        budget: int,
    ) -> OptimizeResult: ...

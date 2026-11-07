"""Bounds of one target's Astra refinement, reserved whole on the cumulative front-door ledger.

For ``N`` calibration examples (``H`` held out), ``K`` rounds and ``L`` labeled cases (judge
targets only), the loop makes at most:

- Astra: ``N`` references, ``N`` judgments of the code prompt, and per round one rewrite and ``N``
  judgments: ``2N + K(N + 1)`` draws;
- the runtime model: ``N`` draws of the code prompt, ``N`` per round and ``L`` gate draws per
  round: ``N(K + 1) + LK`` draws;
- the second reviewer: ``H`` held-out judgments per evaluation: ``H(K + 1)`` draws.

Each draw may use its settings' ``maximum_physical_attempts``. Every attempt of every model is
reserved on the ledger's attempt allowance (the measured settlement counts every recorded
attempt, so the reservation counts them all); the priced models' dollars are also reserved on its
dollar ceiling, at the catalog's trusted whole-generation bound
(:func:`~exp.optimize.evaluation.executable.model_costs.model_cost`). The plan also shows a
declared-size bound: every attempt at the full ``maximum_input_bytes`` of input (a token is at
least one byte) and the full ``output_tokens``, at the catalog's per-token rates.
"""

from __future__ import annotations

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.astra_prompt_contracts import RefinementSettings
from exp.optimize.evaluation.variance.frontdoor_ledger import PhaseLedger
from exp.runtime.models.registry import RuntimeModelCatalog

DEFAULT_EXAMPLES = 24
"""Default calibration size per target for the dry run."""

DEFAULT_HELDOUT = 8
"""Default held-out examples of the default calibration size."""


class ModelBound(ContractModel):
    """One model's share of a target's bound.

    Attributes:
        alias: The model alias.
        draws: Forced draws at most.
        attempts: Physical attempts at most.
        reserved_usd: The trusted dollar bound, or None when the alias is unpriced.
        declared_usd: The declared-size dollar bound, or None when unpriced.
    """

    alias: str
    draws: int = Field(ge=0)
    attempts: int = Field(ge=0)
    reserved_usd: float | None = None
    declared_usd: float | None = None


class TargetBound(ContractModel):
    """One target's whole-loop bound.

    Attributes:
        target_id: The target.
        examples: Calibration examples.
        heldout: Held-out examples.
        rounds: Rewrite rounds at most.
        gate_cases: Labeled cases per round.
        models: Per model: Astra, the runtime model and the reviewer.
    """

    target_id: str
    examples: int
    heldout: int
    rounds: int
    gate_cases: int
    models: tuple[ModelBound, ...]

    @property
    def attempts(self) -> int:
        """Every attempt of every model, reserved on the attempt allowance."""
        return sum(row.attempts for row in self.models)

    @property
    def reserved_usd(self) -> float:
        """The priced models' trusted dollar bound."""
        return sum(row.reserved_usd or 0.0 for row in self.models)

    @property
    def declared_usd(self) -> float:
        """The priced models' declared-size dollar bound."""
        return sum(row.declared_usd or 0.0 for row in self.models)


def draw_counts(examples: int, heldout: int, rounds: int, gate_cases: int) -> tuple[int, int, int]:
    """Astra, runtime and reviewer draws of one target's loop at most."""
    astra = 2 * examples + rounds * (examples + 1)
    runtime = examples * (rounds + 1) + gate_cases * rounds
    reviewer = heldout * (rounds + 1)
    return astra, runtime, reviewer


def _model(project: ProjectStore, settings: AuthorSettings, draws: int) -> ModelBound:
    """Price one model's draws."""
    (row,) = evaluation_costs(project, settings)
    scaled = row.scaled(draws)
    catalog = load_model_catalog(project.model_catalog_path)
    _, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(settings.model_alias)
    rate_in, rate_out = (
        caps.input_cost_per_million_tokens_usd,
        caps.output_cost_per_million_tokens_usd,
    )
    declared = None
    if scaled.usd is not None and rate_in is not None and rate_out is not None:
        per_attempt = (
            settings.maximum_input_bytes * rate_in + settings.output_tokens * rate_out
        ) / 1_000_000
        declared = per_attempt * scaled.attempts
    return ModelBound(
        alias=settings.model_alias,
        draws=draws,
        attempts=scaled.attempts,
        reserved_usd=scaled.usd,
        declared_usd=declared,
    )


def target_bound(
    project: ProjectStore,
    target_id: str,
    settings: RefinementSettings,
    *,
    examples: int,
    heldout: int,
    gate_cases: int,
) -> TargetBound:
    """Price one target's whole loop."""
    astra, runtime, reviewer = draw_counts(examples, heldout, settings.maximum_rounds, gate_cases)
    return TargetBound(
        target_id=target_id,
        examples=examples,
        heldout=heldout,
        rounds=settings.maximum_rounds,
        gate_cases=gate_cases,
        models=(
            _model(project, settings.astra, astra),
            _model(project, settings.runtime, runtime),
            _model(project, settings.reviewer, reviewer),
        ),
    )


def reserve_target(ledger: PhaseLedger | None, phase: str, bound: TargetBound) -> str | None:
    """Reserve a target's bound (every attempt, and the priced dollars), or say why not."""
    if ledger is None:
        return None
    return ledger.admit(phase, bound.reserved_usd, bound.attempts)


def plan_lines(bounds: dict[str, list[TargetBound]]) -> list[str]:
    """Dry-run lines per role and in total: draws, attempts and both dollar bounds."""
    lines: list[str] = []
    total_attempts, total_reserved, total_declared = 0, 0.0, 0.0
    for role, rows in bounds.items():
        attempts = sum(row.attempts for row in rows)
        reserved = sum(row.reserved_usd for row in rows)
        declared = sum(row.declared_usd for row in rows)
        total_attempts += attempts
        total_reserved += reserved
        total_declared += declared
        draws = {
            alias: sum(m.draws for row in rows for m in row.models if m.alias == alias)
            for alias in dict.fromkeys(m.alias for row in rows for m in row.models)
        }
        shown = ", ".join(f"{alias} {count} draws" for alias, count in draws.items())
        lines.append(
            f"{role}: {len(rows)} targets; {shown}; up to {attempts} attempts; Astra USD bound "
            f"${declared:,.2f} at declared sizes, ${reserved:,.2f} reserved (trusted bound)"
        )
    lines.append(
        f"total: up to {total_attempts} attempts on the cumulative allowance; Astra USD "
        f"${total_declared:,.2f} at declared sizes, ${total_reserved:,.2f} reserved"
    )
    return lines

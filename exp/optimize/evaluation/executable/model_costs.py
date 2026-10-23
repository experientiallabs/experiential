"""Credential-free per-role cost components: catalog pricing primitives and fidelity runs.

The pricing primitives import no runtime module, so the learning and audit commands price their
roles without loading the executable inventories in :mod:`exp.optimize.evaluation.executable.costs`.
"""

import math
from dataclasses import dataclass

from exp.common.models import load_model_catalog
from exp.common.models.pricing import completion_cost_reservation
from exp.common.models.token_cost import schedule_maximum_cost_nano_usd
from exp.common.project import ProjectStore
from exp.optimize.evaluation.fidelity.retrieval_binding import index_embedder_alias
from exp.optimize.evaluation.fidelity.runner import FidelitySpec
from exp.optimize.evaluation.spending import DECISION_PROVIDER
from exp.runtime.gateway.decisions_contracts import (
    MAX_DECISION_CRITERIA,
    MAX_DECISION_INPUT_BYTES,
    MAX_DECISION_QUESTIONS,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact


@dataclass(frozen=True)
class CostComponent:
    """One finite schedule component, or an explicitly unpriced dependency.

    Attributes:
        name: Role or dependency displayed before consent.
        attempts: Maximum physical requests included in this component.
        usd: Known-rate planning estimate; None when necessary prices or bounds are unavailable.
    """

    name: str
    attempts: int
    usd: float | None

    def scaled(self, count: int) -> "CostComponent":
        """Repeat an unchanged bounded schedule without assuming saved work is free."""
        return CostComponent(
            self.name, self.attempts * count, None if self.usd is None else self.usd * count
        )


def model_cost(
    project: ProjectStore,
    alias: str,
    attempts: int,
    output: int,
    *,
    input_tokens: int | None = None,
) -> CostComponent:
    """Price known tariff dimensions at full input capacity without resolving credentials."""
    catalog = load_model_catalog(project.model_catalog_path)
    model, caps = RuntimeModelCatalog(catalog, environment={}).snapshot(alias)
    if (
        output
        and catalog.connections[catalog.models[alias].connection].provider != DECISION_PROVIDER
    ):
        # Use the same trusted whole-generation bound as physical admission. Requested
        # visible output limits need not include provider-side reasoning, even at effort none.
        if caps.context_window_tokens is None:
            return CostComponent(alias, attempts, None)
        output = max(output, caps.context_window_tokens)
    context, input_rate, output_rate = (
        caps.context_window_tokens if input_tokens is None else input_tokens,
        caps.input_cost_per_million_tokens_usd,
        caps.output_cost_per_million_tokens_usd,
    )
    if context is None or input_rate is None or (output and output_rate is None):
        return CostComponent(alias, attempts, None)
    if output == 0:
        return CostComponent(alias, attempts, attempts * context * input_rate / 1_000_000)
    tariff = catalog.models[alias].token_prices
    if tariff is not None:
        return CostComponent(
            alias,
            attempts,
            attempts
            * schedule_maximum_cost_nano_usd(tariff, input_tokens=context, output_tokens=output)
            / 1_000_000_000,
        )
    reservation = completion_cost_reservation(
        model=model,
        input_usd_per_million_tokens=input_rate,
        output_usd_per_million_tokens=output_rate or 0,
        cached_input_usd_per_million_tokens=caps.cached_input_cost_per_million_tokens_usd
        or input_rate,
        cache_write_usd_per_million_tokens=caps.cache_write_cost_per_million_tokens_usd
        or input_rate,
        maximum_attempts=attempts,
        maximum_input_tokens=context,
        maximum_output_tokens=output,
    )
    return CostComponent(alias, attempts, reservation.estimated_maximum_call_cost_usd)


def decision_cost(project: ProjectStore, alias: str, attempts: int) -> CostComponent:
    """Price the decision wire's repeated context and answer allowance, not an embedding.

    Every question can repeat the whole bounded input. These are accounting allowances;
    the decision provider has no client-set output-token ceiling. Unknown output prices
    remain unknown even when a deployment usually returns only a short answer.
    """
    return model_cost(
        project,
        alias,
        attempts,
        MAX_DECISION_QUESTIONS * (1024 + 512 * MAX_DECISION_CRITERIA),
        input_tokens=MAX_DECISION_QUESTIONS * (MAX_DECISION_INPUT_BYTES + 1024),
    )


def fidelity_costs(project: ProjectStore, spec: FidelitySpec) -> tuple[CostComponent, ...]:
    """Price one fidelity replay's roles, each at the run's whole physical-attempt ceiling.

    The predictor serves predictions, investigations, evidence reading and dreaming at the
    larger of its two output budgets; the judge runs only when the spec is judged; retrieval
    adds query embeddings and native recall decisions. Like the grounded runtime, every role
    shares one ledger, so pricing each at the ceiling cannot omit an expensive role.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    world = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    attempts = spec.maximum_physical_attempts
    output = max(
        spec.budget("prediction").output_tokens, spec.budget("investigation").output_tokens
    )
    components = [model_cost(project, world.model_alias, attempts, output)]
    if spec.judged:
        components.append(
            model_cost(project, spec.judge_alias, attempts, spec.budget("judging").output_tokens)
        )
    embedder = index_embedder_alias(project, catalog, world)
    if embedder is not None:
        components.append(model_cost(project, embedder, attempts, 0))
    if spec.retrieval is not None:
        components.append(decision_cost(project, spec.retrieval.jev_alias, attempts))
    return tuple(components)


def total_cost(components: tuple[CostComponent, ...]) -> float | None:
    """Keep an incomplete inventory explicitly unknown instead of reporting a partial total."""
    if any(row.usd is None for row in components):
        return None
    return math.fsum(row.usd for row in components if row.usd is not None)

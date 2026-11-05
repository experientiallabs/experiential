"""Canonical admitted System One client composition for rule-memory research."""

from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.tagging import DecisionCache


def rule_memory_decisions(
    project: ProjectStore, stage: RulesStageSpec, transport: AdmittedJsonTransport
) -> DecisionsTraceTagger:
    """Bind the configured native decision alias, admitted transport and shared cache.

    Args:
        project: Owner of the model catalog and shared decision cache.
        stage: Frozen rule-memory profile and concurrency.
        transport: Bounded physical transport already attached to the operation ceiling.

    Returns:
        The native System One collaborator used by rule-memory research.

    Raises:
        ValueError: Memory settings or operation accounting are absent.
        TraceTaggingError: The configured alias has no native decisions wire.
    """
    if stage.profile == "jev_memory" and stage.memory is not None:
        alias = stage.memory.decision_alias
    elif stage.profile == "trajectory_tags" and stage.tagging is not None:
        alias = stage.tagging.decision_alias
    else:
        raise ValueError("System One construction requires a registered decisions profile")
    if transport.spend_ceiling is None:
        raise ValueError("System One transport requires its admitted operation spend ceiling")
    catalog = load_model_catalog(project.model_catalog_path)
    priced = transport.priced_as(alias)
    resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: priced).resolve(alias)
    return DecisionsTraceTagger.for_alias(
        resolved,
        transport=priced,
        cache=DecisionCache(project.paths),
        maximum_concurrency=stage.concurrency,
    )

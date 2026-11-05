"""Native variance decision clients bind physical receipts and an explicit operation stop."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.build_records import VarianceBuildPlan, VarianceBuildSpec
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.tagging import DecisionCache


def _admitted_tagger(
    project: ProjectStore,
    spec: VarianceBuildSpec,
    summary: VarianceBuildPlan,
    code_revision: str,
    *,
    execution_deadline: datetime | None = None,
    before_claim: Callable[[], None] | None = None,
) -> DecisionsTraceTagger:
    """Resolve the decision alias and post its decisions through the admitted transport.

    Every physical attempt is claimed in this run's dispatch ledger and, under a consented
    operation ceiling, reserves its cost priced as the decision alias.

    Args:
        project: Project owning the catalog, cache and receipts.
        spec: The registration.
        summary: The plan whose identity names the ledger and whose attempts bound it.
        code_revision: Producing source identity, recorded on attempt receipts.

    Returns:
        A tagger over the project's decision cache.

    Raises:
        TraceTaggingError: The alias is not a native decision alias.
        ValueError: Its resolved identity differs from the summarized one.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog).resolve(spec.decision_alias)
    if resolved.snapshot != summary.decision_model:
        raise ValueError(f"decision alias {spec.decision_alias!r} changed since it was planned")
    identity = summary.identity
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + "-physical.sqlite"),
            max(1, summary.decision_attempts),
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=code_revision,
        deadline=execution_deadline
        or datetime.now(UTC) + timedelta(hours=spec.tagging_deadline_hours),
        before_claim=before_claim,
    ).priced_as(spec.decision_alias)
    return DecisionsTraceTagger.for_alias(
        resolved,
        transport=transport,
        cache=DecisionCache(project.paths),
        maximum_concurrency=spec.concurrency,
    )

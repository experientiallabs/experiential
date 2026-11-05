"""Compose required preparation collaborators inside the admitted original operation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from exp.common.core.artifacts import sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.rules.decision_client import rule_memory_decisions
from exp.optimize.evaluation.spending import operation_directory
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapSettings
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.spend_ceiling import current_spend_ceiling
from exp.runtime.sessions.dispatch_ledger import DispatchLedger

if TYPE_CHECKING:
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapDependencies, BootstrapPlan


def bootstrap_decisions(
    project: ProjectStore,
    settings: BootstrapSettings,
    plan: BootstrapPlan,
    dependencies: BootstrapDependencies,
) -> BootstrapDependencies:
    """Fill the required System One collaborator before any bootstrap author can dispatch.

    Args:
        project: Owner of the original ledger, catalog and decision cache.
        settings: Verified execution settings, including the finite rules deadline.
        plan: Native full pending inventory under the original operation.
        dependencies: Explicit caller collaborators, otherwise the admitted native defaults.

    Returns:
        Existing collaborators plus a native System One client when pending families need it.

    Raises:
        ValueError: The original spend scope or planned decision allowance is missing.
    """
    stage = settings.rules
    if (
        stage is None
        or stage.profile not in {"jev_memory", "trajectory_tags"}
        or plan.family_bound == 0
        or dependencies.decisions is not None
    ):
        return dependencies
    if stage.profile == "trajectory_tags":
        assert stage.tagging is not None
        decision_alias = stage.tagging.decision_alias
    else:
        assert stage.memory is not None
        decision_alias = stage.memory.decision_alias
    ceiling = current_spend_ceiling()
    expected = operation_directory(project, plan.identity) / "ledger.sqlite"
    if ceiling is None or ceiling.path.resolve() != expected.resolve():
        raise ValueError("bootstrap System One requires the admitted original operation scope")
    attempts = sum(row.attempts for row in plan.costs if row.name == decision_alias)
    if attempts < 1:
        raise ValueError("bootstrap System One has no full pending native decision allowance")
    purpose = f"{plan.identity}-rules-decisions"
    path = project.artifacts.project_directory / f"{purpose}.sqlite"
    ledger = (
        DispatchLedger.reopen(path, sha256_json(purpose))
        if path.exists()
        else DispatchLedger(path, attempts, sha256_json(purpose))
    )
    if attempts > ledger.cap:
        raise ValueError("bootstrap pending decisions exceed the registered immutable dispatch cap")
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=ledger,
        artifacts=project.artifacts,
        code_revision=plan.code_revision,
        deadline=stage.author.deadline,
        spend_ceiling=ceiling,
    )
    return replace(dependencies, decisions=rule_memory_decisions(project, stage, transport))

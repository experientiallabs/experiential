"""Optimizer-owned project activation for frozen automatic router policies."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from openai import OpenAI

from wmo.common.core.artifacts import ArtifactId
from wmo.common.evaluations.evidence import read_evaluation_plan
from wmo.common.project import ArtifactStore, artifact_input
from wmo.common.routing import KnnRouterPolicy
from wmo.optimize.router.router_execution_contract import load_router_execution_contract
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.application import (
    RouterApplicationError,
)
from wmo.runtime.router.application import (
    load_project_router as load_runtime_project_router,
)
from wmo.runtime.router.application import load_router as load_runtime_router
from wmo.runtime.router.capability import (
    load_router_runtime_capability_contract,
    verify_router_runtime_capabilities,
)
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime


def load_project_router(
    project: str,
    root: Path,
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
) -> RouterRuntime:
    """Load a project router with complete automatic artifact verification.

    Args:
        project: Canonical local project identifier.
        root: Local ``.wmo`` root containing project artifacts and model configuration.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping used during runtime client construction.
        runtime_catalog: Optional explicit runtime catalog for deterministic applications.
        decision_sink: Optional aggregate-safe routing-decision recorder.

    Returns:
        Activated immutable router runtime without issuing a model request.

    Raises:
        RouterApplicationError: Project, policy, catalog, or automatic inputs are invalid.
    """
    return load_runtime_project_router(
        project,
        root,
        policy_id=policy_id,
        environment=environment,
        runtime_catalog=runtime_catalog,
        decision_sink=decision_sink,
        policy_verifier=verify_automatic_router_policy,
    )


def load_router(
    project: str,
    root: Path = Path(".wmo"),
    *,
    policy_id: ArtifactId | None = None,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
    decision_sink: DecisionSink | None = None,
    ghost: bool = False,
) -> OpenAI:
    """Load a verified project router as an official synchronous OpenAI client.

    Args:
        project: Canonical project identifier and public OpenAI model name.
        root: Local ``.wmo`` root containing project artifacts and model configuration.
        policy_id: Optional exact policy identity when the project contains several.
        environment: Optional credential mapping used during runtime client construction.
        runtime_catalog: Optional explicit runtime catalog for deterministic applications.
        decision_sink: Optional aggregate-safe routing-decision recorder.
        ghost: Whether completed traffic must bypass durable journal and replay state.

    Returns:
        Official OpenAI client backed by the verified local router and selected traffic mode.

    Raises:
        RouterApplicationError: Project, policy, catalog, or automatic inputs are invalid.
    """
    return load_runtime_router(
        project,
        root,
        policy_id=policy_id,
        environment=environment,
        runtime_catalog=runtime_catalog,
        decision_sink=decision_sink,
        policy_verifier=verify_automatic_router_policy,
        ghost=ghost,
    )


def verify_automatic_router_policy(
    store: ArtifactStore,
    policy: KnnRouterPolicy,
    catalog: RuntimeModelCatalog,
) -> None:
    """Verify automatic execution and capability inputs before credential access.

    Plans without automatic artifacts remain eligible for the provider-free offline path.
    Automatic plans must bind exactly one execution contract and one runtime capability contract,
    and both contracts must agree with the selected policy and credential-free catalog snapshot.

    Args:
        store: Project-local immutable artifact store.
        policy: Selected frozen router policy.
        catalog: Current credential-free runtime model catalog.

    Raises:
        RouterApplicationError: Automatic artifacts are missing, ambiguous, or drifted.
    """
    plan, _plan_input = read_evaluation_plan(store, policy.evaluation_plan_id)
    capability_inputs = []
    execution_inputs = []
    for item in plan.inputs:
        stored = store.read(item.artifact_id)
        if artifact_input(stored.manifest) != item:
            raise RouterApplicationError(
                f"router plan input {item.artifact_id!r} differs from its manifest"
            )
        if stored.manifest.artifact_type == "router-runtime-capabilities":
            capability_inputs.append(item)
        elif stored.manifest.artifact_type == "router-execution-contract":
            execution_inputs.append(item)
    if not capability_inputs and not execution_inputs:
        return
    if len(capability_inputs) != 1 or len(execution_inputs) != 1:
        raise RouterApplicationError(
            "automatic router plan must bind one capability and one execution contract"
        )
    capability_input = capability_inputs[0]
    execution = load_router_execution_contract(store, execution_inputs[0].artifact_id)
    if execution.runtime_capability_input != capability_input:
        raise RouterApplicationError(
            "automatic router execution contract differs from its runtime capability binding"
        )
    execution_candidates = tuple(
        (item.candidate_alias, item.model) for item in execution.candidates
    )
    policy_candidates = tuple((item.alias, item.model) for item in policy.candidates)
    if (
        execution_candidates != policy_candidates
        or execution.incumbent_alias != policy.baseline_alias
    ):
        raise RouterApplicationError(
            "automatic router execution candidates or incumbent differ from the policy"
        )
    contract = load_router_runtime_capability_contract(store, capability_input.artifact_id)
    verify_router_runtime_capabilities(contract, policy.candidates, catalog)

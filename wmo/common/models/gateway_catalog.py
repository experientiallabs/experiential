"""Secret-free gateway deployment views derived from the authored model catalog."""

from __future__ import annotations

from pydantic import Field, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel, Sha256, sha256_json
from wmo.common.models.catalog import (
    GatewayDeploymentMetadata,
    ModelCatalog,
)
from wmo.common.models.model import ModelAlias, ModelCapabilities

ExactModelId = ArtifactId
DeploymentId = ArtifactId
ExactModelPoolId = ArtifactId


class ExactModelDeployment(ContractModel):
    """One callable catalog model with explicit provider and exact-model identity."""

    deployment_id: DeploymentId
    source_alias: ModelAlias
    exact_model_id: ExactModelId
    connection: ArtifactId
    provider: str = Field(min_length=1, max_length=128)
    provider_model: str = Field(min_length=1, max_length=2_048)
    revision: str | None = Field(default=None, max_length=256)
    connection_sha256: Sha256
    capabilities_sha256: Sha256
    capabilities: ModelCapabilities | None = None
    gateway: GatewayDeploymentMetadata = Field(default_factory=GatewayDeploymentMetadata)


class ExactModelPool(ContractModel):
    """An ordered set of deployments certified as one exact logical model."""

    pool_id: ExactModelPoolId
    exact_model_id: ExactModelId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _require_unique_deployments(self) -> ExactModelPool:
        """Reject repeated routes inside one operational pool.

        Returns:
            The validated exact-model pool.

        Raises:
            ValueError: A deployment appears more than once.
        """
        if len(set(self.deployment_ids)) != len(self.deployment_ids):
            raise ValueError("exact-model pool deployments must not repeat")
        return self


class NormalizedGatewayCatalog(ContractModel):
    """Immutable gateway deployment and singleton-pool view of one model catalog."""

    schema_version: int = Field(default=1, ge=1)
    deployments: tuple[ExactModelDeployment, ...] = ()
    pools: tuple[ExactModelPool, ...] = ()

    @model_validator(mode="after")
    def _require_closed_pool_references(self) -> NormalizedGatewayCatalog:
        """Require unique records and pool references to matching exact models.

        Returns:
            The validated normalized catalog.

        Raises:
            ValueError: Deployment or pool identifiers repeat, or a pool reference is invalid.
        """
        by_id = {item.deployment_id: item for item in self.deployments}
        if len(by_id) != len(self.deployments):
            raise ValueError("gateway deployment IDs must be unique")
        pool_ids = tuple(item.pool_id for item in self.pools)
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("exact-model pool IDs must be unique")
        for pool in self.pools:
            for deployment_id in pool.deployment_ids:
                deployment = by_id.get(deployment_id)
                if deployment is None:
                    raise ValueError(
                        f"exact-model pool {pool.pool_id!r} names unknown deployment "
                        f"{deployment_id!r}"
                    )
                if deployment.exact_model_id != pool.exact_model_id:
                    raise ValueError(
                        f"exact-model pool {pool.pool_id!r} contains deployment "
                        f"{deployment_id!r} for another exact model"
                    )
        return self

    def identity_sha256(self) -> Sha256:
        """Return the deterministic digest pinned by a later gateway activation."""
        return sha256_json(self)


def normalize_gateway_catalog(catalog: ModelCatalog) -> NormalizedGatewayCatalog:
    """Derive safe singleton deployments from legacy authored model records.

    Every eligible model alias becomes its own deployment and singleton pool. Derived exact-model
    identity includes normalized connection identity, provider model and revision, and the full
    capability declaration digest. Separate aliases remain separate pools even when those inputs
    are identical. Tinker and SFT sampling handles are intentionally excluded because they are
    training-run provenance, not generally callable gateway deployments.

    Args:
        catalog: Validated authored provider and model catalog.

    Returns:
        Deterministically ordered deployment and singleton-pool records.
    """
    deployments: list[ExactModelDeployment] = []
    pools: list[ExactModelPool] = []
    for alias, record in sorted(catalog.models.items()):
        connection = catalog.connections[record.connection]
        if connection.provider == "tinker" or record.sft_provenance is not None:
            continue
        capabilities_sha256 = _capability_declaration_sha256(record.capabilities)
        exact_model_id = (
            record.gateway.exact_model_id
            if record.gateway is not None and record.gateway.exact_model_id is not None
            else _singleton_exact_model_id(
                connection_sha256=connection.identity_sha256(),
                provider_model=record.model,
                revision=record.revision,
                capabilities_sha256=capabilities_sha256,
            )
        )
        deployment = ExactModelDeployment(
            deployment_id=alias,
            source_alias=alias,
            exact_model_id=exact_model_id,
            connection=record.connection,
            provider=connection.provider,
            provider_model=record.model,
            revision=record.revision,
            connection_sha256=connection.identity_sha256(),
            capabilities_sha256=capabilities_sha256,
            capabilities=record.capabilities,
            gateway=record.gateway or GatewayDeploymentMetadata(),
        )
        deployments.append(deployment)
        pools.append(
            ExactModelPool(
                pool_id=alias,
                exact_model_id=exact_model_id,
                deployment_ids=(deployment.deployment_id,),
            )
        )
    return NormalizedGatewayCatalog(
        deployments=tuple(deployments),
        pools=tuple(pools),
    )


def _capability_declaration_sha256(capabilities: ModelCapabilities | None) -> Sha256:
    """Hash the full catalog declaration without changing frozen capability identity.

    Args:
        capabilities: Existing authored capabilities, or ``None`` when undeclared.

    Returns:
        Digest used only by singleton gateway migration identity.
    """
    return sha256_json(
        None if capabilities is None else capabilities.model_dump(mode="json", exclude_none=False)
    )


def _singleton_exact_model_id(
    *,
    connection_sha256: Sha256,
    provider_model: str,
    revision: str | None,
    capabilities_sha256: Sha256,
) -> ExactModelId:
    """Derive one conservative exact-model ID for a legacy catalog record.

    Args:
        connection_sha256: Normalized secret-free provider endpoint identity.
        provider_model: Exact provider model or deployment spelling.
        revision: Explicit provider revision when authored.
        capabilities_sha256: Full authored capability declaration digest.

    Returns:
        Content-addressed exact-model identifier.
    """
    digest = sha256_json(
        {
            "version": "gateway-singleton-exact-model-v1",
            "connection_sha256": connection_sha256,
            "provider_model": provider_model,
            "revision": revision,
            "capabilities_sha256": capabilities_sha256,
        }
    )
    return f"exact-{digest}"

"""Certified ordered exact-model pool authoring and alias activation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from wmo.cli.gateway.catalog import (
    GatewayCatalogAuthoringError,
    apply_certified_pool_update,
    plan_certified_pool_update,
    rollback_certified_pool_update,
)
from wmo.cli.gateway.receipts import GatewayReceipt, emit_receipt
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.core.locks import FileLockTimeout, file_write_lock
from wmo.common.models import GatewayEquivalenceCertification
from wmo.runtime.gateway.management import GatewayManagement

pool_app = typer.Typer(help="Certify ordered exact-model deployment pools.", no_args_is_help=True)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_DEPLOYMENT_ALIASES_OPTION = typer.Option(..., "--deployment-alias")
_EXACT_MODEL_OPTION = typer.Option(..., "--exact-model")
_CERTIFICATION_ID_OPTION = typer.Option(..., "--certification-id")
_PROVENANCE_OPTION = typer.Option(..., "--provenance")
_EVIDENCE_SHA256_OPTION = typer.Option(..., "--evidence-sha256")
_CERTIFIED_AT_OPTION = typer.Option(..., "--certified-at")
_EXPECTED_CATALOG_SHA256_OPTION = typer.Option(..., "--expected-catalog-sha256")
_REVISION_OPTION = typer.Option(..., "--revision")
_REPLACE_OPTION = typer.Option(False, "--replace")
_REFUSAL_FAILOVER_OPTION = typer.Option(False, "--refusal-failover")


@pool_app.command("certify")
def pool_certify(
    alias: str = typer.Argument(...),
    deployment_aliases: list[str] = _DEPLOYMENT_ALIASES_OPTION,
    exact_model: str = _EXACT_MODEL_OPTION,
    certification_id: str = _CERTIFICATION_ID_OPTION,
    provenance: str = _PROVENANCE_OPTION,
    evidence_sha256: str = _EVIDENCE_SHA256_OPTION,
    certified_at: str = _CERTIFIED_AT_OPTION,
    expected_catalog_sha256: str = _EXPECTED_CATALOG_SHA256_OPTION,
    revision: str = _REVISION_OPTION,
    root: Path = ROOT_OPTION,
    replace: bool = _REPLACE_OPTION,
    refusal_failover: bool = _REFUSAL_FAILOVER_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Certify an ordered pool and atomically activate its public alias revision.

    Args:
        alias: Public alias and stable exact-model pool identifier.
        deployment_aliases: Existing deployment aliases in failover order.
        exact_model: Exact logical model identity shared by every deployment.
        certification_id: Stable operator certification identifier.
        provenance: Secret-free evidence provenance.
        evidence_sha256: Digest of the external equivalence evidence.
        certified_at: Time the operator certified the evidence.
        expected_catalog_sha256: Optimistic digest from the latest catalog receipt.
        revision: Immutable alias revision identifier.
        root: WMO root containing catalog and gateway authority state.
        replace: Whether an existing pool declaration may change.
        refusal_failover: Whether typed precommit refusals may advance within this revision.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether stdout must contain one JSON receipt.
    """
    del non_interactive
    manager = GatewayManagement(root)
    with usage_error(ValueError, FileLockTimeout):
        manager.require_initialized()
        certification = GatewayEquivalenceCertification(
            certification_id=certification_id,
            provenance=provenance,
            evidence_sha256=evidence_sha256,
            certified_at=datetime.fromisoformat(certified_at),
        )
        catalog_path = root / "models.toml"
        with file_write_lock(catalog_path, what="the gateway exact-model pool activation"):
            update = plan_certified_pool_update(
                root,
                pool_id=alias,
                exact_model_id=exact_model,
                deployment_aliases=tuple(deployment_aliases),
                certification=certification,
                expected_catalog_sha256=expected_catalog_sha256,
                replace=replace,
                allow_existing_desired_state=True,
            )
            snapshot_ref = f"catalog-snapshots/{update.snapshot.name}"
            activation_arguments = {
                "alias_id": alias,
                "alias_name": alias,
                "revision_id": revision,
                "pool_id": alias,
                "snapshot_ref": snapshot_ref,
                "catalog_sha256": update.normalized.identity_sha256(),
                "refusal_failover": refusal_failover,
            }
            activation_changed = manager.preflight_direct_alias_activation(**activation_arguments)
            if not update.observed_matches_expected and activation_changed:
                raise GatewayCatalogAuthoringError(
                    "gateway catalog changed; refresh its digest before certifying the pool"
                )
            activation_started = False
            try:
                apply_certified_pool_update(root, update)
                activation_started = True
                activation_changed = manager.activate_direct_alias(**activation_arguments)
            except BaseException as activation_error:
                if not activation_started:
                    rollback_certified_pool_update(root, update)
                    raise
                try:
                    activation_committed = not manager.preflight_direct_alias_activation(
                        **activation_arguments
                    )
                except BaseException:  # noqa: BLE001 - preserve the primary activation failure
                    emit_receipt(
                        GatewayReceipt(
                            operation="pool.certify",
                            resource_kind="alias_revision",
                            resource_id=revision,
                            changed=None,
                            data={
                                "status": "operation_outcome_unknown",
                                "alias": alias,
                                "pool_id": alias,
                                "catalog_sha256": update.normalized.identity_sha256(),
                                "revision": revision,
                                "recovery": "inspect alias status before retrying",
                            },
                        ),
                        json_output=json_output,
                        human=(
                            "operation_outcome_unknown: desired catalog preserved; inspect alias "
                            f"{alias!r} revision {revision!r} before retrying"
                        ),
                    )
                    raise typer.Exit(code=1) from activation_error
                if not activation_committed:
                    rollback_certified_pool_update(root, update)
                    raise
                activation_changed = True
        normalized = update.normalized
        catalog_changed = update.changed
    emit_receipt(
        GatewayReceipt(
            operation="pool.certify",
            resource_kind="alias_revision",
            resource_id=revision,
            changed=catalog_changed or activation_changed,
            data={
                "alias": alias,
                "pool_id": alias,
                "catalog_sha256": normalized.identity_sha256(),
                "exact_model_id": exact_model,
                "deployment_aliases": deployment_aliases,
                "certification": certification.model_dump(mode="json"),
                "refusal_failover": refusal_failover,
            },
        ),
        json_output=json_output,
        human=f"certified pool {alias} and activated revision {revision}",
    )

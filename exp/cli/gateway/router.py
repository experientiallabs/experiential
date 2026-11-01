"""Author router policies that project gateway aliases activate."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import Field

from exp.cli.gateway.receipts import GatewayReceipt, emit_receipt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.core.artifacts import ContractModel, canonical_json_bytes
from exp.common.models import (
    ConnectionConfig,
    ModelAlias,
    ModelSnapshot,
    RoutedCandidateSnapshot,
    load_model_catalog,
)
from exp.common.project import ArtifactStore, ArtifactStoreError, ProjectStore, ProjectStoreError
from exp.common.release_revision import installed_release_revision
from exp.runtime.gateway.guardrails.classification import DecisionModelAdapterDocument
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import load_project_activation
from exp.runtime.gateway.tag_router import (
    TAG_RULES_POLICY_ARTIFACT_TYPE,
    TagRule,
    TagRulesActivation,
    TagRulesRouterPolicy,
    tag_rules_policy_id,
)
from exp.runtime.models import RuntimeModelCatalog

router_app = typer.Typer(help="Author router policies for project aliases.", no_args_is_help=True)

_JSON_OPTION = typer.Option(False, "--json", help="Write one versioned JSON document to stdout.")
_RULES_FILE_ARGUMENT = typer.Argument(
    ..., exists=True, dir_okay=False, readable=True, help="JSON tag-rules file."
)
_PROJECT_OPTION = typer.Option(..., "--project", help="Existing project that stores the policy.")


class TagRulesDocument(ContractModel):
    """An authored tag-rules file: candidate aliases, the classifier, rules, a default.

    Attributes:
        candidates: Catalog aliases the rules may select, in order.
        default: Candidate serving requests no rule matches.
        classifier: Decision-model document, as a classification guardrail declares it.
        rules: Ordered rules; the first match routes the request.
    """

    candidates: tuple[ModelAlias, ...] = Field(min_length=1)
    default: ModelAlias
    classifier: DecisionModelAdapterDocument
    rules: tuple[TagRule, ...] = Field(min_length=1)


def serving_catalog(
    root: Path, serving_connections: dict[str, ConnectionConfig]
) -> RuntimeModelCatalog:
    """Return the authored models with the gateway's current provider connections.

    This is the catalog an alias snapshot serves, so candidate identities pinned
    or checked against it match what the gateway resolves at startup.

    Args:
        root: Gateway root holding the authored ``models.toml``.
        serving_connections: The gateway's active provider connections by ID. They
            replace the connections ``models.toml`` declares.

    Returns:
        A catalog that reads no environment: it pins and checks candidate
        identities but never resolves a credential or builds a provider client.
    """
    authored = load_model_catalog(root / "models.toml")
    return RuntimeModelCatalog(
        authored.model_copy(update={"connections": serving_connections}), environment={}
    )


def tag_rules_policy(
    document: TagRulesDocument,
    catalog: RuntimeModelCatalog,
    *,
    created_at: datetime,
    code_revision: str,
) -> TagRulesRouterPolicy:
    """Pin each candidate's catalog identity and freeze one content-addressed policy.

    Raises:
        ValueError: A candidate is not in the catalog or has no active serving
            connection, or the rules name an unknown candidate or category.
    """
    candidates = tuple(
        RoutedCandidateSnapshot(alias=alias, model=_pinned(catalog, alias))
        for alias in document.candidates
    )
    return TagRulesRouterPolicy(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        policy_id=tag_rules_policy_id(
            candidates=candidates,
            default_alias=document.default,
            classifier=document.classifier,
            rules=document.rules,
        ),
        candidates=candidates,
        default_alias=document.default,
        classifier=document.classifier,
        rules=document.rules,
    )


def _pinned(catalog: RuntimeModelCatalog, alias: str) -> ModelSnapshot:
    """Return one candidate's frozen identity, or say why it cannot be pinned."""
    try:
        return catalog.snapshot(alias)[0]
    except KeyError as exc:
        message = f"candidate {alias!r} has no active serving connection"
        raise ValueError(message) from exc


def _store(store: ArtifactStore, policy: TagRulesRouterPolicy, *, project: str) -> bool:
    """Store the policy, or adopt an already stored one; return whether this run wrote it.

    The identity covers only routing content, so a stored policy with this ID routes
    identically even when another release wrote it. Such a policy is adopted with its
    provenance as stored, but only when the gateway would activate it.
    """
    try:
        stored, _manifest = store.write_or_replay(
            artifact_id=policy.policy_id,
            artifact_type=TAG_RULES_POLICY_ARTIFACT_TYPE,
            envelope=policy,
            envelope_path="policy.json",
            envelope_type=TagRulesRouterPolicy,
            files={"policy.json": canonical_json_bytes(policy)},
        )
    except ValueError:
        # The activation loader is the acceptance check, so a stored policy this run
        # reports as unchanged is one alias create can activate.
        existing = load_project_activation(
            store, project_ref=project, activation_ref=policy.policy_id
        )
        if not isinstance(existing, TagRulesActivation):
            raise
        return False
    return stored.created_at == policy.created_at


@router_app.command("tags")
def router_tags(
    rules_file: Path = _RULES_FILE_ARGUMENT,
    project: str = _PROJECT_OPTION,
    root: Path = ROOT_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Freeze a tag-rules router policy into a project for ``alias create --project``."""
    with usage_error(ValueError, ProjectStoreError, ArtifactStoreError):
        document = TagRulesDocument.model_validate_json(rules_file.read_bytes())
        manager = GatewayManagement(root)
        manager.require_initialized()
        # The same connection authority alias create reads before it snapshots.
        manager.migrate_legacy_provider_connections()
        connections = {item.connection_id: item.config for item in manager.provider_connections()}
        policy = tag_rules_policy(
            document,
            serving_catalog(root, connections),
            created_at=datetime.now(tz=UTC),
            code_revision=installed_release_revision(),
        )
        store = ProjectStore(root, project)
        store.load_project()
        changed = _store(store.artifacts, policy, project=project)
    emit_receipt(
        GatewayReceipt(
            operation="router.tags",
            resource_kind="router_policy",
            resource_id=policy.policy_id,
            changed=changed,
            data={"project": project, "candidates": list(document.candidates)},
        ),
        json_output=json_output,
        human=(
            f"tag-rules policy {policy.policy_id} stored in project {project}; activate it with "
            f"exp config gateway alias create ALIAS --project {project} --policy {policy.policy_id}"
        ),
    )

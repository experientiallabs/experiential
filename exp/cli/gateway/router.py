"""Author router policies that project gateway aliases activate."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from pydantic import Field

from exp.cli.gateway.receipts import GatewayReceipt, emit_receipt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.core.artifacts import ContractModel, canonical_json_bytes, stable_id
from exp.common.models import ModelAlias, RoutedCandidateSnapshot, load_model_catalog
from exp.common.project import ArtifactStoreError, ProjectStore, ProjectStoreError
from exp.common.release_revision import installed_release_revision
from exp.runtime.gateway.guardrails.classification import DecisionModelAdapterDocument
from exp.runtime.gateway.tag_router import TagRule, TagRulesRouterPolicy
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


def tag_rules_policy(
    document: TagRulesDocument,
    catalog: RuntimeModelCatalog,
    *,
    created_at: datetime,
    code_revision: str,
) -> TagRulesRouterPolicy:
    """Pin each candidate's catalog identity and freeze one content-addressed policy.

    The identity covers the candidates' pinned models, the classifier, the rules
    and the default, not the authoring time, so the same file against the same
    catalog always names the same policy.

    Raises:
        ModelConnectionError: A candidate alias is not in the catalog.
        ValueError: The rules name an unknown candidate or category.
    """
    candidates = tuple(
        RoutedCandidateSnapshot(alias=alias, model=catalog.snapshot(alias)[0])
        for alias in document.candidates
    )
    material = {
        "version": "tag-rules-policy-v1",
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "default_alias": document.default,
        "classifier": document.classifier.model_dump(mode="json"),
        "rules": [rule.model_dump(mode="json") for rule in document.rules],
    }
    return TagRulesRouterPolicy(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        policy_id=stable_id("router-policy", material),
        candidates=candidates,
        default_alias=document.default,
        classifier=document.classifier,
        rules=document.rules,
    )


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
        catalog = RuntimeModelCatalog(load_model_catalog(root / "models.toml"), environment={})
        policy = tag_rules_policy(
            document,
            catalog,
            created_at=datetime.now(tz=UTC),
            code_revision=installed_release_revision(),
        )
        store = ProjectStore(root, project)
        store.load_project()
        changed = policy.policy_id not in store.artifacts.list_ids()
        if changed:
            store.artifacts.write_or_verify_exact(
                artifact_id=policy.policy_id,
                artifact_type="router-policy",
                envelope=policy,
                files={"policy.json": canonical_json_bytes(policy)},
            )
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

"""Hosted Project state transitions kept separate from the core local store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from exp.common.core.artifacts import ArtifactInput
from exp.common.core.locks import file_write_lock
from exp.common.project.manifests import ArtifactManifest, artifact_input
from exp.common.project.project import (
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectHostedJudgeEvidence,
    ProjectHostedSetup,
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    load_project_config,
    write_project_config,
)

if TYPE_CHECKING:
    from exp.common.project.store import ProjectStore


class HostedProjectStoreMixin:
    """Add write-once hosted workflow selections to the canonical Project store."""

    def bind_hosted_setup(
        self: ProjectStore,
        setup: ProjectHostedSetup,
    ) -> ProjectConfig:
        """Apply the one late secret-free setup to prepared trace evidence.

        Args:
            setup: Complete built-in system, role, catalog, retrieval, and budget selection.

        Returns:
            Existing identical or newly selected Project configuration.
        """
        return _bind_hosted_setup(self, setup)

    def bind_hosted_completed_build(
        self: ProjectStore,
        build: ProjectBuildArtifacts,
        *,
        spend_ledger: ArtifactInput,
    ) -> ProjectConfig:
        """Select one hosted build and its complete spend evidence atomically.

        Args:
            build: Exact trace, task, retrieval, and grounded-world-model graph.
            spend_ledger: Complete build-stage spend evidence.

        Returns:
            Existing identical or newly selected Project configuration.
        """
        return _bind_hosted_completed_build(self, build, spend_ledger=spend_ledger)

    def bind_hosted_judge_evidence(
        self: ProjectStore,
        evidence: ProjectHostedJudgeEvidence,
    ) -> ProjectConfig:
        """Select provisional machine judge evidence without changing manual review state.

        Args:
            evidence: Exact provisional setup and calibration pointers.

        Returns:
            Existing identical or newly selected Project configuration.
        """
        return _bind_hosted_pointer_state(
            self,
            field_name="hosted_judge",
            value=evidence,
            expected=(
                (evidence.setup, "provisional-judge-setup"),
                (evidence.calibration, "judge-calibration"),
            ),
            what="hosted provisional judge evidence",
        )

    def bind_router_policy(
        self: ProjectStore,
        selection: ProjectRouterPolicyArtifacts,
    ) -> ProjectConfig:
        """Select one immutable fit-only policy lock before held-out reporting.

        Args:
            selection: Exact policy lock, policy, and stage spend pointers.

        Returns:
            Existing identical or newly selected Project configuration.
        """
        return _bind_hosted_pointer_state(
            self,
            field_name="router_policy",
            value=selection,
            expected=(
                (selection.policy_lock, "router-policy-lock"),
                (selection.policy, "router-policy"),
                (selection.spend_ledger, "provider-spend-ledger"),
            ),
            what="hosted router policy",
        )

    def bind_router_report(
        self: ProjectStore,
        selection: ProjectRouterReportArtifacts,
    ) -> ProjectConfig:
        """Select one held-out report and final spend ledger after policy lock.

        Args:
            selection: Exact held-out report and final spend pointers.

        Returns:
            Existing identical or newly selected Project configuration.
        """
        return _bind_hosted_pointer_state(
            self,
            field_name="router_report",
            value=selection,
            expected=(
                (selection.report, "router-report"),
                (selection.spend_ledger, "provider-spend-ledger"),
            ),
            what="hosted router report",
        )


def _bind_hosted_setup(store: ProjectStore, setup: ProjectHostedSetup) -> ProjectConfig:
    """Apply or exactly replay one verified hosted setup."""
    from exp.common.project.catalog import load_project_model_catalog
    from exp.common.project.store import ArtifactStoreError, ProjectStoreError

    with file_write_lock(store.paths.project_toml, what="hosted Project setup"):
        try:
            existing = load_project_config(store.paths.project_toml)
            if existing.provider_free_stage is None:
                raise ValueError("hosted setup requires completed provider-free trace evidence")
            catalog = load_project_model_catalog(store.artifacts, setup.model_catalog)
            if catalog.project_id != existing.project_id:
                raise ValueError("project-scoped model catalog belongs to another Project")
            aliases = {item.alias for item in catalog.models}
            required = {
                setup.models.world_model,
                setup.models.judge,
                setup.models.embedder,
                *setup.models.candidates,
            }
            if setup.models.incumbent is not None:
                required.add(setup.models.incumbent)
            missing = sorted(required - aliases)
            if missing:
                raise ValueError(f"project-scoped model catalog omits setup aliases: {missing}")
            if existing.system is not None:
                models = existing.models
                model_catalog = existing.model_catalog
                retrieval = existing.retrieval
                budgets = existing.budgets
                if models is None or model_catalog is None or retrieval is None or budgets is None:
                    raise ValueError("Project has an incomplete late hosted setup")
                selected = ProjectHostedSetup(
                    system=existing.system,
                    models=models,
                    model_catalog=model_catalog,
                    retrieval=retrieval,
                    budgets=budgets,
                )
                if selected != setup:
                    raise ValueError("Project already has a different late hosted setup")
                return existing
            if existing.build is not None:
                raise ValueError("hosted setup cannot replace an existing completed build")
            updated = ProjectConfig.model_validate(
                {
                    **existing.model_dump(mode="python"),
                    "schema_version": 4,
                    "system": setup.system,
                    "models": setup.models,
                    "model_catalog": setup.model_catalog,
                    "retrieval": setup.retrieval,
                    "budgets": setup.budgets,
                }
            )
            write_project_config(store.paths.project_toml, updated)
        except (ArtifactStoreError, ValueError) as exc:
            raise ProjectStoreError(f"cannot bind hosted Project setup: {exc}") from exc
        return updated


def _bind_hosted_completed_build(
    store: ProjectStore,
    build: ProjectBuildArtifacts,
    *,
    spend_ledger: ArtifactInput,
) -> ProjectConfig:
    """Apply or replay one verified hosted build and its spend evidence."""
    from exp.common.project.store import ArtifactStoreError, ProjectStoreError

    with file_write_lock(store.paths.project_toml, what="hosted completed build"):
        try:
            existing = load_project_config(store.paths.project_toml)
            _verify_completed_build(store, build)
            _verify_artifact_input(store, spend_ledger, artifact_type="provider-spend-ledger")
            if existing.system is None or existing.model_catalog is None:
                raise ValueError("hosted completed build requires bound late setup")
            if existing.provider_free_stage is None:
                raise ValueError("hosted completed build requires provider-free trace evidence")
            if (
                build.trace_dataset != existing.provider_free_stage.trace_dataset
                or build.task_set != existing.provider_free_stage.task_set
            ):
                raise ValueError("hosted build rewrites selected provider-free trace evidence")
            if existing.build == build and existing.build_spend_ledger == spend_ledger:
                return existing
            if existing.build is not None or existing.build_spend_ledger is not None:
                raise ValueError("Project already selects a different hosted completed build")
            updated = ProjectConfig.model_validate(
                {
                    **existing.model_dump(mode="python"),
                    "build": build,
                    "build_spend_ledger": spend_ledger,
                }
            )
            write_project_config(store.paths.project_toml, updated)
        except (ArtifactStoreError, ValueError) as exc:
            raise ProjectStoreError(f"cannot bind hosted completed build: {exc}") from exc
        return updated


def _bind_hosted_pointer_state(
    store: ProjectStore,
    *,
    field_name: str,
    value: BaseModel,
    expected: tuple[tuple[ArtifactInput, str], ...],
    what: str,
) -> ProjectConfig:
    """Bind one verified write-once hosted pointer group under the Project lock."""
    from exp.common.project.store import ArtifactStoreError, ProjectStoreError

    with file_write_lock(store.paths.project_toml, what=what):
        try:
            for pointer, artifact_type in expected:
                _verify_artifact_input(store, pointer, artifact_type=artifact_type)
            existing = load_project_config(store.paths.project_toml)
            current = getattr(existing, field_name)
            if current == value:
                return existing
            if current is not None:
                raise ValueError(f"Project already selects a different {field_name}")
            updated = ProjectConfig.model_validate(
                {**existing.model_dump(mode="python"), field_name: value}
            )
            write_project_config(store.paths.project_toml, updated)
        except (ArtifactStoreError, ValueError) as exc:
            raise ProjectStoreError(f"cannot bind {what}: {exc}") from exc
        return updated


def _verify_artifact_input(
    store: ProjectStore,
    pointer: ArtifactInput,
    *,
    artifact_type: str,
) -> None:
    """Require an exact immutable pointer with one expected artifact type."""
    stored = store.artifacts.read(pointer.artifact_id)
    if stored.manifest.artifact_type != artifact_type:
        raise ValueError(
            f"artifact {pointer.artifact_id} is {stored.manifest.artifact_type!r}, "
            f"not {artifact_type!r}"
        )
    if artifact_input(stored.manifest) != pointer:
        raise ValueError(f"artifact {pointer.artifact_id} manifest digest changed")


def _verify_completed_build(store: ProjectStore, build: ProjectBuildArtifacts) -> None:
    """Verify exact types, pointers, and immediate provenance for a completed build."""
    expected_types = {
        "trace_dataset": "trace-dataset",
        "task_set": "task-set",
        "serving_rag": "trace-rag-index",
        "fit_rag": "trace-rag-index",
        "world_model": "grounded-world-model",
    }
    manifests: dict[str, ArtifactManifest] = {}
    for field_name, artifact_type in expected_types.items():
        pointer = getattr(build, field_name)
        stored = store.artifacts.read(pointer.artifact_id)
        if stored.manifest.artifact_type != artifact_type:
            raise ValueError(
                f"{field_name} artifact is {stored.manifest.artifact_type!r}, not {artifact_type!r}"
            )
        if artifact_input(stored.manifest) != pointer:
            raise ValueError(f"{field_name} artifact manifest digest changed")
        manifests[field_name] = stored.manifest
    expected_inputs = {
        "task_set": (build.trace_dataset,),
        "serving_rag": (build.trace_dataset,),
        "fit_rag": (build.trace_dataset,),
        "world_model": (build.serving_rag,),
    }
    for field_name, inputs in expected_inputs.items():
        if manifests[field_name].inputs != inputs:
            raise ValueError(f"{field_name} artifact does not bind the completed build graph")

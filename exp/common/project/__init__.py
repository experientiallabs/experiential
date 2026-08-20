"""Canonical project configuration and immutable local artifact storage."""

from importlib import import_module
from typing import TYPE_CHECKING

from exp.common.project.events import (
    ProjectStage,
    ProjectStageEvent,
    ProjectStageEventKind,
    ProjectStageFailure,
)
from exp.common.project.manifests import ArtifactFile, ArtifactManifest, artifact_input
from exp.common.project.paths import ProjectPathError, ProjectPaths
from exp.common.project.project import (
    AgentConfiguration,
    ProjectBudgetConfiguration,
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectConfigError,
    ProjectHostedJudgeEvidence,
    ProjectHostedSetup,
    ProjectModelConfiguration,
    ProjectProviderFreeStage,
    ProjectRetrievalConfiguration,
    ProjectRouterPolicyArtifacts,
    ProjectRouterReportArtifacts,
    ProjectSystemConfiguration,
    ProjectTracePreparationSettings,
    load_project_config,
    write_project_config,
)
from exp.common.project.store import (
    ArtifactAlreadyExistsError,
    ArtifactCorruptionError,
    ArtifactStore,
    ArtifactStoreError,
    ProjectStore,
    ProjectStoreError,
    StoredArtifact,
    coordinate_completed_build_selection,
)

if TYPE_CHECKING:
    from exp.common.project.bundle import ExportedProjectBundle as ExportedProjectBundle
    from exp.common.project.bundle import ProjectBundleError as ProjectBundleError
    from exp.common.project.bundle import ProjectBundleManifest as ProjectBundleManifest
    from exp.common.project.bundle import ProjectBundleMember as ProjectBundleMember
    from exp.common.project.bundle import export_project_bundle as export_project_bundle
    from exp.common.project.bundle import restore_project_bundle as restore_project_bundle
    from exp.common.project.catalog import ProjectCatalogModel as ProjectCatalogModel
    from exp.common.project.catalog import ProjectModelCatalog as ProjectModelCatalog
    from exp.common.project.catalog import ProjectModelCatalogError as ProjectModelCatalogError
    from exp.common.project.catalog import load_project_model_catalog as load_project_model_catalog
    from exp.common.project.catalog import (
        persist_project_model_catalog as persist_project_model_catalog,
    )

_LAZY_EXPORT_MODULES = {
    "ExportedProjectBundle": "exp.common.project.bundle",
    "ProjectBundleError": "exp.common.project.bundle",
    "ProjectBundleManifest": "exp.common.project.bundle",
    "ProjectBundleMember": "exp.common.project.bundle",
    "ProjectCatalogModel": "exp.common.project.catalog",
    "ProjectModelCatalog": "exp.common.project.catalog",
    "ProjectModelCatalogError": "exp.common.project.catalog",
    "export_project_bundle": "exp.common.project.bundle",
    "load_project_model_catalog": "exp.common.project.catalog",
    "persist_project_model_catalog": "exp.common.project.catalog",
    "restore_project_bundle": "exp.common.project.bundle",
}

__all__ = [
    "AgentConfiguration",
    "ArtifactAlreadyExistsError",
    "ArtifactCorruptionError",
    "ArtifactFile",
    "ArtifactManifest",
    "ArtifactStore",
    "ArtifactStoreError",
    "ExportedProjectBundle",
    "ProjectBundleError",
    "ProjectBundleManifest",
    "ProjectBundleMember",
    "ProjectCatalogModel",
    "ProjectConfig",
    "ProjectConfigError",
    "ProjectBuildArtifacts",
    "ProjectBudgetConfiguration",
    "ProjectHostedJudgeEvidence",
    "ProjectHostedSetup",
    "ProjectModelConfiguration",
    "ProjectModelCatalog",
    "ProjectModelCatalogError",
    "ProjectProviderFreeStage",
    "ProjectRetrievalConfiguration",
    "ProjectRouterPolicyArtifacts",
    "ProjectRouterReportArtifacts",
    "ProjectSystemConfiguration",
    "ProjectTracePreparationSettings",
    "ProjectPathError",
    "ProjectPaths",
    "ProjectStore",
    "ProjectStoreError",
    "ProjectStage",
    "ProjectStageEvent",
    "ProjectStageEventKind",
    "ProjectStageFailure",
    "StoredArtifact",
    "artifact_input",
    "coordinate_completed_build_selection",
    "export_project_bundle",
    "load_project_config",
    "load_project_model_catalog",
    "persist_project_model_catalog",
    "restore_project_bundle",
    "write_project_config",
]


def __getattr__(name: str) -> object:
    """Resolve one catalog or bundle helper without creating common-package import cycles.

    Args:
        name: Package attribute requested by Python.

    Returns:
        The supported object loaded from its owning Project module.

    Raises:
        AttributeError: The name is not a supported lazy Project export.
    """
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus supported lazy Project exports."""
    return sorted(set(globals()) | set(_LAZY_EXPORT_MODULES))

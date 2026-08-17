"""Canonical sparse evaluation and fidelity contracts."""

from wmo.common.evaluations.build import (
    build_evaluation_dataset,
    load_evaluation_dataset,
)
from wmo.common.evaluations.dataset import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationProtocol,
    EvaluationRow,
    FidelityFailure,
    FidelityPair,
    FidelityReport,
)
from wmo.common.evaluations.evidence import (
    EvaluationCellEvidence,
    EvaluationEvidenceError,
)
from wmo.common.evaluations.fidelity import build_fidelity_report
from wmo.common.evaluations.plan import (
    EvaluationCell,
    EvaluationPlan,
    FidelityGate,
    FidelityThresholds,
)
from wmo.common.evaluations.planning import (
    ObservedProductionCell,
    build_evaluation_plan,
    build_fidelity_evaluation_plan,
    default_fidelity_thresholds,
    persist_fidelity_thresholds,
)

__all__ = [
    "EvaluationCell",
    "EvaluationCellEvidence",
    "EvaluationDataset",
    "EvaluationDatasetManifest",
    "EvaluationPlan",
    "EvaluationProtocol",
    "EvaluationRow",
    "EvaluationEvidenceError",
    "FidelityFailure",
    "FidelityGate",
    "FidelityThresholds",
    "FidelityPair",
    "FidelityReport",
    "ObservedProductionCell",
    "build_evaluation_dataset",
    "build_evaluation_plan",
    "build_fidelity_evaluation_plan",
    "build_fidelity_report",
    "default_fidelity_thresholds",
    "load_evaluation_dataset",
    "persist_fidelity_thresholds",
]

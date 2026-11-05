"""Public read-only preparation provenance, distinct from current execution admission."""

from __future__ import annotations

from typing import Literal

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationSpec
from exp.optimize.evaluation.variance.bootstrap_protocols import read_historical_bootstrap_protocol
from exp.optimize.evaluation.variance.bootstrap_resume import (
    BootstrapResumeBinding,
    bootstrap_execution_spec,
    read_historical_bootstrap_resume,
)
from exp.optimize.evaluation.variance.family_embedding import load_family_embedding_completion
from exp.optimize.evaluation.variance.family_embedding_executor import (
    load_family_embedding_executor_revision,
)

type PreparationSourceType = Literal[
    "family-embedding-executor-revision", "bootstrap-protocol-revision"
]


class BootstrapPreparationProvenance(ContractModel):
    """Historical native facts; never a current producer, grant, resume permission or deadline.

    The digest algorithm is owned by source_type: a full package fingerprint for a family
    executor, or canonical JSON of the closed BootstrapSourceSnapshot for a protocol revision.
    source_ref pins that exact native record and its immutable ancestry.
    """

    binding: BootstrapResumeBinding
    execution_spec: BootstrapPreparationSpec
    producer_revision: str
    source_sha256: Sha256
    source_ref: ArtifactInput
    source_type: PreparationSourceType


def read_bootstrap_preparation_provenance(
    project: ProjectStore,
    reference: ArtifactInput,
    *,
    spec: BootstrapPreparationSpec,
) -> BootstrapPreparationProvenance:
    """Infer actual historical source only from the exact selected native resume lineage.

    Unknown legacy source is refused. This function creates no control, authorizes no old
    intent, constructs no client and never compares a revision label as if it were source bytes.
    """
    binding = read_historical_bootstrap_resume(project, reference, spec=spec)
    source_ref: ArtifactInput
    source_sha256: Sha256
    source_type: PreparationSourceType
    if binding.embedding_revision is not None:
        _, completion = load_family_embedding_completion(
            project, binding.embedding_revision, historical=True
        )
        if completion.executor_revision is None:
            raise ValueError("native family preparation has no recorded executor source proof")
        source_ref = completion.executor_revision
        source = load_family_embedding_executor_revision(project, source_ref, historical=True)
        if (
            source.producer_revision != binding.producer_revision
            or completion.producer_revision != binding.producer_revision
            or source.original_operation != binding.original_operation
            or source.ledger_sha256 != binding.ledger_sha256
        ):
            raise ValueError("native family source provenance belongs to another producer or root")
        source_sha256 = source.source_sha256
        source_type = "family-embedding-executor-revision"
    elif binding.protocol_revision is not None:
        source_ref = binding.protocol_revision
        protocol = read_historical_bootstrap_protocol(project, source_ref)
        if protocol.producer_revision != binding.producer_revision:
            raise ValueError("native protocol source provenance belongs to another producer")
        source_sha256 = sha256_json(protocol.source)
        source_type = "bootstrap-protocol-revision"
    else:
        raise ValueError("native preparation binding has no recorded source provenance")
    return BootstrapPreparationProvenance(
        binding=binding,
        execution_spec=bootstrap_execution_spec(project, binding, spec),
        producer_revision=binding.producer_revision,
        source_sha256=source_sha256,
        source_ref=source_ref,
        source_type=source_type,
    )

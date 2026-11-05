"""Native execution-source witnesses captured before a new preparation binding is registered."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.release_revision import _package_source_digest


class BootstrapExecutionSource(ContractModel):
    """Actual package bytes and artifact producer; never a digest of a revision label."""

    protocol: Literal["bootstrap-execution-source-v1"] = "bootstrap-execution-source-v1"
    producer_revision: str = Field(min_length=1)
    source_sha256: Sha256


def current_bootstrap_execution_source(producer_revision: str) -> BootstrapExecutionSource:
    """Fingerprint installed production source/resources without credentials, Git or tests."""
    return BootstrapExecutionSource(
        producer_revision=producer_revision,
        source_sha256=_package_source_digest(Path(__file__).resolve().parents[3]),
    )

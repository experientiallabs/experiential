"""Serializable launch input shared by local and remotely hosted learning runs."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.service.configuration import RunConfiguration
from exp.optimize.claas.training_contracts import ClaasTrainingSpec


class RunLaunchConfiguration(ContractModel):
    """Name finite compute, durable state, and an environment-only API credential."""

    schema_version: Literal[1] = 1
    directory: Path
    spec: ClaasTrainingSpec
    run: RunConfiguration
    runtime: ResidentVerlSettings
    host: Literal["127.0.0.1", "0.0.0.0"] = "127.0.0.1"
    port: int = Field(default=8000, strict=True, ge=1024, le=65535)
    authentication_env: str = Field(
        default="EXPERIENTIAL_CLAAS_TOKEN", pattern=r"^[A-Z][A-Z0-9_]{0,127}$"
    )
    persistence: Literal["local", "modal-volume"] = "local"
    compute_reservation_usd: float = Field(ge=0, allow_inf_nan=False)
    import_examples_path: Path | None = None
    maximum_import_bytes: int = Field(default=16_777_216, strict=True, ge=1, le=268_435_456)

    @model_validator(mode="after")
    def _validate_paths(self) -> "RunLaunchConfiguration":
        """Keep checkpoint and queue state inside one explicitly owned durable root."""
        if not self.directory.is_absolute():
            raise ValueError("directory must be an absolute durable state directory")
        if ".." in self.directory.parts or ".." in self.runtime.checkpoint_root.parts:
            raise ValueError("durable paths must not contain parent traversal")
        if not self.runtime.checkpoint_root.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError(
                "checkpoint_root must resolve inside the durable state directory; "
                "remove links to another run's storage"
            )
        if self.runtime.maximum_output_tokens > self.spec.max_sequence_tokens:
            raise ValueError("maximum_output_tokens must not exceed max_sequence_tokens")
        return self

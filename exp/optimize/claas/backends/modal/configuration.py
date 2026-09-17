"""Resource and transport configuration, without learning or queue policy."""

from typing import Annotated

from pydantic import Field, field_validator

from exp.common.core.artifacts import ContractModel

ModalName = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")]


class ModalLaunch(ContractModel):
    """Use an existing App, v2 Volume, and immutable prebuilt image for one run.

    The first launch permanently binds the Volume to this App in the environment's
    shared ownership registry. All later launches must use that App. The image must
    contain Experiential with the selected runtime dependencies and GNU ``sync``.
    Secret names identify Modal-managed environment injection, never secret values.
    """

    app_name: ModalName
    environment_name: ModalName
    volume_name: ModalName
    image_id: Annotated[str, Field(pattern=r"^im-[A-Za-z0-9]+$")]
    gpu: Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9.-]+$")]
    cpu: float = Field(default=8, gt=0, le=64, allow_inf_nan=False)
    memory_mib: int = Field(default=32768, strict=True, ge=1024, le=1048576)
    timeout_seconds: int = Field(default=7200, strict=True, ge=60, le=86400)
    startup_timeout_seconds: int = Field(default=600, strict=True, ge=1, le=3600)
    secret_names: tuple[ModalName, ...] = ()
    authentication_secret_name: ModalName | None = None
    maximum_upload_bytes: int = Field(default=1073741824, strict=True, ge=1, le=68719476736)

    @field_validator("secret_names")
    @classmethod
    def _unique_secrets(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        """Reject duplicate secret references rather than changing their precedence."""
        if len(names) != len(set(names)):
            raise ValueError("secret_names must be unique")
        return names

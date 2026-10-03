"""Host-authorized generation waits, separate from the request's hard deadline."""

from pydantic import Field

from exp.common.core.artifacts import ContractModel


class GatewayGenerationTimeouts(ContractModel):
    """Per-request waits issued by the authenticated host, never the caller.

    These replace a deployment's first-token base and stream-progress wait.
    The existing input-size allowance still applies to the first-token base.
    Header/connect timeouts and the total request deadline remain unchanged.
    Each wait is bounded to one hour. A longer wait delays failover from a
    genuinely stalled upstream.

    Attributes:
        first_token_base_seconds: Base wait for the first semantic token.
        progress_seconds: Maximum gap without semantic generation progress.
    """

    first_token_base_seconds: float = Field(gt=0, le=3600, allow_inf_nan=False, strict=True)
    progress_seconds: float = Field(gt=0, le=3600, allow_inf_nan=False, strict=True)

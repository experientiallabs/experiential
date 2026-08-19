"""Public workflow composition failures."""


class RouterCompositionError(ValueError):
    """Explicit workflow inputs cannot safely produce a frozen router."""


class JudgeTranscriptAdmissionError(ValueError):
    """One judge request's counted input exceeds its frozen reserved input-token ceiling.

    The rejection is a property of the rollout transcript, not of the reservation, so the
    workflow excludes that one cell from judging evidence instead of aborting the run.
    """


class JudgeDispatchExhaustedError(ValueError):
    """One admitted judge dispatch exhausted its bounded retries without usable output.

    Every exhausted attempt may have been billed, so the error carries the conservative
    retry-bound cost of the admitted request. The workflow charges that cost against the
    shared provider-spend ledger while excluding the cell from judging evidence.
    """

    def __init__(self, message: str, *, conservative_cost_usd: float) -> None:
        """Bind the failure description to its conservative billed-spend ceiling.

        Args:
            message: Concise operator-facing failure description.
            conservative_cost_usd: Retry-bound worst-case spend of the exhausted dispatch.
        """
        super().__init__(message)
        self.conservative_cost_usd = conservative_cost_usd

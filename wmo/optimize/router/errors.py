"""Public workflow composition failures."""


class RouterCompositionError(ValueError):
    """Explicit workflow inputs cannot safely produce a frozen router."""


class JudgeTranscriptAdmissionError(ValueError):
    """One judge request's counted input exceeds its frozen reserved input-token ceiling.

    The rejection is a property of the rollout transcript, not of the reservation, so the
    workflow excludes that one cell from judging evidence instead of aborting the run.
    """

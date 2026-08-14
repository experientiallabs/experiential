"""Public workflow composition failures."""


class RouterCompositionError(ValueError):
    """Explicit workflow inputs cannot safely produce a frozen router."""

"""Selection-only project router activation errors."""


class RouterApplicationError(ValueError):
    """A project cannot select one verified frozen router for local execution."""

"""Provider-neutral errors for malformed completed provider responses."""


class ProviderResponseError(ValueError):
    """A provider returned a completed response that violates WMO's typed contract."""

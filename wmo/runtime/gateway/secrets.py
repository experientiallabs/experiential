"""Environment-only provider secret resolution for the local gateway."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

_ENV_REFERENCE = re.compile(r"env://([A-Z_][A-Z0-9_]*)\Z")


class SecretResolutionError(ValueError):
    """A provider secret reference is unsupported, absent, or empty."""


class EnvironmentSecretResolver:
    """Resolve only explicit environment references without retaining values."""

    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        """Initialize the resolver.

        Args:
            environment: Optional injected environment for deterministic tests.
        """
        self._environment = os.environ if environment is None else environment

    def resolve(self, reference: str) -> str:
        """Resolve one ``env://VARIABLE`` reference.

        Args:
            reference: Opaque provider credential reference.

        Returns:
            The current non-empty environment value.

        Raises:
            SecretResolutionError: The reference is unsupported or not populated.
        """
        match = _ENV_REFERENCE.fullmatch(reference)
        if match is None:
            raise SecretResolutionError("provider secret reference must use env://VARIABLE_NAME")
        value = self._environment.get(match.group(1))
        if not value:
            raise SecretResolutionError("provider secret environment variable is not populated")
        return value

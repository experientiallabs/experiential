"""Resident runtime settings reject ambiguous ownership before acquiring a GPU."""

from pathlib import Path

import pytest

from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings


def test_requires_absolute_checkpoint_root() -> None:
    """A relative path cannot silently change a run's durable checkpoint owner."""
    with pytest.raises(ValueError, match="absolute"):
        ResidentVerlSettings(checkpoint_root=Path("relative"))

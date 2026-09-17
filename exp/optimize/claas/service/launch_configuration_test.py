"""Checkpoint ownership follows resolved storage identity, including directory aliases."""

from pathlib import Path

import pytest

from exp.optimize.claas.service.execution_test import configuration
from exp.optimize.claas.service.launch_configuration import RunLaunchConfiguration


def test_checkpoint_symlink_cannot_escape_owned_state(tmp_path: Path) -> None:
    """Reject an apparently nested checkpoint path that aliases another durable root."""
    owned = tmp_path / "owned"
    foreign = tmp_path / "foreign"
    owned.mkdir()
    foreign.mkdir()
    (owned / "checkpoints").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="resolve inside"):
        configuration(owned)


def test_root_alias_uses_the_same_containment_identity(tmp_path: Path) -> None:
    """A root alias is valid when its checkpoint path resolves within that same root."""
    owned = tmp_path / "owned"
    owned.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(owned, target_is_directory=True)
    original = configuration(owned)
    updated = original.model_dump()
    updated["directory"] = alias
    assert RunLaunchConfiguration.model_validate(updated).directory == alias

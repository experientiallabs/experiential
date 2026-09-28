"""Input-file and cost authority checks require no Modal SDK or provider calls."""

from pathlib import Path

import pytest

from exp.optimize.claas.backends.modal.validation import validate_import_file


def test_import_rejects_symlink_and_oversize(tmp_path: Path) -> None:
    """Reject unsafe input before any Modal allocation or remote mutation."""
    path = tmp_path / "input.jsonl"
    path.write_text("1234")
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        validate_import_file(alias, 100)
    with pytest.raises(ValueError, match="maximum_upload_bytes"):
        validate_import_file(path, 3)

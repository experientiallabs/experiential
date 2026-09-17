"""Bounded inputs reject pipes and symlinks before parsing or starting compute."""

import os
from pathlib import Path

import pytest

from exp.optimize.claas.service.files import read_regular_file


def test_regular_file_limits_and_special_files(tmp_path: Path) -> None:
    """Only a complete bounded regular file is admitted, with no FIFO blocking."""
    source = tmp_path / "source"
    source.write_bytes(b"abcd")
    assert read_regular_file(source, 4) == b"abcd"
    with pytest.raises(ValueError, match="regular file"):
        read_regular_file(source, 3)
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        read_regular_file(link, 4)
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match="regular file"):
        read_regular_file(pipe, 4)

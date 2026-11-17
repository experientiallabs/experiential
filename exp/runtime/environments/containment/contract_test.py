"""Tests for the shared local process containment contract."""

from __future__ import annotations

import os

from exp.runtime.environments.containment.contract import close_fd


def test_close_fd_treats_an_already_closed_descriptor_as_complete() -> None:
    """Closing twice is a completed cleanup, never a second failure."""
    read_end, write_end = os.pipe()
    os.close(write_end)
    close_fd(read_end)
    close_fd(read_end)
    close_fd(write_end)

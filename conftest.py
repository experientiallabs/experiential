"""Repo-wide pytest configuration."""

from __future__ import annotations

import os

# Rich consoles snapshot color support when constructed, and `wmh.cli.app` builds its console at
# import time — so color-forcing vars must go before any test module imports it, or a dev shell
# exporting FORCE_COLOR would inject ANSI codes into CliRunner captures and fail assertions.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

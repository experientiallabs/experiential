"""Repo-wide pytest configuration."""

from __future__ import annotations

import importlib
import os

# Rich consoles snapshot color support when constructed, and `wmh.cli.app` builds its console at
# import time — so color-forcing vars must go before anything imports it, or a dev shell
# exporting FORCE_COLOR would inject ANSI codes into CliRunner captures and fail assertions.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

# `wmh.cli.app` also loads the developer's real `.env` at import time (so wizard-saved keys
# persist across sessions). Tests must see the shell environment, not the `.env`: otherwise a
# local OPENAI_API_KEY/AWS key would un-skip the live provider tests and leak into CLI
# assertions. Import it here — before any test module — and scrub whatever the load injected.
_shell_env = dict(os.environ)
importlib.import_module("wmh.cli.app")
for _var in set(os.environ) - set(_shell_env):
    del os.environ[_var]

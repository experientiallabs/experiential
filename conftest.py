"""Repo-wide pytest configuration."""

from __future__ import annotations

import os
import sys

# Rich consoles snapshot color support when constructed, and `exp.cli.app` builds its console at
# import time — so color-forcing vars must go before any test module imports it, or a dev shell
# exporting FORCE_COLOR would inject ANSI codes into CliRunner captures and fail assertions.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

# The SDK and Capture command help support Python 3.12. Only the TLS capture
# engine requires 3.13, matching its conditional distribution dependencies. Ignore
# these exact engine tests before import; auth, policy, control, CLI help, and
# read-only backend checks still run on the minimum supported SDK interpreter.
collect_ignore = (
    [
        "exp/cli/capture/display_test.py",
        "exp/cli/capture/runner_test.py",
        "exp/cli/capture/session_test.py",
        "exp/runtime/capture/certificates_test.py",
        "exp/runtime/capture/normalization_test.py",
        "exp/runtime/capture/proxy_test.py",
        "exp/runtime/capture/redirector_test.py",
        "exp/runtime/capture/transports_test.py",
        "exp/runtime/capture/upload_test.py",
    ]
    if sys.version_info < (3, 13)
    else []
)

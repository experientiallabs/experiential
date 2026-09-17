# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`exp optimize`: the one switch over the product optimizers.

The group exposes `router`, `model`, and bounded `claas` learning runs.
This module owns the switch only,
so no optimization logic lives here.
"""

from __future__ import annotations

import typer

from exp.cli.optimize.claas import claas_app
from exp.cli.optimize.model import optimize_model
from exp.cli.optimize.router import router

optimize_app = typer.Typer(
    help="Router, supervised model, and finite continual-learning optimization.",
    no_args_is_help=True,
)

optimize_app.command(
    "router",
    help="Optimize a guarded router automatically from one completed project build.",
)(router)
optimize_app.command(
    "model", help="Build routed interactions into W12 and run bounded W13 Tinker SFT."
)(optimize_model)

optimize_app.add_typer(claas_app, name="claas")

# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmo optimize`: the one switch over the product optimizers.

The group is exactly two commands: `router` (the guarded offline kNN path) and
`model` (automatic routed-interaction SFT). This module owns the switch only,
so no optimization logic lives here.
"""

from __future__ import annotations

import typer

from wmo.cli.optimize.model import optimize_model
from wmo.cli.optimize.router import router

optimize_app = typer.Typer(
    help="Offline optimization of frozen project artifacts.",
    no_args_is_help=True,
)

optimize_app.command(
    "router",
    help="Optimize a guarded router automatically from one completed project build.",
)(router)
optimize_app.command(
    "model", help="Build routed interactions into W12 and run bounded W13 Tinker SFT."
)(optimize_model)

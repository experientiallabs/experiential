# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmo optimize`: the one switch over the product optimizers.

Extracted from the former `harness_app` module, which co-located the product's
optimize switch with the harness-search CLI; the search program now lives in the
agent-optimization repo, and this module owns only what the product ships:
`router` (the guarded offline kNN path) and `model` (offline SFT from persisted artifacts).
"""

from __future__ import annotations

import typer

from wmo.cli.model_optimize import optimize_model
from wmo.cli.router_app import router

optimize_app = typer.Typer(
    help="Offline optimization of frozen project artifacts.",
    no_args_is_help=True,
)

optimize_app.command(
    "router",
    help="Fit, freeze, and report one guarded router from completed evidence.",
)(router)
optimize_app.command(
    "model", help="Run W13 Tinker SFT from an explicit persisted W12 dataset configuration."
)(optimize_model)

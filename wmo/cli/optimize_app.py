# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmo optimize`: the one switch over the product optimizers.

Extracted from the former `harness_app` module, which co-located the product's
optimize switch with the harness-search CLI; the search program now lives in the
agent-optimization repo, and this module owns only what the product ships:
`router` (the guarded offline kNN path) and `model` (offline SFT from persisted artifacts).
"""

from __future__ import annotations

import typer

optimize_app = typer.Typer(
    help="Optimizers behind one switch. `router` fits routing policy; `model` runs offline SFT "
    "from a persisted W12 dataset.",
    no_args_is_help=True,
)

# `router` loads its real Typer app only on first use, so `wmo --help` never pays for the routing
# optimizer import chain. `model` is registered directly from its narrow command owner.
from wmo.cli.defer import add_deferred_typer  # noqa: E402
from wmo.cli.model_optimize import optimize_model  # noqa: E402

add_deferred_typer(
    optimize_app,
    name="router",
    module="wmo.cli.router_app",
    attr="router_app",
    help="Resume sparse evaluation and fit the single guarded offline kNN router.",
    known_names=("fit", "report"),
)
optimize_app.command(
    "model", help="Run W13 Tinker SFT from an explicit persisted W12 dataset configuration."
)(optimize_model)

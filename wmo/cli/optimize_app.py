# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmo optimize`: the one switch over the product optimizers.

Extracted from the former `harness_app` module, which co-located the product's
optimize switch with the harness-search CLI; the search program now lives in the
agent-optimization repo, and this module owns only what the product ships:
`router` (the single guarded offline kNN path) and `distill` (legacy model training).
"""

from __future__ import annotations

import typer

optimize_app = typer.Typer(
    help="Offline guarded router optimization and model distillation.",
    no_args_is_help=True,
)

# Commands load only on first use so root help remains provider-free.
from wmo.cli.defer import add_deferred_typer  # noqa: E402

add_deferred_typer(
    optimize_app,
    name="router",
    module="wmo.cli.router_app",
    attr="router_app",
    help="Resume sparse evaluation and fit the single guarded offline kNN router.",
    known_names=("fit", "report"),
)
add_deferred_typer(
    optimize_app,
    name="distill",
    module="wmo.cli.model_app",
    attr="model_app",
    help="Train the agent model itself: distillation of a Tinker LoRA student from real "
    "benchmark rollouts (harbor or tau2, config-selected), gated on held-out solve rates.",
    known_names=("run", "report"),
)

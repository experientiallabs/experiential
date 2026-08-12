# Copyright (c) 2026 Experiential Labs. All rights reserved.

"""`wmo optimize`: the one switch over the product optimizers.

Extracted from the former `harness_app` module, which co-located the product's
optimize switch with the harness-search CLI; the search program now lives in the
agent-optimization repo, and this module owns only what the product ships:
`route` (fit/tune/report policies), `model` (the staged one-command path), and
`distill` (train a Tinker LoRA student).
"""

from __future__ import annotations

import typer

optimize_app = typer.Typer(
    help="Optimizers behind one switch. `model` is the staged one-command path (preflight, "
    "sweep, fit, tune, report); `route` is those steps individually; `harness` searches the "
    "agent scaffold; `distill` trains an adapter.",
    no_args_is_help=True,
)

# `route` and `distill` load their real Typer app only on first use, so `wmo --help` never pays
# for the optimize/engine/distill import chains those modules pull in. `model` is registered
# directly from its command owner, whose heavy imports remain local to its functions, which keeps
# `optimize model --help` at full fidelity without a compatibility import.
from wmo.cli.defer import add_deferred_typer  # noqa: E402
from wmo.cli.optimize_model_cmd import optimize_model  # noqa: E402

add_deferred_typer(
    optimize_app,
    name="route",
    module="wmo.cli.route_app",
    attr="route_app",
    help="Make models routable, measure them closed-loop, then fit, tune, and report policies.",
    known_names=("student", "sweep", "fit", "tune", "report", "pin"),
)
add_deferred_typer(
    optimize_app,
    name="distill",
    module="wmo.cli.model_app",
    attr="model_app",
    help="Train the agent model itself: distillation of a Tinker LoRA student from real "
    "benchmark rollouts (harbor or tau2, config-selected), gated on held-out solve rates.",
    known_names=("run", "probe", "report"),
)
optimize_app.command(
    "model", help="Measure, fit, tune, and report a routing policy in one command."
)(optimize_model)

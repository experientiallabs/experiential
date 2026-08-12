"""Compose the supported route-optimization command group."""

from __future__ import annotations

import typer

from wmo.cli.route_deepswe_cmd import register as _register_deepswe_command
from wmo.cli.route_fit_cmd import register as _register_fit_commands
from wmo.cli.route_pool_cmd import register as _register_pool_commands
from wmo.cli.route_sweep_cmd import register as _register_sweep_command

route_app = typer.Typer(
    help="Make models routable, measure them closed-loop, then fit, tune, and report policies.",
    no_args_is_help=True,
)
_register_sweep_command(route_app)
_register_pool_commands(route_app)
_register_fit_commands(route_app)
_register_deepswe_command(route_app)

"""Deferred Typer sub-app loading so root CLI import stays light.

Parent ``--help`` only needs the group's name and help string. Subcommand resolution
(``wmo optimize …``, ``wmo optimize --help``) imports the real Typer app once via
``importlib`` and then delegates to it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from typer import _click
from typer.core import TyperGroup
from typer.main import get_group

if TYPE_CHECKING:
    import typer


class DeferredTyperGroup(TyperGroup):
    """Click group that imports a Typer app only when a subcommand is resolved.

    ``list_commands`` returns ``known_names`` without importing so a parent ``--help`` that
    never descends into this group stays cheap. Resolving any child (including this group's
    own ``--help``, which walks children for short help) loads the real app once.
    """

    def __init__(
        self,
        *args: object,
        import_path: str,
        attr: str,
        known_names: tuple[str, ...],
        **kwargs: object,
    ) -> None:
        # Typer/Click pass a wide kwargs bag; forwarding as typed kwargs is not practical here.
        super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type]
        self._import_path = import_path
        self._attr = attr
        self._known_names = known_names
        self._real: TyperGroup | None = None

    def _ensure(self) -> TyperGroup:
        if self._real is None:
            module = importlib.import_module(self._import_path)
            typer_app = getattr(module, self._attr)
            loaded = get_group(typer_app)
            assert isinstance(loaded, TyperGroup)
            self._real = loaded
        return self._real

    def list_commands(self, ctx: _click.Context) -> list[str]:
        """Return known command names without loading the deferred application."""
        if self._real is not None:
            return self._ensure().list_commands(ctx)
        return list(self._known_names)

    def get_command(self, ctx: _click.Context, cmd_name: str) -> _click.Command | None:
        """Load the deferred application and resolve one command."""
        return self._ensure().get_command(ctx, cmd_name)


def add_deferred_typer(
    parent: typer.Typer,
    *,
    name: str,
    module: str,
    attr: str,
    help: str,
    known_names: tuple[str, ...],
    no_args_is_help: bool = True,
) -> None:
    """Register a named sub-app whose module loads only on first use.

    Creates an empty Typer placeholder so Typer's registration bookkeeping stays intact, then
    passes a ``DeferredTyperGroup`` subclass via ``cls`` so Click resolves children lazily.
    """
    import typer as typer_mod

    placeholder = typer_mod.Typer(help=help, no_args_is_help=no_args_is_help)

    class _Deferred(DeferredTyperGroup):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(
                *args,
                import_path=module,
                attr=attr,
                known_names=known_names,
                **kwargs,
            )

    parent.add_typer(
        placeholder,
        name=name,
        help=help,
        no_args_is_help=no_args_is_help,
        cls=_Deferred,
    )

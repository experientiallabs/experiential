"""A replay panel stops readably when a provider request fails or runs out of time."""

from pathlib import Path

import pytest
import typer
from rich.console import Console

from exp.cli.evaluation import replay_view
from exp.common.project import ProjectStore
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.transport import ProviderTransportError


@pytest.mark.parametrize(
    "failure",
    [
        ProviderTransportError("provider returned HTTP 400", status_code=400),
        ProviderDeadlineExceeded("provider request deadline exceeded"),
    ],
    ids=["transport", "deadline"],
)
def test_a_provider_stop_keeps_completed_traces_and_names_the_resume_flag(
    failure: Exception, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stopped trace is never bought again silently, whichever way the request failed."""

    def stop(*_args: object, **_kwargs: object) -> None:
        """Fail the panel the way the provider failed."""
        raise failure

    monkeypatch.setattr(replay_view, "require_resumable", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(replay_view, "execute_panel", stop)
    console = Console(record=True, width=200)
    with pytest.raises(typer.Exit) as stopped:
        replay_view.run_panel(console, ProjectStore(tmp_path, "panel"), ())
    assert stopped.value.exit_code == 1
    output = console.export_text()
    assert f"error a provider request failed: {failure}" in output
    assert "--retry-incomplete" in output

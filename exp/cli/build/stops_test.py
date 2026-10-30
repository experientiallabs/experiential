"""Paid build work stops readably with exit 1 instead of a traceback."""

import pytest
import typer
from rich.console import Console

from exp.cli.build.stops import paid_work_stops
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.transport import ProviderTransportError


@pytest.mark.parametrize(
    "failure",
    [
        ProviderTransportError("provider returned HTTP 503", status_code=503),
        ProviderDeadlineExceeded("provider request deadline exceeded"),
    ],
    ids=["transport", "deadline"],
)
def test_a_failed_or_timed_out_provider_request_stops_readably(failure: Exception) -> None:
    """A request that ran out of time is reported like any other provider failure."""
    console = Console(record=True, width=200)
    with pytest.raises(typer.Exit) as stopped, paid_work_stops(console):
        raise failure
    assert stopped.value.exit_code == 1
    assert stopped.value.__cause__ is failure
    output = console.export_text()
    assert f"error a provider request failed: {failure}" in output
    assert "Run the same exp build command again to resume" in output
    assert "Traceback" not in output

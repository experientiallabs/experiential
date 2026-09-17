"""Finite resources, immutable images, and credential-reference validation."""

import pytest
from pydantic import ValidationError

from exp.optimize.claas.backends.modal.configuration import ModalLaunch


def launch() -> ModalLaunch:
    """Return inert resource references without constructing any Modal clients."""
    return ModalLaunch(
        app_name="learning",
        environment_name="main",
        volume_name="learner-state",
        image_id="im-fixture123",
        gpu="A100-80GB",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"image_id": "latest"},
        {"timeout_seconds": 86401},
        {"gpu": "A100:2"},
        {"secret_names": ("auth", "auth")},
        {"api_key": "do-not-store"},
    ],
)
def test_rejects_ambiguous_or_unbounded_resources(changes: dict[str, object]) -> None:
    """Refuse mutable images, multiple GPUs, secrets, and unlimited run resources."""
    with pytest.raises(ValidationError):
        ModalLaunch.model_validate(launch().model_dump() | changes)

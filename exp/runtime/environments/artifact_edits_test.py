"""Artifact handoff contracts reject ambiguous output identities."""

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.runtime.environments.artifact_edits import NativeArtifactBinding


@pytest.mark.parametrize("paths", [("../outside",), ("/absolute",), ("a", "a"), ("a", "a/b")])
def test_native_outputs_are_confined_files(paths: tuple[str, ...]) -> None:
    """Unsafe or overlapping file contracts fail before any adapter is selected."""
    with pytest.raises(ValueError):
        NativeArtifactBinding(
            input_artifact=ArtifactInput(artifact_id="input", sha256="0" * 64),
            adapter_sha256="1" * 64,
            output_paths=paths,
        )

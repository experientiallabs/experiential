"""Verifier reruns cannot publish rewards inherited from earlier attempts."""

import base64
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from exp.optimize.evaluation.harbor import client as harbor_client


@pytest.mark.parametrize("failure", ["incomplete", "malformed", "transport", "credential"])
def test_failed_verification_removes_prior_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Every failed verifier attempt must erase a previous successful reward first."""
    reward = tmp_path / "reward.txt"
    reward.write_text("1\n")
    monkeypatch.setattr(sys, "argv", ["tools.py", "verify"])
    monkeypatch.setenv("EXP_HARBOR_ENDPOINT", "http://localhost:1234")
    monkeypatch.setenv("EXP_HARBOR_VERIFIER_TOKEN", "test-token")
    monkeypatch.setattr(harbor_client, "Path", lambda path: tmp_path)

    def response(request: Request, *, timeout: int) -> io.BytesIO:
        """Fail after a prior reward already exists, without using the network."""
        if failure == "transport":
            raise URLError("connection lost")
        return io.BytesIO(
            b"not JSON" if failure == "malformed" else b'{"status":"incomplete","reward":null}'
        )

    monkeypatch.setattr(harbor_client, "urlopen", response)
    if failure == "credential":
        monkeypatch.delenv("EXP_HARBOR_VERIFIER_TOKEN")
    if failure == "incomplete":
        assert harbor_client.main() == 2
    else:
        with pytest.raises((ValueError, URLError, KeyError)):
            harbor_client.main()
    assert not reward.exists(), "A failed attempt left a stale passing reward"


def test_completed_verification_writes_current_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed failing grade replaces a prior successful grade."""
    (tmp_path / "reward.txt").write_text("1\n")
    monkeypatch.setattr(sys, "argv", ["tools.py", "verify"])
    monkeypatch.setenv("EXP_HARBOR_ENDPOINT", "http://localhost:1234")
    monkeypatch.setenv("EXP_HARBOR_VERIFIER_TOKEN", "test-token")
    monkeypatch.setattr(harbor_client, "Path", lambda path: tmp_path)

    def response(request: Request, *, timeout: int) -> io.BytesIO:
        """Return a completed current grade through the actual client parser."""
        return io.BytesIO(b'{"status":"completed","reward":0}')

    monkeypatch.setattr(harbor_client, "urlopen", response)
    assert harbor_client.main() == 0
    assert (tmp_path / "reward.txt").read_text() == "0\n"
    assert json.loads((tmp_path / "experiential.json").read_bytes())["reward"] == 0


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "a/../../outside", "a\\outside", "C:output.txt"]
)
def test_native_download_rejects_path_escape(tmp_path: Path, name: str) -> None:
    """A declared output cannot write outside its new actor output directory."""

    result = {
        "files": [
            {
                "path": name,
                "content_base64": base64.b64encode(b"data").decode(),
                "sha256": hashlib.sha256(b"data").hexdigest(),
            }
        ]
    }
    with pytest.raises(ValueError, match="unsafe"):
        harbor_client.write_outputs(result, tmp_path / "outputs")
    assert not list(tmp_path.iterdir())


def test_native_download_rejects_corrupt_bytes(tmp_path: Path) -> None:
    """Transport damage cannot create a seemingly completed local artifact."""
    with pytest.raises(ValueError, match="digest"):
        harbor_client.write_outputs(
            {"files": [{"path": "output.txt", "content_base64": "ZGF0YQ==", "sha256": "0" * 64}]},
            tmp_path / "outputs",
        )
    assert not list(tmp_path.iterdir())

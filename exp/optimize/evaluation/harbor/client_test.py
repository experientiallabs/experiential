"""Verifier reruns publish only current verdicts and never rewards from earlier attempts."""

import base64
import hashlib
import io
import json
import sys
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.harbor import client as harbor_client


def _verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: Callable[..., io.BytesIO]
) -> int:
    """Run the packaged verifier against a scripted grading response, without a network."""
    monkeypatch.setattr(sys, "argv", ["tools.py", "verify", "--directory", str(tmp_path)])
    monkeypatch.setenv("EXP_HARBOR_ENDPOINT", "http://localhost:1234")
    monkeypatch.setenv("EXP_HARBOR_VERIFIER_TOKEN", "test-token")
    monkeypatch.setattr(harbor_client, "urlopen", response)
    return harbor_client.main()


def _prior_pass(directory: Path) -> None:
    """Leave both Harbor reward forms from an earlier successful attempt."""
    (directory / "reward.txt").write_text("1\n")
    (directory / "reward.json").write_text('{"reward": 1, "scored": 1}\n')


@pytest.mark.parametrize("failure", ["malformed", "transport", "http"])
def test_grading_failure_retains_detail_and_removes_prior_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """No verdict means no reward file, so Harbor reports an exception rather than a score."""
    _prior_pass(tmp_path)

    def response(request: Request, *, timeout: int) -> io.BytesIO:
        """Fail after a prior reward already exists, without using the network."""
        if failure == "transport":
            raise URLError("connection lost")
        if failure == "http":
            raise HTTPError(
                request.full_url, 503, "unavailable", Message(), io.BytesIO(b'{"error":"X"}')
            )
        return io.BytesIO(b"not JSON")

    assert _verify(tmp_path, monkeypatch, response) == 2
    assert not (tmp_path / "reward.txt").exists() and not (tmp_path / "reward.json").exists()
    retained = (tmp_path / "experiential.json").read_bytes()
    if failure == "malformed":
        assert retained == b"not JSON"
    else:
        assert json.loads(retained)["status"] == "grading_failed"


def test_missing_verifier_credential_fails_before_any_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A misconfigured verifier environment cannot leave an inherited passing reward."""
    _prior_pass(tmp_path)
    monkeypatch.delenv("EXP_HARBOR_VERIFIER_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["tools.py", "verify", "--directory", str(tmp_path)])
    monkeypatch.setenv("EXP_HARBOR_ENDPOINT", "http://localhost:1234")
    with pytest.raises(KeyError):
        harbor_client.main()
    assert not (tmp_path / "reward.txt").exists() and not (tmp_path / "reward.json").exists()


@pytest.mark.parametrize(
    "verdict",
    [
        {"status": "incomplete", "reward": None, "qualification": "diagnostic_only"},
        {"status": "incomplete", "reward": None, "qualification": "qualified_observable_contract"},
        {"status": "completed", "reward": None},
        {"status": "completed", "reward": True},
        {"status": "completed", "reward": 2},
    ],
)
def test_unscored_verdict_writes_harbor_reward_file_without_a_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: JsonObject
) -> None:
    """Harbor always finds a reward file; an unscored verdict never claims a reward value."""
    _prior_pass(tmp_path)

    def response(request: Request, *, timeout: int) -> io.BytesIO:
        """Return an unscored but valid backend verdict."""
        return io.BytesIO(json.dumps(verdict).encode())

    assert _verify(tmp_path, monkeypatch, response) == 0
    assert not (tmp_path / "reward.txt").exists()
    assert json.loads((tmp_path / "reward.json").read_text()) == {"scored": 0}
    assert json.loads((tmp_path / "experiential.json").read_bytes()) == verdict


@pytest.mark.parametrize("reward", [0, 1])
def test_completed_verification_writes_current_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reward: int
) -> None:
    """A completed grade replaces a prior successful grade with the current scored reward."""
    _prior_pass(tmp_path)

    def response(request: Request, *, timeout: int) -> io.BytesIO:
        """Return a completed current grade through the actual client parser."""
        return io.BytesIO(json.dumps({"status": "completed", "reward": reward}).encode())

    assert _verify(tmp_path, monkeypatch, response) == 0
    assert not (tmp_path / "reward.txt").exists()
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": reward, "scored": 1}
    assert json.loads((tmp_path / "experiential.json").read_bytes())["reward"] == reward


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


def test_explicit_unix_failure_never_falls_back_to_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing IPC endpoint is a retained failure even if HTTP credentials are available."""
    monkeypatch.setenv("EXP_HARBOR_SOCKET", str(tmp_path / "missing.sock"))
    monkeypatch.setenv("EXP_HARBOR_ENDPOINT", "http://localhost:1234")
    monkeypatch.setenv("EXP_HARBOR_VERIFIER_TOKEN", "verifier")
    monkeypatch.setattr(sys, "argv", ["tools.py", "verify", "--directory", str(tmp_path / "grade")])

    def forbid_http(*args: object, **kwargs: object) -> None:
        """No implicit HTTP dispatch is allowed for an explicit Unix selection."""
        pytest.fail("Unix client attempted HTTP fallback")

    monkeypatch.setattr(harbor_client, "urlopen", forbid_http)
    assert harbor_client.main() == 2
    assert not (tmp_path / "grade/reward.json").exists()
    assert json.loads((tmp_path / "grade/experiential.json").read_text())["error"] == "URLError"

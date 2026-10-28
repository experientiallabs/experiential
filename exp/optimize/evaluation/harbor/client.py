"""Standalone actor/verifier transport copied into isolated Harbor environments.

Harbor requires every verified trial to leave ``reward.json`` or ``reward.txt`` in its verifier
log directory. The verifier therefore writes ``reward.json`` for every verdict the trusted backend
returns: a scored verdict carries ``{"reward": 0 or 1, "scored": 1}``, while a diagnostic,
unsubmitted, failed or unknown-outcome verdict carries only ``{"scored": 0}``, never a reward
value the backend did not grant. The complete verdict is retained in ``experiential.json``.
Only a grading failure, where no verdict exists, leaves no reward file, so Harbor records that
trial as an infrastructure exception instead of an actor outcome.
"""

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPResponse
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

VERIFIER_FILES = ("reward.json", "reward.txt", "experiential.json")


def main() -> int:
    """Send a fixed public operation, or publish a private completed verifier result."""
    parser = argparse.ArgumentParser(description="Access this trial's frozen simulator")
    parser.add_argument("operation", choices=("tools", "call", "submit", "verify", "outputs"))
    parser.add_argument(
        "--body", default="{}", help="JSON call or final answer, as declared by tools"
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="New directory for declared native outputs, or the verifier log directory "
        "(default /logs/verifier)",
    )
    args = parser.parse_args()
    if args.operation == "outputs" and args.directory is None:
        parser.error("outputs requires --directory")
    if args.operation == "verify":
        directory = args.directory or Path("/logs/verifier")
        directory.mkdir(parents=True, exist_ok=True)
        for name in VERIFIER_FILES:
            (directory / name).unlink(missing_ok=True)
    endpoint = (
        "http://localhost"
        if "EXP_HARBOR_SOCKET" in os.environ
        else os.environ["EXP_HARBOR_ENDPOINT"].rstrip("/")
    )
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("a credential-free HTTP endpoint is required")
    credential = os.environ[
        "EXP_HARBOR_VERIFIER_TOKEN" if args.operation == "verify" else "EXP_HARBOR_ACTOR_TOKEN"
    ]
    path = "/grade" if args.operation == "verify" else "/" + args.operation
    request = Request(
        endpoint + path,
        data=None if args.operation == "tools" else args.body.encode(),
        headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"},
        method="GET" if args.operation == "tools" else "POST",
    )
    if args.operation == "verify":
        return verify(request, directory)
    with open_request(request) as response:
        raw = response.read()
    if args.operation == "outputs":
        write_outputs(json.loads(raw), args.directory)
        return 0
    sys.stdout.buffer.write(raw + b"\n")
    return 0


class UnixHTTPConnection(HTTPConnection):
    """HTTP framing over only the explicitly selected local Unix socket."""

    def __init__(self, path: str) -> None:
        if not Path(path).is_absolute():
            raise ValueError("an absolute Unix backend socket path is required")
        super().__init__("localhost", timeout=120)
        self.socket_path = path

    def connect(self) -> None:
        """Never resolve a host, consult a proxy or fall back to TCP."""
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.sock.settimeout(self.timeout)
            self.sock.connect(self.socket_path)
        except BaseException:
            self.sock.close()
            raise


@contextmanager
def open_request(request: Request) -> Iterator[HTTPResponse]:
    """Use the selected explicit transport, retaining the same HTTP failure semantics."""
    if "EXP_HARBOR_SOCKET" not in os.environ:
        with urlopen(request, timeout=120) as response:
            yield response
        return
    connection = UnixHTTPConnection(os.environ["EXP_HARBOR_SOCKET"])
    try:
        connection.request(
            request.get_method(), request.selector, request.data, dict(request.header_items())
        )
        response = connection.getresponse()
        if response.status >= 400:
            raise HTTPError(
                request.full_url,
                response.status,
                response.reason,
                response.headers,
                BytesIO(response.read()),
            )
        with response:
            yield response
    except OSError as error:
        if isinstance(error, HTTPError):
            raise
        raise URLError(error) from error
    finally:
        connection.close()


def verify(request: Request, directory: Path) -> int:
    """Retain the backend verdict and publish the Harbor reward file it supports.

    Args:
        request: Authorized private grading request.
        directory: Verifier log directory, already cleared of earlier attempts.

    Returns:
        Zero when a verdict was recorded, two when grading failed and no reward exists.
    """
    try:
        with open_request(request) as response:
            raw = response.read()
    except HTTPError as error:
        body = error.read().decode(errors="replace")
        return grading_failed(directory, {"http_status": error.code, "response": body})
    except URLError as error:
        return grading_failed(directory, {"error": type(error).__name__, "reason": str(error)})
    (directory / "experiential.json").write_bytes(raw)
    try:
        result = json.loads(raw)
    except ValueError:
        return grading_failed(directory, None, "grading response is not JSON")
    if not isinstance(result, dict):
        return grading_failed(directory, None, "grading response is not a JSON object")
    reward = result.get("reward")
    scored = result.get("status") == "completed" and type(reward) is int and reward in (0, 1)
    rewards = {"reward": reward, "scored": 1} if scored else {"scored": 0}
    (directory / "reward.json").write_text(json.dumps(rewards) + "\n")
    if not scored:
        sys.stderr.write(
            "Experiential verdict is unscored ("
            + str(result.get("status"))
            + ", "
            + str(result.get("qualification"))
            + "); see experiential.json for the retained outcome.\n"
        )
    return 0


def grading_failed(directory: Path, detail: dict[str, int | str] | None, reason: str = "") -> int:
    """Retain a grading failure without any reward file, so Harbor reports an exception.

    Args:
        directory: Verifier log directory.
        detail: Transport failure to retain, or None to keep an already saved raw response.
        reason: Human-readable failure summary.

    Returns:
        The verifier failure exit status.
    """
    if detail is not None:
        record = {"status": "grading_failed", **detail}
        (directory / "experiential.json").write_text(json.dumps(record) + "\n")
        reason = reason or json.dumps(detail)
    sys.stderr.write(
        "Experiential grading failed: "
        + reason
        + ". No reward was written; confirm the trusted trial backend is still serving, "
        "then verify again.\n"
    )
    return 2


def write_outputs(result: dict[str, list[dict[str, str]]], directory: Path) -> None:
    """Verify declared file bytes and stage them without overwriting local actor files."""
    if not isinstance(result, dict) or not isinstance(result.get("files"), list):
        raise ValueError("native output response has no file inventory")
    files: dict[str, bytes] = {}
    for row in result["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("native output entry is malformed")
        name = row["path"]
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or PureWindowsPath(name).drive
            or ".." in path.parts
            or "\\" in name
            or path.as_posix() != name
            or name in files
        ):
            raise ValueError("native output path is unsafe or repeated")
        payload = base64.b64decode(row["content_base64"], validate=True)
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("native output digest mismatch")
        files[name] = payload
    if not files or len(files) > 32 or sum(map(len, files.values())) > 32 * 1024 * 1024:
        raise ValueError("native output exceeds the file allowance")
    if directory.exists() or directory.is_symlink():
        raise ValueError("native output directory exists; choose a new directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".native-outputs-", dir=directory.parent))
    try:
        for name, payload in files.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    sys.exit(main())

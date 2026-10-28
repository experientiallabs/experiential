"""Standalone actor/verifier transport copied into isolated Harbor environments."""

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> int:
    """Send a fixed public operation, or publish a private completed verifier result."""
    parser = argparse.ArgumentParser(description="Access this trial's frozen simulator")
    parser.add_argument("operation", choices=("tools", "call", "submit", "verify", "outputs"))
    parser.add_argument(
        "--body", default="{}", help="JSON call or final answer, as declared by tools"
    )
    parser.add_argument("--directory", type=Path, help="New directory for declared native outputs")
    args = parser.parse_args()
    if args.operation == "outputs" and args.directory is None:
        parser.error("outputs requires --directory")
    if args.operation == "verify":
        directory = Path("/logs/verifier")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reward.txt").unlink(missing_ok=True)
        (directory / "experiential.json").unlink(missing_ok=True)
    endpoint = os.environ["EXP_HARBOR_ENDPOINT"].rstrip("/")
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
    with urlopen(request, timeout=120) as response:
        raw = response.read()
    if args.operation == "outputs":
        write_outputs(json.loads(raw), args.directory)
        return 0
    if args.operation != "verify":
        sys.stdout.buffer.write(raw + b"\n")
        return 0
    result = json.loads(raw)
    (directory / "experiential.json").write_bytes(raw)
    reward = result.get("reward") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("status") != "completed"
        or type(reward) is not int
        or reward not in (0, 1)
    ):
        return 2
    (directory / "reward.txt").write_text(str(reward) + "\n")
    return 0


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

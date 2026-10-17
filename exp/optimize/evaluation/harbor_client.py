"""Standalone actor/verifier transport copied into isolated Harbor environments."""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main() -> int:
    """Send a fixed public operation, or publish a private completed verifier result."""
    parser = argparse.ArgumentParser(description="Access this trial's frozen simulator")
    parser.add_argument("operation", choices=("tools", "call", "submit", "verify"))
    parser.add_argument(
        "--body", default="{}", help="JSON call or final answer, as declared by tools"
    )
    args = parser.parse_args()
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
    if args.operation != "verify":
        sys.stdout.buffer.write(raw + b"\n")
        return 0
    result = json.loads(raw)
    directory = Path("/logs/verifier")
    directory.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    sys.exit(main())

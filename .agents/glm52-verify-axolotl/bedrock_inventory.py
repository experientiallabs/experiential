"""List Bedrock inference profiles relevant to trajectory judging."""

from __future__ import annotations

import argparse
import json
import logging

import boto3


LOGGER = logging.getLogger(__name__)


def list_profiles(region: str) -> None:
    """Log active Anthropic system inference profiles in a region."""
    client = boto3.client("bedrock", region_name=region)
    profiles: list[dict[str, str]] = []
    next_token: str | None = None
    while True:
        request: dict[str, str] = {"typeEquals": "SYSTEM_DEFINED"}
        if next_token:
            request["nextToken"] = next_token
        response = client.list_inference_profiles(**request)
        for summary in response.get("inferenceProfileSummaries", []):
            profile_id = str(summary.get("inferenceProfileId", ""))
            profile_name = str(summary.get("inferenceProfileName", ""))
            if "anthropic" in profile_id.lower() or "claude" in profile_name.lower():
                profiles.append(
                    {
                        "id": profile_id,
                        "name": profile_name,
                        "status": str(summary.get("status", "")),
                        "type": str(summary.get("type", "")),
                    }
                )
        next_token = response.get("nextToken")
        if not next_token:
            break
    LOGGER.info("%s", json.dumps(profiles, indent=2, sort_keys=True))


def main() -> None:
    """Parse command-line arguments and list profiles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-west-1")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    list_profiles(args.region)


if __name__ == "__main__":
    main()

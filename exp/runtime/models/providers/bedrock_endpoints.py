"""Package-owned Amazon Bedrock endpoint metadata resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import NamedTuple, Protocol, cast

from exp.runtime.models.providers.transport import ProviderTransportError


class BotocoreLoader(Protocol):
    """Built-in-only botocore data loader."""

    def load_data(self, name: str) -> Mapping[str, object]:
        """Load one bundled botocore metadata document."""


class _EndpointResolver(Protocol):
    """Botocore partition resolver used for the native streaming origin."""

    def construct_endpoint(
        self,
        service_name: str,
        region_name: str,
        *,
        use_fips_endpoint: bool = False,
    ) -> Mapping[str, object] | None:
        """Resolve one service endpoint without constructing a client."""


class BedrockRuntimeEndpoint(NamedTuple):
    """Official runtime origin and canonical SigV4 region."""

    origin: str
    signing_region: str


@cache
def resolve_bedrock_runtime_endpoint(region_name: str) -> BedrockRuntimeEndpoint:
    """Resolve the official origin and signing region from bundled metadata."""
    try:
        from botocore.regions import EndpointResolver
    except ImportError as exc:
        raise RuntimeError("Bedrock requires botocore; install experiential") from exc
    resolver = cast(
        "_EndpointResolver",
        EndpointResolver(built_in_botocore_loader().load_data("endpoints")),
    )
    canonical_region, use_fips = _canonical_fips_region(region_name)
    endpoint = resolver.construct_endpoint(
        "bedrock-runtime",
        canonical_region,
        use_fips_endpoint=use_fips,
    )
    hostname = None if endpoint is None else endpoint.get("hostname")
    protocols = () if endpoint is None else endpoint.get("protocols", ())
    if (
        not isinstance(hostname, str)
        or not isinstance(protocols, Sequence)
        or "https" not in protocols
    ):
        raise ProviderTransportError(
            f"Bedrock has no HTTPS runtime endpoint for region {region_name!r}"
        )
    return BedrockRuntimeEndpoint(
        origin=f"https://{hostname}",
        signing_region=canonical_region,
    )


def bedrock_runtime_origin(region_name: str) -> str:
    """Return the official HTTPS Bedrock Runtime origin."""
    return resolve_bedrock_runtime_endpoint(region_name).origin


def bedrock_signing_region(region_name: str) -> str:
    """Return the canonical region used in Bedrock SigV4 scope."""
    return resolve_bedrock_runtime_endpoint(region_name).signing_region


def _canonical_fips_region(region_name: str) -> tuple[str, bool]:
    """Normalize either botocore-supported FIPS pseudo-region spelling."""
    if region_name.startswith("fips-"):
        return region_name.removeprefix("fips-"), True
    if region_name.endswith("-fips"):
        return region_name.removesuffix("-fips"), True
    return region_name, False


def built_in_botocore_loader() -> BotocoreLoader:
    """Create a loader restricted to package-owned endpoint metadata."""
    try:
        from botocore.loaders import Loader
    except ImportError as exc:
        raise RuntimeError("Bedrock requires botocore; install experiential") from exc
    return cast(
        "BotocoreLoader",
        Loader(
            extra_search_paths=[Loader.BUILTIN_DATA_PATH],
            include_default_search_paths=False,
        ),
    )

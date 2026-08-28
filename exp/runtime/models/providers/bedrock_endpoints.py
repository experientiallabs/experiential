"""Package-owned Amazon Bedrock endpoint metadata resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import cache
from typing import Protocol, cast

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
    ) -> Mapping[str, object] | None:
        """Resolve one service endpoint without constructing a client."""


@cache
def bedrock_runtime_origin(region_name: str) -> str:
    """Resolve the HTTPS Bedrock Runtime origin from bundled AWS metadata."""
    try:
        from botocore.regions import EndpointResolver
    except ImportError as exc:
        raise RuntimeError("Bedrock requires botocore; install experiential") from exc
    resolver = cast(
        "_EndpointResolver",
        EndpointResolver(built_in_botocore_loader().load_data("endpoints")),
    )
    endpoint = resolver.construct_endpoint("bedrock-runtime", region_name)
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
    return f"https://{hostname}"


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

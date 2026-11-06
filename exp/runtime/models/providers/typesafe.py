"""TypeSafe SystemOne connection facts for native decision dispatch only."""

from __future__ import annotations

from urllib.parse import urlsplit

from exp.common.models import BillingSource, ModelRequest, ModelResponse, ModelSnapshot
from exp.common.models.self_hosted import is_loopback_url
from exp.runtime.gateway.organization_tag import organization_tag_key
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.transport import JsonHttpTransport

TYPESAFE_BASE_URL = "https://api.typesafe.ai/v1"
TYPESAFE_IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"
"""Opt-in header TypeSafe requires on every request that carries an image asset."""


class TypeSafeClient:
    """Describe SystemOne's native wire without pretending it is a chat client.

    The native gateway owns decision HTTP requests, deadlines, and settlement.
    This object satisfies the catalog's ModelClient seam only to reject accidental
    completion use locally; it never adapts decisions into assistant messages.
    """

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str | None,
        base_url: str = TYPESAFE_BASE_URL,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        self_hosted: bool = False,
    ) -> None:
        """Bind one explicit SystemOne connection without network work.

        Args:
            model: Exact provider model identity selected by the catalog.
            api_key: Credential already released by the named connection; a self-hosted server
                may take none.
            base_url: Official API root, a catalog-approved trusted HTTPS origin, or, when
                ``self_hosted``, a loopback HTTP(S) root.
            transport: Registry construction seam, unused because Rust owns dispatch.
            self_hosted: The catalog marked the connection as an operator-run loopback server.

        Raises:
            ValueError: The credential is empty where one is required, or the endpoint is not a
                clean HTTPS URL (or, self-hosted, a clean loopback URL).
        """
        del transport
        if not api_key and not self_hosted:
            raise ValueError("TypeSafeClient requires a non-empty API key")
        parsed = urlsplit(base_url)
        schemes = {"http", "https"} if self_hosted else {"https"}
        if (
            parsed.scheme not in schemes
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or (self_hosted and not is_loopback_url(base_url))
        ):
            raise ValueError(
                "TypeSafe requires an HTTPS base URL (or, self-hosted, a loopback URL) without "
                "credentials or query"
            )
        self._model = model
        self._api_key = api_key or ""
        self._base_url = base_url.rstrip("/")

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Refuse chat completions before any provider dispatch.

        Args:
            request: A conversational request, unsupported by SystemOne.

        Raises:
            ProviderCapabilityError: Always; use the gateway's decisions surface instead.
        """
        del request
        raise ProviderCapabilityError(capability="completions")

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the native decision endpoint and bearer-authenticated wire facts.

        A catalog-approved origin other than the official API is a self-hosted
        SystemOne server, which fair-queues by organization, so its profile
        carries the private key for the per-organization scheduling tag. A
        self-hosted server reached without a credential gets neither the bearer
        header nor the tag.
        """
        url = f"{self._base_url}/systemone"
        tagged = self._base_url != TYPESAFE_BASE_URL and bool(self._api_key)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return GatewayWireProfile(
            dialect="typesafe_systemone",
            url=url,
            decisions_url=url,
            headers=headers,
            model_id=self._model.model_id,
            billing_customer_managed=self._model.billing_source == BillingSource.CUSTOMER_MANAGED,
            supports_temperature=False,
            supports_top_p=False,
            organization_tag_key=organization_tag_key(self._api_key) if tagged else None,
        )

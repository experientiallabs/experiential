"""TypeSafe SystemOne connection facts for native decision dispatch only."""

from __future__ import annotations

from urllib.parse import urlsplit

from exp.common.models import BillingSource, ModelRequest, ModelResponse, ModelSnapshot
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.transport import JsonHttpTransport

TYPESAFE_BASE_URL = "https://api.typesafe.ai/v1"


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
        api_key: str,
        base_url: str = TYPESAFE_BASE_URL,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
    ) -> None:
        """Bind one explicit, authenticated SystemOne connection without network work.

        Args:
            model: Exact provider model identity selected by the catalog.
            api_key: Credential already released by the named connection.
            base_url: Official API root, or a catalog-approved trusted HTTPS origin.
            transport: Registry construction seam, unused because Rust owns dispatch.

        Raises:
            ValueError: The credential is empty or the endpoint is not a clean HTTPS URL.
        """
        del transport
        if not api_key:
            raise ValueError("TypeSafeClient requires a non-empty API key")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("TypeSafe requires an HTTPS base URL without credentials or query")
        self._model = model
        self._api_key = api_key
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
        """Return the native decision endpoint and bearer-authenticated wire facts."""
        url = f"{self._base_url}/systemone"
        return GatewayWireProfile(
            dialect="typesafe_systemone",
            url=url,
            decisions_url=url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            model_id=self._model.model_id,
            billing_customer_managed=self._model.billing_source == BillingSource.CUSTOMER_MANAGED,
            supports_temperature=False,
            supports_top_p=False,
        )

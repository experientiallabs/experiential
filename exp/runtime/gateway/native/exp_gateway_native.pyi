"""Type stubs for the exp_gateway_native extension module."""

from typing import Protocol

__version__: str

class _ControlPlane(Protocol):
    """The callback surface the data plane requires (see NativeControlPlane)."""

    def authenticate(self, argument: str) -> str: ...
    def admit(self, argument: str) -> str: ...
    def settle(self, argument: str) -> str: ...
    def models(self, argument: str) -> str: ...
    def model_detail(self, argument: str) -> str: ...
    def usage_json(self, argument: str) -> str: ...
    def readiness(self, argument: str) -> str: ...

def serve(control_plane: _ControlPlane, config_json: str) -> None: ...
def decode_chat_canonical(
    body: str,
    idempotency_key: str | None = None,
    client_request_id: str | None = None,
) -> str: ...
def build_upstream_payload(
    dialect: str,
    canonical_request: str,
    model_id: str,
    supports_temperature: bool = True,
    reasoning_effort: str | None = None,
    token_limit_key: str = "max_tokens",
) -> str: ...
def encode_chat_fixture(
    request_id: str,
    model: str,
    created_at: int,
    include_usage: bool,
    events_json: str,
) -> list[str]: ...

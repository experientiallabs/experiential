"""Administrative tool search forwards to the same native request owner."""

from unittest.mock import Mock

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_tool_search import NativeToolSearchMixin


def test_tool_search_round_forwards_without_reading_python_request_state() -> None:
    """The administrative adapter neither finds a Python entry nor executes a round."""
    owner = Mock(spec=NativeAttemptAccounting)
    owner.call.return_value = '{"loaded":true}'
    plane = NativeToolSearchMixin()
    plane._accounting = owner  # noqa: SLF001 - compose the adapter boundary under test.
    argument = '{"request_id":"request","route_depth":0,"calls":[]}'
    assert plane.tool_search_round(argument) == '{"loaded":true}'
    owner.call.assert_called_once_with("tool_search_round", argument)

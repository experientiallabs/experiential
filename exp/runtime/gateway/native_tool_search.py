"""Administrative access to the admitted native request owner."""

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


class NativeToolSearchMixin:
    """Use the same native implementation as serving callbacks."""

    _accounting: NativeAttemptAccounting

    def tool_search_round(self, argument: str) -> str:
        """Dispatch this operation through the admitted native owner."""
        return self._accounting.call("tool_search_round", argument)

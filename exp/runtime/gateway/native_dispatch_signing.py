"""Administrative access to the admitted native request owner."""

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting


class NativeDispatchSigningMixin:
    """Use the same native implementation as serving callbacks."""

    _accounting: NativeAttemptAccounting

    def sign_dispatch(self, argument: str) -> str:
        """Dispatch this operation through the admitted native owner."""
        return self._accounting.call("sign_dispatch", argument)

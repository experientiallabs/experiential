"""The fidelity judge, resumable after an unknown outcome without losing its profile."""

from collections.abc import Callable
from functools import partial

from exp.common.core.artifacts import JsonObject, Sha256
from exp.common.models.unknown_outcome import resume_unknown
from exp.runtime.models.structured_completion import StructuredCompletion


class ResumableJudge:
    """The profiled judge, re-dispatched after an unknown outcome only under resumption.

    It keeps the judge's model and completion-profile identity, so the evaluator still
    records which inference configuration produced the judgment.
    """

    def __init__(
        self,
        inner: StructuredCompletion,
        retried: Callable[[str], None] | None,
        started: Callable[[], None],
    ) -> None:
        """Wrap one profiled judge completion, its resumption recorder and progress."""
        self._inner, self._retried, self._started = inner, retried, started

    @property
    def model_identity(self) -> str:
        """The judge's configured completion model."""
        return self._inner.model_identity

    @property
    def completion_profile_sha256(self) -> Sha256:
        """The judge's inference configuration identity."""
        return self._inner.completion_profile_sha256

    def __call__(self, body: JsonObject, logical_id: str) -> JsonObject:
        """Send or replay the judgment; a lost one is re-sent only when resuming."""
        self._started()
        return resume_unknown(partial(self._inner, body), logical_id, self._retried)

"""Tests for finalized manual judge protocol execution."""

import pytest

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelResponse,
    ModelSnapshot,
    OperationEconomics,
)
from wmo.optimize.router.judging.contracts import ManualJudgeError
from wmo.optimize.router.judging.protocol import _raw_response


def _response(content: str) -> ModelResponse:
    """Wrap one visible judge text body in a provider response.

    Args:
        content: Visible assistant text returned by the judge.

    Returns:
        Response carrying the text under a fixed configured model identity.
    """
    return ModelResponse(
        output=AssistantAction(content=content),
        model=ModelSnapshot(
            provider="bedrock",
            model_id="judge-model",
            capabilities_sha256=sha256_json(ModelCapabilities()),
            connection_sha256=sha256_json({"provider": "bedrock"}),
        ),
        economics=OperationEconomics(),
    )


def test_fenced_judge_json_is_accepted_and_unfenced_prose_is_rejected() -> None:
    """Providers that fence schema-valid judge JSON remain usable for calibration."""
    fenced = '```json\n{"dimensions": [{"dimension_id": "task-success"}]}\n```'
    explained = 'The agent failed the task.\n\n```json\n{"dimensions": []}\n```'

    assert _raw_response(_response(fenced)) == {"dimensions": [{"dimension_id": "task-success"}]}
    assert _raw_response(_response(explained)) == {"dimensions": []}
    with pytest.raises(ManualJudgeError, match="malformed structured JSON"):
        _raw_response(_response('My verdict is {"dimensions": []} overall.'))
    with pytest.raises(ManualJudgeError, match="must be a JSON object"):
        _raw_response(_response("[1, 2]"))

"""Public official OpenAI Python client over a loaded project router."""

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

from wmo.runtime.router.application import load_router
from wmo.runtime.router.runtime_test import _runtime


def test_load_router_exposes_official_chat_and_responses_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy-path Python API needs no WMO request or message types."""
    runtime, model_client = _runtime()
    monkeypatch.setattr(
        "wmo.runtime.router.application.load_project_router",
        lambda project, root, **kwargs: runtime,
    )

    with load_router("support-agent") as router:
        assert isinstance(router, OpenAI)
        chat = router.chat.completions.create(
            model="support-agent",
            messages=[{"role": "user", "content": "Help me"}],
        )
        response = router.responses.create(model="support-agent", input="Help me")

    assert isinstance(chat, ChatCompletion)
    assert isinstance(response, Response)
    assert model_client.complete_calls == 2

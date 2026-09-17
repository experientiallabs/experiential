"""SDK request validation for exact learner sampling."""

import pytest

from exp.optimize.claas.service.http.inputs import parse_generation


def test_requests_preserve_explicit_token_bound() -> None:
    """Responses and raw completions keep their distinct input semantics."""
    response = parse_generation(
        {"model": "student", "input": "hello", "max_output_tokens": 17}, "responses", "resp-1"
    )
    completion = parse_generation(
        {"model": "student", "prompt": "hello", "max_tokens": 19}, "completions", "cmpl-1"
    )
    assert response.messages[0].content == "hello"
    assert response.maximum_output_tokens == 17
    assert completion.prompt == "hello"
    assert completion.maximum_output_tokens == 19


def test_unsupported_sampling_does_not_become_training_evidence() -> None:
    """Never silently change a caller's decoding distribution."""
    with pytest.raises(ValueError, match="temperature=1"):
        parse_generation(
            {"model": "student", "prompt": "hello", "temperature": 0.7}, "completions", "id"
        )
    with pytest.raises(ValueError, match="unsupported learner"):
        parse_generation(
            {"model": "student", "input": "hello", "previous_response_id": "old"}, "responses", "id"
        )

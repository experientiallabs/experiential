"""Tests for visible structured-JSON provider text normalization."""

from __future__ import annotations

import json

from wmo.common.models.structured import structured_json_text


def test_labeled_fence_is_unwrapped() -> None:
    """A single labeled Markdown fence yields the JSON body."""
    content = '```json\n{"dimensions": []}\n```'
    assert json.loads(structured_json_text(content)) == {"dimensions": []}


def test_unlabeled_fence_and_trailing_whitespace_are_unwrapped() -> None:
    """An unlabeled fence with surrounding whitespace yields the JSON body."""
    content = '\n```\n{"message": "next", "terminal": false}\n```\n'
    assert json.loads(structured_json_text(content)) == {"message": "next", "terminal": False}


def test_bare_json_is_unchanged() -> None:
    """Unfenced JSON text passes through with whitespace stripped only."""
    assert structured_json_text(' {"a": 1} ') == '{"a": 1}'


def test_explanation_before_one_fenced_block_is_discarded() -> None:
    """A single fenced block is unambiguous even when the provider explains itself first."""
    content = 'Here is my answer:\n```json\n{"a": 1}\n```'
    assert json.loads(structured_json_text(content)) == {"a": 1}


def test_unfenced_prose_stays_invalid() -> None:
    """Prose with no fenced block is preserved so strict parsing still fails."""
    content = 'Here is my answer: {"a": 1} and some trailing thoughts.'
    assert structured_json_text(content) == content


def test_several_fenced_blocks_stay_invalid() -> None:
    """More than one fenced block is preserved so strict parsing still fails."""
    content = '```json\n{"a": 1}\n```\n```json\n{"a": 2}\n```'
    assert structured_json_text(content) == content


def test_text_after_the_closing_fence_stays_invalid() -> None:
    """A fenced block followed by prose is preserved so strict parsing still fails."""
    content = '```json\n{"a": 1}\n```\nLet me know if you need anything else.'
    assert structured_json_text(content) == content


def test_non_json_fence_label_stays_invalid() -> None:
    """A fence labeled with another language is preserved so strict parsing still fails."""
    content = '```python\n{"a": 1}\n```'
    assert structured_json_text(content) == content


def test_single_line_fence_is_unwrapped() -> None:
    """A fence without an inner newline yields the JSON body."""
    assert structured_json_text('```{"a": 1}```') == '{"a": 1}'

"""Prompt versions record one edit each and reproduce from their parent."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.project import ArtifactStore, ProjectPaths
from exp.optimize.evaluation.hillclimb.prompts import (
    PromptEdit,
    baseline_version,
    edited_version,
    load_prompt_version,
    load_reference,
    persist_prompt_version,
)


def test_append_and_replace_edits_chain_into_saved_versions(tmp_path: Path) -> None:
    """Each version names its parent and edit; saving twice returns the same reference."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    root = baseline_version("climb", None)
    first = edited_version(root, PromptEdit(kind="append", text=" Check. ", change_note="a"))
    assert first.system_prompt == "Check." and first.parent == root.identity
    second = edited_version(first, PromptEdit(kind="append", text="Verify.", change_note="b"))
    assert second.system_prompt == "Check.\n\nVerify."
    third = edited_version(second, PromptEdit(kind="replace", text="Only this.", change_note="c"))
    assert third.system_prompt == "Only this."
    now = datetime.now(UTC)
    for version in (root, first, second, third):
        reference = persist_prompt_version(store, version, created_at=now, code_revision="t")
        assert persist_prompt_version(store, version, created_at=now, code_revision="t") == (
            reference
        )
        assert load_prompt_version(store, reference) == version
    manifest = store.read(third.identity).manifest
    assert manifest.inputs == (load_reference(store, second.identity),)


def test_an_edit_that_changes_nothing_is_refused() -> None:
    """Replacing a prompt with itself is not a round."""
    root = baseline_version("climb", "Same.")
    with pytest.raises(ValueError, match="leaves the system prompt unchanged"):
        edited_version(root, PromptEdit(kind="replace", text="Same. ", change_note="noop"))

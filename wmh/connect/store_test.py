"""Tests for context bundle persistence and markdown rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wmh.connect.store import BundleManifest, ContextStore, render_markdown
from wmh.connect.types import ContextItem, ItemKind, PullQuery


def _item(n: int, body: str = "hello world") -> ContextItem:
    return ContextItem(
        id=str(n),
        source="github",
        kind=ItemKind.ISSUE,
        title=f"Issue {n}",
        body=body,
        url=f"https://github.test/i/{n}",
        created_at="2026-07-01T00:00:00+00:00",
        updated_at="2026-07-02T00:00:00+00:00",
    )


def _manifest(name: str = "github", count: int = 2) -> BundleManifest:
    return BundleManifest(
        name=name,
        connector="github",
        query=PullQuery(target="octo/repo", limit=50),
        pulled_at="2026-07-15T10:00:00+00:00",
        item_count=count,
        account="octocat",
    )


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    items = [_item(1), _item(2)]

    bundle_dir = store.save(_manifest(), items)

    assert bundle_dir == tmp_path / ".wmh" / "context" / "github"
    manifest_doc = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_doc["account"] == "octocat"  # the singular on-disk key
    lines = (bundle_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["id"] == "1"

    manifest, loaded = store.load("github")
    assert manifest == _manifest()
    assert loaded == items


def test_save_refuses_overwrite_unless_asked(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    store.save(_manifest(), [_item(1)])

    with pytest.raises(FileExistsError, match="overwrite=True"):
        store.save(_manifest(), [_item(2)])

    store.save(_manifest(count=1), [_item(3)], overwrite=True)
    _, items = store.load("github")
    assert [item.id for item in items] == ["3"]


def test_list_bundles_sorted_and_delete(tmp_path: Path) -> None:
    store = ContextStore(tmp_path)
    assert store.list_bundles() == []

    store.save(_manifest("z-bundle"), [])
    store.save(_manifest("a-bundle"), [])

    assert [manifest.name for manifest in store.list_bundles()] == ["a-bundle", "z-bundle"]
    assert store.delete("z-bundle")
    assert not store.delete("z-bundle")
    assert [manifest.name for manifest in store.list_bundles()] == ["a-bundle"]


def test_load_missing_bundle_is_actionable(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ContextStore.save"):
        ContextStore(tmp_path).load("nope")


def test_bundle_names_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context bundle name"):
        ContextStore(tmp_path).save(_manifest("../evil"), [])


def test_render_markdown_includes_provenance_and_item_sections() -> None:
    text = render_markdown(_manifest(), [_item(1), _item(2)])

    assert text.startswith("# Context bundle: github\n")
    assert "- connector: github" in text
    assert "- pulled_at: 2026-07-15T10:00:00+00:00" in text
    assert "- account: octocat" in text
    assert "- query: target=octo/repo, limit=50" in text
    assert "## Issue 1" in text and "## Issue 2" in text
    assert (
        "issue | created 2026-07-01T00:00:00+00:00 | updated 2026-07-02T00:00:00+00:00"
        " | https://github.test/i/1"
    ) in text
    assert "hello world" in text
    # Deterministic: rendering the same bundle twice yields byte-identical output.
    assert render_markdown(_manifest(), [_item(1), _item(2)]) == text


def test_render_markdown_within_budget_is_untruncated() -> None:
    items = [_item(1), _item(2)]
    full = render_markdown(_manifest(), items)
    assert render_markdown(_manifest(), items, max_chars=len(full)) == full
    assert "items omitted" not in full


def test_render_markdown_truncates_whole_items_from_the_tail() -> None:
    items = [_item(n, body="x" * 400) for n in range(1, 6)]
    full = render_markdown(_manifest(count=5), items)
    budget = len(full) - 300

    text = render_markdown(_manifest(count=5), items, max_chars=budget)

    assert len(text) <= budget
    assert "## Issue 1" in text
    assert "## Issue 5" not in text
    assert "items omitted" in text
    assert text.rstrip().endswith("items omitted")

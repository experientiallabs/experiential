"""Tests for the versioned harness store: append-only versions, aliases, load paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.store import CHAMPION_ALIAS, HarnessStore


def _variant(name: str, prompt: str) -> HarnessDoc:
    base = HarnessDoc.baseline(name)
    surfaces = [
        s.model_copy(update={"content": prompt}) if s.id == "prompt:core" else s
        for s in base.surfaces
    ]
    return HarnessDoc(name=name, surfaces=surfaces)


def test_save_assigns_incrementing_versions(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    v1 = store.save_version(HarnessDoc.baseline("h"))
    v2 = store.save_version(_variant("h", "be careful"))
    assert (v1.version, v2.version) == (1, 2)
    assert store.versions("h") == [1, 2]
    # Load round-trips the exact document, stamped with its version.
    assert store.load("h", "v1") == v1
    assert store.load("h", "2") == v2


def test_default_load_prefers_champion_alias_else_latest(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    store.save_version(HarnessDoc.baseline("h"))
    store.save_version(_variant("h", "v2 prompt"))
    assert store.load("h").version == 2  # no alias yet -> latest
    store.set_alias("h", CHAMPION_ALIAS, 1)
    assert store.load("h").version == 1  # champion wins over latest
    # Rollback/promotion is just re-pointing.
    store.set_alias("h", CHAMPION_ALIAS, 2)
    assert store.load("h").version == 2


def test_alias_to_missing_version_rejected(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    store.save_version(HarnessDoc.baseline("h"))
    with pytest.raises(ValueError, match="no version v9"):
        store.set_alias("h", CHAMPION_ALIAS, 9)


def test_unknown_ref_and_missing_harness_are_friendly(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="no harness named"):
        store.load("ghost")
    store.save_version(HarnessDoc.baseline("h"))
    with pytest.raises(ValueError, match="no version or alias"):
        store.load("h", "prod")


def test_versions_are_append_only(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    store.save_version(HarnessDoc.baseline("h"))
    # The rendered export exists beside the authoritative doc.json.
    v1_dir = store.dir_for("h") / "v1"
    assert (v1_dir / "doc.json").exists()
    assert (v1_dir / "SYSTEM.md").exists()
    assert (v1_dir / "config.toml").exists()
    before = (v1_dir / "doc.json").read_text(encoding="utf-8")
    store.save_version(_variant("h", "new"))
    assert (v1_dir / "doc.json").read_text(encoding="utf-8") == before  # v1 untouched


def test_hand_authored_dir_without_doc_json_loads(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    directory = store.dir_for("hand") / "v1"
    directory.mkdir(parents=True)
    (directory / "SYSTEM.md").write_text("You are careful.", encoding="utf-8")
    (directory / "config.toml").write_text(
        '[harness]\ntools = ["bash", "submit"]\nmax_turns = 7\ntemperature = 0.2\n',
        encoding="utf-8",
    )
    doc = store.load("hand")
    assert doc.system_prompt() == "You are careful."
    assert doc.tools() == ["bash", "submit"]
    assert doc.max_turns() == 7
    assert doc.temperature() == 0.2


def test_list_names_only_counts_dirs_with_versions(tmp_path: Path) -> None:
    store = HarnessStore(tmp_path)
    store.save_version(HarnessDoc.baseline("real"))
    (store.harnesses_dir / "empty").mkdir(parents=True)
    assert store.list_names() == ["real"]


def test_skill_surfaces_render_and_reload(tmp_path: Path) -> None:
    from wmh.harness.skills import Skill

    skill = Skill(name="count-words", description="count words", body="wc -w <p>")
    doc = HarnessDoc(
        name="h",
        surfaces=[
            *HarnessDoc.baseline("h").surfaces,
            Surface(id="skill:count-words", kind=SurfaceKind.SKILL, content=skill.to_markdown()),
        ],
    )
    store = HarnessStore(tmp_path)
    saved = store.save_version(doc)
    assert (store.dir_for("h") / "v1" / "skills" / "count-words.md").exists()
    reloaded = store.load("h")
    assert reloaded == saved
    assert [s.name for s in reloaded.skills()] == ["count-words"]


def test_code_surface_round_trips_through_render_and_parse(tmp_path) -> None:  # noqa: ANN001
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    store = HarnessStore(tmp_path)
    saved = store.save_version(code_baseline("coded"))
    exported = store.dir_for("coded") / f"v{saved.version}" / "runtime.py"
    assert exported.exists()
    # Hand-authored dirs (no doc.json) recover the code surface from the rendered file.
    (store.dir_for("coded") / f"v{saved.version}" / "doc.json").unlink()
    loaded = store.load("coded")
    code = loaded.surface(CODE_RUNTIME_ID)
    assert code is not None
    assert code.content == exported.read_text(encoding="utf-8")

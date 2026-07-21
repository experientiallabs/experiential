"""Tests for the versioned harness store: append-only versions, aliases, load paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.harness.doc import MAX_OUTPUT_TOKENS_ID, HarnessDoc, Surface, SurfaceKind
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
    assert doc.max_output_tokens() == 4096  # old exports keep the runtime default
    assert doc.temperature() == 0.2


def test_model_output_budget_renders_and_reparses_without_doc_json(tmp_path: Path) -> None:
    """The editable Pi model budget survives the portable config.toml representation."""
    base = HarnessDoc.baseline("budgeted")
    doc = HarnessDoc(
        name="budgeted",
        surfaces=[
            *base.surfaces,
            Surface(id=MAX_OUTPUT_TOKENS_ID, kind=SurfaceKind.PARAM, content="16384"),
        ],
    )
    store = HarnessStore(tmp_path)
    saved = store.save_version(doc)
    version_dir = store.dir_for(doc.name) / f"v{saved.version}"
    assert "max_output_tokens = 16384" in (version_dir / "config.toml").read_text()

    (version_dir / "doc.json").unlink()
    assert store.load(doc.name).max_output_tokens() == 16384


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


def test_pi_node_source_tree_renders_and_round_trips(tmp_path: Path) -> None:
    """A pi-node harness renders its full source tree; a doc.json-less dir recovers it.

    This is what the agent Harness view browses and what a sandbox downloads: the export must be
    the harness's real directory structure (`src/...`), not just SYSTEM.md/config.toml.
    """
    from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID
    from wmh.harness.pi_vendor import pi_agent_code_surfaces

    code_surfaces = pi_agent_code_surfaces()
    assert code_surfaces, "vendored pi tree should yield code surfaces"
    doc = HarnessDoc(
        name="pi",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            *code_surfaces,
        ],
    )
    store = HarnessStore(tmp_path)
    saved = store.save_version(doc)
    version_dir = store.dir_for("pi") / f"v{saved.version}"

    # Every code surface rendered to its own path — a real, nested directory tree.
    for surface in code_surfaces:
        assert surface.path is not None
        rendered = version_dir / surface.path
        assert rendered.exists(), f"{surface.path} missing from the export"
        assert rendered.read_text(encoding="utf-8") == surface.content
    assert any("/" in (s.path or "") for s in code_surfaces)  # genuinely nested, not flat

    # doc.json is authoritative when present.
    assert store.load("pi").doc_hash == doc.doc_hash
    # Without it, the rendered tree still reconstructs a pi-node harness with the identical code
    # files (id + path + content). The full doc hash need not match — config.toml materializes the
    # effective scalar defaults the sparse doc omitted — but the source tree is preserved.
    (version_dir / "doc.json").unlink()
    reloaded = store.load("pi")
    assert reloaded.runtime_kind() == "pi-node"
    original_code = {(s.id, s.path): s.content for s in doc.code_files()}
    reloaded_code = {(s.id, s.path): s.content for s in reloaded.code_files()}
    assert reloaded_code == original_code


def test_code_surface_path_colliding_with_a_reserved_file_is_rejected(tmp_path: Path) -> None:
    """A pathful code surface may not shadow SYSTEM.md/config.toml/etc. in the export."""
    doc = HarnessDoc(
        name="x",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(
                id="code:config-toml", kind=SurfaceKind.CODE, path="config.toml", content="# nope"
            ),
        ],
    )
    store = HarnessStore(tmp_path)
    with pytest.raises(ValueError, match="reserved file"):
        store.save_version(doc)


def test_hand_authored_dir_load_skips_dotfiles(tmp_path: Path) -> None:
    """Finder metadata (.DS_Store, binary) must not break a rendered-only load."""
    store = HarnessStore(tmp_path)
    directory = store.dir_for("hand") / "v1"
    directory.mkdir(parents=True)
    (directory / "SYSTEM.md").write_text("You are careful.", encoding="utf-8")
    (directory / ".DS_Store").write_bytes(b"\x00\x01Bud1\x00")
    (directory / ".hidden").mkdir()
    (directory / ".hidden" / "note.txt").write_text("ignored", encoding="utf-8")

    doc = store.load("hand")

    assert doc.system_prompt() == "You are careful."
    assert doc.code_files() == []

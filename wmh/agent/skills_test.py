"""Tests for the skill library: round-trip, normalization, progressive-disclosure index, subset."""

from __future__ import annotations

from pathlib import Path

from wmh.agent.skills import Skill, SkillLibrary, normalize_skill_name


def test_skill_markdown_roundtrip() -> None:
    skill = Skill(name="grep-logs", description="find errors in logs", body="grep -i error *.log")
    restored = Skill.from_markdown(skill.to_markdown())
    assert restored == skill


def test_normalize_skill_name() -> None:
    assert normalize_skill_name("Grep The Logs!") == "grep-the-logs"
    assert normalize_skill_name("") == "skill"


def test_library_persists_and_reloads(tmp_path: Path) -> None:
    lib = SkillLibrary(tmp_path)
    lib.save("Find Big Files", "locate large files", "du -ah . | sort -h | tail")
    # A fresh library over the same dir sees the saved skill.
    reloaded = SkillLibrary(tmp_path)
    assert reloaded.names() == ["find-big-files"]
    got = reloaded.get("find-big-files")
    assert got is not None and "du -ah" in got.body


def test_render_index_is_names_and_descriptions_only() -> None:
    lib = SkillLibrary()
    lib.save("a-skill", "does A", "secret body A")
    index = lib.render_index()
    assert "a-skill: does A" in index
    assert "secret body A" not in index  # bodies are disclosed only via read_skill


def test_subset_keeps_named_skills() -> None:
    lib = SkillLibrary()
    lib.save("keep", "k", "b1")
    lib.save("drop", "d", "b2")
    subset = lib.subset(["keep", "missing"])
    assert subset.names() == ["keep"]

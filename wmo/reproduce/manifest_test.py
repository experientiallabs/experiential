"""Manifests are shipped data: every one must load, and the schema must refuse nonsense."""

from pathlib import Path

import pytest

from wmo.reproduce.manifest import Manifest, load_manifest, load_manifest_file, manifest_names


def test_every_shipped_manifest_loads_and_names_a_real_cookbook() -> None:
    """A manifest that does not validate is a packaging bug a user would hit first."""
    names = manifest_names()
    assert "routerbench" in names
    assert "tau-bench" in names
    repo_root = Path(__file__).resolve().parents[2]
    for name in names:
        manifest = load_manifest(name)
        assert manifest.name == name
        assert (repo_root / manifest.cookbook).is_file(), (
            f"manifest '{name}' points at cookbook {manifest.cookbook}, which does not exist"
        )


def test_pool_files_are_not_listed_as_benchmarks() -> None:
    assert not any(name.endswith(".pool") for name in manifest_names())


def test_unknown_manifest_names_the_available_ones() -> None:
    with pytest.raises(KeyError, match="routerbench"):
        load_manifest("definitely-not-a-benchmark")


def test_matrix_kind_requires_its_protocol_table(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
name = "bad"
title = "t"
cookbook = "docs/cookbook/tau-bench.md"
exactness = "bit-exact"
kind = "matrix"

[data]
hf_repo = "org/repo"
files = ["matrix.json"]

[[published]]
label = "row"
accuracy = 0.5
cost_per_run_usd = 0.1
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no \\[matrix\\] table"):
        load_manifest_file(path)


def test_commands_kind_may_not_claim_bit_exact() -> None:
    """Live providers are nondeterministic; a manifest claiming otherwise is lying."""
    with pytest.raises(ValueError, match="never bit-exact"):
        Manifest.model_validate(
            {
                "name": "bad",
                "title": "t",
                "cookbook": "docs/cookbook/tau-bench.md",
                "exactness": "bit-exact",
                "kind": "commands",
                "data": {"hf_repo": "org/repo", "files": ["traces.jsonl"]},
                "commands": {
                    "steps": [["build"]],
                    "report_file": "report.json",
                    "estimated_spend_usd": 10.0,
                },
                "published": [{"label": "row", "accuracy": 0.5, "cost_per_run_usd": 0.1}],
            }
        )


def test_shipped_tau_manifest_carries_wide_tolerances_and_spend() -> None:
    """The live manifest must state its nondeterminism and its price, not imply precision."""
    manifest = load_manifest("tau-bench")
    assert manifest.exactness == "protocol-exact"
    assert manifest.commands is not None
    assert manifest.commands.estimated_spend_usd > 0
    for row in manifest.published:
        assert row.tolerance_accuracy > 0
        assert row.tolerance_cost > 0

"""Tests for the reproducible production LOC reporting command."""

from __future__ import annotations

import json
import subprocess
import sys

from wmo.cli import repo_metrics

BASELINE_REVISION = "e7aad17b2f5041769ad8107ab25e77d4e88729ca"


def test_production_loc_boundary_matches_the_approved_baseline() -> None:
    """The documented production predicate reproduces 303 files and 98,489 physical lines."""
    snapshot = repo_metrics.production_snapshot(BASELINE_REVISION)
    assert snapshot.file_count == 303
    assert snapshot.line_count == 98_489


def test_production_path_boundary_excludes_tests_testdata_and_test_configuration() -> None:
    """The LOC predicate implements every exclusion named by the approved counting boundary."""
    assert repo_metrics.is_production_path("wmo/common/core/files.py")
    assert repo_metrics.is_production_path("wmo/runtime/harness/vendor/pi-agent/package.json")
    assert not repo_metrics.is_production_path("wmo/common/core/files_test.py")
    assert not repo_metrics.is_production_path("wmo/conftest.py")
    assert not repo_metrics.is_production_path("wmo/simulation/ingest/testdata/sample_otlp.json")
    assert not repo_metrics.is_production_path(
        "wmo/runtime/harness/vendor/pi-agent/vitest.config.ts"
    )


def test_production_loc_uses_physical_final_newline_semantics() -> None:
    """The reporting command counts an unterminated final line without a phantom newline."""
    assert repo_metrics._physical_line_count("") == 0
    assert repo_metrics._physical_line_count("line") == 1
    assert repo_metrics._physical_line_count("line\n") == 1
    assert repo_metrics._physical_line_count("line\n\n") == 2


def test_production_loc_report_has_no_delta_for_one_revision() -> None:
    """A same-revision report is a stable zero-delta baseline report."""
    report = repo_metrics.production_loc_report(BASELINE_REVISION, BASELINE_REVISION)
    assert report.production_files_added == 0
    assert report.production_files_removed == 0
    assert report.production_files_net == 0
    assert report.production_lines_added == 0
    assert report.production_lines_removed == 0
    assert report.production_lines_net == 0
    assert not report.direct_dependencies_added
    assert not report.direct_dependencies_removed


def test_production_loc_command_emits_machine_readable_report() -> None:
    """The documented command can be reproduced without a root CLI surface change."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "wmo.cli.repo_metrics",
            "--base",
            BASELINE_REVISION,
            "--head",
            BASELINE_REVISION,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["base"]["file_count"] == 303
    assert payload["base"]["line_count"] == 98_489
    assert payload["production_lines_net"] == 0

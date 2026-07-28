"""Contract: `wmo.hub` and `environment_capture.hub` describe the SAME bundles in the SAME place.

`wmo/hub.py` is a vendored, narrowed copy of the member's Hub read core — the flagship wheel
must install with no dependency on anything under `packages/` (AGENTS.md § Monorepo), so `wmo`
cannot import it. Vendoring trades a release coupling for a drift risk, and this file is the
thing that makes the trade safe.

Both sides of the drift are silent in production. A benchmark added to one registry only makes
`wmo download` and the member's own fetch disagree about what exists. A data root that resolves
differently makes the member's capture write where `wmo examples list` never looks. Neither
raises; the user just sees an empty list.

Lives in `contracts/` because it imports both packages and so belongs to neither tree.
"""

from __future__ import annotations

from dataclasses import fields

import environment_capture.hub as member_hub
import pytest

import wmo.hub as wmo_hub


def test_the_vendored_registry_matches_the_member() -> None:
    """Same benchmarks, and the same CorpusSpec fields for each one.

    Compared field by field so a failure names the drifted attribute instead of printing two
    dataclass reprs. `data_dirs` and `published` matter as much as the benchmark set: `wmo eval
    run` reads both to decide whether to promise the user a `wmo download` that will actually
    land the missing file.
    """
    assert sorted(wmo_hub.CORPORA) == sorted(member_hub.CORPORA), (
        "the vendored corpus registry and the member's have drifted apart; a benchmark is "
        "registered in exactly one of wmo/hub.py and "
        "packages/environment-capture/environment_capture/hub.py — add it to both"
    )
    drifted = {
        f"{name}.{spec_field.name}": (ours, theirs)
        for name, spec in wmo_hub.CORPORA.items()
        for spec_field in fields(spec)
        if (ours := getattr(spec, spec_field.name))
        != (theirs := getattr(member_hub.CORPORA[name], spec_field.name))
    }
    assert not drifted, f"vendored CorpusSpec fields differ from the member's: {drifted}"


def test_the_vendored_data_root_agrees_with_the_member(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both copies resolve the same corpus path, in a checkout and under the shared override."""
    assert wmo_hub.corpus_path("gaia2") == member_hub.corpus_path("gaia2")

    monkeypatch.setenv("ENVCAP_DATA_ROOT", "/tmp/wmo-shared-data-root")
    assert wmo_hub.corpus_path("gaia2") == member_hub.corpus_path("gaia2")


def test_the_vendored_copy_reads_the_same_repo_ids() -> None:
    """A bundle the member pushed must be a bundle `wmo download` can find.

    The publish side lives only in the member (`hub_push`), so the two must agree on the repo
    naming convention or `wmo` would fetch from an id nobody publishes to. Includes the legacy
    `wmh-` fallback: it is the whole reason the ordering, not just the set, is pinned.
    """
    for benchmark in wmo_hub.CORPORA:
        assert wmo_hub.candidate_repo_ids(benchmark) == member_hub.candidate_repo_ids(benchmark)

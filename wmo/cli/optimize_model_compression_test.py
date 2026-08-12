"""Staged model optimizer compression and embedder tests."""

# ruff: noqa: F403, F405
from wmo.cli.optimize_model_fixtures_test import *


def test_a_compressed_run_shows_the_compact_row_and_charges_it_to_the_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compaction is represented honestly: an arm, the two stages it configures, no second bill."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", *_ARM)
    assert result.exit_code == 0, result.output
    assert _says(result.output, f"{_ARM_LINE}, configures sweep and fit")
    assert _says(result.output, "included in sweep")
    # The compact row sits between sweep and fit, as STAGE_ORDER promised.
    assert (
        _stage_rows(result.output).index("compact") == _stage_rows(result.output).index("sweep") + 1
    )
    # The candidate projection is now an over-estimate AND misses the compressor: both said.
    assert _says(result.output, "that candidate figure is an OVER-estimate")

    _, _, _, manifest_path = _paths(root)
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert [record.stage.value for record in manifest.stages] == [
        "sweep",
        "compact",
        "fit",
        "tune",
        "report",
    ]
    compact = manifest.record_for(Stage.COMPACT)
    assert compact is not None
    # No artifact of its own, no bill of its own: the arm's fingerprint is the whole record.
    assert compact.artifact_path is None
    assert compact.spend_usd == 0.0 and compact.world_model_spend_usd == 0.0
    assert compact.fingerprint == {"compression": _ARM_LINE}


def test_a_compressed_run_fits_the_policy_in_the_geometry_it_will_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Representation consistency end to end: the served arm and the fitted arm are stamped equal.

    This is what the mount gate re-checks, and why the orchestrator cannot fit a compressed
    endpoint on a raw bank: one flag configures the sweep and the fit, so both halves agree by
    construction rather than by the operator remembering to pass it twice.
    """
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", *_ARM).exit_code == 0

    policy = RoutingPolicy.load(_paths(root)[1])
    assert policy.fit_compression is not None
    assert policy.compression is not None
    assert policy.fit_compression.compressor_id == "truncate"
    assert policy.fit_compression.aggressiveness == pytest.approx(0.3)
    assert policy.fit_compression.compressor_version == "1"
    assert policy.compression == policy.fit_compression
    # The matrix's episodes really ran that arm, which is what the fit is allowed to stamp.
    measured = OutcomeMatrix.load(_paths(root)[0]).measured_compression()
    assert measured is not None and measured == policy.fit_compression


def test_the_uncompressed_plan_table_is_untouched_by_the_new_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control: without --compressor this is byte-for-byte the command it was before.

    Same stage rows, same projected spend string, no compact row advertising a stage with nothing
    to do, and no compression key in the fit's fingerprint (its absence is what says "raw", so
    adding one would re-fit every model whose manifest predates the flag).
    """
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--dry-run")
    assert result.exit_code == 0, result.output
    assert _stage_rows(result.output) == ["preflight", "sweep", "fit", "tune", "report"]
    assert "compact" not in _flat(result.output)
    assert "includedinsweep" not in _flat(result.output)
    assert "~$0.33" in _flat(result.output)  # the same projection as before this change
    assert "OVER-estimate" not in result.output

    assert _run(tmp_path, root, "--yes").exit_code == 0
    manifest = RunManifest.model_validate_json(_paths(root)[3].read_text(encoding="utf-8"))
    fit = manifest.record_for(Stage.FIT)
    assert fit is not None and "compression" not in fit.fingerprint
    assert RoutingPolicy.load(_paths(root)[1]).fit_compression is None


def test_an_unchanged_arm_skips_the_whole_run_compaction_included(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", *_ARM).exit_code == 0
    again = _run(tmp_path, root, "--yes", *_ARM)
    assert again.exit_code == 0, again.output
    # Nothing runs at all, so the whole verdict is the plan table's: compaction reads as done
    # rather than as a stage that quietly did nothing.
    assert _says(again.output, "SKIP (already done: same compressor, same aggressiveness)")
    assert _says(again.output, "matrix.json is current")
    assert _says(again.output, "policy.json is current")
    assert _says(again.output, "every stage is current")


def test_moving_the_dial_re_measures_the_arm_and_refits_on_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different aggressiveness is different evidence, so the cells are bought again."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", *_ARM).exit_code == 0
    moved = _run(tmp_path, root, "--yes", "--compressor", "truncate", "--aggressiveness", "0.6")
    assert moved.exit_code == 0, moved.output
    assert _says(moved.output, "compression changed")
    # The fit is dirtied BY the compaction row, not by the sweep above it: the arm it embeds
    # through is what changed, and the reason printed says which stage owns that.
    assert _says(moved.output, "runs after compact, which will change its input")
    assert RoutingPolicy.load(_paths(root)[1]).fit_compression is not None
    assert (
        OutcomeMatrix.load(_paths(root)[0]).measured_compression()
        == RoutingPolicy.load(_paths(root)[1]).fit_compression
    )


def test_dropping_the_compressor_re_measures_the_raw_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turning compression off is a change of arm like any other, in the other direction."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes", *_ARM).exit_code == 0
    raw = _run(tmp_path, root, "--yes")
    assert raw.exit_code == 0, raw.output
    assert _says(raw.output, "compression changed")
    assert "compact" not in _flat(raw.output)  # the row is gone with the flag
    policy = RoutingPolicy.load(_paths(root)[1])
    assert policy.fit_compression is None and policy.compression is None
    manifest = RunManifest.model_validate_json(_paths(root)[3].read_text(encoding="utf-8"))
    # The compact record from the compressed run is stale, and the fit no longer claims an arm.
    fit = manifest.record_for(Stage.FIT)
    assert fit is not None and "compression" not in fit.fingerprint


def test_an_unknown_compressor_is_refused_before_the_plan_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world_model = _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--compressor", "gzip-but-for-prompts")
    assert result.exit_code != 0
    assert _says(result.output, "unknown compressor 'gzip-but-for-prompts'")
    assert _says(result.output, "known compressors:")
    assert not _says(result.output, "optimize model: support")  # no plan table was printed
    assert world_model.tasks == [] and not _paths(root)[0].exists()


def test_aggressiveness_without_a_compressor_says_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--aggressiveness", "0.4")
    assert result.exit_code != 0
    assert _says(result.output, "--aggressiveness needs --compressor to apply it")
    assert not _paths(root)[0].exists()


def test_the_embedder_resolution_is_printed_and_lands_in_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Auto with no azure resource resolves to hashing, says so, and quotes what that costs."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes")
    assert result.exit_code == 0, result.output
    assert _says(result.output, "embedder: hashing-512 (auto;")
    assert _says(result.output, "AZURE_OPENAI_API_KEY")
    assert _says(result.output, "measured +0.60pt")
    manifest = RunManifest.model_validate_json(_paths(root)[3].read_text(encoding="utf-8"))
    fit = manifest.record_for(Stage.FIT)
    assert fit is not None and fit.fingerprint["embedder"] == "hashing-512"
    assert RoutingPolicy.load(_paths(root)[1]).embedder.kind == "hashing"


def test_an_explicit_hashing_embedder_is_the_same_fit_auto_already_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fingerprint records the RESOLVED embedder, not the word the operator typed, so naming
    # the backend auto had already chosen is not a refit.
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    assert _run(tmp_path, root, "--yes").exit_code == 0
    again = _run(tmp_path, root, "--yes", "--embedder", "hashing")
    assert again.exit_code == 0, again.output
    assert _says(again.output, "embedder: hashing-512 (explicit)")
    assert _says(again.output, "policy.json is current")


def test_embedder_azure_without_the_env_pair_names_the_pair_and_the_manual_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This command has no --deployment/--endpoint, so its refusal must not name them as the fix."""
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--embedder", "azure")
    assert result.exit_code != 0
    assert _says(result.output, "needs AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT")
    assert _says(result.output, "wmo optimize route fit")
    assert not _paths(root)[0].exists()


def test_an_unknown_embedder_lists_the_real_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch)
    root = _project(tmp_path)
    result = _run(tmp_path, root, "--yes", "--embedder", "word2vec")
    assert result.exit_code != 0
    assert _says(result.output, "unknown embedder 'word2vec'; use auto, hashing, azure or local")


def test_auto_takes_the_semantic_embedder_when_the_resource_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of auto, on the pure resolver so no test ever bills an embedding API."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    spec, line = optimize_plan_module._resolve_embedder_choice("auto")
    assert spec.kind == "azure"
    assert spec.deployment == AZURE_EMBEDDER_DEPLOYMENT
    assert spec.dim == 3072  # the deployment's native width, not the hashing default
    assert spec.endpoint == "https://example.openai.azure.com"
    assert "auto; AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT present" in line
    # Spend nobody typed a flag for is stated, because `auto` chose it.
    assert "billed to that resource" in line

    explicit, explicit_line = optimize_plan_module._resolve_embedder_choice("azure")
    assert explicit == spec  # same resource, same deployment, same width
    assert "(explicit)" in explicit_line


def test_the_fit_fingerprint_moves_with_the_embedder_and_with_the_arm(tmp_path: Path) -> None:
    """What makes a resume re-fit: a different embedder or a different arm, each on its own."""
    matrix = tmp_path / "matrix.json"
    matrix.write_text("{}", encoding="utf-8")

    def fingerprint(
        *, embedder: EmbedderSpec, compression: CompressionConfig | None
    ) -> dict[str, str]:
        return optimize_plan_module._fit_fingerprint(
            matrix=matrix,
            embedder=embedder,
            compression=compression,
            fallback=None,
            allow_uneven=False,
        )

    hashing = fingerprint(embedder=EmbedderSpec(), compression=None)
    azure = fingerprint(
        embedder=EmbedderSpec(
            kind="azure", dim=3072, deployment="text-embedding-3-large", endpoint="https://r/"
        ),
        compression=None,
    )
    assert hashing["embedder"] != azure["embedder"]
    assert "compression" not in hashing  # absence is how raw is spelled

    arm = CompressionConfig(compressor_id="truncate", aggressiveness=0.3)
    keener = CompressionConfig(compressor_id="truncate", aggressiveness=0.6)
    assert fingerprint(embedder=EmbedderSpec(), compression=arm) != hashing
    assert fingerprint(embedder=EmbedderSpec(), compression=arm) != fingerprint(
        embedder=EmbedderSpec(), compression=keener
    )

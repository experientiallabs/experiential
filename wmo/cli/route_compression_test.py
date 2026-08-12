"""Route compression CLI tests."""

# ruff: noqa: F403, F405
from wmo.cli.route_fixtures_test import *


def test_route_fit_writes_compression_config(tmp_path: Path) -> None:
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    matrix_file = _matrix_file(tmp_path, compression=config)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "rank",
            "--out",
            str(policy_file),
            "--clusters",
            "2",
            "--dim",
            "64",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.compression is not None
    assert policy.compression.compressor_id == "truncate"
    assert policy.compression.aggressiveness == 0.5


def test_route_fit_rejects_unknown_compressor(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(tmp_path / "policy.json"),
            "--compressor",
            "llmzip",
        ],
    )
    assert result.exit_code != 0
    assert "unknown compressor" in result.output


def test_route_fit_rejects_orphan_aggressiveness(tmp_path: Path) -> None:
    matrix_file = _matrix_file(tmp_path)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--out",
            str(tmp_path / "policy.json"),
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "--compressor" in result.output


def test_route_fit_stamps_what_the_evidence_was_fitted_under(tmp_path: Path) -> None:
    # D-COMPRESS requirement A: --compressor does not just switch serving on, it moves the FIT
    # onto the compressed representation and records that on the artifact, which is what makes
    # the resulting policy mountable at all.
    policy_file = tmp_path / "policy.json"
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_knn_matrix_file(tmp_path, compression=config)),
            "--kind",
            "knn",
            "--out",
            str(policy_file),
            "--dim",
            "64",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)  # loads, so the mount gate is satisfied
    assert policy.fit_compression is not None
    assert policy.fit_compression == policy.compression


def test_route_fit_refuses_to_stamp_an_arm_the_matrix_never_measured(tmp_path: Path) -> None:
    # The contract gap: --compressor moved the fit-side embeddings but could not retroactively
    # change what the EPISODES ran under, so a compressed policy could be stamped over rewards
    # measured uncompressed. That is a joint fit over an arm nobody ran.
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path)),  # uncompressed rewards
            "--out",
            str(tmp_path / "policy.json"),
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code != 0
    assert "measured with raw text" in result.output
    assert "one matrix per arm" in result.output


def test_route_fit_refuses_compressed_rewards_under_a_raw_fit(tmp_path: Path) -> None:
    # The mirror image, and the same error: rewards produced under compression do not describe
    # an endpoint that serves raw text.
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path, compression=config)),
            "--out",
            str(tmp_path / "policy.json"),
        ],
    )
    assert result.exit_code != 0
    flat = "".join(ch for ch in result.output if not ch.isspace() and ch not in "│┌┐└┘─╔╗╚╝║═")
    assert "wouldstamprawtext" in flat


def test_route_sweep_rejects_an_unservable_compressor_before_spending(tmp_path: Path) -> None:
    # The arm has to be one that could actually be served, and the check lands before any
    # episode is paid for.
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--compressor",
            "llmzip",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "unknown compressor" in result.output


def test_route_sweep_rejects_orphan_aggressiveness(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "sweep",
            "support",
            "--aggressiveness",
            "0.5",
            "--root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "--compressor" in result.output


def test_route_fit_stamps_the_running_compressor_version(tmp_path: Path) -> None:
    # Regression: the config was built without a version, so it defaulted to "1" and any fit
    # against a version-bumped compressor stamped a lie. The mount gate then hard-stopped the
    # result with a remedy the CLI has no flag to carry out. Latent while everything is v1,
    # which is exactly why it needs a test.
    class _V3(TruncateCompressor):
        id = "cli-v3-for-tests"
        version = "3"

    register_compressor(cast("Compressor", _V3()))
    config = CompressionConfig(
        compressor_id="cli-v3-for-tests", compressor_version="3", aggressiveness=0.5
    )
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(_matrix_file(tmp_path, compression=config)),
            "--out",
            str(policy_file),
            "--dim",
            "64",
            "--compressor",
            "cli-v3-for-tests",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)  # loads, so the version gate is satisfied
    assert policy.compression is not None
    assert policy.compression.compressor_version == "3"
    assert policy.fit_compression == policy.compression
    assert policy.serving_compressor() is not None  # and it mounts


def test_route_fit_knn_stamps_the_compression_it_was_fitted_under(tmp_path: Path) -> None:
    """The knn path must carry the stamp, not just the rank path.

    `--kind knn` returns from `fit_knn_artifact` before the rank path's stamping line, so a fit
    that attached compression only after the branch would write a knn policy with no
    `fit_compression` at all. That is the representation-consistency failure in a new costume:
    the endpoint would serve compressed while its bank claimed to be raw, and the mount gate
    would have nothing to compare.
    """
    config = CompressionConfig(compressor_id="truncate", aggressiveness=0.5)
    matrix_file = _knn_matrix_file(tmp_path, compression=config)
    policy_file = tmp_path / "policy.json"
    result = runner.invoke(
        app,
        [
            "optimize",
            "route",
            "fit",
            str(matrix_file),
            "--kind",
            "knn",
            "--out",
            str(policy_file),
            "--fallback",
            "a",
            "--min-pairs",
            "0",
            "--compressor",
            "truncate",
            "--aggressiveness",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.kind == "knn"
    assert policy.compression is not None
    assert policy.compression.compressor_id == "truncate"
    # Both halves, so the mount gate has an identity to check rather than a null.
    assert policy.fit_compression is not None
    assert same_compression(policy.compression, policy.fit_compression)


def test_route_fit_knn_leaves_the_stamp_null_without_the_flag(tmp_path: Path) -> None:
    """The negative control: an uncompressed knn fit is byte-identical to before this seam."""
    policy_file = tmp_path / "policy.json"
    result = _fit_knn(_knn_matrix_file(tmp_path), policy_file)
    assert result.exit_code == 0, result.output
    policy = RoutingPolicy.load(policy_file)
    assert policy.compression is None and policy.fit_compression is None

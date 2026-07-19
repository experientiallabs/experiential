"""Tests for the Harbor 0.18 job-config adapter."""

from __future__ import annotations

from pathlib import Path

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from pydantic import ValidationError

from wmh.evals.harbor.config import (
    SUPPORTED_HARBOR_VERSION,
    HarborEnvironmentBackend,
    HarborJobSpec,
    build_harbor_job_config,
    validate_controlled_harbor_environment,
)
from wmh.evals.harbor.docker_environment import REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH
from wmh.evals.harbor.e2b_environment import EXACT_E2B_ENVIRONMENT_IMPORT_PATH
from wmh.tracking.rate_limit import (
    ExternalDispatchRateAuthority,
    ExternalDispatchRatePolicy,
    bind_external_dispatch_rate_authority,
)

_DATASET_REF = "sha256:" + "a" * 64


def _spec(tmp_path: Path, **updates: object) -> HarborJobSpec:
    values: dict[str, object] = {
        "job_name": "ground-truth-evaluation",
        "jobs_dir": tmp_path,
        "datasets": [
            DatasetConfig(
                name="example/benchmark",
                ref=_DATASET_REF,
                task_names=["task-a", "task-b"],
            )
        ],
        "n_attempts": 5,
        "n_concurrent_trials": 3,
    }
    values.update(updates)
    return HarborJobSpec.model_validate(values)


def _agent() -> AgentConfig:
    return AgentConfig(
        import_path="wmh.evals.harbor.agent:WmhPiAgent",
        model_name="bedrock/us.anthropic.claude-opus-4-1-v1:0",
        kwargs={"harness_name": "candidate", "harness_version": 7},
    )


def _rate_policy() -> ExternalDispatchRatePolicy:
    return ExternalDispatchRatePolicy(
        provider="e2b",
        operation="sandbox_create",
        maximum_dispatches=4,
        period_milliseconds=1000,
    )


def test_local_default_maps_explicitly_to_harbor_docker(tmp_path: Path) -> None:
    config = build_harbor_job_config(_spec(tmp_path), agent=_agent())

    assert config.environment.type is EnvironmentType.DOCKER
    assert config.environment.import_path == REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH
    assert config.environment.delete is True
    assert config.environment.force_build is False
    assert config.environment.mounts is None
    assert config.environment.extra_docker_compose == []
    assert config.environment.env == {}
    assert config.environment.kwargs == {}
    assert config.environment.extra_allowed_hosts == []
    assert config.n_attempts == 5
    assert config.n_concurrent_trials == 3
    assert config.datasets[0].name == "example/benchmark"
    assert config.datasets[0].ref == _DATASET_REF
    assert config.datasets[0].task_names == ["task-a", "task-b"]
    assert config.agents[0].import_path == "wmh.evals.harbor.agent:WmhPiAgent"
    assert config.agents[0].kwargs == {"harness_name": "candidate", "harness_version": 7}

    authority = ExternalDispatchRateAuthority.bootstrap(tmp_path / "rate.json", _rate_policy())
    with pytest.raises(ValueError, match="local.*create-rate authority"):
        build_harbor_job_config(
            _spec(tmp_path),
            agent=_agent(),
            create_rate_binding=bind_external_dispatch_rate_authority(authority),
        )
    with pytest.raises(ValueError, match="unused E2B create-rate policy"):
        build_harbor_job_config(
            _spec(tmp_path, create_rate_policy=_rate_policy()),
            agent=_agent(),
        )


def test_e2b_is_an_explicit_environment_with_no_docker_fallback(tmp_path: Path) -> None:
    authority = ExternalDispatchRateAuthority.bootstrap(tmp_path / "rate.json", _rate_policy())
    binding = bind_external_dispatch_rate_authority(authority)
    config = build_harbor_job_config(
        _spec(
            tmp_path,
            environment_backend=HarborEnvironmentBackend.E2B,
            create_rate_policy=_rate_policy(),
        ),
        agent=_agent(),
        create_rate_binding=binding,
    )

    assert config.environment.type is EnvironmentType.E2B
    assert config.environment.import_path == EXACT_E2B_ENVIRONMENT_IMPORT_PATH
    assert config.environment.kwargs == {
        "allow_preexisting_e2b_builds": False,
        "resource_budget_bindings": [],
        "create_rate_binding": binding.model_dump(mode="json"),
    }


def test_e2b_job_rejects_missing_or_mismatched_create_rate_authority(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        environment_backend=HarborEnvironmentBackend.E2B,
        create_rate_policy=_rate_policy(),
    )
    with pytest.raises(ValueError, match="create-rate authority"):
        build_harbor_job_config(spec, agent=_agent())

    other_policy = _rate_policy().model_copy(update={"maximum_dispatches": 3})
    authority = ExternalDispatchRateAuthority.bootstrap(tmp_path / "other-rate.json", other_policy)
    with pytest.raises(ValueError, match="create-rate policy"):
        build_harbor_job_config(
            spec,
            agent=_agent(),
            create_rate_binding=bind_external_dispatch_rate_authority(authority),
        )

    with pytest.raises(ValidationError, match="four-per-second"):
        _spec(
            tmp_path,
            environment_backend=HarborEnvironmentBackend.E2B,
            create_rate_policy=_rate_policy().model_copy(update={"maximum_dispatches": 5}),
        )


def test_preexisting_e2b_build_admission_is_explicit_and_frozen(tmp_path: Path) -> None:
    authority = ExternalDispatchRateAuthority.bootstrap(tmp_path / "rate.json", _rate_policy())
    config = build_harbor_job_config(
        _spec(
            tmp_path,
            environment_backend=HarborEnvironmentBackend.E2B,
            create_rate_policy=_rate_policy(),
            allow_preexisting_e2b_builds=True,
        ),
        agent=_agent(),
        create_rate_binding=bind_external_dispatch_rate_authority(authority),
    )

    assert config.environment.kwargs["allow_preexisting_e2b_builds"] is True
    with pytest.raises(ValidationError, match="only with the E2B backend"):
        _spec(tmp_path, allow_preexisting_e2b_builds=True)


def test_native_local_and_git_dataset_sources_are_preserved(tmp_path: Path) -> None:
    local = DatasetConfig(path=tmp_path / "tasks")
    git = DatasetConfig(
        repo="https://github.com/example/benchmark.git",
        path=Path("datasets/tasks"),
        version="527d50deb63a5d279e8c20593c18a2cbc7f61f9e",
    )

    config = build_harbor_job_config(_spec(tmp_path, datasets=[local, git]), agent=_agent())

    assert config.datasets == [local, git]


def test_dataset_configs_are_copied_before_harbor_resolves_them(tmp_path: Path) -> None:
    dataset = DatasetConfig(
        name="example/benchmark",
        ref=_DATASET_REF,
    )

    config = build_harbor_job_config(_spec(tmp_path, datasets=[dataset]), agent=_agent())
    config.datasets[0].ref = "sha256:" + "b" * 64

    assert dataset.ref == _DATASET_REF


def test_retries_require_an_explicit_exception_allowlist(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unsupported.*attempt ledger"):
        _spec(tmp_path, max_retries=2)


def test_explicit_retry_allowlist_does_not_enable_unaudited_retries(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unsupported.*attempt ledger"):
        _spec(
            tmp_path,
            max_retries=2,
            retry_exceptions={"EnvironmentStartTimeoutError"},
        )

    bypassed_validation = _spec(tmp_path).model_copy(
        update={
            "max_retries": 2,
            "retry_exceptions": {"EnvironmentStartTimeoutError"},
        },
        deep=True,
    )
    with pytest.raises(ValidationError, match="unsupported.*attempt ledger"):
        build_harbor_job_config(bypassed_validation, agent=_agent())


def test_retry_allowlist_without_retries_is_rejected_as_misleading(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="retry_exceptions are unsupported"):
        _spec(tmp_path, retry_exceptions={"EnvironmentStartTimeoutError"})


def test_default_job_config_has_retries_disabled(tmp_path: Path) -> None:
    config = build_harbor_job_config(_spec(tmp_path), agent=_agent())

    assert config.retry.max_retries == 0
    assert config.retry.include_exceptions is None


def test_agent_concurrency_must_be_final_before_job_construction(tmp_path: Path) -> None:
    agent = _agent().model_copy(update={"n_concurrent": 2}, deep=True)
    config = build_harbor_job_config(
        _spec(tmp_path, agent_n_concurrent=2),
        agent=agent,
    )

    assert config.agents[0] == agent

    with pytest.raises(ValueError, match="agent n_concurrent"):
        build_harbor_job_config(
            _spec(tmp_path, agent_n_concurrent=2),
            agent=_agent(),
        )


def test_supported_harbor_version_is_exact() -> None:
    assert SUPPORTED_HARBOR_VERSION == "0.18.0"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"import_path": "evil:Environment"}, "import_path"),
        ({"type": EnvironmentType.DAYTONA}, "Docker or E2B"),
        ({"force_build": True}, "force_build"),
        ({"delete": False}, "cleanup"),
        (
            {
                "mounts": [
                    {
                        "type": "bind",
                        "source": "/host",
                        "target": "/mnt/host",
                    }
                ]
            },
            "host mounts",
        ),
        ({"extra_docker_compose": [Path("overlay.yml")]}, "Compose overlays"),
        ({"env": {"TOKEN": "${TOKEN}"}}, "host variables"),
        ({"kwargs": {"privileged": True}}, "backend kwargs"),
        ({"extra_allowed_hosts": ["host.docker.internal"]}, "extra allowed hosts"),
    ],
)
def test_controlled_environment_rejects_host_facing_overrides(
    update: dict[str, object],
    message: str,
) -> None:
    environment = EnvironmentConfig(
        type=EnvironmentType.DOCKER,
        import_path=REAPING_DOCKER_ENVIRONMENT_IMPORT_PATH,
    ).model_copy(update=update)

    with pytest.raises(ValueError, match=message):
        validate_controlled_harbor_environment(environment)


def test_controlled_environment_rejects_backend_drift() -> None:
    environment = EnvironmentConfig(type=EnvironmentType.E2B)

    with pytest.raises(ValueError, match="differs from the frozen job backend"):
        validate_controlled_harbor_environment(
            environment,
            expected_type=EnvironmentType.DOCKER,
        )


def test_controlled_e2b_environment_requires_exact_build_adapter() -> None:
    with pytest.raises(ValueError, match="trusted WMH adapter"):
        validate_controlled_harbor_environment(EnvironmentConfig(type=EnvironmentType.E2B))

    validate_controlled_harbor_environment(
        EnvironmentConfig(
            type=EnvironmentType.E2B,
            import_path=EXACT_E2B_ENVIRONMENT_IMPORT_PATH,
        )
    )

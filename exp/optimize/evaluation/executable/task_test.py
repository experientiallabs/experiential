"""Executable package evidence stays private and the shared Harbor file inventory stays exact."""

import json
from pathlib import Path

import pytest

from exp.optimize.evaluation.executable.package import export_executable_harbor
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.harbor.export import verify_harbor_export
from exp.optimize.model.sft.training_test import _TIME


def test_executable_export_uses_common_client_without_private_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actor image receives its prefix and interfaces, never the verifier or workbook store."""
    project, task, spec = qualified_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unqualified"):
        export_executable_harbor(project, task, tmp_path / "refused", name="test/workbook")
    qualification = qualify_executable_task(project, spec, created_at=_TIME, code_revision="test")
    output = export_executable_harbor(
        project, task, tmp_path / "package", name="test/workbook", qualification=qualification
    )
    binding = verify_harbor_export(output)
    assert binding["qualification"] == "qualified_observable_contract"
    context = json.loads((output / "environment/actor-context.json").read_text())
    assert context["actor_prefix"]
    assert "verifier" not in context and "world" not in context
    assert not tuple(output.rglob("*.xlsx"))
    assert "from a frozen checkpoint" in (output / "instruction.md").read_text()
    from exp.optimize.evaluation.harbor.export import PYTHON_IMAGE

    assert (output / "environment/Dockerfile").read_text() == (
        f"FROM {PYTHON_IMAGE}\nWORKDIR /app\nCOPY tools.py /app/tools.py\n"
        "COPY actor-context.json /app/actor-context.json\n"
    )
    (output / "environment/actor-context.json").write_text("{}")
    with pytest.raises(ValueError, match="changed after export"):
        verify_harbor_export(output)


def test_unix_no_network_task_is_frozen_and_canonical_in_both_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in transport binds both separate images and cannot be changed after qualification."""
    import tomllib

    from exp.common.core.artifacts import sha256_json
    from exp.optimize.evaluation.executable.package import verify_executable_package
    from exp.optimize.evaluation.executable.task import (
        load_executable_task,
        persist_executable_task,
    )

    project, original, spec = qualified_fixture(tmp_path, monkeypatch)
    legacy = load_executable_task(project, original)
    old_payload = json.loads(project.artifacts.read_bytes(original.artifact_id, "task.json"))
    assert "harbor_transport" not in old_payload
    assert legacy.profile() == sha256_json(old_payload)
    explicit_http = legacy.model_copy(update={"harbor_transport": "http"})
    assert explicit_http.model_dump(mode="json") == old_payload
    unix = legacy.model_copy(update={"harbor_transport": "unix-no-network"})
    assert unix.profile() != legacy.profile()
    reference = persist_executable_task(project, unix, created_at=_TIME, code_revision="test")
    qualification = qualify_executable_task(
        project, spec.model_copy(update={"task": reference}), created_at=_TIME, code_revision="test"
    )
    directory = export_executable_harbor(
        project, reference, tmp_path / "unix", name="test/unix", qualification=qualification
    )
    assert verify_executable_package(project, directory) == reference
    from exp.optimize.evaluation.harbor.export import PYTHON_IMAGE

    actor_dockerfile = (directory / "environment/Dockerfile").read_text()
    verifier_dockerfile = (directory / "tests/Dockerfile").read_text()
    assert actor_dockerfile.startswith(f"FROM {PYTHON_IMAGE}\n")
    assert actor_dockerfile.endswith("RUN chown 65534:65534 /app\nUSER 65534:65534\n")
    assert verifier_dockerfile.startswith(f"FROM {PYTHON_IMAGE}\n")
    assert "USER" not in verifier_dockerfile and "chown" not in verifier_dockerfile
    configuration = tomllib.loads((directory / "task.toml").read_text())
    assert configuration["verifier"]["environment_mode"] == "separate"
    for part in (
        configuration["environment"],
        configuration["agent"],
        configuration["verifier"],
        configuration["verifier"]["environment"],
    ):
        assert part["network_mode"] == "no-network"
    assert set(configuration["environment"]["env"]) == {
        "EXP_HARBOR_SOCKET",
        "EXP_HARBOR_ACTOR_TOKEN",
    }
    assert set(configuration["verifier"]["env"]) == {
        "EXP_HARBOR_SOCKET",
        "EXP_HARBOR_VERIFIER_TOKEN",
    }
    assert not tuple(directory.rglob("*.xlsx"))
    for role in ["environment", "tests"]:
        config = json.loads((directory / role / "docker-compose.yaml").read_text())["services"][
            "main"
        ]
        assert config["network_mode"] == "none" and config["cap_drop"] == ["ALL"]
        assert config["security_opt"] == ["no-new-privileges:true"]
        assert config["volumes"] == [
            {
                "type": "bind",
                "source": "${EXP_HARBOR_SOCKET_DIR:?trial socket directory required}",
                "target": "/run/experiential",
                "read_only": True,
                "bind": {"create_host_path": False},
            }
        ]
    # Even repairing the untrusted file inventory cannot override the trusted task export.
    path = directory / "environment/docker-compose.yaml"
    config = json.loads(path.read_text())
    config["services"]["main"]["network_mode"] = "host"
    path.write_text(json.dumps(config))
    import hashlib

    binding_path = directory / "frozen-binding.json"
    binding = json.loads(binding_path.read_text())
    binding["files"]["environment/docker-compose.yaml"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    binding_path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_executable_package(project, directory)

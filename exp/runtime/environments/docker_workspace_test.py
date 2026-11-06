"""Native workspace controls for exact files, real execution and isolated restoration."""

import json
import os
import shlex

import pytest

from exp.common.models import ToolCall
from exp.common.tasks import TaskCase, ToolSchema
from exp.runtime.environments.docker_workspace import (
    DockerWorkspaceConfig,
    DockerWorkspaceFactory,
)
from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest
from exp.runtime.environments.worlds import ExecutableWorldRuntime


def test_configuration_requires_finite_resources_and_immutable_image() -> None:
    """Mutable tags and unbounded resources cannot define checkpoint identity."""
    with pytest.raises(ValueError, match="immutable"):
        DockerWorkspaceConfig(image="python:latest")
    with pytest.raises(ValueError, match="limit"):
        DockerWorkspaceConfig(image="sha256:" + "0" * 64, maximum_seconds=0)


def test_real_workspace_execution_restore_and_false_green_control() -> None:
    """A selected local image executes real tests and restores isolated native bytes."""
    image = os.environ.get("EXP_TEST_WORKSPACE_IMAGE")
    if image is None:
        pytest.skip("requires an explicitly selected local Docker workspace image")
    factory = DockerWorkspaceFactory(
        DockerWorkspaceConfig(
            image=image,
            docker=tuple(shlex.split(os.environ.get("EXP_TEST_DOCKER", "docker"))),
            maximum_seconds=5,
        )
    )
    tools = ("Bash", "Read", "Write", "Edit")
    manifest = WorldManifest(
        domains=(StateDomain(name="workspace", owner="native"),),
        tools=tuple(
            ToolEffectPolicy(
                name=name,
                reads=("workspace",),
                writes=() if name == "Read" else ("workspace",),
            )
            for name in tools
        ),
    )
    task = TaskCase(
        task_id="native-workspace-control",
        lineage_group_id="native-workspace-control",
        partition="fit",
        instruction="Exercise real files and Python.",
        workload_weight=1,
        origin="authored",
        tools=tuple(
            ToolSchema(name=name, description=name, input_schema={"type": "object"})
            for name in tools
        ),
    )
    runtime = ExecutableWorldRuntime(manifest, factory)
    with runtime.open(task) as world:
        initial = world.checkpoint()
        payload = "def answer():\n    return 'λ ` literal'\n"
        reply = world.execute(
            ToolCall(
                call_id="write", name="Write", arguments={"path": "answer.py", "content": payload}
            )
        )
        assert not reply.is_error
        assert json.loads(reply.content)["written_bytes"] == len(payload.encode())
        saved = world.checkpoint()
        reply = world.execute(
            ToolCall(call_id="read", name="Read", arguments={"path": "answer.py"})
        )
        assert json.loads(reply.content)["content"] == payload
        reply = world.execute(
            ToolCall(
                call_id="invalid-edit",
                name="Edit",
                arguments={
                    "path": "answer.py",
                    "edits": [
                        {"oldText": "λ", "newText": "μ"},
                        {"oldText": "absent", "newText": "bad"},
                    ],
                },
            )
        )
        assert reply.is_error and world.checkpoint().image == saved.image
        reply = world.execute(
            ToolCall(
                call_id="tests",
                name="Bash",
                arguments={"command": "python -c 'import answer; assert answer.answer() == 0'"},
            )
        )
        output = json.loads(reply.content)
        assert reply.is_error and output["exit_code"] != 0 and "AssertionError" in output["stderr"]
        reply = world.execute(
            ToolCall(
                call_id="outside",
                name="Write",
                arguments={"path": "/etc/forbidden", "content": "bad"},
            )
        )
        assert reply.is_error
        reply = world.execute(
            ToolCall(
                call_id="timeout", name="Bash", arguments={"command": "sleep 4", "timeout": 100}
            )
        )
        assert json.loads(reply.content)["timed_out"]
        reply = world.execute(
            ToolCall(
                call_id="large",
                name="Bash",
                arguments={"command": "python -c 'import sys; sys.stdout.write(\"x\" * 1000000)'"},
            )
        )
        assert json.loads(reply.content)["output_truncated"]
        with runtime.restore(task, saved) as restored:
            assert restored.checkpoint().image == saved.image
            reply = restored.execute(
                ToolCall(
                    call_id="edit",
                    name="Edit",
                    arguments={"path": "answer.py", "edits": [{"oldText": "λ", "newText": "μ"}]},
                )
            )
            assert not reply.is_error
            assert restored.checkpoint().image != saved.image
        with runtime.restore(task, initial) as empty:
            assert empty.checkpoint().image == initial.image
            assert empty.execute(
                ToolCall(call_id="missing", name="Read", arguments={"path": "answer.py"})
            ).is_error

        before_background = world.checkpoint()
        with pytest.raises(ValueError, match="background processes"):
            world.execute(
                ToolCall(
                    call_id="background",
                    name="Bash",
                    arguments={"command": "sleep 30 >/dev/null 2>&1 &"},
                )
            )
        assert world.checkpoint().image == before_background.image
        result = world.execute(
            ToolCall(
                call_id="root", name="Bash", arguments={"command": "touch /etc/workspace-control"}
            )
        )
        assert result.is_error and json.loads(result.content)["exit_code"] != 0

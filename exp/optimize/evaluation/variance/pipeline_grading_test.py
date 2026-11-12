"""Grading by the rubric judge and behavior tags, and the one revision from adversarial linters.

The judging and difference mining themselves run end to end in ``pipeline_test.py`` and
``pipeline_behaviors_test.py``; the revision decision is pinned here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance import pipeline_grading
from exp.optimize.evaluation.variance.pipeline_inputs import GeneralRubric


def ref(name: str) -> ArtifactInput:
    """A saved-artifact reference named ``name``."""
    return ArtifactInput(artifact_id=name, sha256="0" * 64)


def context(*, adversarial: object | None, library_id: str | None = "agent-behavior") -> Any:  # noqa: ANN401
    """A stage context whose spec has the given adversarial and library settings."""
    authoring = SimpleNamespace(
        adversarial=adversarial,
        library_id=library_id,
        adversarial_spec=lambda: adversarial,
    )
    spec = SimpleNamespace(mining=SimpleNamespace(rubric=None), authoring=authoring)
    build = SimpleNamespace(outputs={})
    return SimpleNamespace(
        project=SimpleNamespace(artifacts=None),
        spec=spec,
        code_revision="test",
        upstream={"variance-build": build},
    )


def rules() -> Any:  # noqa: ANN401
    """Rules outputs holding the general rubric and no library version."""
    job = SimpleNamespace(study=SimpleNamespace(checkpoint=ref("start")))
    return SimpleNamespace(
        rubric=ref("rubric-0"),
        rubric_packet=ref("packet-0"),
        rubric_library=None,
        job=job,
        rulebook=ref("book"),
    )


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[object]]:
    """Replace the judging, library and author owners with recorders."""
    calls: dict[str, list[object]] = {"judged_by": [], "library": [], "authored": []}
    library = SimpleNamespace(reference=ref("library-1"))

    def judge_tiers(
        context: object, groups: object, rubric: ArtifactInput
    ) -> tuple[dict[str, ArtifactInput], dict[str, ArtifactInput]]:
        calls["judged_by"].append(rubric.artifact_id)
        return {"actor": ref(f"judged-{rubric.artifact_id}")}, {
            "actor": ref(f"differences-{rubric.artifact_id}")
        }

    def adversarial_library(
        project: object, spec: object, differences: object, base: object, **kwargs: object
    ) -> object:
        calls["library"].append(differences)
        return base if "unchanged" in calls else library

    def mining_rubric(
        context: object, build: object, job: object, rulebook: object, binding: object
    ) -> tuple[ArtifactInput, ArtifactInput]:
        calls["authored"].append(binding)
        return ref("rubric-1"), ref("packet-1")

    monkeypatch.setattr(pipeline_grading, "_judge_tiers", judge_tiers)
    monkeypatch.setattr(pipeline_grading, "adversarial_library", adversarial_library)
    monkeypatch.setattr(pipeline_grading, "mining_rubric", mining_rubric)
    monkeypatch.setattr(
        pipeline_grading,
        "load_continuation_checkpoint",
        lambda store, checkpoint: SimpleNamespace(task=SimpleNamespace(task_id="task")),
    )
    monkeypatch.setattr(
        pipeline_grading.BuildOutputs, "model_validate", lambda outputs: SimpleNamespace()
    )
    return calls


def test_groups_are_graded_by_the_general_rubric_without_adversarial_linters(
    wired: dict[str, list[object]],
) -> None:
    """With linters off the general rubric grades every group once and is never revised."""
    grading = pipeline_grading.grade_groups(context(adversarial=None), rules(), {})
    assert wired["judged_by"] == ["rubric-0"] and not wired["library"]
    assert grading.rubric == GeneralRubric(rubric=ref("rubric-0"), packet=ref("packet-0"))
    assert grading.judged == {"actor": ref("judged-rubric-0")}
    assert grading.summary()["rubric_revised"] is False


def test_accepted_adversarial_linters_revise_the_rubric_before_the_grading_is_frozen(
    wired: dict[str, list[object]],
) -> None:
    """Mined hack families become linters; the rubric is re-authored and every group re-judged."""
    grading = pipeline_grading.grade_groups(context(adversarial=object()), rules(), {})
    assert wired["library"] == [(ref("differences-rubric-0"),)]
    assert [cast(Any, binding).reference for binding in wired["authored"]] == [ref("library-1")]
    assert wired["judged_by"] == ["rubric-0", "rubric-1"]
    assert grading.rubric.revised and grading.rubric.library == ref("library-1")
    assert grading.judged == {"actor": ref("judged-rubric-1")}
    assert grading.behavior == {"actor": ref("differences-rubric-1")}


def test_no_new_linter_keeps_the_general_rubric(wired: dict[str, list[object]]) -> None:
    """A library unchanged by the groups' families leaves the general rubric frozen."""
    wired["unchanged"] = [True]
    grading = pipeline_grading.grade_groups(context(adversarial=object()), rules(), {})
    assert wired["judged_by"] == ["rubric-0"] and not wired["authored"]
    assert not grading.rubric.revised
    assert (
        pipeline_grading.revised_rubric(
            context(adversarial=object(), library_id=None), rules(), {"actor": ref("d")}
        )
        is None
    )

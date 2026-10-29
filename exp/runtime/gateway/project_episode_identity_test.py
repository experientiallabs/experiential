"""Project episode keys retain public identity and explicit component boundaries."""

from exp.runtime.gateway import routing
from exp.runtime.gateway.project_episode_identity import project_episode_identity


def test_project_episode_export_and_component_boundaries() -> None:
    """Moving the owner leaves the public function identity and collision protection intact."""
    assert routing.project_episode_identity is project_episode_identity
    assert project_episode_identity(("a:b", "c", "revision", "episode")) != (
        project_episode_identity(("a", "b:c", "revision", "episode"))
    )

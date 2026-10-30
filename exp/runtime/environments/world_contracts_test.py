"""Declaration rules of executable-world manifests, checked before any backend runs."""

import pytest
from pydantic import ValidationError

from exp.runtime.environments.world_contracts import StateDomain, ToolEffectPolicy, WorldManifest


def test_manifest_refuses_ambiguous_or_unowned_declarations() -> None:
    """Domains and tools are unique, closed, single-owner, and external effects are sandboxed."""
    domain = StateDomain(name="a", owner="native")
    tool = ToolEffectPolicy(name="t", writes=("a",))
    with pytest.raises(ValidationError, match="unique"):
        WorldManifest(domains=(domain, domain), tools=(tool,))
    with pytest.raises(ValidationError, match="unique"):
        WorldManifest(domains=(domain,), tools=(tool, tool))
    with pytest.raises(ValidationError, match="unknown domains"):
        WorldManifest(domains=(domain,), tools=(ToolEffectPolicy(name="t", reads=("z",)),))
    with pytest.raises(ValidationError, match="spans native and simulated"):
        WorldManifest(
            domains=(domain, StateDomain(name="s", owner="simulated")),
            tools=(ToolEffectPolicy(name="t", reads=("a",), writes=("s",)),),
        )
    with pytest.raises(ValidationError, match="not sandboxed"):
        StateDomain(name="mail", owner="native", kind="effect")

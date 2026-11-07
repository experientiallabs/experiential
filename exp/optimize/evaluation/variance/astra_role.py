"""The Astra role split: one declared closed alias for build-time tag authoring, nothing else.

A front-door registration may declare ``astra_alias``, the one approved closed model. It may fill
only the build-time tag roles, before any training:

- the corpus and family tag-bank author (``slots["tag-banks"].author``);
- the state-tag author (``slots["state-rag"].author_alias``);
- the tag-monitor reviewer at the build-time checkpoints 1 to 5 (``slots["tag-monitor-*"].astra``).

The point-bar meta-rules come from a published authored bank (``PointQualification.bank``), so
they are written by the tag-bank author and need no role of their own.

Every other role stays open-weight: actors, judges, selectors, critics, world-model, rule and
authoring roles, the families reviewer, the rubric-evolution author and auditors, the epoch
monitor (checkpoint 6), and every decision alias (which must be a native decision model).
:func:`refuse_astra_outside_its_roles` walks the whole registration and refuses the declared
alias anywhere outside the paths above. The paths are declared here, never inferred from names.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonValue

ASTRA_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("astra_alias",),
        ("slots", "tag-banks", "author", "model_alias"),
        ("slots", "state-rag", "author_alias"),
        *(
            ("slots", f"tag-monitor-{name}", "astra", "astra_alias")
            for name in ("corpus", "families", "rules", "mining", "dense")
        ),
        *(
            ("slots", f"tag-monitor-{name}", "astra", "author", "model_alias")
            for name in ("corpus", "families", "rules", "mining", "dense")
        ),
    }
)
"""Every place the declared Astra alias may appear in a front-door registration."""

ROLE_KEYS = frozenset(
    {
        "model_alias",
        "author_alias",
        "namer_alias",
        "decision_alias",
        "jev_alias",
        "actor_alias",
        "astra_alias",
        "alias",
    }
)
"""Keys whose string value names the model filling a role."""

ROLE_LISTS = frozenset({"actors"})
"""Keys whose list of strings names the models filling a role."""


def _uses(value: JsonValue, path: tuple[str, ...], alias: str) -> list[tuple[str, ...]]:
    """Every role path under ``value`` naming ``alias``."""
    found: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = (*path, key)
            if key in ROLE_KEYS and item == alias:
                found.append(here)
            elif key in ROLE_LISTS and isinstance(item, list) and alias in item:
                found.append(here)
            else:
                found.extend(_uses(item, here, alias))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_uses(item, (*path, str(index)), alias))
    return found


def refuse_astra_outside_its_roles(registration: JsonValue, astra_alias: str | None) -> None:
    """Refuse the declared Astra alias outside the build-time tag roles.

    Args:
        registration: The front-door registration as JSON.
        astra_alias: The declared closed Astra alias, or None when none is declared.

    Raises:
        ValueError: The alias fills any other role, or Astra settings declare another alias.
    """
    if isinstance(registration, dict):
        for name, slot in (registration.get("slots") or {}).items():
            declared = slot.get("astra", {}) if isinstance(slot, dict) else {}
            named = declared.get("astra_alias") if isinstance(declared, dict) else None
            if named is not None and named != astra_alias:
                raise ValueError(
                    f"slot {name} declares Astra alias {named!r}, but the registration's "
                    f"astra_alias is {astra_alias!r}"
                )
    if astra_alias is None:
        return
    outside = [path for path in _uses(registration, (), astra_alias) if path not in ASTRA_PATHS]
    if outside:
        where = ", ".join(".".join(path) for path in outside[:5])
        raise ValueError(
            f"the closed Astra alias {astra_alias!r} may author build-time tags only; it is "
            f"named in another role at {where}"
        )

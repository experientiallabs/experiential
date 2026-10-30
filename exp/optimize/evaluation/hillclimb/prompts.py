"""Versioned actor system prompts: each version is its parent plus one noted edit.

A climb's baseline is the system prompt its cases already run under. Every later version names
its parent version, the one edit that produced it and, when an analyst proposed the edit, the
proposal set it was chosen from. Versions are content-addressed and never deleted, so a reverted
edit stays inspectable beside the kept ones.
"""

from datetime import datetime
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, ContractModel, stable_id
from exp.common.project import ArtifactStore, artifact_input
from exp.runtime.agents.chat import normalize_chat_system_prompt

PROMPT_VERSION_TYPE = "agent-prompt-version"


class PromptEdit(ContractModel):
    """One change to the actor's system prompt.

    Attributes:
        kind: ``append`` adds one instruction block after the current prompt (or starts one);
            ``replace`` substitutes the complete prompt.
        text: The added block or the complete replacement.
        change_note: Why the change should help; kept with the version whether kept or reverted.
    """

    kind: Literal["append", "replace"]
    text: str = Field(min_length=1, max_length=20_000)
    change_note: str = Field(min_length=1, max_length=2_000)

    def apply(self, prompt: str | None) -> str:
        """Return the complete prompt after this edit.

        Args:
            prompt: Current complete prompt, or None when the actor has none.

        Returns:
            The edited prompt, normalized as the chat actor would deliver it.

        Raises:
            ValueError: The result is blank or longer than the chat actor accepts.
        """
        edited = (
            self.text if self.kind == "replace" or prompt is None else prompt + "\n\n" + self.text
        )
        normalized = normalize_chat_system_prompt(edited)
        assert normalized is not None
        return normalized


class PromptVersion(ContractModel):
    """One saved system prompt of a climb.

    Attributes:
        climb: Identity of the climb the version belongs to.
        parent: Identity of the version this one edits; None for the climb's baseline.
        edit: The one change from the parent; None for the baseline.
        proposal: Analyst proposal set the edit was chosen from; None for an operator edit.
        system_prompt: Complete instruction; None only for a baseline that runs without one.
    """

    climb: str = Field(min_length=1)
    parent: str | None = None
    edit: PromptEdit | None = None
    proposal: ArtifactInput | None = None
    system_prompt: str | None

    @model_validator(mode="after")
    def lineage(self) -> Self:
        """Require a parent exactly for edited versions and an edit behind every proposal."""
        if (self.parent is None) != (self.edit is None):
            raise ValueError("a prompt version has a parent exactly when it records an edit")
        if self.proposal is not None and self.edit is None:
            raise ValueError("a proposal names the edit it supplied")
        if self.edit is not None and self.system_prompt is None:
            raise ValueError("an edited prompt version carries its complete prompt")
        return self

    @property
    def identity(self) -> str:
        """Content identity of this version."""
        return stable_id(PROMPT_VERSION_TYPE, self)


def baseline_version(climb: str, system_prompt: str | None) -> PromptVersion:
    """The climb's starting version: the instruction its cases already run under."""
    return PromptVersion(climb=climb, system_prompt=system_prompt)


def edited_version(
    parent: PromptVersion, edit: PromptEdit, *, proposal: ArtifactInput | None = None
) -> PromptVersion:
    """Apply one edit to a version without saving the result.

    Args:
        parent: Version the edit changes.
        edit: The one change.
        proposal: Analyst proposal set that supplied the edit, if any.

    Returns:
        The child version; save its parent first, then the child, with
        ``persist_prompt_version``.

    Raises:
        ValueError: The edit leaves the prompt unchanged or produces an invalid prompt.
    """
    prompt = edit.apply(parent.system_prompt)
    if prompt == parent.system_prompt:
        raise ValueError("the edit leaves the system prompt unchanged; propose a real change")
    return PromptVersion(
        climb=parent.climb,
        parent=parent.identity,
        edit=edit,
        proposal=proposal,
        system_prompt=prompt,
    )


def persist_prompt_version(
    store: ArtifactStore, version: PromptVersion, *, created_at: datetime, code_revision: str
) -> ArtifactInput:
    """Save a version once under its content identity and return its reference.

    Args:
        store: Project artifacts.
        version: Version to save; its parent and proposal must already be saved.
        created_at: Publication time of a first save.
        code_revision: Producing implementation of a first save.

    Returns:
        The saved version's reference; saving the same content again returns the same one.
    """
    identity = version.identity
    if not store.exists(identity):
        sources = [] if version.parent is None else [load_reference(store, version.parent)]
        if version.proposal is not None:
            sources.append(version.proposal)
        inputs = tuple(sorted(sources, key=lambda ref: ref.artifact_id))
        store.write_json(
            artifact_id=identity,
            artifact_type=PROMPT_VERSION_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=created_at, code_revision=code_revision, inputs=inputs
            ),
            files={"version.json": version},
        )
    return load_reference(store, identity)


def load_reference(store: ArtifactStore, identity: str) -> ArtifactInput:
    """The verified reference of a saved version by identity."""
    return artifact_input(store.read(identity).manifest)


def load_prompt_version(store: ArtifactStore, reference: ArtifactInput) -> PromptVersion:
    """Read a saved version and check it reproduces from its parent and edit.

    Args:
        store: Project artifacts.
        reference: Saved version.

    Returns:
        The verified version.

    Raises:
        ValueError: The reference, type, identity or edit lineage differs from the saved bytes.
    """
    manifest = store.read(reference.artifact_id).manifest
    if artifact_input(manifest) != reference or manifest.artifact_type != PROMPT_VERSION_TYPE:
        raise ValueError("prompt version reference differs from its saved artifact")
    version = PromptVersion.model_validate_json(
        store.read_bytes(reference.artifact_id, "version.json")
    )
    if version.identity != reference.artifact_id:
        raise ValueError("prompt version content differs from its identity")
    if version.parent is not None and version.edit is not None:
        parent = load_prompt_version(store, load_reference(store, version.parent))
        if parent.climb != version.climb or version.edit.apply(parent.system_prompt) != (
            version.system_prompt
        ):
            raise ValueError("prompt version does not reproduce from its parent and edit")
    return version

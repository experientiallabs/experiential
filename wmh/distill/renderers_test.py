"""Tests for the wmh verbatim renderers, driven through real harbor code.

The interesting assertions here are end-to-end on purpose. A synthetic
multi-turn terminus-2 episode runs through the REAL `harbor.llms.tinker`,
`harbor.llms.chat.Chat` and terminus-2 response parsers, with only the Tinker
transport stubbed by a scripted sampler, and the recorded token spans then go
through the REAL `wmh.distill.data.build_datums`. Nothing here talks to Tinker,
E2B or the network beyond loading a cached tokenizer.

Five invariants are what this module exists to protect:

1. tokens-in-tokens-out: the loss-mask-1.0 ids are exactly the recorded
   completion ids, concatenated;
2. verbatim content, ephemeral reasoning: a turn's reasoning is in its OWN
   loss mask and in no later prompt, while its action content is replayed into
   history as the exact ids the sampler issued. The episode is therefore one
   datum PER TURN, which is the deliberate cost of the context it saves;
3. the sampling prompt and the training prompt are the same tokens: what the
   sampler saw is what the datum trains on, so `sampled_logprobs` describe the
   sequence they are attached to. This is the one way to get the change silently
   wrong, so it is asserted rather than assumed;
4. harbor's terminus-2 parsers accept the content, including when the model's
   reasoning itself contains a JSON blob that looks like an action;
5. a per-command observation is clipped to `rollout.observation_clip_tokens`
   with its head, tail and a marker kept, and the task instruction never is.
"""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
from harbor.agents.terminus_2.terminus_xml_plain_parser import TerminusXMLPlainParser
from harbor.llms.chat import Chat
from harbor.llms.tinker import TinkerLLM
from tinker_cookbook.renderers import get_registered_renderer_names, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from wmh.distill.config import (
    DistillConfig,
    HarborConfig,
    RolloutConfig,
    SamplingConfig,
    StudentConfig,
    TeacherConfig,
)
from wmh.distill.data import build_datums
from wmh.distill.renderers import (
    NEMOTRON3_ULTRA_VERBATIM,
    NEMOTRON3_VERBATIM,
    OBSERVATION_CLIP_MARKER,
    QWEN3_5_VERBATIM,
    QWEN3_VERBATIM,
    VERBATIM_RENDERERS,
    Nemotron3UltraVerbatimRenderer,
    Nemotron3VerbatimRenderer,
    Qwen3_5VerbatimRenderer,
    Qwen3VerbatimRenderer,
    VerbatimContent,
    VerbatimHistoryMixin,
    is_known_renderer,
    register_verbatim_renderers,
)
from wmh.distill.tokens import TrialRecord
from wmh.providers.tinker import TokenSpan

if TYPE_CHECKING:
    from wmh.distill.data import TrainDatum

# Each pair is (base model, the wmh renderer a run config would name for it).
QWEN3_5_MODEL = "Qwen/Qwen3.5-9B"
QWEN3_6_MODEL = "Qwen/Qwen3.6-27B"
NEMOTRON_NANO_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
NEMOTRON_ULTRA_MODEL = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

REASONING_RENDERERS = [
    (QWEN3_5_MODEL, QWEN3_5_VERBATIM),
    (QWEN3_6_MODEL, QWEN3_5_VERBATIM),
    (NEMOTRON_NANO_MODEL, NEMOTRON3_VERBATIM),
    (NEMOTRON_ULTRA_MODEL, NEMOTRON3_ULTRA_VERBATIM),
]

TURNS = 6
COMMANDS = ["ls -la\n", "make\n", "cat src/parser.c\n", "make\n", "./run_tests.sh\n", "echo ok\n"]
_OBSERVATION_HEAD = "=== first line of the observation ==="
_FILLER_LINE = "gcc -c -O2 -Wall -Wextra -Isrc -o build/obj/unit"
REASONING = [
    "The terminal is at a fresh prompt in /app and nothing has run yet, so I should list the "
    "files before deciding anything.",
    "There is a src directory and a Makefile, so building is the obvious next step; I will run "
    "make and read the compiler errors rather than guess.",
    "Compilation failed on line 42 with an implicit declaration, which means a missing include; "
    "let me read the top of the file.",
    "The include is there after all, so the failure is the prototype order; rebuild and see "
    "whether the same error survives.",
    "The build is clean and the test binary exists, but a clean build is not a passing suite, so "
    "run the tests before claiming anything.",
    "All 14 tests pass with exit status zero, which is both acceptance criteria, so the task is "
    "done and I can stop here.",
]


def _tokenizer_or_skip(base_model: str) -> object:
    """The base model's real tokenizer, skipping when it is not available offline."""
    try:
        return get_tokenizer(base_model)
    except OSError as exc:  # pragma: no cover - only on a machine with no cached tokenizer
        pytest.skip(f"the {base_model} tokenizer is not available here: {exc}")


# -- the scripted sampler ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Sequence:
    """The sampled-sequence slice `TinkerLLM.call` reads."""

    tokens: list[int]
    logprobs: list[float]


@dataclass(frozen=True)
class _SampleResponse:
    """The sample-response slice `TinkerLLM.call` reads."""

    sequences: list[_Sequence]


class _ScriptedSamplingClient:
    """Answers every sample with the next scripted assistant turn."""

    def __init__(self, llm: _ScriptedTinkerLLM) -> None:
        self._llm = llm

    async def sample_async(
        self, prompt: object, num_samples: int, sampling_params: object
    ) -> _SampleResponse:
        """Return the scripted completion for this turn, ignoring the sampling params."""
        tokens = self._llm.next_completion(prompt)
        return _SampleResponse(sequences=[_Sequence(tokens=tokens, logprobs=[-0.5] * len(tokens))])


class _ScriptedTinkerLLM(TinkerLLM):
    """Harbor's real `TinkerLLM` with its transport replaced by a script.

    Everything the invariants depend on stays real: prompt construction through
    the renderer, `parse_response`, the rollout-detail recording, and the
    message history harbor keeps between turns.
    """

    def __init__(
        self,
        *,
        script: Sequence[str],
        model_name: str,
        renderer_name: str,
        max_tokens: int,
        context_limit: int,
        truncate_turn: int | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            renderer_name=renderer_name,
            max_tokens=max_tokens,
            context_limit=context_limit,
        )
        self._script = list(script)
        self._turn = 0
        self._truncate_turn = truncate_turn

    # The stub answers the one `sample_async` shape TinkerLLM.call uses; it is not a
    # tinker.SamplingClient, which is the whole point of the substitution.
    async def _ensure_client(self) -> _ScriptedSamplingClient:  # ty: ignore[invalid-method-override]
        """Never contact Tinker."""
        return _ScriptedSamplingClient(self)

    def next_completion(self, prompt: object) -> list[int]:
        """Encode this turn's scripted body, minus whatever the renderer prefilled."""
        tokenizer = self._renderer.tokenizer
        prompt_ids = prompt.to_ints()  # ty: ignore[unresolved-attribute]
        decoded = str(tokenizer.decode(prompt_ids))
        marker = "<|im_start|>assistant\n"
        prefill = decoded[decoded.rindex(marker) + len(marker) :]
        body = self._script[self._turn]
        assert body.startswith(prefill), (
            f"turn {self._turn} of the script must continue the renderer's prefill {prefill!r}"
        )
        tokens = list(tokenizer.encode(body[len(prefill) :], add_special_tokens=False))
        self._turn += 1
        if self._truncate_turn == self._turn - 1:
            # A turn that hit the output cap: no stop token, cut mid-answer.
            return tokens[: len(tokens) // 2]
        return tokens + [self._renderer._end_message_token]


def _json_action(index: int, command: str, *, complete: bool) -> str:
    payload: dict[str, object] = {
        "analysis": f"Turn {index}: reading the terminal state.",
        "plan": f"Turn {index}: run the next command.",
        "commands": [{"keystrokes": command, "duration": 1.0}],
    }
    if complete:
        payload["task_complete"] = True
    return json.dumps(payload, indent=2)


def _xml_action(index: int, command: str) -> str:
    return (
        f"<response>\n<analysis>Turn {index}.</analysis>\n<plan>Run it.</plan>\n<commands>\n"
        f"<command>\n<keystrokes>{command}</keystrokes>\n<duration>1.0</duration>\n</command>\n"
        "</commands>\n</response>"
    )


def _script(
    *, fmt: str = "json", reasoning: Sequence[str] | None = None, turns: int = TURNS
) -> list[str]:
    """One scripted assistant turn per step, each a think block plus an action."""
    thoughts = list(reasoning) if reasoning is not None else REASONING[:turns]
    bodies = []
    for index in range(turns):
        action = (
            _json_action(index, COMMANDS[index], complete=index == turns - 1)
            if fmt == "json"
            else _xml_action(index, COMMANDS[index])
        )
        bodies.append(f"<think>\n{thoughts[index]}\n</think>\n\n{action}")
    return bodies


# -- the episode driver --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Episode:
    """One driven episode: the parser verdicts plus the recorded token spans."""

    prompts: list[list[int]]
    completions: list[list[int]]
    errors: list[str]
    command_counts: list[int]
    keystrokes: list[list[str]]
    tokenizer: object
    histories: list[list[dict[str, object]]]
    """The message list each turn was sampled from, exactly as harbor's
    `TinkerLLM.call` assembles it (the chat history plus this turn's user
    message), so a test can re-render a turn's prompt the way training would."""

    renderer_name: str
    base_model: str

    def text(self, token_ids: Sequence[int]) -> str:
        """Decode ids keeping the template's special tokens."""
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=False))  # ty: ignore[unresolved-attribute]

    @property
    def spans(self) -> list[TokenSpan]:
        """The recorded spans, shaped exactly as the rollout collector assembles them."""
        return [
            TokenSpan(
                call_index=index,
                prompt_token_ids=list(prompt),
                sampled_token_ids=list(completion),
                sampled_logprobs=[-0.5] * len(completion),
            )
            for index, (prompt, completion) in enumerate(
                zip(self.prompts, self.completions, strict=True)
            )
        ]

    def prefix_breaks(self) -> list[int]:
        """Turn indices whose prompt does not extend the previous prompt plus its tokens."""
        breaks = []
        accumulated = list(self.prompts[0]) + list(self.completions[0])
        for index in range(1, len(self.prompts)):
            prompt = self.prompts[index]
            if list(prompt[: len(accumulated)]) != accumulated:
                breaks.append(index)
            accumulated = list(prompt) + list(self.completions[index])
        return breaks


def _run_episode(
    base_model: str,
    renderer_name: str,
    *,
    fmt: str = "json",
    script: Sequence[str] | None = None,
    truncate_turn: int | None = None,
    observation_clip_tokens: int = 0,
    observation_filler: int = 0,
    instruction: str = "Fix the build in /app and make the tests pass.",
) -> _Episode:
    """Drive a synthetic terminus-2 episode through real harbor code.

    Args:
        base_model: The Tinker lineup model whose tokenizer and renderer run.
        renderer_name: The registered renderer terminus-2 resolves.
        fmt: `json` or `xml`, selecting the terminus-2 response parser.
        script: Per-turn assistant bodies; defaults to `_script(fmt=fmt)`.
        truncate_turn: Turn index to cut off mid-answer (no stop token).
        observation_clip_tokens: The per-observation clip the renderers are
            registered with; 0 disables it.
        observation_filler: How many filler lines each command observation
            carries, for driving the clip.
        instruction: The opening user message (never an observation).
    """
    _tokenizer_or_skip(base_model)
    register_verbatim_renderers(observation_clip_tokens)
    bodies = list(script) if script is not None else _script(fmt=fmt)
    llm = _ScriptedTinkerLLM(
        script=bodies,
        truncate_turn=truncate_turn,
        model_name=base_model,
        renderer_name=renderer_name,
        max_tokens=4096,
        context_limit=1_000_000,
    )
    chat = Chat(llm)
    parser = TerminusJSONPlainParser() if fmt == "json" else TerminusXMLPlainParser()
    prompt = instruction
    errors: list[str] = []
    counts: list[int] = []
    keystrokes: list[list[str]] = []
    histories: list[list[dict[str, object]]] = []
    for index in range(len(bodies)):
        # Exactly what `TinkerLLM.call` renders: the chat history so far plus
        # this turn's user message.
        histories.append(
            [
                {"role": message.get("role", "user"), "content": message.get("content", "")}
                for message in chat.messages
            ]
            + [{"role": "user", "content": prompt}]
        )
        response = asyncio.run(chat.chat(prompt))
        result = parser.parse_response(response.content)
        errors.append(result.error)
        counts.append(len(result.commands))
        keystrokes.append([command.keystrokes for command in result.commands])
        feedback = f"\n\nERROR: {result.error}" if result.error else ""
        filler = "".join(f"{_FILLER_LINE} {line}\n" for line in range(observation_filler))
        prompt = (
            f"root@host:/app# {COMMANDS[index]}\n{_OBSERVATION_HEAD}\n"
            f"{filler}(output of turn {index})\n" + feedback
        )
    details = chat.rollout_details[0]
    return _Episode(
        prompts=[list(ids) for ids in details["prompt_token_ids"]],
        completions=[list(ids) for ids in details["completion_token_ids"]],
        errors=errors,
        command_counts=counts,
        keystrokes=keystrokes,
        tokenizer=llm._renderer.tokenizer,
        histories=histories,
        renderer_name=renderer_name,
        base_model=base_model,
    )


def _datums(episode: _Episode) -> list[TrainDatum]:
    """The episode's training datums, through the real `build_datums`."""
    record = TrialRecord(
        task_id="synthetic",
        attempt=1,
        trial_name="synthetic.1",
        reward=1.0,
        passed=True,
        spans=episode.spans,
        artifact_dir="/trials/synthetic",
    )
    cfg = DistillConfig(
        student=StudentConfig(base_model=QWEN3_5_MODEL),
        teacher=TeacherConfig(model=QWEN3_6_MODEL),
        harbor=HarborConfig(job_template="job.yaml"),
        rollout=RolloutConfig(context_budget_tokens=1_000_000),
        sampling=SamplingConfig(max_tokens=4096),
    )
    datums, _stats = build_datums([record], cfg)
    return datums


def _masked_text(datum: TrainDatum, tokenizer: object, mask: float) -> str:
    """The datum's tokens at one mask value, decoded."""
    ids = [
        token
        for token, weight in zip(datum.model_input_tokens, datum.loss_mask, strict=True)
        if weight == mask
    ]
    return str(tokenizer.decode(ids))  # ty: ignore[unresolved-attribute]


# -- registration --------------------------------------------------------------------------------


def test_the_four_verbatim_renderers_register_under_wmh_names() -> None:
    """The names a run config points `[rollout.renderers]` at."""
    register_verbatim_renderers()
    registered = get_registered_renderer_names()
    assert set(VERBATIM_RENDERERS) == {
        QWEN3_VERBATIM,
        QWEN3_5_VERBATIM,
        NEMOTRON3_VERBATIM,
        NEMOTRON3_ULTRA_VERBATIM,
    }
    for name in VERBATIM_RENDERERS:
        assert name in registered
        assert name.startswith("wmh/"), "namespaced so a cookbook name can never be shadowed"


def test_registration_is_idempotent() -> None:
    """The rollout collector calls it on every batch, so repeats must be free."""
    register_verbatim_renderers()
    before = sorted(get_registered_renderer_names())
    register_verbatim_renderers()
    register_verbatim_renderers()
    assert sorted(get_registered_renderer_names()) == before


@pytest.mark.parametrize(
    "renderer_class",
    [
        Qwen3VerbatimRenderer,
        Qwen3_5VerbatimRenderer,
        Nemotron3VerbatimRenderer,
        Nemotron3UltraVerbatimRenderer,
    ],
)
def test_the_mixin_comes_first_in_every_mro(renderer_class: type) -> None:
    """Mixin second would give the base renderer's overrides back and undo the fix."""
    mro = renderer_class.__mro__
    assert mro[1] is VerbatimHistoryMixin, f"{renderer_class.__name__} MRO is {mro}"


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_each_registered_renderer_builds_and_disclaims_the_extension_property(
    base_model: str, renderer_name: str
) -> None:
    """`get_renderer` must resolve the name against the model's real tokenizer.

    The extension property is deliberately NOT claimed: turn N's reasoning is
    absent from turn N+1's prompt, so the prompts are not prefixes of one
    another. The cookbook's own merge paths gate on this flag, so reporting True
    would merge sequences that are not extensions.
    """
    tokenizer = _tokenizer_or_skip(base_model)
    register_verbatim_renderers()
    renderer = get_renderer(renderer_name, tokenizer)  # ty: ignore[invalid-argument-type]
    assert isinstance(renderer, VerbatimHistoryMixin)
    assert not renderer.has_extension_property


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_registered_factory_carries_the_configured_observation_clip(
    base_model: str, renderer_name: str
) -> None:
    """The clip reaches the renderer through registration, not a global.

    That is what makes one config value apply identically in every arm: every
    rollout path builds its renderer from this registry (see
    `wmh.distill.rollouts.terminus_2_agent_kwargs`).
    """
    tokenizer = _tokenizer_or_skip(base_model)
    register_verbatim_renderers(1234)
    assert get_renderer(renderer_name, tokenizer).observation_clip_tokens == 1234  # ty: ignore[invalid-argument-type, unresolved-attribute]
    register_verbatim_renderers()
    assert get_renderer(renderer_name, tokenizer).observation_clip_tokens == 0  # ty: ignore[invalid-argument-type, unresolved-attribute]


def test_is_known_renderer_accepts_wmh_and_builtin_names_and_rejects_typos() -> None:
    """Config load is where a renderer typo must die, not the first paid rollout."""
    for name in VERBATIM_RENDERERS:
        assert is_known_renderer(name)
    for name in ("qwen3", "qwen3_5", "nemotron3", "nemotron3_ultra_disable_thinking"):
        assert is_known_renderer(name)
    for name in ("qwen3_5_verbatim", "wmh/qwen35_verbatim", "nemotron_3", ""):
        assert not is_known_renderer(name)


# -- the content channel -------------------------------------------------------------------------


def test_verbatim_content_is_a_string_that_survives_copies() -> None:
    """The ids ride from parse_response to render_message inside harbor's own plumbing."""
    content = VerbatimContent('{"analysis": "a"}', [11, 22, 33], 2)
    assert isinstance(content, str)
    assert content == '{"analysis": "a"}'
    assert json.dumps({"content": content}) == '{"content": "{\\"analysis\\": \\"a\\"}"}'
    for survivor in (
        copy.copy(content),
        copy.deepcopy(content),
        pickle.loads(pickle.dumps(content)),
    ):
        assert survivor.ids == [11, 22, 33]
        # The reasoning boundary has to survive too: a copy that forgot it would
        # replay the reasoning back into history.
        assert survivor.content_start == 2
        assert survivor.content_ids == [33]
    # Derived strings lose the ids, which is the safe direction: a turn whose ids
    # were lost re-renders through the base renderer instead of emitting wrong tokens.
    assert not hasattr(content.strip(), "ids")
    assert not hasattr(f"{content}", "ids")


def test_verbatim_content_with_no_reasoning_block_replays_whole() -> None:
    """content_start 0 means nothing was dropped, so every id replays."""
    content = VerbatimContent("plain", [7, 8, 9])
    assert content.content_start == 0
    assert content.content_ids == [7, 8, 9]


def test_verbatim_content_copies_the_ids_it_is_given() -> None:
    """A later mutation of the caller's list must not rewrite a recorded turn."""
    ids = [1, 2, 3]
    content = VerbatimContent("x", ids)
    ids.append(4)
    assert content.ids == [1, 2, 3]


# -- invariant 1: tokens in, tokens out --------------------------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_loss_mask_ids_are_exactly_the_recorded_completions(
    base_model: str, renderer_name: str
) -> None:
    """TITO: the tokens trained on are the tokens the sampler returned, in order.

    Anything else means the loss and the sampled logprobs describe different
    sequences, which no metric downstream would reveal. The merge is now one
    datum per turn, so the check runs over the datums CONCATENATED: every
    sampled id is trained on exactly once, reasoning included, and nothing else
    is.
    """
    episode = _run_episode(base_model, renderer_name)
    datums = _datums(episode)
    trained = [token for datum in datums for token in datum.sampled_token_ids()]
    expected = [token for completion in episode.completions for token in completion]
    assert trained == expected
    assert len(trained) == sum(len(completion) for completion in episode.completions)
    # Per datum, too: a datum's loss mask covers its own turn's completion whole.
    for datum, completion in zip(datums, episode.completions, strict=True):
        assert datum.sampled_token_ids() == completion


# -- invariant 2: verbatim content, ephemeral reasoning ------------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_every_turn_is_its_own_datum_because_history_drops_its_reasoning(
    base_model: str, renderer_name: str
) -> None:
    """The deliberate reversal: prompt(N+1) is no longer an extension of turn N.

    Turn N's reasoning is not in turn N+1's prompt, so the prefix test fails at
    every boundary and the episode becomes one datum per turn. That is the
    accepted cost (re-prefilled context, quadratic in turns) of episodes that
    stay inside the context ceiling; each datum is still a self-contained
    prompt-to-completion pair, and each one records which span it holds.
    """
    episode = _run_episode(base_model, renderer_name)
    assert len(episode.prompts) == TURNS
    assert episode.prefix_breaks() == list(range(1, TURNS))
    datums = _datums(episode)
    assert len(datums) == TURNS
    assert [datum.fragment_index for datum in datums] == list(range(TURNS))
    assert [datum.span_indices for datum in datums] == [[turn] for turn in range(TURNS)]
    # Each datum is exactly its turn's recorded call: that prompt, then that
    # completion, and nothing else.
    for datum, prompt, completion in zip(datums, episode.prompts, episode.completions, strict=True):
        assert datum.model_input_tokens == [*prompt, *completion]
        assert datum.loss_mask == [0.0] * len(prompt) + [1.0] * len(completion)


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_a_historical_turn_replays_its_exact_action_ids_without_its_reasoning(
    base_model: str, renderer_name: str
) -> None:
    """The mechanism, at renderer level: which ids a history turn contributes.

    The replayed ids must be a SUFFIX of the sampled ids (never a re-encoding of
    decoded text, which could retokenize differently) and must start after the
    turn's `</think>`.
    """
    tokenizer = _tokenizer_or_skip(base_model)
    register_verbatim_renderers()
    renderer = get_renderer(renderer_name, tokenizer)  # ty: ignore[invalid-argument-type]
    body = _script(turns=1)[0]
    # The Qwen3.5/Nemotron generation header prefills `<think>\n`, so the sampled
    # ids start inside the block, exactly as the sampler would return them.
    sampled = list(
        tokenizer.encode(  # ty: ignore[unresolved-attribute]
            body.split("<think>\n", 1)[1], add_special_tokens=False
        )
    ) + [renderer._end_message_token]  # noqa: SLF001 - the stop token the sampler appends
    message, termination = renderer.parse_response(sampled)
    assert termination.is_clean
    content = message["content"]
    assert isinstance(content, VerbatimContent)

    close_think = renderer.close_think_ids  # ty: ignore[unresolved-attribute]
    assert content.content_start == sampled.index(close_think[0]) + len(close_think)
    # A suffix of the sampled ids, so nothing was re-encoded.
    assert content.content_ids == sampled[len(sampled) - len(content.content_ids) :]
    replayed = str(tokenizer.decode(content.content_ids, skip_special_tokens=False))  # ty: ignore[unresolved-attribute]
    assert REASONING[0] not in replayed
    assert "</think>" not in replayed
    assert '"keystrokes": "ls -la\\n"' in replayed


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_stock_reasoning_renderer_is_the_thing_being_fixed(
    base_model: str, renderer_name: str
) -> None:
    """The regression this module exists for, measured against the same episode.

    The cookbook's own reasoning renderer for these models cannot even reach the
    prefix comparison: harbor's terminus-2 parser is handed a LIST and raises.
    """
    del renderer_name
    _tokenizer_or_skip(base_model)
    from tinker_cookbook.model_info import get_recommended_renderer_name

    stock = get_recommended_renderer_name(base_model)
    with pytest.raises((TypeError, AttributeError)):
        _run_episode(base_model, stock)


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_reasoning_is_trained_on_once_and_appears_in_no_prompt(
    base_model: str, renderer_name: str
) -> None:
    """The point of the change: think fully every turn, re-send it never.

    Per turn: that turn's reasoning is inside its own loss mask exactly once, and
    no prompt in the episode contains any turn's reasoning. The ACTION content of
    every earlier turn is still there, which is what makes the history usable at
    all.
    """
    episode = _run_episode(base_model, renderer_name)
    datums = _datums(episode)
    for turn, datum in enumerate(datums):
        trained_text = _masked_text(datum, episode.tokenizer, 1.0)
        context_text = _masked_text(datum, episode.tokenizer, 0.0)
        assert trained_text.count("</think>") == 1
        assert trained_text.count(REASONING[turn]) == 1
        for thought in REASONING[:TURNS]:
            assert thought not in context_text
        # Every earlier turn's action survives in the context verbatim.
        for earlier in range(turn):
            assert f'"keystrokes": "{COMMANDS[earlier]}"'.replace("\n", "\\n") in context_text


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_context_shrinks_by_the_reasoning_it_no_longer_carries(
    base_model: str, renderer_name: str
) -> None:
    """The measured effect, in miniature: per-turn growth without the reasoning.

    Each prompt grows by (this turn's observation + the previous turn's ACTION),
    never by the previous turn's reasoning. Asserted as a token count so the test
    fails if the drop silently stops happening, which decoded-text checks alone
    would not catch (a re-encoded action could look right and tokenize wrong).
    """
    episode = _run_episode(base_model, renderer_name)
    reasoning_tokens = [
        len(episode.tokenizer.encode(thought, add_special_tokens=False))  # ty: ignore[unresolved-attribute]
        for thought in REASONING[:TURNS]
    ]
    for turn in range(1, TURNS):
        growth = len(episode.prompts[turn]) - len(episode.prompts[turn - 1])
        # The previous turn's whole completion (reasoning included) would have
        # been at least this much bigger.
        assert growth < len(episode.completions[turn - 1]) + reasoning_tokens[turn - 1]


# -- invariant 3: the sampling prompt is the training prompt --------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_sampling_prompt_and_the_training_prompt_are_the_same_tokens(
    base_model: str, renderer_name: str
) -> None:
    """The one way to get this change silently wrong, closed by assertion.

    If the reasoning drop happened when ASSEMBLING training data rather than when
    building the sampling prompt, the datum's context would not be the context
    `sampled_logprobs` were measured under. Importance sampling would then
    reweight against logprobs that do not describe the sequence, and nothing
    would error: the run would simply train on mismatched weights.

    Two halves, both required:

    1. every datum's mask-0 prefix IS the recorded prompt of the call it holds
       (what training consumes is what the sampler consumed);
    2. re-rendering that call's conversation with a FRESH renderer built from the
       same name reproduces the recorded prompt id for id and byte for byte, so
       the drop is a property of the renderer rather than of one instance's
       state, and no later re-render can diverge from it.
    """
    episode = _run_episode(base_model, renderer_name)
    datums = _datums(episode)
    tokenizer = _tokenizer_or_skip(base_model)
    register_verbatim_renderers()
    fresh = get_renderer(renderer_name, tokenizer)  # ty: ignore[invalid-argument-type]

    for turn, (datum, prompt) in enumerate(zip(datums, episode.prompts, strict=True)):
        context = [
            token
            for token, weight in zip(datum.model_input_tokens, datum.loss_mask, strict=True)
            if weight == 0.0
        ]
        assert context == prompt, f"turn {turn}: the datum's context is not the sampled prompt"
        rerendered = fresh.build_generation_prompt(episode.histories[turn]).to_ints()  # ty: ignore[invalid-argument-type]
        assert rerendered == prompt, f"turn {turn}: re-render diverged from the sampled prompt"
        assert episode.text(rerendered).encode("utf-8") == episode.text(prompt).encode("utf-8")


# -- invariant 4: harbor's parsers accept the content ------------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_json_parser_reads_every_turn(base_model: str, renderer_name: str) -> None:
    """`error == ''` is not enough: a silent `ncmd == 0` is a no-op episode."""
    episode = _run_episode(base_model, renderer_name, fmt="json")
    assert episode.errors == [""] * TURNS
    assert all(count >= 1 for count in episode.command_counts)
    assert [keys[0] for keys in episode.keystrokes] == COMMANDS[:TURNS]


def test_the_xml_parser_reads_every_turn() -> None:
    """The other terminus-2 response format, whose failure mode was AttributeError."""
    episode = _run_episode(QWEN3_5_MODEL, QWEN3_5_VERBATIM, fmt="xml")
    assert episode.errors == [""] * TURNS
    assert all(count >= 1 for count in episode.command_counts)
    assert len(_datums(episode)) == TURNS


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_reasoning_that_contains_a_json_action_never_reaches_the_parser(
    base_model: str, renderer_name: str
) -> None:
    """The hostile case: the model talks through a candidate action before choosing.

    If the think block reached the parser, its JSON is what would be extracted
    (the parser's mixed-content auto-fix scans for the FIRST valid object), and
    the agent would run a command the model rejected.
    """
    decoy = json.dumps(
        {
            "analysis": "decoy",
            "plan": "decoy",
            "commands": [{"keystrokes": "rm -rf /\n", "duration": 1.0}],
        }
    )
    thoughts = [
        f"I could answer with {decoy} but that is wrong, so I will not. "
        f"Consider also struct {{ int port; struct {{ char *n; }} inner; }} cfg;"
        for _ in range(TURNS)
    ]
    episode = _run_episode(base_model, renderer_name, script=_script(reasoning=thoughts))
    assert episode.errors == [""] * TURNS
    assert all(count == 1 for count in episode.command_counts)
    assert [keys[0] for keys in episode.keystrokes] == COMMANDS[:TURNS]
    assert all("rm -rf /\n" not in keys for keys in episode.keystrokes)
    assert len(_datums(episode)) == TURNS
    # The decoy, which lives in the reasoning, reaches no later prompt either.
    for prompt in episode.prompts:
        assert "decoy" not in episode.text(prompt)


def test_a_turn_that_never_stopped_is_not_carried_verbatim() -> None:
    """A truncated turn has no stop token, so replaying its ids would corrupt the context.

    It falls back to the base renderer, which re-renders it from text rather than
    from the ids the sampler issued, and is counted by
    `RolloutStats.truncated_spans`.
    """
    tokenizer = _tokenizer_or_skip(QWEN3_5_MODEL)
    register_verbatim_renderers()
    renderer = get_renderer(QWEN3_5_VERBATIM, tokenizer)  # ty: ignore[invalid-argument-type]
    body = "reasoning that never closes and never terminates"
    ids = list(tokenizer.encode(body, add_special_tokens=False))  # ty: ignore[unresolved-attribute]
    message, termination = renderer.parse_response(ids)
    assert not termination.is_clean
    assert not isinstance(message["content"], VerbatimContent)


def test_a_turn_with_no_reasoning_block_replays_verbatim_and_still_merges() -> None:
    """No `</think>` means nothing to drop, so that boundary keeps the old behavior.

    The turn replays under its OWN sampling header with all of its ids, so
    prompt(N+1) really does extend prompt(N) plus sampled(N) and the two turns
    merge into ONE datum. Worth pinning twice over: it is the branch that proves
    the fragmentation is caused by the reasoning drop and nothing else, and it is
    why a datum can hold more than one span (hence `span_indices`, plural).
    """
    # Qwen3.5's generation header prefills `<think>\n`, so a turn that never
    # emits `</think>` is a turn whose sampled ids carry no boundary at all.
    unclosed = [f"<think>\nturn {index} never closes its reasoning" for index in range(2)]
    episode = _run_episode(QWEN3_5_MODEL, QWEN3_5_VERBATIM, script=unclosed)
    assert episode.prefix_breaks() == []
    datums = _datums(episode)
    assert len(datums) == 1
    assert datums[0].span_indices == [0, 1]

    # An EMPTY think block still leaves a `</think>` in the sampled ids, so the
    # drop fires and the boundary breaks, exactly as for real reasoning.
    empty_block = [
        f"<think>\n\n</think>\n\n{_json_action(index, COMMANDS[index], complete=False)}"
        for index in range(2)
    ]
    assert _run_episode(QWEN3_5_MODEL, QWEN3_5_VERBATIM, script=empty_block).prefix_breaks() == [1]


# -- invariant 5: the per-command observation clip -----------------------------------------------


def test_a_long_observation_is_clipped_to_the_configured_budget() -> None:
    """Head, tail and a marker survive; the middle is dropped and named."""
    budget = 200
    episode = _run_episode(
        QWEN3_5_MODEL,
        QWEN3_5_VERBATIM,
        observation_clip_tokens=budget,
        observation_filler=400,
    )
    unclipped = _run_episode(
        QWEN3_5_MODEL, QWEN3_5_VERBATIM, observation_clip_tokens=0, observation_filler=400
    )
    # Growth per turn is bounded by the clip plus the marker and the replayed
    # action, so the whole episode stays far shorter than the unclipped one.
    assert len(episode.prompts[-1]) < len(unclipped.prompts[-1]) / 2

    later = episode.text(episode.prompts[2])
    assert "tokens of terminal output omitted" in later
    assert _OBSERVATION_HEAD in later, "the head of the observation is kept"
    assert "(output of turn 0)" in later, "the tail of the observation is kept"
    assert later.count(_FILLER_LINE) < 400, "the middle is dropped"
    # The clip is bounded: one observation's own tokens cannot exceed the budget
    # plus the marker, however long the command printed.
    marker_tokens = len(
        episode.tokenizer.encode(  # ty: ignore[unresolved-attribute]
            OBSERVATION_CLIP_MARKER.format(dropped=999_999), add_special_tokens=False
        )
    )
    growth = len(episode.prompts[2]) - len(episode.prompts[1])
    assert growth <= budget + marker_tokens + len(episode.completions[1])


def test_the_task_instruction_is_never_clipped() -> None:
    """The opening message is not an observation; clipping it would change the task."""
    instruction = "Fix the build in /app. " + " ".join(
        f"Requirement {index}: keep every acceptance criterion." for index in range(300)
    )
    episode = _run_episode(
        QWEN3_5_MODEL, QWEN3_5_VERBATIM, observation_clip_tokens=50, instruction=instruction
    )
    for prompt in episode.prompts:
        assert instruction in episode.text(prompt)
        assert "Requirement 299" in episode.text(prompt)


def test_a_short_observation_is_left_alone() -> None:
    """Under budget means untouched: no marker, nothing dropped."""
    episode = _run_episode(QWEN3_5_MODEL, QWEN3_5_VERBATIM, observation_clip_tokens=2000)
    text = episode.text(episode.prompts[-1])
    assert "omitted" not in text
    for turn in range(TURNS - 1):
        assert f"(output of turn {turn})" in text


def test_the_clip_is_stable_across_the_turns_that_replay_it() -> None:
    """A clipped observation must clip identically every time it is re-rendered.

    Otherwise the training prompt of a later turn would disagree with the one the
    sampler saw for THAT turn, which is the same silent mismatch the reasoning
    drop has to avoid: every turn's prompt is a fresh full render.
    """
    episode = _run_episode(
        QWEN3_5_MODEL, QWEN3_5_VERBATIM, observation_clip_tokens=120, observation_filler=300
    )
    tokenizer = _tokenizer_or_skip(QWEN3_5_MODEL)
    register_verbatim_renderers(120)
    fresh = get_renderer(QWEN3_5_VERBATIM, tokenizer)  # ty: ignore[invalid-argument-type]
    for turn, prompt in enumerate(episode.prompts):
        assert fresh.build_generation_prompt(episode.histories[turn]).to_ints() == prompt  # ty: ignore[invalid-argument-type]
    # Turn 0's clipped observation is still byte-identical in the last prompt.
    first = episode.text(episode.prompts[1]).split("<|im_start|>user\n")[-1]
    assert first.split("<|im_end|>")[0] in episode.text(episode.prompts[-1])

"""Tests for the wmh verbatim renderers, driven through real harbor code.

The interesting assertions here are end-to-end on purpose. A synthetic
multi-turn terminus-2 episode runs through the REAL `harbor.llms.tinker`,
`harbor.llms.chat.Chat` and terminus-2 response parsers, with only the Tinker
transport stubbed by a scripted sampler, and the recorded token spans then go
through the REAL `wmh.distill.data.build_datums`. Nothing here talks to Tinker,
E2B or the network beyond loading a cached tokenizer.

Four invariants are what this module exists to protect:

1. tokens-in-tokens-out: the loss-mask-1.0 ids are exactly the recorded
   completion ids, concatenated;
2. the prefix property: every turn's prompt extends the previous prompt plus
   its sampled tokens, so an episode is ONE datum;
3. the reasoning survives exactly once, inside the loss mask and nowhere in the
   mask-0 context;
4. harbor's terminus-2 parsers accept the content, including when the model's
   reasoning itself contains a JSON blob that looks like an action.
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
    QWEN3_5_STRIP_HISTORY,
    QWEN3_5_VERBATIM,
    QWEN3_VERBATIM,
    VERBATIM_RENDERERS,
    WMH_RENDERERS,
    Nemotron3UltraVerbatimRenderer,
    Nemotron3VerbatimRenderer,
    Qwen3_5StripHistoryRenderer,
    Qwen3_5VerbatimRenderer,
    Qwen3VerbatimRenderer,
    StripHistoryMixin,
    VerbatimContent,
    VerbatimHistoryMixin,
    is_known_renderer,
    register_wmh_renderers,
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
) -> _Episode:
    """Drive a synthetic terminus-2 episode through real harbor code."""
    _tokenizer_or_skip(base_model)
    register_wmh_renderers()
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
    prompt = "Fix the build in /app and make the tests pass."
    errors: list[str] = []
    counts: list[int] = []
    keystrokes: list[list[str]] = []
    for index in range(len(bodies)):
        response = asyncio.run(chat.chat(prompt))
        result = parser.parse_response(response.content)
        errors.append(result.error)
        counts.append(len(result.commands))
        keystrokes.append([command.keystrokes for command in result.commands])
        feedback = f"\n\nERROR: {result.error}" if result.error else ""
        prompt = f"root@host:/app# {COMMANDS[index]}\n(output of turn {index})\n" + feedback
    details = chat.rollout_details[0]
    return _Episode(
        prompts=[list(ids) for ids in details["prompt_token_ids"]],
        completions=[list(ids) for ids in details["completion_token_ids"]],
        errors=errors,
        command_counts=counts,
        keystrokes=keystrokes,
        tokenizer=llm._renderer.tokenizer,
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
    register_wmh_renderers()
    registered = get_registered_renderer_names()
    assert set(VERBATIM_RENDERERS) == {
        QWEN3_VERBATIM,
        QWEN3_5_VERBATIM,
        NEMOTRON3_VERBATIM,
        NEMOTRON3_ULTRA_VERBATIM,
    }
    for name in WMH_RENDERERS:
        assert name in registered
        assert name.startswith("wmh/"), "namespaced so a cookbook name can never be shadowed"


def test_registration_is_idempotent() -> None:
    """The rollout collector calls it on every batch, so repeats must be free."""
    register_wmh_renderers()
    before = sorted(get_registered_renderer_names())
    register_wmh_renderers()
    register_wmh_renderers()
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
def test_each_registered_renderer_builds_and_claims_the_extension_property(
    base_model: str, renderer_name: str
) -> None:
    """`get_renderer` must resolve the name against the model's real tokenizer."""
    tokenizer = _tokenizer_or_skip(base_model)
    register_wmh_renderers()
    renderer = get_renderer(renderer_name, tokenizer)  # ty: ignore[invalid-argument-type]
    assert isinstance(renderer, VerbatimHistoryMixin)
    assert renderer.has_extension_property


def test_strip_history_keeps_reasoning_in_the_turn_and_out_of_the_next_prompt() -> None:
    """The whole point of the strip-history renderer, asserted on real token ids.

    Reasoning must stay in what we TRAIN on (the sampled turn, which the loss
    masks over) and stay out of what we PROMPT with (every later turn), because
    carrying it forward is what overflowed the context window.
    """
    episode = _run_episode(QWEN3_5_MODEL, QWEN3_5_STRIP_HISTORY)

    def decode(ids: list[int]) -> str:
        return str(episode.tokenizer.decode(ids))  # ty: ignore[unresolved-attribute]

    # The `<think>` OPEN tag lives in the renderer's generation prefill, not in the sampled
    # ids, so the reasoning TEXT is what to assert on at both ends.
    for index, completion in enumerate(episode.completions):
        assert REASONING[index] in decode(completion), (
            f"turn {index} must still reason -- that is what we distill"
        )
    for index, prompt in enumerate(episode.prompts):
        carried = [thought for thought in REASONING[:index] if thought in decode(prompt)]
        assert not carried, (
            f"prompt {index} carries {len(carried)} prior turn(s) of reasoning; "
            "context grows with every turn and the episode overflows"
        )
    assert all(error == "" for error in episode.errors), (
        "harbor's parser must accept the content, which is why it is a plain str"
    )
    assert all(count > 0 for count in episode.command_counts), "actions must survive the strip"


def test_strip_history_fragments_the_episode_into_per_turn_datums() -> None:
    """The cost side of the trade, made explicit so a regression cannot hide it.

    Dropping thinking from history means turn N's sampled ids are not a
    substring of prompt N+1, so the prefix merge cannot fire. This is expected,
    not a bug -- but it multiplies teacher prefill, so it is pinned.
    """
    episode = _run_episode(QWEN3_5_MODEL, QWEN3_5_STRIP_HISTORY)
    datums = _datums(episode)
    assert len(datums) == len(episode.completions) > 1, "one datum per turn"

    merged = _datums(_run_episode(QWEN3_5_MODEL, QWEN3_5_VERBATIM))
    assert len(merged) == 1, "the verbatim renderer still merges, for contexts that fit"


def test_strip_history_renderer_disclaims_the_extension_property() -> None:
    """It must report False, or `build_datums` would merge non-prefix turns."""
    tokenizer = _tokenizer_or_skip(QWEN3_5_MODEL)
    register_wmh_renderers()
    renderer = get_renderer(QWEN3_5_STRIP_HISTORY, tokenizer)  # ty: ignore[invalid-argument-type]
    assert isinstance(renderer, StripHistoryMixin)
    assert not renderer.has_extension_property
    assert Qwen3_5StripHistoryRenderer.__mro__[1] is StripHistoryMixin


def test_is_known_renderer_accepts_wmh_and_builtin_names_and_rejects_typos() -> None:
    """Config load is where a renderer typo must die, not the first paid rollout."""
    for name in WMH_RENDERERS:
        assert is_known_renderer(name)
    for name in ("qwen3", "qwen3_5", "nemotron3", "nemotron3_ultra_disable_thinking"):
        assert is_known_renderer(name)
    for name in ("qwen3_5_verbatim", "wmh/qwen35_verbatim", "nemotron_3", ""):
        assert not is_known_renderer(name)


# -- the content channel -------------------------------------------------------------------------


def test_verbatim_content_is_a_string_that_survives_copies() -> None:
    """The ids ride from parse_response to render_message inside harbor's own plumbing."""
    content = VerbatimContent('{"analysis": "a"}', [11, 22, 33])
    assert isinstance(content, str)
    assert content == '{"analysis": "a"}'
    assert json.dumps({"content": content}) == '{"content": "{\\"analysis\\": \\"a\\"}"}'
    assert copy.copy(content).ids == [11, 22, 33]
    assert copy.deepcopy(content).ids == [11, 22, 33]
    assert pickle.loads(pickle.dumps(content)).ids == [11, 22, 33]
    # Derived strings lose the ids, which is the safe direction: a turn whose ids
    # were lost re-renders through the base renderer instead of emitting wrong tokens.
    assert not hasattr(content.strip(), "ids")
    assert not hasattr(f"{content}", "ids")


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
    sequences, which no metric downstream would reveal.
    """
    episode = _run_episode(base_model, renderer_name)
    datums = _datums(episode)
    assert len(datums) == 1
    trained = [
        token
        for token, weight in zip(datums[0].model_input_tokens, datums[0].loss_mask, strict=True)
        if weight == 1.0
    ]
    expected = [token for completion in episode.completions for token in completion]
    assert trained == expected
    assert len(trained) == sum(len(completion) for completion in episode.completions)


# -- invariant 2: the prefix property ------------------------------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_every_turn_extends_the_previous_one_so_the_episode_is_one_datum(
    base_model: str, renderer_name: str
) -> None:
    """The whole cost model: one datum per episode, not one per turn."""
    episode = _run_episode(base_model, renderer_name)
    assert len(episode.prompts) == TURNS
    assert episode.prefix_breaks() == []
    datums = _datums(episode)
    assert len(datums) == 1
    assert datums[0].fragment_index == 0


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


# -- invariant 3: the reasoning survives, exactly once ---------------------------------------


@pytest.mark.parametrize(("base_model", "renderer_name"), REASONING_RENDERERS)
def test_the_reasoning_is_inside_the_loss_mask_and_never_in_the_context(
    base_model: str, renderer_name: str
) -> None:
    """Kion's requirement: distillation trains on the reasoning, once per turn."""
    episode = _run_episode(base_model, renderer_name)
    datum = _datums(episode)[0]
    trained_text = _masked_text(datum, episode.tokenizer, 1.0)
    context_text = _masked_text(datum, episode.tokenizer, 0.0)
    assert trained_text.count("</think>") == TURNS
    assert context_text.count("</think>") == 0
    for thought in REASONING[:TURNS]:
        assert trained_text.count(thought) == 1
        assert thought not in context_text


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
    assert episode.prefix_breaks() == []


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
    assert episode.prefix_breaks() == []
    assert len(_datums(episode)) == 1


def test_a_turn_that_never_stopped_is_not_carried_verbatim() -> None:
    """A truncated turn has no stop token, so replaying its ids would corrupt the context.

    It falls back to the base renderer, which is visible downstream as a prefix
    break, and is counted by `RolloutStats.truncated_spans`.
    """
    tokenizer = _tokenizer_or_skip(QWEN3_5_MODEL)
    register_wmh_renderers()
    renderer = get_renderer(QWEN3_5_VERBATIM, tokenizer)  # ty: ignore[invalid-argument-type]
    body = "reasoning that never closes and never terminates"
    ids = list(tokenizer.encode(body, add_special_tokens=False))  # ty: ignore[unresolved-attribute]
    message, termination = renderer.parse_response(ids)
    assert not termination.is_clean
    assert not isinstance(message["content"], VerbatimContent)

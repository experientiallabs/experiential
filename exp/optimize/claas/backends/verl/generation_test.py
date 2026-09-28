"""Original-token provenance and prompt/action rendering with a real local tokenizer."""

from pathlib import Path
from typing import Literal

import pytest
from tokenizers import AddedToken

from exp.common.claas.generation import GenerationRequest
from exp.common.models import AssistantAction, ModelMessage, ToolCall
from exp.common.tasks import ToolSchema
from exp.optimize.claas.backends.verl.configuration import ResidentVerlSettings
from exp.optimize.claas.backends.verl.generation import _message, generation_result, prompt_ids
from exp.optimize.claas.backends.verl.native_test import tokenizer
from exp.optimize.claas.training_contracts_test import spec


def test_original_tokens_and_probability_values_survive_text_decoding(tmp_path: Path) -> None:
    """Decoded text does not replace IDs, log probabilities, or the sampled policy identity."""
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, decoder="text")
    request = GenerationRequest(
        request_id="r", model=spec().adapter_id, prompt="a b", maximum_output_tokens=3
    )
    model_tokenizer = tokenizer()
    model_tokenizer.eos_token = "g"
    prompt = prompt_ids(request, model_tokenizer, spec(), settings)
    assert prompt == (1, 2)
    result = generation_result(
        request, prompt, (3, 7), (-1.25, -2.5), model_tokenizer, spec(), settings, "policy-7"
    )
    assert result.raw_text == "c"
    assert result.exact_tokens.response_token_ids == (3, 7)
    assert result.exact_tokens.response_logprobs == (-1.25, -2.5)
    assert result.exact_tokens.policy_revision == "policy-7"
    assert result.exact_tokens.sampling_temperature == 1
    assert result.action.content == "c"


def test_prompt_limits_are_rejected_without_truncation(tmp_path: Path) -> None:
    """The sampled input must fit the exact requested output bound and model context."""
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, maximum_output_tokens=4)
    request = GenerationRequest(
        request_id="r", model=spec().base_model, prompt="a b", maximum_output_tokens=5
    )
    with pytest.raises(ValueError, match="maximum_output_tokens"):
        prompt_ids(request, tokenizer(), spec(), settings)
    with pytest.raises(ValueError, match="sequence limit"):
        prompt_ids(
            request.model_copy(update={"maximum_output_tokens": 4}),
            tokenizer(),
            spec().model_copy(update={"max_sequence_tokens": 5}),
            settings,
        )


def test_chat_template_receives_original_tool_linkage(tmp_path: Path) -> None:
    """The actual tokenizer template consumes tool metadata and assistant calls once."""
    model_tokenizer = tokenizer()
    model_tokenizer.chat_template = (
        "{% for message in messages %}{{message['content']}} {% endfor %}"
        "{% if tools %}{{tools[0]['function']['name']}}{% endif %}"
    )
    tool = ToolSchema(name="c", description="tool", input_schema={"type": "object"})
    action = AssistantAction(tool_calls=(ToolCall(call_id="call-1", name="c", arguments={}),))
    message = ModelMessage(role="assistant", content="", assistant_action=action)
    assert _message(message)["tool_calls"] == [
        {"id": "call-1", "type": "function", "function": {"name": "c", "arguments": {}}}
    ]
    request = GenerationRequest(
        request_id="r",
        model=spec().base_model,
        messages=(ModelMessage(role="user", content="a b"),),
        tools=(tool,),
        maximum_output_tokens=4,
    )
    assert prompt_ids(
        request, model_tokenizer, spec(), ResidentVerlSettings(checkpoint_root=tmp_path)
    ) == (1, 2, 3)


def test_tool_syntax_special_tokens_survive_visible_decoding(tmp_path: Path) -> None:
    """Tool syntax may be special tokens; only trailing EOS/padding may disappear from text."""
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<tool_call>", "</tool_call>"]}
    )
    text = '<tool_call>{"name":"c","arguments":{}}</tool_call>'
    response = tuple(model_tokenizer.encode(text, add_special_tokens=False))
    # The tiny vocabulary leaves JSON unknown, but preserves declared special tool markers.
    rendered = model_tokenizer.decode(list(response), skip_special_tokens=False)
    assert "<tool_call>" in rendered and "</tool_call>" in rendered
    request = GenerationRequest(
        request_id="r", model=spec().base_model, prompt="a", maximum_output_tokens=64
    )
    result = generation_result(
        request,
        (1,),
        response,
        tuple(-1.0 for _ in response),
        model_tokenizer,
        spec(),
        ResidentVerlSettings(checkpoint_root=tmp_path, decoder="text"),
        "policy-1",
    )
    assert result.raw_text == rendered
    assert result.exact_tokens.response_token_ids == response


@pytest.mark.parametrize("include_declared_call", [False, True])
def test_undeclared_sampled_tools_preserve_original_learning_evidence(
    tmp_path: Path, include_declared_call: bool
) -> None:
    """A model's wrong tool name remains an unchanged sampled action for caller feedback."""
    raw = '<tool_call>{"name":"functions.missing","arguments":{"key":"value"}}</tool_call>'
    if include_declared_call:
        raw = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>\n' + raw
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
    )
    response = tuple(model_tokenizer.encode(raw, add_special_tokens=False)) + (7,)
    logprobs = tuple(-0.25 - index for index in range(len(response)))
    request = GenerationRequest(
        request_id="sample-1",
        model=spec().base_model,
        messages=(ModelMessage(role="user", content="look up the record"),),
        tools=(ToolSchema(name="lookup", description="Find a record", input_schema={}),),
        maximum_output_tokens=8,
    )
    result = generation_result(
        request,
        (1, 2),
        response,
        logprobs,
        model_tokenizer,
        spec(),
        ResidentVerlSettings(checkpoint_root=tmp_path, decoder="hermes"),
        "policy-7",
    )
    expected = ["lookup", "functions.missing"] if include_declared_call else ["functions.missing"]
    assert [call.name for call in result.action.tool_calls] == expected
    assert result.action.tool_calls[-1].arguments == {"key": "value"}
    assert result.raw_text == raw
    assert result.exact_tokens.prompt_token_ids == (1, 2)
    assert result.exact_tokens.response_token_ids == response
    assert result.exact_tokens.response_logprobs == logprobs
    assert result.exact_tokens.policy_revision == "policy-7"
    assert result.response_id == request.request_id


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_incomplete_tool_syntax_is_not_repaired_into_an_action(
    tmp_path: Path, finish_reason: Literal["stop", "length"]
) -> None:
    """Transporting a parsed wrong name does not invent structure for an incomplete sample."""
    raw = '<tool_call>{"name":"lookup","arguments":{}}'
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
    )
    response = tuple(model_tokenizer.encode(raw, add_special_tokens=False))
    request = GenerationRequest(
        request_id="incomplete", model=spec().base_model, prompt="a", maximum_output_tokens=8
    )
    with pytest.raises(ValueError, match="incomplete tool call"):
        generation_result(
            request,
            (1,),
            response,
            tuple(-1.0 for _ in response),
            model_tokenizer,
            spec(),
            ResidentVerlSettings(checkpoint_root=tmp_path, decoder="hermes"),
            "policy-0",
            finish_reason,
        )


@pytest.mark.parametrize("decoder", ["hermes", "qwen35"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_unfinished_reasoning_requires_native_length_to_retain_evidence(
    tmp_path: Path,
    decoder: Literal["hermes", "qwen35"],
    finish_reason: Literal["stop", "length"],
) -> None:
    """Identical sampled reasoning is a retained length failure only with that native reason."""
    raw = "<think>private unfinished reasoning"
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
    )
    response = tuple(model_tokenizer.encode(raw, add_special_tokens=False))
    request = GenerationRequest(
        request_id="reasoning", model=spec().base_model, prompt="a", maximum_output_tokens=1
    )
    settings = ResidentVerlSettings(checkpoint_root=tmp_path, decoder=decoder)
    if finish_reason == "stop":
        with pytest.raises(ValueError, match="ended inside reasoning"):
            generation_result(
                request,
                (1,),
                response,
                (-0.25,),
                model_tokenizer,
                spec(),
                settings,
                "policy-7",
                finish_reason,
            )
        return
    result = generation_result(
        request,
        (1,),
        response,
        (-0.25,),
        model_tokenizer,
        spec(),
        settings,
        "policy-7",
        finish_reason,
    )
    assert result.action == AssistantAction(content="")
    assert result.finish_reason == "length"
    assert result.response_id == request.request_id
    assert result.raw_text == raw
    assert result.exact_tokens.prompt_token_ids == (1,)
    assert result.exact_tokens.response_token_ids == response
    assert result.exact_tokens.response_logprobs == (-0.25,)
    assert result.exact_tokens.policy_revision == "policy-7"


@pytest.mark.parametrize("probabilities", [(-0.25, -0.5), (float("nan"),)])
def test_length_reasoning_does_not_bypass_original_token_validation(
    tmp_path: Path, probabilities: tuple[float, ...]
) -> None:
    """Recognized truncated reasoning still needs valid original probability evidence."""
    raw = "<think>private unfinished reasoning"
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
    )
    request = GenerationRequest(
        request_id="reasoning", model=spec().base_model, prompt="a", maximum_output_tokens=1
    )
    with pytest.raises(ValueError, match="response_logprobs"):
        generation_result(
            request,
            (1,),
            tuple(model_tokenizer.encode(raw)),
            probabilities,
            model_tokenizer,
            spec(),
            ResidentVerlSettings(checkpoint_root=tmp_path, decoder="hermes"),
            "policy-7",
            "length",
        )


@pytest.mark.parametrize("decoder", ["hermes", "qwen35"])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
@pytest.mark.parametrize("prefix", ["visible answer", "tool"])
def test_visible_prefix_before_unfinished_reasoning_remains_fatal(
    tmp_path: Path,
    decoder: Literal["hermes", "qwen35"],
    finish_reason: Literal["stop", "length"],
    prefix: str,
) -> None:
    """A sampled public prefix cannot be silently discarded by the empty-action fallback."""
    if prefix == "tool":
        prefix = (
            '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
            if decoder == "hermes"
            else "<tool_call><function=lookup></function></tool_call>"
        )
    raw = prefix + "<think>unfinished reasoning"
    model_tokenizer = tokenizer()
    model_tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken(raw, normalized=False)]}
    )
    response = tuple(model_tokenizer.encode(raw, add_special_tokens=False))
    request = GenerationRequest(
        request_id="mixed", model=spec().base_model, prompt="a", maximum_output_tokens=1
    )
    with pytest.raises(ValueError, match="visible output before unfinished reasoning"):
        generation_result(
            request,
            (1,),
            response,
            (-0.25,),
            model_tokenizer,
            spec(),
            ResidentVerlSettings(checkpoint_root=tmp_path, decoder=decoder),
            "policy-7",
            finish_reason,
        )

"""Tests for the real tau2-episode runner: split resolution, routing, metering, resume.

Everything here runs offline. The fake capture directory is synthesized from the REAL committed
`scenarios_eval.jsonl`, so the split-resolution tests exercise the actual pinned blobs without
needing the gitignored tau2 clone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from real_episodes import (
    CANONICAL_EPISODE_TIMEOUT_S,
    CANONICAL_MAX_TOKENS,
    CANONICAL_MAX_TURNS,
    CANONICAL_TAU2_MAX_RETRIES,
    EVAL_SCENARIOS,
    UNLABELED_COHORT,
    ProtocolPins,
    RealEpisodeRow,
    Tau2Results,
    agent_llm_args,
    append_rows,
    batch_command,
    batch_timeout_s,
    build_env,
    domains_dir,
    latest_per_cell,
    litellm_route,
    load_pinned_scenarios,
    load_rows,
    main,
    next_save_to,
    price_order,
    resume_keys,
    rows_from_results,
    save_to_name,
    spend_usd,
    to_matrix,
)

from wmo.providers.base import ProviderKind
from wmo.providers.pool import PoolEntry

_AZURE_OPENAI = PoolEntry(
    name="gpt-5.4-mini",
    kind=ProviderKind.AZURE_OPENAI,
    model="gpt-5.4-mini",
    deployment="gpt-5.4-mini",
    endpoint="https://google-sheets.openai.azure.com",
    api_key_env="AZURE_GOOGLE_SHEETS_API_KEY",
    input_per_mtok=0.75,
    output_per_mtok=3.0,
)
_AZURE_AI = PoolEntry(
    name="glm-5.2",
    kind=ProviderKind.AZURE_OPENAI,
    model="FW-GLM-5.2",
    deployment="FW-GLM-5.2",
    endpoint="https://silen-resource.services.ai.azure.com",
    api_key_env="AZURE_SILEN_RESOURCE_API_KEY",
    tier="open",
    input_per_mtok=1.54,
    output_per_mtok=4.84,
)
_ANTHROPIC = PoolEntry(
    name="fable-5",
    kind=ProviderKind.ANTHROPIC,
    model="claude-fable-5",
    input_per_mtok=15.0,
    output_per_mtok=75.0,
)

_ENVIRON = {
    "AZURE_GOOGLE_SHEETS_API_KEY": "sheets-key",
    "AZURE_SILEN_RESOURCE_API_KEY": "silen-key",
    "ANTHROPIC_API_KEY": "anthropic-key",
}


def _write_tasks(capture_dir: Path, domain: str, tasks: list[dict[str, object]]) -> None:
    path = domains_dir(capture_dir) / domain / "tasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def _task(task_id: str, instructions: dict[str, object], basis: list[str]) -> dict[str, object]:
    return {
        "id": task_id,
        "user_scenario": {"instructions": instructions},
        "evaluation_criteria": {"reward_basis": basis},
    }


def _fake_capture_from_pinned_split(capture_dir: Path) -> list[dict[str, object]]:
    """Build a tau2 task tree that contains exactly the committed pinned eval scenarios."""
    pinned = [
        json.loads(line)
        for line in EVAL_SCENARIOS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_domain: dict[str, list[dict[str, object]]] = {}
    for index, row in enumerate(pinned):
        by_domain.setdefault(str(row["domain"]), []).append(
            _task(f"t{index}", json.loads(str(row["task"])), ["DB"])
        )
    for domain, tasks in by_domain.items():
        _write_tasks(capture_dir, domain, tasks)
    return pinned


def test_every_committed_pinned_scenario_resolves(tmp_path: Path) -> None:
    pinned = _fake_capture_from_pinned_split(tmp_path)
    scenarios = load_pinned_scenarios(tmp_path)
    assert len(scenarios) == len(pinned)
    assert all(s.scenario_id == f"{s.domain}:{s.task_id}" for s in scenarios)
    assert {s.domain for s in scenarios} == {row["domain"] for row in pinned}


def test_scenario_ids_disambiguate_colliding_task_ids(tmp_path: Path) -> None:
    # Airline and retail both number tasks from "0"; bare ids would merge two different
    # tasks into one matrix cell.
    airline = {"domain": "airline", "reason_for_call": "cancel"}
    retail = {"domain": "retail", "reason_for_call": "return"}
    _write_tasks(tmp_path, "airline", [_task("0", airline, ["DB"])])
    _write_tasks(tmp_path, "retail", [_task("0", retail, ["DB"])])
    split = tmp_path / "split.jsonl"
    split.write_text(
        json.dumps({"domain": "airline", "provenance": ["a"], "task": json.dumps(airline)})
        + "\n"
        + json.dumps({"domain": "retail", "provenance": ["b"], "task": json.dumps(retail)})
        + "\n",
        encoding="utf-8",
    )
    ids = [s.scenario_id for s in load_pinned_scenarios(tmp_path, split)]
    assert ids == ["airline:0", "retail:0"]


def test_nl_assertion_scoring_is_recorded(tmp_path: Path) -> None:
    deterministic = {"domain": "airline", "reason_for_call": "a"}
    judged = {"domain": "airline", "reason_for_call": "b"}
    _write_tasks(
        tmp_path,
        "airline",
        [
            _task("0", deterministic, ["DB", "COMMUNICATE"]),
            _task("1", judged, ["DB", "NL_ASSERTION"]),
        ],
    )
    split = tmp_path / "split.jsonl"
    split.write_text(
        "\n".join(
            json.dumps({"domain": "airline", "provenance": [f"p{i}"], "task": json.dumps(task)})
            for i, task in enumerate((deterministic, judged))
        )
        + "\n",
        encoding="utf-8",
    )
    flags = {s.task_id: s.nl_assertion_reward for s in load_pinned_scenarios(tmp_path, split)}
    assert flags == {"0": False, "1": True}


def test_unmatched_pinned_scenario_aborts(tmp_path: Path) -> None:
    # Running a different task set than the world model was evaluated on is the one failure a
    # sim-to-real comparison cannot survive, so it must abort rather than silently shrink.
    _write_tasks(tmp_path, "airline", [_task("0", {"domain": "airline", "x": 1}, ["DB"])])
    split = tmp_path / "split.jsonl"
    split.write_text(
        json.dumps({"domain": "airline", "provenance": ["ghost"], "task": json.dumps({"x": 2})})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="do not match any tau2 task"):
        load_pinned_scenarios(tmp_path, split)


def test_missing_tau2_clone_says_what_to_do(tmp_path: Path) -> None:
    split = tmp_path / "split.jsonl"
    split.write_text(
        json.dumps({"domain": "airline", "provenance": ["a"], "task": "{}"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="--capture-dir"):
        load_pinned_scenarios(tmp_path, split)


def test_litellm_route_splits_the_two_azure_families() -> None:
    assert litellm_route(_AZURE_OPENAI) == "azure/gpt-5.4-mini"
    assert litellm_route(_AZURE_AI) == "azure_ai/FW-GLM-5.2"
    assert litellm_route(_ANTHROPIC) == "anthropic/claude-fable-5"


def test_unrecognized_azure_endpoint_is_refused() -> None:
    # Guessing the family would send the call to the wrong service with the wrong credential
    # variable and surface as an opaque 401 inside tau2.
    entry = _AZURE_AI.model_copy(update={"endpoint": "https://example.invalid"})
    with pytest.raises(SystemExit, match="neither"):
        litellm_route(entry)


def test_build_env_carries_both_azure_families(tmp_path: Path) -> None:
    env = build_env(tmp_path, _AZURE_AI, _AZURE_OPENAI, _ENVIRON)
    assert env["AZURE_AI_API_KEY"] == "silen-key"
    assert env["AZURE_AI_API_BASE"] == "https://silen-resource.services.ai.azure.com/models"
    assert env["AZURE_API_KEY"] == "sheets-key"
    assert env["TAU2_DATA_DIR"] == str(tmp_path / "tau2-bench" / "data")


def test_build_env_rejects_two_accounts_on_one_family(tmp_path: Path) -> None:
    other = _AZURE_OPENAI.model_copy(
        update={"name": "gpt-5.5", "api_key_env": "AZURE_SILEN_RESOURCE_API_KEY"}
    )
    with pytest.raises(SystemExit, match="AZURE_API_KEY"):
        build_env(tmp_path, other, _AZURE_OPENAI, _ENVIRON)


def test_build_env_names_the_missing_credential(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="AZURE_SILEN_RESOURCE_API_KEY"):
        build_env(tmp_path, _AZURE_AI, _AZURE_OPENAI, {"AZURE_GOOGLE_SHEETS_API_KEY": "k"})


def _command(tmp_path: Path, pins: ProtocolPins | None = None) -> list[str]:
    return batch_command(
        tmp_path,
        _AZURE_AI,
        _AZURE_OPENAI,
        "airline",
        ["0", "3"],
        "s",
        4,
        pins or ProtocolPins(),
    )


def test_batch_command_pins_the_environment(tmp_path: Path) -> None:
    command = _command(tmp_path)
    assert command[command.index("--user-llm") + 1] == "azure/gpt-5.4-mini"
    assert command[command.index("--agent-llm") + 1] == "azure_ai/FW-GLM-5.2"
    # No temperature on either stream; the agent stream carries only the token pin.
    assert command[command.index("--agent-llm-args") + 1] == '{"max_tokens": 8192}'
    assert command[command.index("--user-llm-args") + 1] == "{}"
    assert command[command.index("--num-trials") + 1] == "1"
    assert command[command.index("--task-ids") + 1 : command.index("--max-concurrency")] == [
        "0",
        "3",
    ]


def test_canonical_pins_are_the_decisions_values() -> None:
    # DECISIONS.md 2026-07-27 "cross-lane weave" ack 2: max_turns=100, episode_timeout_s=1800,
    # max_tokens=8192, user sim azure/gpt-5.4-mini, tau2 --max-retries 0. Every real-tau2 leg in
    # the project runs these exact values, so a drift here silently splits the cohort.
    assert (CANONICAL_MAX_TURNS, CANONICAL_MAX_TOKENS, CANONICAL_TAU2_MAX_RETRIES) == (100, 8192, 0)
    assert CANONICAL_EPISODE_TIMEOUT_S == 1800.0
    pins = ProtocolPins()
    assert pins.user_sim == "gpt-5.4-mini"
    assert pins.is_canonical
    assert pins.label == "turns100-t1800-tok8192-r0-sim-gpt-5.4-mini"


def test_canonical_pins_reach_the_tau2_argv(tmp_path: Path) -> None:
    # tau2's own defaults are 200 steps, no timeout, and 3 retries, so an absent flag is not a
    # neutral choice: it is a different capture cohort.
    command = _command(tmp_path)
    assert command[command.index("--max-steps") + 1] == "100"
    assert command[command.index("--timeout") + 1] == "1800"
    assert command[command.index("--max-retries") + 1] == "0"
    # A headless batch must not block on tau2's interactive resume prompt.
    assert "--auto-resume" in command


def test_moved_pins_reach_the_argv_and_rename_the_cohort(tmp_path: Path) -> None:
    pins = ProtocolPins(max_turns=200, episode_timeout_s=600, max_tokens=4096, tau2_max_retries=3)
    command = _command(tmp_path, pins)
    assert command[command.index("--max-steps") + 1] == "200"
    assert command[command.index("--timeout") + 1] == "600"
    assert command[command.index("--max-retries") + 1] == "3"
    assert command[command.index("--agent-llm-args") + 1] == '{"max_tokens": 4096}'
    assert not pins.is_canonical
    assert pins.label == "turns200-t600-tok4096-r3-sim-gpt-5.4-mini"


def test_a_different_user_simulator_is_a_different_cohort() -> None:
    # The user simulator IS the environment, so it belongs in the label even though it is also
    # recorded per row.
    assert not ProtocolPins(user_sim="gpt-5.5").is_canonical
    assert ProtocolPins(user_sim="gpt-5.5").label.endswith("-sim-gpt-5.5")


def test_max_tokens_zero_omits_the_key_for_a_strict_deployment() -> None:
    # litellm rewrites max_tokens to max_completion_tokens only for reasoning models its table
    # knows; an unrecognized deployment rejects the key outright, and a grid that cannot start is
    # worse than a labelled second cohort.
    assert agent_llm_args(ProtocolPins(max_tokens=0)) == "{}"
    assert ProtocolPins(max_tokens=0).label == "turns100-t1800-tok0-r0-sim-gpt-5.4-mini"


def test_an_impossible_pin_is_refused_not_run(tmp_path: Path) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    assert (
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--max-turns",
                "0",  # an episode with no turns is not a cohort, it is a typo
                "--dry-run",
            ]
        )
        == 2
    )


def test_batch_deadline_covers_every_wave_of_the_episode_pin() -> None:
    # A hard kill forfeits the episodes tau2 has not written yet, so the deadline has to cover the
    # whole batch, not one episode: 20 tasks at concurrency 4 is five 1800s waves.
    assert batch_timeout_s(20, 4, 1800.0) == 5 * 1800.0 + 300.0
    assert batch_timeout_s(3, 4, 1800.0) == 1800.0 + 300.0
    assert batch_timeout_s(0, 0, 1800.0) == 1800.0 + 300.0


def test_save_to_name_is_unique_per_cell_and_attempt() -> None:
    assert save_to_name(_AZURE_AI, "airline", 0) == "real_glm_5_2_airline_e0_a0"
    assert save_to_name(_AZURE_AI, "airline", 1) != save_to_name(_AZURE_AI, "airline", 0)
    assert save_to_name(_AZURE_AI, "airline", 0, 1) == "real_glm_5_2_airline_e0_a1"


def test_next_save_to_skips_a_cells_used_attempts(tmp_path: Path) -> None:
    # Sharing a save directory across attempts makes the retry impossible: tau2 reads the old one
    # as a checkpoint and refuses a task list that is a subset of it, which is exactly what a
    # resumed cell asks for.
    simulations = tmp_path / "tau2-bench" / "data" / "simulations"
    (simulations / save_to_name(_AZURE_AI, "airline", 0, 0)).mkdir(parents=True)
    assert next_save_to(tmp_path, _AZURE_AI, "airline", 0) == "real_glm_5_2_airline_e0_a1"
    # A different cell is untouched by another cell's attempts.
    assert next_save_to(tmp_path, _AZURE_AI, "airline", 1) == "real_glm_5_2_airline_e1_a0"


def _results_payload(reward: float | None) -> Tau2Results:
    reward_info = {"reward": reward} if reward is not None else None
    return Tau2Results.model_validate(
        {
            "simulations": [
                {
                    "task_id": "0",
                    "duration": 42.5,
                    "termination_reason": "user_stop",
                    "agent_cost": 0.019,
                    "user_cost": 0.002,
                    "reward_info": reward_info,
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "looking that up",
                            "generation_time_seconds": 1.5,
                            "tool_calls": [{"name": "get_user_details"}],
                            "usage": {"prompt_tokens": 30000, "completion_tokens": 1000},
                        },
                        {"role": "user", "usage": {"prompt_tokens": 800, "completion_tokens": 40}},
                    ],
                }
            ]
        }
    )


def _scenario_index(tmp_path: Path) -> dict[str, object]:
    task = {"domain": "airline", "reason_for_call": "cancel"}
    _write_tasks(tmp_path, "airline", [_task("0", task, ["DB", "NL_ASSERTION"])])
    split = tmp_path / "split.jsonl"
    split.write_text(
        json.dumps({"domain": "airline", "provenance": ["p"], "task": json.dumps(task)}) + "\n",
        encoding="utf-8",
    )
    return {s.task_id: s for s in load_pinned_scenarios(tmp_path, split)}


def test_rows_from_results_meters_the_episode(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    [row] = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    assert row.scenario_id == "airline:0"
    assert row.reward == 1.0
    assert row.nl_assertion_reward is True
    assert row.duration_s == 42.5
    assert row.call_seconds == [1.5]
    assert row.steps == 1
    assert (row.agent_input_tokens, row.agent_output_tokens) == (30000, 1000)
    assert (row.user_input_tokens, row.user_output_tokens) == (800, 40)
    # Our pool prices, not tau2's litellm guess, which is kept alongside for audit.
    assert row.cost_usd_pool == pytest.approx(30 * 1.54 / 1000 + 1 * 4.84 / 1000)
    assert row.cost_usd_tau2_agent == 0.019
    assert row.user_sim == "gpt-5.4-mini"


def test_unscored_episode_is_never_zeroed(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    [row] = rows_from_results(_results_payload(None), _AZURE_AI, 0, index, _AZURE_OPENAI)
    assert row.reward is None
    [outcome] = to_matrix([row], [_AZURE_AI, _AZURE_OPENAI]).outcomes
    assert outcome.reward is None
    assert outcome.scored is False
    assert outcome.success is False


def test_matrix_carries_cost_and_latency(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    [row] = rows_from_results(_results_payload(0.5), _AZURE_AI, 0, index, _AZURE_OPENAI)
    [outcome] = to_matrix([row], [_AZURE_AI]).outcomes
    assert outcome.cost_usd == row.cost_usd_pool
    assert outcome.call_seconds == [1.5]
    assert outcome.usage.input_tokens == 30000


def test_off_split_simulations_are_dropped(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    payload = _results_payload(1.0)
    payload.simulations[0].task_id = "99"
    assert rows_from_results(payload, _AZURE_AI, 0, index, _AZURE_OPENAI) == []


def test_rows_round_trip_and_resume_keys(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    rows = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    path = tmp_path / "rows.jsonl"
    append_rows(path, rows)
    append_rows(path, rows_from_results(_results_payload(0.0), _ANTHROPIC, 1, index, _AZURE_OPENAI))
    reloaded = load_rows(path)
    assert {row.key for row in reloaded} == {
        ("airline:0", "glm-5.2", 0),
        ("airline:0", "fable-5", 1),
    }
    assert spend_usd(reloaded) == pytest.approx(
        sum(row.cost_usd_pool for row in reloaded) + 2 * 0.002
    )


def test_load_rows_on_a_fresh_run() -> None:
    assert load_rows(Path("/nonexistent/rows.jsonl")) == []


def test_cohort_label_round_trips_through_rows_jsonl(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    label = ProtocolPins().label
    rows = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI, label)
    assert [row.cohort for row in rows] == [label]
    path = tmp_path / "rows.jsonl"
    append_rows(path, rows)
    assert [row.cohort for row in load_rows(path)] == [label]


def test_rows_written_before_the_pins_still_load(tmp_path: Path) -> None:
    # A row with no cohort field is readable evidence of a DIFFERENT cohort (it ran on tau2's
    # defaults); dropping it as unreadable would lose episodes that were paid for.
    index = _scenario_index(tmp_path)
    [row] = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    payload = json.loads(row.model_dump_json())
    del payload["cohort"]
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert [r.cohort for r in load_rows(path)] == [UNLABELED_COHORT]


def test_a_run_refuses_to_append_to_another_cohorts_rows(tmp_path: Path) -> None:
    out = tmp_path / "out"
    index = _scenario_index(tmp_path)
    _fake_capture_from_pinned_split(tmp_path)
    append_rows(
        out / "rows.jsonl",
        rows_from_results(
            _results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI, "turns200-t0-tok0-r3-sim-x"
        ),
    )
    assert (
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(out),
                "--only",
                "glm-5.2",
                "--dry-run",
            ]
        )
        == 2
    )


def test_retry_failed_re_runs_only_the_unscored_cells(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    scored = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    unscored = rows_from_results(_results_payload(None), _ANTHROPIC, 0, index, _AZURE_OPENAI)
    rows = scored + unscored
    assert resume_keys(rows, retry_failed=False) == {row.key for row in rows}
    # Without tau2's own retries, this flag is the only second chance a dead episode gets.
    assert resume_keys(rows, retry_failed=True) == {row.key for row in scored}


def test_a_retried_cell_counts_as_one_episode_but_two_purchases(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    failed = rows_from_results(_results_payload(None), _AZURE_AI, 0, index, _AZURE_OPENAI)
    retried = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    rows = failed + retried
    assert [row.reward for row in latest_per_cell(rows)] == [1.0]  # the retry supersedes
    [outcome] = to_matrix(rows, [_AZURE_AI]).outcomes
    assert outcome.reward == 1.0
    # Both attempts were billed, so the spend meter counts both.
    assert spend_usd(rows) == pytest.approx(2 * spend_usd(retried))


def test_price_order_is_cheapest_first() -> None:
    ordered = price_order([_ANTHROPIC, _AZURE_AI, _AZURE_OPENAI])
    assert [entry.name for entry in ordered] == ["gpt-5.4-mini", "glm-5.2", "fable-5"]


def _pool_file(path: Path) -> Path:
    path.write_text(
        "\n".join(
            f"[[model]]\nname = '{e.name}'\nkind = '{e.kind.value}'\nmodel = '{e.model}'\n"
            + (f"deployment = '{e.deployment}'\n" if e.deployment else "")
            + (f"endpoint = '{e.endpoint}'\n" if e.endpoint else "")
            + (f"api_key_env = '{e.api_key_env}'\n" if e.api_key_env else "")
            + f"input_per_mtok = {e.input_per_mtok}\noutput_per_mtok = {e.output_per_mtok}\n"
            for e in (_AZURE_OPENAI, _AZURE_AI, _ANTHROPIC)
        ),
        encoding="utf-8",
    )
    return path


def test_dry_run_resolves_the_split_and_spends_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    with caplog.at_level("INFO", logger="tau-real"):
        exit_code = main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--only",
                "glm-5.2",
                "--dry-run",
            ]
        )
    assert exit_code == 0
    assert not (tmp_path / "out").exists()  # nothing executed, nothing written
    printed = caplog.text
    assert "pinned eval split: 20 scenarios" in printed
    assert "azure_ai/FW-GLM-5.2" in printed
    assert "--user-llm azure/gpt-5.4-mini" in printed
    # The cohort the operator is about to buy is stated before anything is bought.
    assert f"protocol pins: cohort '{ProtocolPins().label}' (canonical)" in printed
    assert "--max-steps 100 --timeout 1800 --max-retries 0" in printed
    assert "--agent-llm azure/gpt-5.4-mini" not in printed


def test_scenario_flag_restricts_the_run(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    scenarios = load_pinned_scenarios(tmp_path)
    wanted = scenarios[0].scenario_id
    with caplog.at_level("INFO", logger="tau-real"):
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--only",
                "glm-5.2",
                "--scenario",
                wanted,
                "--dry-run",
            ]
        )
    assert "pinned eval split: 1 scenarios" in caplog.text
    assert caplog.text.count("would run:") == 1


def test_scenario_flag_rejects_ids_outside_the_split(tmp_path: Path) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    assert (
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--scenario",
                "airline:999",
                "--dry-run",
            ]
        )
        == 2
    )


def test_user_simulator_model_can_also_be_scored_as_candidate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    with caplog.at_level("INFO", logger="tau-real"):
        assert (
            main(
                [
                    "--capture-dir",
                    str(tmp_path),
                    "--pool",
                    str(_pool_file(tmp_path / "pool.toml")),
                    "--out-dir",
                    str(tmp_path / "out"),
                    "--only",
                    "gpt-5.4-mini",
                    "--dry-run",
                ]
            )
            == 0
        )
    assert "--agent-llm azure/gpt-5.4-mini" in caplog.text
    assert "--user-llm azure/gpt-5.4-mini" in caplog.text


def test_write_matrix_only_rebuilds_from_rows(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    out = tmp_path / "out"
    append_rows(
        out / "rows.jsonl",
        rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI),
    )
    exit_code = main(
        [
            "--capture-dir",
            str(tmp_path),
            "--pool",
            str(_pool_file(tmp_path / "pool.toml")),
            "--out-dir",
            str(out),
            "--write-matrix-only",
        ]
    )
    assert exit_code == 0
    matrix = json.loads((out / "matrix.json").read_text(encoding="utf-8"))
    assert [o["scenario_id"] for o in matrix["outcomes"]] == ["airline:0"]


def test_unknown_user_simulator_is_reported(tmp_path: Path) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    assert (
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--user-sim",
                "not-in-the-pool",
                "--dry-run",
            ]
        )
        == 2
    )


def test_resolves_against_the_real_tau2_clone_when_present() -> None:
    """End-to-end split resolution against a real clone, when one is available locally."""
    capture_dir = Path(__file__).resolve().parent.parent
    if not (domains_dir(capture_dir) / "airline" / "tasks.json").is_file():
        pytest.skip("no local tau2-bench clone (see ../README.md § Setup)")
    scenarios = load_pinned_scenarios(capture_dir)
    assert len(scenarios) == 20
    assert all(s.task_id for s in scenarios)


def test_row_model_rejects_a_missing_meter() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        RealEpisodeRow.model_validate({"scenario_id": "airline:0"})


def test_torn_final_line_does_not_brick_resume(tmp_path: Path) -> None:
    # A kill during the append leaves a truncated record. Refusing to load it would lose every
    # episode already bought, including for --write-matrix-only.
    index = _scenario_index(tmp_path)
    path = tmp_path / "rows.jsonl"
    append_rows(path, rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"scenario_id": "airline:0", "domain": "air')  # torn write, no newline
    rows = load_rows(path)
    assert len(rows) == 1
    assert rows[0].reward == 1.0


def test_append_rows_writes_a_batch_atomically(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    path = tmp_path / "rows.jsonl"
    batch = rows_from_results(_results_payload(1.0), _AZURE_AI, 0, index, _AZURE_OPENAI)
    append_rows(path, batch * 3)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    append_rows(path, [])  # an empty batch must not leave a stray newline
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_null_token_meters_do_not_lose_the_episode() -> None:
    # A provider that omits usage on an errored turn serializes nulls; rejecting them would
    # discard a whole paid batch.
    results = Tau2Results.model_validate(
        {
            "simulations": [
                {
                    "task_id": "0",
                    "duration": None,
                    "reward_info": {"reward": 1.0},
                    "messages": [
                        {
                            "role": "assistant",
                            "usage": {"prompt_tokens": None, "completion_tokens": 5},
                        }
                    ],
                }
            ]
        }
    )
    [sim] = results.simulations
    assert sim.duration == 0.0
    assert sim.messages[0].usage is not None
    assert sim.messages[0].usage.prompt_tokens == 0


def test_structured_content_blocks_do_not_lose_the_episode(tmp_path: Path) -> None:
    index = _scenario_index(tmp_path)
    payload = _results_payload(1.0)
    payload.simulations[0].messages[0].content = [{"type": "text", "text": "hi"}]
    [row] = rows_from_results(payload, _AZURE_AI, 0, index, _AZURE_OPENAI)
    assert row.reward == 1.0
    assert row.replies == []  # non-string content is not a text reply


def test_telecom_task_ids_survive_argv_construction(tmp_path: Path) -> None:
    # Telecom is 5 of the 20 pinned scenarios and its ids carry | [ ] : characters.
    telecom_id = (
        "[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off[PERSONA:Hard]"
    )
    command = batch_command(
        tmp_path, _AZURE_AI, _AZURE_OPENAI, "telecom", [telecom_id], "s", 4, ProtocolPins()
    )
    assert command[command.index("--task-ids") + 1] == telecom_id


def test_only_rejects_a_model_that_is_not_in_the_pool(tmp_path: Path) -> None:
    _fake_capture_from_pinned_split(tmp_path)
    assert (
        main(
            [
                "--capture-dir",
                str(tmp_path),
                "--pool",
                str(_pool_file(tmp_path / "pool.toml")),
                "--out-dir",
                str(tmp_path / "out"),
                "--only",
                "glm-5.2",
                "gml-5.2",  # typo
                "--dry-run",
            ]
        )
        == 2
    )


def test_self_hosted_openai_endpoint_is_not_dropped(tmp_path: Path) -> None:
    student = PoolEntry(
        name="student",
        kind=ProviderKind.OPENAI,
        model="qwen3.5-9b",
        endpoint="https://tinker.example/v1",
        api_key_env="TINKER_API_KEY",
        input_per_mtok=0.1,
        output_per_mtok=0.2,
    )
    env = build_env(tmp_path, student, _AZURE_OPENAI, {**_ENVIRON, "TINKER_API_KEY": "tk"})
    assert env["OPENAI_API_KEY"] == "tk"
    assert env["OPENAI_BASE_URL"] == "https://tinker.example/v1"


def test_missing_pinned_split_says_how_to_regenerate(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="pin_scenarios.py"):
        load_pinned_scenarios(tmp_path, tmp_path / "absent.jsonl")

"""Monotonic public-docstring guardrails for the W1 migration."""

from __future__ import annotations

import ast
import functools
import subprocess
import tomllib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCSTRING_BASELINE_REVISION = "e7aad17b2f5041769ad8107ab25e77d4e88729ca"


@dataclass(frozen=True, order=True)
class DocstringViolation:
    """One public API docstring requirement that the current source does not satisfy."""

    path: str
    symbol: str
    kind: str
    reason: str


# A migration owner appends every fixed baseline violation here. The compact rows preserve the
# exact path, symbol, kind, and reason while keeping this executable inventory below the W1 limit.
# fmt: off
_DOCSTRING_TOMBSTONE_ROWS: Final[tuple[str, ...]] = (
    'wmo/cli/app.py|build|function|missing-args-section',
    'wmo/cli/app.py|build|function|missing-raises-section',
    'wmo/cli/app.py|config_telemetry|function|missing-args-section',
    'wmo/cli/app.py|config_telemetry|function|missing-raises-section',
    'wmo/cli/app.py|config_telemetry|function|nontrivial-one-line-docstring',
    'wmo/cli/app.py|demo|function|missing-args-section',
    'wmo/cli/app.py|demo|function|missing-raises-section',
    'wmo/cli/app.py|demo|function|nontrivial-one-line-docstring',
    'wmo/cli/app.py|download|function|missing-args-section',
    'wmo/cli/app.py|download|function|missing-raises-section',
    'wmo/cli/app.py|eval_|function|missing-args-section',
    'wmo/cli/app.py|eval_|function|missing-raises-section',
    'wmo/cli/app.py|knowledge_|function|missing-args-section',
    'wmo/cli/app.py|list_models|function|missing-args-section',
    'wmo/cli/app.py|list_models|function|missing-raises-section',
    'wmo/cli/app.py|play|function|missing-args-section',
    'wmo/cli/app.py|play|function|nontrivial-one-line-docstring',
    'wmo/cli/app.py|providers_set|function|missing-args-section',
    'wmo/cli/app.py|providers_set|function|missing-raises-section',
    'wmo/cli/app.py|providers_verify|function|missing-args-section',
    'wmo/cli/app.py|providers_verify|function|missing-raises-section',
    'wmo/cli/app.py|research_concurrency|function|missing-args-section',
    'wmo/cli/app.py|research_concurrency|function|missing-raises-section',
    'wmo/cli/app.py|research_deepswe_holdout|function|missing-args-section',
    'wmo/cli/app.py|research_deepswe_holdout|function|missing-raises-section',
    'wmo/cli/app.py|research_plot_concurrency|function|missing-args-section',
    'wmo/cli/app.py|research_plot_concurrency|function|missing-raises-section',
    'wmo/cli/app.py|research_plot_concurrency_combined|function|missing-args-section',
    'wmo/cli/app.py|research_plot_concurrency_combined|function|missing-raises-section',
    'wmo/cli/app.py|scenarios_build|function|missing-args-section',
    'wmo/cli/app.py|scenarios_build|function|missing-raises-section',
    'wmo/cli/app.py|scenarios_verify|function|missing-args-section',
    'wmo/cli/app.py|serve|function|missing-args-section',
    'wmo/cli/app.py|serve|function|missing-raises-section',
    'wmo/cli/optimize_model_app.py|build_endpoint_scorecard|function|missing-args-section',
    'wmo/cli/optimize_model_app.py|build_endpoint_scorecard|function|missing-returns-section',
    'wmo/cli/optimize_model_app.py|optimize_model|function|missing-args-section',
    'wmo/cli/optimize_model_app.py|optimize_model|function|missing-raises-section',
    'wmo/cli/platform_cmds.py|login|function|missing-args-section',
    'wmo/cli/platform_cmds.py|login|function|missing-raises-section',
    'wmo/cli/platform_cmds.py|login|function|nontrivial-one-line-docstring',
    'wmo/cli/platform_cmds.py|logout|function|nontrivial-one-line-docstring',
    'wmo/cli/platform_cmds.py|pull|function|missing-args-section',
    'wmo/cli/platform_cmds.py|pull|function|missing-raises-section',
    'wmo/cli/platform_cmds.py|push|function|missing-args-section',
    'wmo/cli/platform_cmds.py|push|function|missing-raises-section',
    'wmo/cli/platform_cmds.py|register|function|missing-args-section',
    'wmo/cli/platform_cmds.py|register|function|nontrivial-one-line-docstring',
    'wmo/cli/platform_cmds.py|status|function|missing-raises-section',
    'wmo/cli/platform_cmds.py|status|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|cell_progress|function|missing-args-section',
    'wmo/cli/route_app.py|cell_progress|function|missing-returns-section',
    'wmo/cli/route_app.py|cell_progress|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|convert_deepswe_cmd|function|missing-args-section',
    'wmo/cli/route_app.py|convert_deepswe_cmd|function|missing-raises-section',
    'wmo/cli/route_app.py|fit|function|missing-args-section',
    'wmo/cli/route_app.py|fit|function|missing-raises-section',
    'wmo/cli/route_app.py|pin|function|missing-args-section',
    'wmo/cli/route_app.py|pin|function|missing-raises-section',
    'wmo/cli/route_app.py|print_cost_estimate|function|missing-args-section',
    'wmo/cli/route_app.py|print_coverage|function|missing-args-section',
    'wmo/cli/route_app.py|print_deferred_risks|function|missing-args-section',
    'wmo/cli/route_app.py|print_deferred_risks|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|print_dial|function|missing-args-section',
    'wmo/cli/route_app.py|print_dial|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|print_knn_fit|function|missing-args-section',
    'wmo/cli/route_app.py|print_knn_fit|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|print_tiny_corpus_note|function|missing-args-section',
    'wmo/cli/route_app.py|print_tiny_corpus_note|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|print_world_model_spend|function|missing-args-section',
    'wmo/cli/route_app.py|push|function|missing-args-section',
    'wmo/cli/route_app.py|push|function|missing-raises-section',
    'wmo/cli/route_app.py|report|function|missing-args-section',
    'wmo/cli/route_app.py|report|function|missing-raises-section',
    'wmo/cli/route_app.py|report|function|nontrivial-one-line-docstring',
    'wmo/cli/route_app.py|student|function|missing-args-section',
    'wmo/cli/route_app.py|student|function|missing-raises-section',
    'wmo/cli/route_app.py|sweep|function|missing-args-section',
    'wmo/cli/route_app.py|sweep|function|missing-raises-section',
    'wmo/cli/route_app.py|tune|function|missing-args-section',
    'wmo/cli/route_app.py|tune|function|missing-raises-section',
    'wmo/cli/route_app.py|uneven_warning|function|missing-args-section',
    'wmo/cli/route_app.py|uneven_warning|function|missing-returns-section',
    'wmo/cli/run_cmd.py|LocalPiRunRecorder.finish|method|missing-args-section',
    'wmo/cli/run_cmd.py|LocalPiRunRecorder.finish|method|nontrivial-one-line-docstring',
    'wmo/cli/run_cmd.py|LocalPiRunRecorder.flush|method|nontrivial-one-line-docstring',
    'wmo/cli/run_cmd.py|LocalPiRunRecorder.record|method|missing-args-section',
    'wmo/cli/run_cmd.py|LocalPiRunRecorder.record|method|nontrivial-one-line-docstring',
    'wmo/cli/run_cmd.py|RemoteWorldModelDriver.run|method|missing-raises-section',
    'wmo/cli/run_cmd.py|RemoteWorldModelDriver.run|method|nontrivial-one-line-docstring',
    'wmo/cli/run_cmd.py|RunRecorder.finish|method|missing-docstring',
    'wmo/cli/run_cmd.py|RunRecorder.flush|method|missing-docstring',
    'wmo/cli/run_cmd.py|RunRecorder.record|method|missing-docstring',
    'wmo/cli/runs_app.py|backfill|function|missing-args-section',
    'wmo/cli/runs_app.py|list_runs|function|missing-args-section',
    'wmo/cli/runs_app.py|list_runs|function|missing-raises-section',
    'wmo/cli/runs_app.py|list_runs|function|nontrivial-one-line-docstring',
    'wmo/cli/runs_app.py|register|function|missing-args-section',
    'wmo/cli/runs_app.py|register|function|nontrivial-one-line-docstring',
    'wmo/cli/runs_app.py|retry_run|function|missing-args-section',
    'wmo/cli/runs_app.py|show_run|function|missing-args-section',
    'wmo/cli/runs_app.py|show_run|function|missing-raises-section',
    'wmo/cli/runs_app.py|show_run|function|nontrivial-one-line-docstring',
    'wmo/cli/runs_app.py|stop_run|function|missing-args-section',
    'wmo/cli/runs_app.py|tail_run|function|missing-args-section',
    'wmo/cli/runs_app.py|tail_run|function|missing-raises-section',
    'wmo/cli/session_state.py|SessionStateStore.current_session_id|method|missing-returns-section',
    ('wmo/cli/session_state.py|SessionStateStore.current_session_id|method|'
     'nontrivial-one-line-docstring'),
    'wmo/cli/session_state.py|SessionStateStore.delete|method|missing-args-section',
    'wmo/cli/session_state.py|SessionStateStore.delete|method|nontrivial-one-line-docstring',
    'wmo/cli/session_state.py|SessionStateStore.load|method|missing-args-section',
    'wmo/cli/session_state.py|SessionStateStore.load|method|missing-raises-section',
    'wmo/cli/session_state.py|SessionStateStore.load|method|missing-returns-section',
    'wmo/cli/session_state.py|SessionStateStore.load|method|nontrivial-one-line-docstring',
    'wmo/cli/session_state.py|SessionStateStore.load_base_archive|method|missing-args-section',
    'wmo/cli/session_state.py|SessionStateStore.load_base_archive|method|missing-raises-section',
    'wmo/cli/session_state.py|SessionStateStore.load_base_archive|method|missing-returns-section',
    ('wmo/cli/session_state.py|SessionStateStore.load_base_archive|method|'
     'nontrivial-one-line-docstring'),
    'wmo/cli/session_state.py|SessionStateStore.save|method|missing-args-section',
    'wmo/cli/session_state.py|SessionStateStore.set_current|method|missing-args-section',
    'wmo/cli/session_state.py|SessionStateStore.set_current|method|nontrivial-one-line-docstring',
    ('wmo/cli/session_state.py|SessionStateStore.write_recovery_archive|method|'
     'missing-args-section'),
    ('wmo/cli/session_state.py|SessionStateStore.write_recovery_archive|method|'
     'missing-returns-section'),
    'wmo/cli/ui.py|models_table|function|missing-args-section',
    'wmo/cli/ui.py|models_table|function|missing-returns-section',
    'wmo/cli/ui.py|run_play_repl|function|missing-args-section',
    'wmo/optimize/research/ablation.py|Ablation.conditions|method|missing-docstring',
    'wmo/optimize/research/ablation.py|Ablation.name|method|missing-docstring',
    'wmo/optimize/research/ablation.py|ConditionReport.summary|method|missing-docstring',
    'wmo/optimize/research/ablation.py|aggregate|function|missing-args-section',
    'wmo/optimize/research/ablation.py|aggregate|function|missing-returns-section',
    'wmo/optimize/research/ablation.py|aggregate|function|nontrivial-one-line-docstring',
    'wmo/optimize/research/ablation.py|as_int|function|missing-args-section',
    'wmo/optimize/research/ablation.py|as_int|function|missing-returns-section',
    'wmo/optimize/research/ablation.py|run_ablation|function|missing-args-section',
    'wmo/optimize/research/ablation.py|run_ablation|function|missing-returns-section',
    'wmo/optimize/research/concurrency_plot.py|render_cost|function|missing-args-section',
    'wmo/optimize/research/concurrency_plot.py|render_cost|function|missing-raises-section',
    'wmo/optimize/research/concurrency_plot.py|render_cost|function|missing-returns-section',
    'wmo/optimize/research/concurrency_plot.py|render_report|function|missing-args-section',
    'wmo/optimize/research/concurrency_plot.py|render_report|function|missing-returns-section',
    'wmo/optimize/research/concurrency_plot.py|render_speedup|function|missing-args-section',
    'wmo/optimize/research/concurrency_plot.py|render_speedup|function|missing-returns-section',
    'wmo/optimize/research/concurrency_run.py|build_real_runner|function|missing-args-section',
    'wmo/optimize/research/concurrency_run.py|build_real_runner|function|missing-raises-section',
    'wmo/optimize/research/concurrency_run.py|build_real_runner|function|missing-returns-section',
    'wmo/optimize/research/concurrency_run.py|build_world_runner|function|missing-args-section',
    'wmo/optimize/research/concurrency_run.py|build_world_runner|function|missing-returns-section',
    ('wmo/optimize/research/concurrency_scaling.py|ConcurrencyPoint.summary|method|'
     'missing-docstring'),
    ('wmo/optimize/research/concurrency_scaling.py|run_concurrency_scaling|function|'
     'missing-args-section'),
    ('wmo/optimize/research/concurrency_scaling.py|run_concurrency_scaling|function|'
     'missing-raises-section'),
    ('wmo/optimize/research/concurrency_scaling.py|run_concurrency_scaling|function|'
     'missing-returns-section'),
    ('wmo/optimize/research/gepa_scaling.py|GepaScalingAblation.conditions|method|'
     'missing-docstring'),
    'wmo/optimize/research/gepa_scaling.py|GepaScalingAblation.run|method|missing-args-section',
    'wmo/optimize/research/gepa_scaling.py|GepaScalingAblation.run|method|missing-returns-section',
    ('wmo/optimize/research/gepa_scaling.py|GepaScalingAblation.run|method|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/research/pipeline.py|optimize_prompt|function|missing-args-section',
    'wmo/optimize/research/pipeline.py|optimize_prompt|function|missing-returns-section',
    'wmo/optimize/research/pipeline.py|score_prompt|function|missing-args-section',
    'wmo/optimize/research/pipeline.py|score_prompt|function|missing-returns-section',
    'wmo/optimize/research/scaling_split.py|partition_corpus|function|missing-args-section',
    'wmo/optimize/research/scaling_split.py|partition_corpus|function|missing-raises-section',
    'wmo/optimize/research/scaling_split.py|partition_corpus|function|missing-returns-section',
    'wmo/optimize/research/scaling_split.py|subsample_train|function|missing-args-section',
    'wmo/optimize/research/scaling_split.py|subsample_train|function|missing-returns-section',
    'wmo/optimize/research/scenario_fidelity.py|fidelity_report|function|missing-args-section',
    'wmo/optimize/research/scenario_fidelity.py|fidelity_report|function|missing-raises-section',
    'wmo/optimize/research/scenario_fidelity.py|fidelity_report|function|missing-returns-section',
    'wmo/optimize/research/scenario_fidelity.py|kendall|function|missing-args-section',
    'wmo/optimize/research/scenario_fidelity.py|kendall|function|missing-returns-section',
    'wmo/optimize/research/scenario_fidelity.py|kendall|function|nontrivial-one-line-docstring',
    'wmo/optimize/research/scenario_fidelity.py|random_subsets|function|missing-args-section',
    'wmo/optimize/research/scenario_fidelity.py|random_subsets|function|missing-returns-section',
    ('wmo/optimize/research/scenario_fidelity.py|random_subsets|function|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/research/scenario_fidelity.py|score_matrix|function|missing-args-section',
    'wmo/optimize/research/scenario_fidelity.py|score_matrix|function|missing-returns-section',
    'wmo/optimize/research/scenario_fidelity.py|spearman|function|missing-args-section',
    'wmo/optimize/research/scenario_fidelity.py|spearman|function|missing-returns-section',
    'wmo/optimize/research/scenario_fidelity.py|spearman|function|nontrivial-one-line-docstring',
    'wmo/optimize/research/scenario_recovery.py|ground_truth_labels|function|missing-args-section',
    ('wmo/optimize/research/scenario_recovery.py|ground_truth_labels|function|'
     'missing-returns-section'),
    'wmo/optimize/research/scenario_recovery.py|recovery_report|function|missing-args-section',
    'wmo/optimize/research/scenario_recovery.py|recovery_report|function|missing-raises-section',
    'wmo/optimize/research/scenario_recovery.py|recovery_report|function|missing-returns-section',
    ('wmo/optimize/research/scenario_recovery.py|recovery_report|function|'
     'nontrivial-one-line-docstring'),
    ('wmo/optimize/research/seed_stability.py|SeedStabilityAblation.conditions|method|'
     'missing-docstring'),
    ('wmo/optimize/research/seed_stability.py|SeedStabilityAblation.run|method|'
     'missing-args-section'),
    ('wmo/optimize/research/seed_stability.py|SeedStabilityAblation.run|method|'
     'missing-returns-section'),
    ('wmo/optimize/research/seed_stability.py|SeedStabilityAblation.run|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/optimize/research/trace_scaling.py|TraceScalingAblation.conditions|method|'
     'missing-docstring'),
    'wmo/optimize/research/trace_scaling.py|TraceScalingAblation.run|method|missing-args-section',
    ('wmo/optimize/research/trace_scaling.py|TraceScalingAblation.run|method|'
     'missing-returns-section'),
    ('wmo/optimize/research/trace_scaling.py|TraceScalingAblation.run|method|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/research/trace_scaling.py|split_mode|function|missing-args-section',
    'wmo/optimize/research/trace_scaling.py|split_mode|function|missing-raises-section',
    'wmo/optimize/research/trace_scaling.py|split_mode|function|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|build_ladder|function|missing-args-section',
    'wmo/optimize/routing/scorecard.py|build_ladder|function|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|build_scorecard|function|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|CompletionRule.completed|method|missing-args-section',
    'wmo/optimize/routing/scorecard.py|CompletionRule.completed|method|missing-returns-section',
    ('wmo/optimize/routing/scorecard.py|CompletionRule.completed|method|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/routing/scorecard.py|ConditionLabel.replace|method|missing-args-section',
    'wmo/optimize/routing/scorecard.py|ConditionLabel.replace|method|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|Ladder.operating_points|method|missing-args-section',
    'wmo/optimize/routing/scorecard.py|Ladder.operating_points|method|missing-returns-section',
    ('wmo/optimize/routing/scorecard.py|Ladder.operating_points|method|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/routing/scorecard.py|Ladder.pareto|method|missing-args-section',
    'wmo/optimize/routing/scorecard.py|Ladder.pareto|method|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|OperatingPoint.as_cost_quality_anchor|method|missing-returns-section',
    'wmo/optimize/routing/scorecard.py|rows_for_model|function|missing-args-section',
    'wmo/optimize/routing/scorecard.py|rows_for_model|function|missing-raises-section',
    'wmo/optimize/routing/scorecard.py|rows_for_model|function|missing-returns-section',
    ('wmo/optimize/routing/scorecard.py|rows_for_model|function|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/routing/scorecard.py|rows_for_policy|function|missing-returns-section',
    'wmo/optimize/telemetry/hooks.py|GridEmitter.create|method|missing-returns-section',
    'wmo/optimize/telemetry/hooks.py|GridEmitter.on_arm_start|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|GridEmitter.on_outcome|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|GridEmitter.on_status|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|GridEmitter.send_cells|method|nontrivial-one-line-docstring',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.create|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.create|method|missing-returns-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.create|method|nontrivial-one-line-docstring',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.finished|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.heartbeat|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.stage_completed|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.stage_running|method|missing-args-section',
    ('wmo/optimize/telemetry/hooks.py|PipelineEmitter.stage_running|method|'
     'nontrivial-one-line-docstring'),
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.stage_skipped|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|PipelineEmitter.start|method|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|frontier_from_reader|function|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|frontier_from_reader|function|missing-returns-section',
    'wmo/optimize/telemetry/hooks.py|platform_frontier|function|missing-args-section',
    'wmo/optimize/telemetry/hooks.py|platform_frontier|function|missing-returns-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.authorize_url|method|missing-args-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.authorize_url|method|missing-returns-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.authorize_url|method|nontrivial-one-line-docstring',
    'wmo/runtime/platform/auth.py|BrowserLogin.close|method|nontrivial-one-line-docstring',
    'wmo/runtime/platform/auth.py|BrowserLogin.port|method|missing-docstring',
    'wmo/runtime/platform/auth.py|BrowserLogin.start|method|missing-returns-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.start|method|nontrivial-one-line-docstring',
    'wmo/runtime/platform/auth.py|BrowserLogin.wait|method|missing-args-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.wait|method|missing-returns-section',
    'wmo/runtime/platform/auth.py|BrowserLogin.wait|method|nontrivial-one-line-docstring',
    'wmo/runtime/platform/client.py|PlatformClient.ack_run_control|method|missing-args-section',
    'wmo/runtime/platform/client.py|PlatformClient.ack_run_control|method|missing-returns-section',
    ('wmo/runtime/platform/client.py|PlatformClient.ack_run_control|method|'
     'nontrivial-one-line-docstring'),
    'wmo/runtime/platform/client.py|PlatformClient.close|method|missing-docstring',
    ('wmo/runtime/platform/client.py|PlatformClient.complete_local_pi_worker|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.complete_local_pi_worker|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.complete_local_pi_worker|method|'
     'nontrivial-one-line-docstring'),
    'wmo/runtime/platform/client.py|PlatformClient.create_endpoint|method|missing-raises-section',
    ('wmo/runtime/platform/client.py|PlatformClient.create_local_pi_run|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.create_local_pi_run|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.create_local_pi_run|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/runtime/platform/client.py|PlatformClient.create_world_model_session|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.create_world_model_session|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.create_world_model_session|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/runtime/platform/client.py|PlatformClient.download_endpoint_policy|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.download_endpoint_policy|method|'
     'missing-raises-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.download_model_bundle|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.download_model_bundle|method|'
     'missing-raises-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.finish_local_pi_run|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.finish_local_pi_run|method|'
     'nontrivial-one-line-docstring'),
    'wmo/runtime/platform/client.py|PlatformClient.get_endpoint|method|missing-args-section',
    'wmo/runtime/platform/client.py|PlatformClient.get_endpoint|method|missing-raises-section',
    'wmo/runtime/platform/client.py|PlatformClient.get_endpoint|method|missing-returns-section',
    ('wmo/runtime/platform/client.py|PlatformClient.install_endpoint_artifacts|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.install_endpoint_artifacts|method|'
     'missing-raises-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.install_endpoint_policy|method|'
     'missing-raises-section'),
    'wmo/runtime/platform/client.py|PlatformClient.list_org_run_cells|method|missing-args-section',
    ('wmo/runtime/platform/client.py|PlatformClient.list_org_run_cells|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.list_org_run_cells|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/runtime/platform/client.py|PlatformClient.list_org_run_events|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.list_org_run_events|method|'
     'missing-returns-section'),
    'wmo/runtime/platform/client.py|PlatformClient.list_org_runs|method|missing-args-section',
    'wmo/runtime/platform/client.py|PlatformClient.list_org_runs|method|missing-returns-section',
    'wmo/runtime/platform/client.py|PlatformClient.list_world_models|method|missing-docstring',
    'wmo/runtime/platform/client.py|PlatformClient.push_model_bundle|method|missing-args-section',
    ('wmo/runtime/platform/client.py|PlatformClient.push_model_bundle|method|'
     'missing-raises-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.push_model_bundle|method|'
     'missing-returns-section'),
    'wmo/runtime/platform/client.py|PlatformClient.push_run_events|method|missing-args-section',
    'wmo/runtime/platform/client.py|PlatformClient.push_run_events|method|missing-returns-section',
    ('wmo/runtime/platform/client.py|PlatformClient.request_org_run_control|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.request_org_run_control|method|'
     'missing-returns-section'),
    'wmo/runtime/platform/client.py|PlatformClient.resolve_run_target|method|missing-args-section',
    ('wmo/runtime/platform/client.py|PlatformClient.resolve_run_target|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.resolve_run_target|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/runtime/platform/client.py|PlatformClient.step_world_model_session|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.step_world_model_session|method|'
     'missing-returns-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.step_world_model_session|method|'
     'nontrivial-one-line-docstring'),
    ('wmo/runtime/platform/client.py|PlatformClient.stream_org_run_events|method|'
     'missing-args-section'),
    ('wmo/runtime/platform/client.py|PlatformClient.stream_org_run_events|method|'
     'missing-yields-section'),
    'wmo/runtime/platform/client.py|PlatformClient.whoami|method|missing-docstring',
    'wmo/runtime/platform/client.py|fetch_cli_config|function|missing-args-section',
    'wmo/runtime/platform/client.py|fetch_cli_config|function|missing-raises-section',
    'wmo/runtime/platform/client.py|fetch_cli_config|function|missing-returns-section',
    'wmo/runtime/platform/credentials.py|clear_credentials|function|missing-returns-section',
    'wmo/runtime/platform/credentials.py|clear_credentials|function|nontrivial-one-line-docstring',
    'wmo/runtime/platform/credentials.py|load_credentials|function|missing-returns-section',
    'wmo/runtime/platform/credentials.py|save_credentials|function|missing-args-section',
    'wmo/runtime/platform/credentials.py|save_credentials|function|missing-returns-section',
    'wmo/runtime/platform/transfer.py|extract_push_meta|function|missing-args-section',
    'wmo/runtime/platform/transfer.py|extract_push_meta|function|missing-returns-section',
    'wmo/runtime/platform/transfer.py|sha256_file|function|missing-args-section',
    'wmo/runtime/platform/transfer.py|sha256_file|function|missing-returns-section',
    'wmo/runtime/platform/transfer.py|sha256_file|function|nontrivial-one-line-docstring',
    'wmo/runtime/runs/reader.py|RunsReader.close|method|nontrivial-one-line-docstring',
    'wmo/runtime/runs/reader.py|RunsReader.event_count|method|missing-args-section',
    'wmo/runtime/runs/reader.py|RunsReader.event_count|method|missing-raises-section',
    'wmo/runtime/runs/reader.py|RunsReader.event_count|method|missing-returns-section',
    'wmo/runtime/runs/reader.py|RunsReader.open|method|missing-raises-section',
    'wmo/runtime/runs/reader.py|RunsReader.open|method|missing-returns-section',
    'wmo/runtime/runs/reader.py|RunsReader.request_control|method|missing-args-section',
    'wmo/runtime/runs/reader.py|RunsReader.request_control|method|missing-raises-section',
    'wmo/runtime/runs/reader.py|RunsReader.request_control|method|missing-returns-section',
    'wmo/runtime/runs/reader.py|RunsReader.tail|method|missing-args-section',
    'wmo/runtime/runs/reader.py|RunsReader.tail|method|missing-yields-section',
    'wmo/simulation/context/apps.py|get_app|function|missing-args-section',
    'wmo/simulation/context/apps.py|get_app|function|missing-returns-section',
    'wmo/simulation/context/brave.py|BraveConnector.connect|method|missing-args-section',
    'wmo/simulation/context/brave.py|BraveConnector.connect|method|missing-returns-section',
    'wmo/simulation/context/brave.py|BraveConnector.pull|method|missing-args-section',
    'wmo/simulation/context/brave.py|BraveConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/brave.py|BraveConnector.verify|method|missing-args-section',
    'wmo/simulation/context/brave.py|BraveConnector.verify|method|missing-returns-section',
    'wmo/simulation/context/connector.py|get_connector|function|missing-args-section',
    'wmo/simulation/context/connector.py|get_connector|function|missing-raises-section',
    'wmo/simulation/context/connector.py|get_connector|function|missing-returns-section',
    'wmo/simulation/context/connector.py|get_connector|function|nontrivial-one-line-docstring',
    'wmo/simulation/context/connector.py|register_connector|function|missing-args-section',
    ('wmo/simulation/context/connector.py|register_connector|function|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/credentials.py|connectors_path|function|missing-returns-section',
    'wmo/simulation/context/credentials.py|connectors_path|function|nontrivial-one-line-docstring',
    'wmo/simulation/context/credentials.py|delete_connector_auth|function|missing-args-section',
    'wmo/simulation/context/credentials.py|delete_connector_auth|function|missing-returns-section',
    'wmo/simulation/context/credentials.py|load_connector_auth|function|missing-args-section',
    'wmo/simulation/context/credentials.py|load_connector_auth|function|missing-returns-section',
    'wmo/simulation/context/credentials.py|resolve_env_token|function|missing-args-section',
    'wmo/simulation/context/credentials.py|resolve_env_token|function|missing-returns-section',
    'wmo/simulation/context/credentials.py|save_connector_auth|function|missing-args-section',
    'wmo/simulation/context/credentials.py|save_connector_auth|function|missing-returns-section',
    'wmo/simulation/context/github.py|GitHubConnector.connect|method|missing-args-section',
    'wmo/simulation/context/github.py|GitHubConnector.connect|method|missing-returns-section',
    ('wmo/simulation/context/github.py|GitHubConnector.connect|method|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/github.py|GitHubConnector.pull|method|missing-args-section',
    'wmo/simulation/context/github.py|GitHubConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/github.py|GitHubConnector.verify|method|missing-args-section',
    'wmo/simulation/context/github.py|GitHubConnector.verify|method|missing-raises-section',
    'wmo/simulation/context/github.py|GitHubConnector.verify|method|missing-returns-section',
    'wmo/simulation/context/github.py|GitHubConnector.verify|method|nontrivial-one-line-docstring',
    'wmo/simulation/context/google.py|GmailConnector.pull|method|missing-args-section',
    'wmo/simulation/context/google.py|GmailConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/google.py|GmailConnector.verify|method|missing-args-section',
    'wmo/simulation/context/google.py|GmailConnector.verify|method|missing-returns-section',
    'wmo/simulation/context/google.py|GmailConnector.verify|method|nontrivial-one-line-docstring',
    'wmo/simulation/context/google.py|GoogleCalendarConnector.pull|method|missing-args-section',
    'wmo/simulation/context/google.py|GoogleCalendarConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/google.py|GoogleCalendarConnector.verify|method|missing-args-section',
    ('wmo/simulation/context/google.py|GoogleCalendarConnector.verify|method|'
     'missing-returns-section'),
    ('wmo/simulation/context/google.py|GoogleCalendarConnector.verify|method|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/google.py|GoogleDriveConnector.pull|method|missing-args-section',
    'wmo/simulation/context/google.py|GoogleDriveConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/google.py|GoogleDriveConnector.verify|method|missing-args-section',
    'wmo/simulation/context/google.py|GoogleDriveConnector.verify|method|missing-returns-section',
    ('wmo/simulation/context/google.py|GoogleDriveConnector.verify|method|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/notion.py|NotionConnector.connect|method|missing-args-section',
    'wmo/simulation/context/notion.py|NotionConnector.connect|method|missing-returns-section',
    ('wmo/simulation/context/notion.py|NotionConnector.connect|method|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/notion.py|NotionConnector.pull|method|missing-args-section',
    'wmo/simulation/context/notion.py|NotionConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/notion.py|NotionConnector.pull|method|nontrivial-one-line-docstring',
    'wmo/simulation/context/notion.py|NotionConnector.verify|method|missing-args-section',
    'wmo/simulation/context/notion.py|NotionConnector.verify|method|missing-returns-section',
    'wmo/simulation/context/notion.py|NotionConnector.verify|method|nontrivial-one-line-docstring',
    'wmo/simulation/context/oauth.py|ensure_fresh|function|missing-args-section',
    'wmo/simulation/context/oauth.py|ensure_fresh|function|missing-returns-section',
    'wmo/simulation/context/oauth.py|pkce_challenge|function|missing-returns-section',
    'wmo/simulation/context/oauth.py|refresh_auth|function|missing-args-section',
    'wmo/simulation/context/oauth.py|refresh_auth|function|missing-returns-section',
    'wmo/simulation/context/oauth.py|serve_until|function|missing-args-section',
    'wmo/simulation/context/oauth.py|serve_until|function|nontrivial-one-line-docstring',
    'wmo/simulation/context/slack.py|SlackConnector.connect|method|missing-args-section',
    'wmo/simulation/context/slack.py|SlackConnector.connect|method|missing-returns-section',
    'wmo/simulation/context/slack.py|SlackConnector.pull|method|missing-args-section',
    'wmo/simulation/context/slack.py|SlackConnector.pull|method|missing-returns-section',
    'wmo/simulation/context/slack.py|SlackConnector.verify|method|missing-args-section',
    'wmo/simulation/context/slack.py|SlackConnector.verify|method|missing-returns-section',
    'wmo/simulation/context/store.py|ContextStore.delete|method|missing-args-section',
    'wmo/simulation/context/store.py|ContextStore.delete|method|missing-returns-section',
    'wmo/simulation/context/store.py|ContextStore.delete|method|nontrivial-one-line-docstring',
    'wmo/simulation/context/store.py|ContextStore.list_bundles|method|missing-returns-section',
    ('wmo/simulation/context/store.py|ContextStore.list_bundles|method|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/context/store.py|ContextStore.load|method|missing-args-section',
    'wmo/simulation/context/store.py|ContextStore.load|method|missing-returns-section',
    'wmo/simulation/context/store.py|ContextStore.save|method|missing-args-section',
    'wmo/simulation/context/store.py|ContextStore.save|method|missing-returns-section',
    'wmo/simulation/context/store.py|render_markdown|function|missing-args-section',
    'wmo/simulation/context/store.py|render_markdown|function|missing-returns-section',
    'wmo/simulation/context/types.py|capped|function|missing-args-section',
    'wmo/simulation/context/types.py|capped|function|missing-returns-section',
    'wmo/simulation/context/types.py|capped|function|nontrivial-one-line-docstring',
    'wmo/simulation/context/types.py|strip_html|function|missing-args-section',
    'wmo/simulation/context/types.py|strip_html|function|missing-returns-section',
    'wmo/simulation/context/types.py|strip_html|function|nontrivial-one-line-docstring',
    'wmo/simulation/context/types.py|transport_errors|function|missing-yields-section',
    'wmo/simulation/evaluation/failover.py|SameModelFailover.complete|method|missing-docstring',
    'wmo/simulation/evaluation/failover.py|SameModelFailover.embed|method|missing-docstring',
    'wmo/simulation/evaluation/failover.py|SameModelFailover.verify|method|missing-docstring',
    'wmo/simulation/evaluation/failover.py|anthropic_direct_id|function|missing-args-section',
    'wmo/simulation/evaluation/failover.py|anthropic_direct_id|function|missing-returns-section',
    'wmo/simulation/evaluation/failover.py|same_model_chain|function|missing-args-section',
    'wmo/simulation/evaluation/failover.py|same_model_chain|function|missing-returns-section',
    ('wmo/simulation/evaluation/failover.py|same_model_chain|function|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/evaluation/grid.py|CappedProvider.complete|method|missing-docstring',
    'wmo/simulation/evaluation/grid.py|CappedProvider.embed|method|missing-docstring',
    'wmo/simulation/evaluation/grid.py|CappedProvider.verify|method|missing-docstring',
    'wmo/simulation/evaluation/grid.py|merge_results|function|missing-args-section',
    'wmo/simulation/evaluation/grid.py|merge_results|function|missing-raises-section',
    'wmo/simulation/evaluation/grid.py|merge_results|function|missing-returns-section',
    'wmo/simulation/evaluation/grid.py|run_grid|function|missing-args-section',
    'wmo/simulation/evaluation/grid.py|run_grid|function|missing-returns-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid|function|missing-args-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid|function|missing-raises-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid|function|missing-returns-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid_heatmap|function|missing-args-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid_heatmap|function|missing-raises-section',
    'wmo/simulation/evaluation/grid_plot.py|plot_grid_heatmap|function|missing-returns-section',
    'wmo/simulation/model/demo.py|DemoStep.exact_match|method|missing-docstring',
    'wmo/simulation/model/demo.py|run_demo|function|missing-args-section',
    'wmo/simulation/model/demo.py|run_demo|function|missing-raises-section',
    'wmo/simulation/model/demo.py|run_demo|function|missing-returns-section',
    'wmo/simulation/model/eval_suites.py|EvalSuite.aliases|method|missing-docstring',
    'wmo/simulation/model/eval_suites.py|EvalSuite.resolve_files|method|missing-docstring',
    'wmo/simulation/model/eval_suites.py|EvalSuite.resolve_prompt|method|missing-docstring',
    'wmo/simulation/model/eval_suites.py|discover_eval_suites|function|missing-args-section',
    'wmo/simulation/model/eval_suites.py|discover_eval_suites|function|missing-returns-section',
    'wmo/simulation/model/eval_suites.py|list_eval_results|function|missing-args-section',
    'wmo/simulation/model/eval_suites.py|list_eval_results|function|missing-returns-section',
    'wmo/simulation/model/eval_suites.py|list_eval_results|function|nontrivial-one-line-docstring',
    'wmo/simulation/model/eval_suites.py|load_eval_suite|function|missing-args-section',
    'wmo/simulation/model/eval_suites.py|load_eval_suite|function|missing-raises-section',
    'wmo/simulation/model/eval_suites.py|load_eval_suite|function|missing-returns-section',
    'wmo/simulation/model/eval_suites.py|load_eval_suite|function|nontrivial-one-line-docstring',
    'wmo/simulation/model/eval_suites.py|resolve_eval_suite|function|missing-args-section',
    'wmo/simulation/model/eval_suites.py|resolve_eval_suite|function|missing-raises-section',
    'wmo/simulation/model/eval_suites.py|resolve_eval_suite|function|missing-returns-section',
    ('wmo/simulation/model/eval_suites.py|resolve_eval_suite|function|'
     'nontrivial-one-line-docstring'),
    'wmo/simulation/model/play.py|parse_action|function|missing-args-section',
    'wmo/simulation/model/play.py|parse_action|function|missing-raises-section',
    'wmo/simulation/model/play.py|parse_action|function|missing-returns-section',
    'wmo/simulation/model/play.py|play_turn|function|missing-args-section',
    'wmo/simulation/model/play.py|play_turn|function|missing-returns-section',
    'wmo/simulation/model/play.py|play_turn|function|nontrivial-one-line-docstring',
    'wmo/optimize/judge.py|Judge.score|method|missing-docstring',
    'wmo/optimize/judge.py|RubricJudge.score|method|missing-docstring',
    'wmo/optimize/judge_quality.py|JudgeQualityReport.failed|method|missing-docstring',
    'wmo/optimize/judge_quality.py|JudgeQualityReport.n_passed|method|missing-docstring',
    'wmo/optimize/judge_quality.py|JudgeQualityReport.n_total|method|missing-docstring',
    'wmo/optimize/judge_quality.py|JudgeQualityReport.summary|method|missing-docstring',
    'wmo/optimize/judge_quality.py|ScoreBand.holds|method|missing-docstring',
    'wmo/optimize/judge_quality.py|run_judge_quality|function|missing-args-section',
    'wmo/optimize/judge_quality.py|run_judge_quality|function|missing-returns-section',
    'wmo/optimize/numeric.py|NumericJudge.score|method|missing-docstring',
    'wmo/optimize/reward.py|EpisodeRewardJudge.score|method|missing-args-section',
    'wmo/optimize/reward.py|EpisodeRewardJudge.score|method|missing-returns-section',
    'wmo/simulation/evaluation/gold.py|GoldJudge.score|method|missing-docstring',
    'wmo/simulation/scenarios/verification/judge.py|ChecklistJudge.score|method|missing-docstring',
    'wmo/simulation/scenarios/verification/judge.py|ChecklistResult.pass_rate|method|missing-docstring',
    *(
        row
        for group in (
        "wmo/cli/ingest_cmd.py|ingest|function|missing-args-section\x1fwmo/cli/ingest_cmd.py|ingest|function|missing-raises-section\x1fwmo/cli/ui.py|RichBuildReporter.activity|method|missing-docstring",
        "wmo/cli/ui.py|RichBuildReporter.close|method|nontrivial-one-line-docstring\x1fwmo/cli/ui.py|RichBuildReporter.index_done|method|missing-docstring\x1fwmo/cli/ui.py|RichBuildReporter.ingest_done|method|missing-docstring",
        "wmo/cli/ui.py|RichBuildReporter.optimize_done|method|missing-docstring\x1fwmo/cli/ui.py|RichBuildReporter.optimize_start|method|missing-docstring\x1fwmo/cli/ui.py|RichBuildReporter.rollout|method|missing-docstring",
        "wmo/cli/ui.py|RichBuildReporter.split_done|method|missing-docstring\x1fwmo/cli/ui.py|build_summary_panel|function|missing-args-section\x1fwmo/cli/ui.py|build_summary_panel|function|missing-returns-section",
        "wmo/cli/ui.py|build_summary_panel|function|nontrivial-one-line-docstring\x1fwmo/cli/ui.py|run_build_wizard|function|missing-args-section\x1fwmo/cli/ui.py|run_build_wizard|function|missing-returns-section",
        "wmo/cli/ui.py|serve_model_default|function|missing-args-section\x1fwmo/cli/ui.py|serve_model_default|function|missing-returns-section\x1fwmo/optimize/routing/evaluation.py|scenario_id|function|missing-args-section",
        "wmo/optimize/routing/evaluation.py|scenario_id|function|missing-returns-section\x1fwmo/optimize/routing/sweep.py|resolve_config|function|missing-args-section\x1fwmo/optimize/routing/sweep.py|resolve_config|function|missing-raises-section",
        "wmo/optimize/routing/sweep.py|resolve_config|function|missing-returns-section\x1fwmo/optimize/routing/sweep.py|resolve_config|function|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/adapter.py|get_adapter|function|missing-docstring",
        "wmo/simulation/ingest/adapter.py|register_adapter|function|missing-docstring\x1fwmo/simulation/ingest/base.py|BaseTraceAdapter.collect_all|method|missing-args-section\x1fwmo/simulation/ingest/base.py|BaseTraceAdapter.collect_all|method|missing-returns-section",
        "wmo/simulation/ingest/base.py|BaseTraceAdapter.from_file|method|missing-docstring\x1fwmo/simulation/ingest/base.py|BaseTraceAdapter.from_vendor|method|missing-docstring\x1fwmo/simulation/ingest/base.py|load_payloads|function|missing-args-section",
        "wmo/simulation/ingest/base.py|load_payloads|function|missing-returns-section\x1fwmo/simulation/ingest/braintrust.py|BraintrustAdapter.spans_from_payload|method|missing-args-section\x1fwmo/simulation/ingest/braintrust.py|BraintrustAdapter.spans_from_payload|method|missing-returns-section",
        "wmo/simulation/ingest/braintrust.py|BraintrustAdapter.spans_from_payload|method|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/detect.py|detect_format|function|missing-args-section\x1fwmo/simulation/ingest/detect.py|detect_format|function|missing-raises-section",
        "wmo/simulation/ingest/detect.py|detect_format|function|missing-returns-section\x1fwmo/simulation/ingest/langfuse.py|LangfuseAdapter.spans_from_payload|method|missing-args-section\x1fwmo/simulation/ingest/langfuse.py|LangfuseAdapter.spans_from_payload|method|missing-returns-section",
        "wmo/simulation/ingest/langfuse.py|LangfuseAdapter.spans_from_payload|method|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/langsmith.py|LangSmithAdapter.spans_from_payload|method|missing-args-section\x1fwmo/simulation/ingest/langsmith.py|LangSmithAdapter.spans_from_payload|method|missing-returns-section",
        "wmo/simulation/ingest/langsmith.py|LangSmithAdapter.spans_from_payload|method|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/mastra.py|MastraAdapter.spans_from_payload|method|missing-docstring\x1fwmo/simulation/ingest/messages.py|ChatMessagesAdapter.spans_from_payload|method|missing-docstring",
        "wmo/simulation/ingest/normalize.py|SpanEmitter.emit|method|missing-args-section\x1fwmo/simulation/ingest/normalize.py|SpanEmitter.emit|method|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/normalize.py|action_from_llm_span|function|missing-docstring",
        "wmo/simulation/ingest/normalize.py|any_value|function|missing-args-section\x1fwmo/simulation/ingest/normalize.py|any_value|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|any_value|function|nontrivial-one-line-docstring",
        "wmo/simulation/ingest/normalize.py|as_text|function|missing-args-section\x1fwmo/simulation/ingest/normalize.py|as_text|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|as_text|function|nontrivial-one-line-docstring",
        "wmo/simulation/ingest/normalize.py|attrs_to_dict|function|missing-args-section\x1fwmo/simulation/ingest/normalize.py|attrs_to_dict|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|collect_spans|function|missing-args-section",
        "wmo/simulation/ingest/normalize.py|collect_spans|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|collect_spans|function|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/normalize.py|group_spans|function|missing-args-section",
        "wmo/simulation/ingest/normalize.py|group_spans|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|is_llm_span|function|missing-docstring\x1fwmo/simulation/ingest/normalize.py|is_tool_span|function|missing-docstring",
        "wmo/simulation/ingest/normalize.py|iso_to_ordinal|function|missing-args-section\x1fwmo/simulation/ingest/normalize.py|iso_to_ordinal|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|observation_from_tool_span|function|missing-docstring",
        "wmo/simulation/ingest/normalize.py|openai_call_name_args|function|missing-args-section\x1fwmo/simulation/ingest/normalize.py|openai_call_name_args|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|openai_tool_calls|function|missing-args-section",
        "wmo/simulation/ingest/normalize.py|openai_tool_calls|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|openai_tool_calls|function|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/normalize.py|parse_span|function|missing-args-section",
        "wmo/simulation/ingest/normalize.py|parse_span|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|parse_span|function|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/normalize.py|to_int|function|missing-args-section",
        "wmo/simulation/ingest/normalize.py|to_int|function|missing-returns-section\x1fwmo/simulation/ingest/normalize.py|to_int|function|nontrivial-one-line-docstring\x1fwmo/simulation/ingest/normalize.py|tool_call_action_from_tool_span|function|missing-docstring",
        "wmo/simulation/ingest/otel_writer.py|trace_to_spans|function|missing-args-section\x1fwmo/simulation/ingest/otel_writer.py|trace_to_spans|function|missing-returns-section\x1fwmo/simulation/ingest/otel_writer.py|trace_to_spans|function|nontrivial-one-line-docstring",
        "wmo/simulation/ingest/otel_writer.py|write_traces_jsonl|function|missing-args-section\x1fwmo/simulation/ingest/otel_writer.py|write_traces_jsonl|function|missing-returns-section\x1fwmo/simulation/ingest/otel_writer.py|write_traces_jsonl|function|nontrivial-one-line-docstring",
        "wmo/simulation/ingest/phoenix.py|PhoenixAdapter.spans_from_payload|method|missing-args-section\x1fwmo/simulation/ingest/phoenix.py|PhoenixAdapter.spans_from_payload|method|missing-returns-section\x1fwmo/simulation/ingest/postgres.py|PostgresAdapter.from_file|method|missing-docstring",
        "wmo/simulation/ingest/postgres.py|PostgresAdapter.spans_from_pull|method|missing-docstring\x1fwmo/simulation/ingest/posthog.py|PostHogAdapter.spans_from_payload|method|missing-docstring\x1fwmo/simulation/ingest/quality.py|drop_degenerate_traces|function|missing-args-section",
        "wmo/simulation/ingest/quality.py|drop_degenerate_traces|function|missing-returns-section\x1fwmo/simulation/ingest/stream.py|event_json|function|missing-args-section\x1fwmo/simulation/ingest/stream.py|event_json|function|missing-returns-section",
        "wmo/simulation/ingest/stream.py|ingest_events|function|missing-args-section\x1fwmo/simulation/ingest/stream.py|ingest_events|function|missing-raises-section\x1fwmo/simulation/ingest/stream.py|ingest_events|function|missing-yields-section",
        "wmo/simulation/model/build.py|build|function|missing-args-section\x1fwmo/simulation/model/build.py|build|function|missing-raises-section\x1fwmo/simulation/model/build.py|build|function|missing-returns-section",
        "wmo/simulation/model/build.py|ingest|function|missing-args-section\x1fwmo/simulation/model/build.py|ingest|function|missing-raises-section\x1fwmo/simulation/model/build.py|ingest|function|missing-returns-section",
        "wmo/simulation/model/build.py|ingest|function|nontrivial-one-line-docstring\x1fwmo/simulation/model/build.py|split_holdout|function|missing-args-section\x1fwmo/simulation/model/build.py|split_holdout|function|missing-returns-section",
        "wmo/simulation/model/build.py|split_traces|function|missing-args-section\x1fwmo/simulation/model/build.py|split_traces|function|missing-returns-section\x1fwmo/simulation/model/build.py|split_traces_3way|function|missing-args-section",
        "wmo/simulation/model/build.py|split_traces_3way|function|missing-raises-section\x1fwmo/simulation/model/build.py|split_traces_3way|function|missing-returns-section\x1fwmo/simulation/scenarios/builder.py|build_scenario_set|function|missing-args-section",
        "wmo/simulation/scenarios/builder.py|build_scenario_set|function|missing-raises-section\x1fwmo/simulation/scenarios/builder.py|build_scenario_set|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/clustering.py|cluster_facets|function|missing-args-section",
        "wmo/simulation/scenarios/mining/clustering.py|cluster_facets|function|missing-raises-section\x1fwmo/simulation/scenarios/mining/clustering.py|cluster_facets|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/clustering.py|cluster_facets|function|nontrivial-one-line-docstring",
        "wmo/simulation/scenarios/mining/clustering.py|default_k|function|missing-args-section\x1fwmo/simulation/scenarios/mining/clustering.py|default_k|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/clustering.py|default_k|function|nontrivial-one-line-docstring",
        "wmo/simulation/scenarios/mining/clustering.py|kmeans_labels|function|missing-args-section\x1fwmo/simulation/scenarios/mining/clustering.py|kmeans_labels|function|missing-raises-section\x1fwmo/simulation/scenarios/mining/clustering.py|kmeans_labels|function|missing-returns-section",
        "wmo/simulation/scenarios/mining/clustering.py|name_clusters|function|missing-args-section\x1fwmo/simulation/scenarios/mining/clustering.py|name_clusters|function|nontrivial-one-line-docstring\x1fwmo/simulation/scenarios/mining/clustering.py|normalize_rows|function|missing-args-section",
        "wmo/simulation/scenarios/mining/clustering.py|normalize_rows|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/clustering.py|normalize_rows|function|nontrivial-one-line-docstring\x1fwmo/simulation/scenarios/mining/facets.py|FacetExtractor.extract|method|missing-args-section",
        "wmo/simulation/scenarios/mining/facets.py|FacetExtractor.extract|method|missing-returns-section\x1fwmo/simulation/scenarios/mining/facets.py|FacetExtractor.extract|method|nontrivial-one-line-docstring\x1fwmo/simulation/scenarios/mining/facets.py|FacetExtractor.extract_all|method|missing-args-section",
        "wmo/simulation/scenarios/mining/facets.py|FacetExtractor.extract_all|method|missing-returns-section\x1fwmo/simulation/scenarios/mining/facets.py|TraceFacet.embed_text|method|missing-returns-section\x1fwmo/simulation/scenarios/mining/facets.py|tool_signature|function|missing-args-section",
        "wmo/simulation/scenarios/mining/facets.py|tool_signature|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/facets.py|trace_digest|function|missing-args-section\x1fwmo/simulation/scenarios/mining/facets.py|trace_digest|function|missing-returns-section",
        "wmo/simulation/scenarios/mining/facets.py|trace_domain|function|missing-args-section\x1fwmo/simulation/scenarios/mining/facets.py|trace_domain|function|missing-returns-section\x1fwmo/simulation/scenarios/mining/facets.py|trace_domain|function|nontrivial-one-line-docstring",
        "wmo/simulation/scenarios/mining/selection.py|hybrid_select|function|missing-args-section\x1fwmo/simulation/scenarios/mining/selection.py|hybrid_select|function|missing-raises-section\x1fwmo/simulation/scenarios/mining/selection.py|hybrid_select|function|missing-returns-section",
        "wmo/simulation/scenarios/mining/selection.py|semdedup_keep|function|missing-args-section\x1fwmo/simulation/scenarios/mining/selection.py|semdedup_keep|function|missing-returns-section\x1fwmo/simulation/scenarios/spec.py|scenarios_from_traces|function|missing-args-section",
        "wmo/simulation/scenarios/spec.py|scenarios_from_traces|function|missing-returns-section\x1fwmo/simulation/scenarios/spec.py|tools_hint_from_traces|function|missing-args-section\x1fwmo/simulation/scenarios/spec.py|tools_hint_from_traces|function|missing-returns-section",
        "wmo/simulation/scenarios/synthesis/scenario_set.py|ScenarioSet.load|method|missing-docstring\x1fwmo/simulation/scenarios/synthesis/scenario_set.py|ScenarioSet.retain|method|missing-args-section\x1fwmo/simulation/scenarios/synthesis/scenario_set.py|ScenarioSet.save|method|missing-docstring",
        "wmo/simulation/scenarios/synthesis/synthesizer.py|ScenarioSynthesizer.synthesize|method|missing-args-section\x1fwmo/simulation/scenarios/synthesis/synthesizer.py|ScenarioSynthesizer.synthesize|method|missing-returns-section\x1fwmo/simulation/scenarios/verification/judge.py|ChecklistJudge.score|method|missing-docstring",
        "wmo/simulation/scenarios/verification/judge.py|ChecklistResult.pass_rate|method|missing-docstring\x1fwmo/simulation/scenarios/verification/verify.py|VerificationReport.back_agreement_rate|method|missing-docstring\x1fwmo/simulation/scenarios/verification/verify.py|VerificationReport.solvable_rate|method|missing-docstring",
        "wmo/simulation/scenarios/verification/verify.py|verify_scenarios|function|missing-args-section\x1fwmo/simulation/scenarios/verification/verify.py|verify_scenarios|function|missing-returns-section\x1fwmo/simulation/serving/builds.py|BuildManager.snapshot|method|missing-docstring",
        "wmo/simulation/serving/builds.py|BuildManager.sse_events|method|missing-args-section\x1fwmo/simulation/serving/builds.py|BuildManager.sse_events|method|missing-yields-section\x1fwmo/simulation/serving/builds.py|BuildManager.start|method|missing-args-section",
        "wmo/simulation/serving/builds.py|BuildManager.start|method|missing-raises-section\x1fwmo/simulation/serving/builds.py|BuildManager.start|method|missing-returns-section\x1fwmo/simulation/serving/builds.py|BuildManager.start|method|nontrivial-one-line-docstring",
        "wmo/simulation/serving/builds.py|BuildManager.uploads_dir|method|missing-docstring\x1fwmo/simulation/serving/builds.py|BuildManager.wait|method|missing-args-section\x1fwmo/simulation/serving/builds.py|BuildManager.wait|method|missing-raises-section",
        "wmo/simulation/serving/builds.py|BuildManager.wait|method|missing-returns-section\x1fwmo/simulation/serving/traces_source.py|scenarios_from_traces|function|missing-args-section\x1fwmo/simulation/serving/traces_source.py|scenarios_from_traces|function|missing-returns-section",
        "wmo/simulation/serving/traces_source.py|scenarios_from_traces|function|nontrivial-one-line-docstring",
        )
        for row in group.split("\x1f")
    ),
)
# fmt: on
DOCSTRING_TOMBSTONES: frozenset[DocstringViolation] = frozenset(
    DocstringViolation(*row.split("|", maxsplit=3)) for row in _DOCSTRING_TOMBSTONE_ROWS
)


@functools.lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    """Return every tracked repository path or skip outside a Git checkout."""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; public-docstring guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("not a git checkout; public-docstring guardrails require the repository")
    return tuple(result.stdout.splitlines())


def _git_output(arguments: list[str]) -> str:
    """Return Git output for a local read command or skip outside a checkout."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git not available; public-docstring guardrails require a git checkout")
    if result.returncode != 0:
        pytest.skip("the frozen docstring baseline is unavailable in this checkout")
    return result.stdout


@functools.cache
def _tracked_files_at_revision(revision: str) -> tuple[str, ...]:
    """Return all repository paths at one immutable Git revision."""
    return tuple(_git_output(["ls-tree", "-r", "--name-only", revision]).splitlines())


def _is_production_python_path(relative_path: str) -> bool:
    """Return whether a path contains production Python subject to public-docstring checks."""
    return (
        relative_path.startswith("wmo/")
        and relative_path.endswith(".py")
        and not relative_path.endswith("_test.py")
        and Path(relative_path).name != "conftest.py"
    )


def _is_public_name(name: str) -> bool:
    """Return whether a Python name is public under the migration convention."""
    return not name.startswith("_")


def _is_protocol_class(node: ast.ClassDef) -> bool:
    """Return whether a class explicitly extends typing.Protocol."""
    for base in node.bases:
        candidate = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(candidate, ast.Name) and candidate.id == "Protocol":
            return True
        if isinstance(candidate, ast.Attribute) and candidate.attr == "Protocol":
            return True
    return False


def _body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """Return a function body after its optional leading docstring expression."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _is_trivial_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return whether one public function is eligible for a one-line docstring."""
    body = _body_without_docstring(node)
    if len(body) != 1:
        return False
    statement = body[0]
    return isinstance(statement, (ast.Pass, ast.Return)) or (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _descendants_without_nested_definitions(node: ast.AST) -> Iterator[ast.AST]:
    """Yield descendants of one API without crossing into nested definitions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield child
        yield from _descendants_without_nested_definitions(child)


def _public_argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    """Return documented argument names after excluding the receiver convention."""
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    return tuple(argument.arg for argument in arguments if argument.arg not in {"self", "cls"})


def _has_google_section(docstring: str, section: str) -> bool:
    """Return whether a docstring contains one exact Google-style section heading."""
    return any(line.strip() == f"{section}:" for line in docstring.splitlines())


def _function_violations(
    relative_path: str,
    qualified_name: str,
    kind: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[DocstringViolation]:
    """Return public-docstring violations for one function or method."""
    docstring = ast.get_docstring(node, clean=False)
    if docstring is None:
        return {DocstringViolation(relative_path, qualified_name, kind, "missing-docstring")}
    if _is_trivial_function(node):
        return set()
    violations: set[DocstringViolation] = set()
    if len(docstring.splitlines()) == 1:
        violations.add(
            DocstringViolation(relative_path, qualified_name, kind, "nontrivial-one-line-docstring")
        )
    if _public_argument_names(node) and not _has_google_section(docstring, "Args"):
        violations.add(
            DocstringViolation(relative_path, qualified_name, kind, "missing-args-section")
        )
    descendants = tuple(_descendants_without_nested_definitions(node))
    if any(
        isinstance(descendant, ast.Return) and descendant.value is not None
        for descendant in descendants
    ):
        if not _has_google_section(docstring, "Returns"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-returns-section")
            )
    if any(isinstance(descendant, (ast.Yield, ast.YieldFrom)) for descendant in descendants):
        if not _has_google_section(docstring, "Yields"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-yields-section")
            )
    if any(isinstance(descendant, ast.Raise) for descendant in descendants):
        if not _has_google_section(docstring, "Raises"):
            violations.add(
                DocstringViolation(relative_path, qualified_name, kind, "missing-raises-section")
            )
    return violations


def _docstring_violations_in_source(
    relative_path: str, source: str
) -> frozenset[DocstringViolation]:
    """Return public module, class, protocol, function, and method violations from source."""
    if relative_path.endswith("_test.py"):
        return frozenset()
    tree = ast.parse(source, filename=relative_path)
    violations: set[DocstringViolation] = set()
    if ast.get_docstring(tree, clean=False) is None:
        violations.add(DocstringViolation(relative_path, "<module>", "module", "missing-docstring"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public_name(node.name):
            class_kind = "protocol" if _is_protocol_class(node) else "class"
            if ast.get_docstring(node, clean=False) is None:
                violations.add(
                    DocstringViolation(relative_path, node.name, class_kind, "missing-docstring")
                )
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(
                    member.name
                ):
                    violations.update(
                        _function_violations(
                            relative_path,
                            f"{node.name}.{member.name}",
                            "method",
                            member,
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public_name(
            node.name
        ):
            violations.update(_function_violations(relative_path, node.name, "function", node))
    return frozenset(violations)


def _docstring_violations(paths: Iterable[str]) -> frozenset[DocstringViolation]:
    """Return current public-docstring violations from tracked production Python modules."""
    violations: set[DocstringViolation] = set()
    for relative_path in paths:
        if not _is_production_python_path(relative_path):
            continue
        path = REPO_ROOT / relative_path
        if path.is_file():
            violations.update(
                _docstring_violations_in_source(relative_path, path.read_text(encoding="utf-8"))
            )
    return frozenset(violations)


@functools.cache
def _baseline_docstring_violations() -> frozenset[DocstringViolation]:
    """Return the exact W1 baseline violations used as the active transition inventory."""
    violations: set[DocstringViolation] = set()
    for relative_path in _tracked_files_at_revision(DOCSTRING_BASELINE_REVISION):
        if not _is_production_python_path(relative_path):
            continue
        source = _git_output(["show", f"{DOCSTRING_BASELINE_REVISION}:{relative_path}"])
        violations.update(_docstring_violations_in_source(relative_path, source))
    return frozenset(violations)


def test_ruff_selects_public_google_docstring_rules() -> None:
    """Ruff selects public module, class, function, and method docstring presence checks."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as file_handle:
        config = tomllib.load(file_handle)
    selected = set(config["tool"]["ruff"]["lint"]["extend-select"])
    assert {"D100", "D101", "D102", "D103", "D104"} <= selected
    assert config["tool"]["ruff"]["lint"]["pydocstyle"]["convention"] == "google"


def test_public_docstring_transition_inventory_is_monotonic() -> None:
    """New violations fail and fixed baseline violations become permanent tombstones."""
    baseline = _baseline_docstring_violations()
    current = _docstring_violations(_tracked_files())
    new_violations = current - baseline
    fixed_violations = baseline - current
    missing_tombstones = fixed_violations - DOCSTRING_TOMBSTONES
    stale_tombstones = DOCSTRING_TOMBSTONES - fixed_violations
    reintroduced = current & DOCSTRING_TOMBSTONES
    assert not new_violations, f"new public-docstring violations: {sorted(new_violations)}"
    assert not missing_tombstones, (
        "fixed baseline public-docstring violations must be tombstoned: "
        f"{sorted(missing_tombstones)}"
    )
    assert not stale_tombstones, f"stale public-docstring tombstones: {sorted(stale_tombstones)}"
    assert not reintroduced, f"reintroduced public-docstring tombstones: {sorted(reintroduced)}"


def test_google_docstrings_accept_trivial_and_nontrivial_public_apis() -> None:
    """One-line trivial APIs and full Google sections are both direct passing fixtures."""
    source = '''"""Fixture module."""

from typing import Protocol

class CustomerProtocol(Protocol):
    """Provides a customer extension point."""

    def execute(self, request: str) -> str:
        """Normalize one request.

        Args:
            request: Request text supplied by the customer.

        Returns:
            The normalized request.

        Raises:
            ValueError: If the request is blank.
        """
        if not request:
            raise ValueError("request is required")
        return request.strip()


def identity(value: str) -> str:
    """Return the supplied value."""
    return value


def stream(values: list[str]):
    """Yield normalized values.

    Args:
        values: Values to normalize.

    Yields:
        Normalized values.
    """
    yield from (value.strip() for value in values)
'''
    assert not _docstring_violations_in_source("wmo/fixture.py", source)


def test_google_docstrings_reject_missing_and_nontrivial_one_line_public_apis() -> None:
    """Direct failing fixtures cover public modules, classes, protocols, functions, and methods."""
    source = '''from typing import Protocol


class UndocumentedProtocol(Protocol):
    ...


class CustomerProtocol(Protocol):
    """Provides a customer extension point."""

    def execute(self, request: str) -> str:
        """Execute a request."""
        if not request:
            raise ValueError("request is required")
        return request.strip()


def build(request: str) -> str:
    """Build a result."""
    normalized = request.strip()
    return normalized


def stream(values: list[str]):
    """Stream values."""
    yield from values
'''
    violations = _docstring_violations_in_source("wmo/fixture.py", source)
    reasons = {violation.reason for violation in violations}
    kinds = {violation.kind for violation in violations}
    assert "missing-docstring" in reasons
    assert "nontrivial-one-line-docstring" in reasons
    assert "missing-args-section" in reasons
    assert "missing-returns-section" in reasons
    assert "missing-raises-section" in reasons
    assert "missing-yields-section" in reasons
    assert {"module", "protocol", "method", "function"} <= kinds


def test_private_and_test_helpers_are_not_public_docstring_apis() -> None:
    """Private helpers and test fixtures remain outside the public API contract."""
    source = '''"""Fixture module."""

def _helper() -> None:
    pass
'''
    assert not _docstring_violations_in_source("wmo/fixture.py", source)
    assert not _docstring_violations_in_source("wmo/fixture_test.py", "def test_case(): pass\n")

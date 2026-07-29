"""Build leak-free WMO artifacts from frozen training-side real traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from wmo.config import ArtifactPaths, HarnessConfig, save_config
from wmo.core.files import write_text_atomic
from wmo.core.types import Action, ActionKind, Observation, Step, Trace
from wmo.engine.prompts import BASE_ENV_PROMPT
from wmo.ingest import get_adapter
from wmo.providers.base import ProviderConfig, ProviderKind
from wmo.retrieval import EmbeddingRetriever, HashingEmbedder

EMBED_DIM = 512


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected object, got {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value).__name__}")
    return list(value)


def _objects(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append({str(key): item for key, item in value.items()})
    return rows


def _string_set(path: Path, key: str) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        value: list[object] = []
    elif text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"{path} must contain a list or JSONL objects")
        value = list(parsed)
    else:
        value = list(_objects(path))
    selected = set()
    for raw in value:
        row = _dict(raw)
        item = row.get(key)
        if not isinstance(item, str):
            raise ValueError(f"{path} contains a row without string {key}")
        selected.add(item)
    return selected


def _split_ids(path: Path, benchmark: str) -> set[str]:
    value = _json(path)
    root = _dict(value)
    if not isinstance(root.get(benchmark), dict):
        raise ValueError(f"{path} has no {benchmark} split")
    fit = _dict(root[benchmark]).get("fit")
    if not isinstance(fit, list) or not all(isinstance(item, str) for item in fit):
        raise ValueError(f"{path} has no string fit ids for {benchmark}")
    return {str(item) for item in fit}


def _trace_task(trace: Trace) -> str | None:
    return next((step.task for step in trace.steps if step.task), None)


def _tau_traces(source: Path, allowed: Path) -> tuple[list[Trace], dict[str, object]]:
    traces = get_adapter("otel-genai").from_file(str(source))
    tasks = _string_set(allowed, "task")
    selected = [trace for trace in traces if _trace_task(trace) in tasks]
    covered = {_trace_task(trace) for trace in selected}
    if covered != tasks:
        missing = sorted(tasks - covered)
        raise ValueError(f"Tau training corpus lacks {len(missing)} frozen tasks: {missing[:3]}")
    return selected, {
        "source_kind": "otel-genai",
        "source": str(source),
        "source_sha256": _sha256(source),
        "allowed_tasks": str(allowed),
        "allowed_tasks_sha256": _sha256(allowed),
        "allowed_task_count": len(tasks),
    }


def _terminal_traces(rows_path: Path, fit_ids: set[str]) -> tuple[list[Trace], dict[str, object]]:
    traces = []
    covered: set[str] = set()
    for row in _objects(rows_path):
        raw_task_id = row.get("task_id")
        if not isinstance(raw_task_id, str) or raw_task_id not in fit_ids:
            continue
        raw_paths = row.get("trace_paths")
        if not isinstance(raw_paths, list):
            continue
        for raw_path in raw_paths:
            if not isinstance(raw_path, str):
                continue
            path = Path(raw_path)
            payload = _dict(_json(path))
            if not isinstance(payload.get("steps"), list):
                raise ValueError(f"{path} is not a WMO run trace")
            steps = [Step.model_validate(item) for item in _list(payload["steps"])]
            model = str(row.get("model", "unknown"))
            traces.append(
                Trace(
                    trace_id=f"terminal-bench-2:{raw_task_id}:{model}",
                    steps=steps,
                    source=str(path),
                    metadata={
                        "benchmark": "terminal-bench-2",
                        "task_id": raw_task_id,
                        "model": model,
                    },
                )
            )
            covered.add(raw_task_id)
    if covered != fit_ids:
        missing = sorted(fit_ids - covered)
        raise ValueError(f"Terminal-Bench training traces lack {len(missing)} tasks: {missing[:3]}")
    return traces, {
        "source_kind": "harbor-wmo-run",
        "source": str(rows_path),
        "source_sha256": _sha256(rows_path),
        "allowed_task_count": len(fit_ids),
    }


def _routerbench_traces(
    rows_path: Path,
    fit_ids: set[str],
) -> tuple[list[Trace], dict[str, object]]:
    traces = []
    covered: set[str] = set()
    for row in _objects(rows_path):
        scenario_id = row.get("scenario_id")
        reward = row.get("reward")
        if (
            not isinstance(scenario_id, str)
            or scenario_id not in fit_ids
            or not isinstance(reward, (int, float))
        ):
            continue
        raw_replies = row.get("replies")
        replies = raw_replies if isinstance(raw_replies, list) else []
        answer = next((value for value in replies if isinstance(value, str)), "<no answer>")
        model = str(row.get("model", "unknown"))
        task = str(row.get("task", scenario_id))
        observation = {
            "correct": bool(float(reward) >= 0.5),
            "reward": float(reward),
        }
        traces.append(
            Trace(
                trace_id=f"routerbench:{scenario_id}:{model}",
                steps=[
                    Step(
                        task=task,
                        action=Action(kind=ActionKind.MESSAGE, content=answer),
                        observation=Observation(
                            content=json.dumps(observation, sort_keys=True),
                            reward=float(reward),
                        ),
                    )
                ],
                source=str(rows_path),
                metadata={
                    "benchmark": "routerbench",
                    "task_id": scenario_id,
                    "model": model,
                },
            )
        )
        covered.add(scenario_id)
    if covered != fit_ids:
        missing = sorted(fit_ids - covered)
        raise ValueError(f"RouterBench training traces lack {len(missing)} tasks: {missing[:3]}")
    return traces, {
        "source_kind": "objective-answer-cells",
        "source": str(rows_path),
        "source_sha256": _sha256(rows_path),
        "allowed_task_count": len(fit_ids),
    }


def _config() -> HarnessConfig:
    return HarnessConfig(
        providers=[
            ProviderConfig(
                kind=ProviderKind.AZURE_OPENAI,
                model="gpt-5.5",
                model_type="gpt-5.5",
                api_version="2024-10-21",
                reasoning_effort="high",
            )
        ],
        serve_provider=ProviderKind.AZURE_OPENAI,
        embed_dim=EMBED_DIM,
        top_k=5,
        train_split=0.8,
        gepa_budget=0,
        trace_adapter="otel-genai",
    )


def _write_artifact(
    out_dir: Path,
    traces: list[Trace],
    provenance: dict[str, object],
    *,
    force: bool,
) -> None:
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force only before Phase 2 starts")
        shutil.rmtree(out_dir)
    paths = ArtifactPaths(out_dir)
    retriever = EmbeddingRetriever(HashingEmbedder(dim=EMBED_DIM))
    retriever.index(traces)
    retriever.save(paths.index)
    save_config(_config(), out_dir)
    paths.base_prompt.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(paths.base_prompt, BASE_ENV_PROMPT)
    write_text_atomic(paths.optimized_prompt, BASE_ENV_PROMPT)
    write_text_atomic(paths.frontier, json.dumps([BASE_ENV_PROMPT], indent=2) + "\n")
    write_text_atomic(
        paths.metrics,
        json.dumps(
            {
                "mode": "leak-free-rag-only",
                "gepa_rollouts": 0,
                "traces": len(traces),
                "steps": sum(len(trace.steps) for trace in traces),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    trace_ids = sorted(trace.trace_id for trace in traces)
    write_text_atomic(
        out_dir / "provenance.json",
        json.dumps(
            {
                **provenance,
                "world_model_provider": "azure",
                "world_model": "gpt-5.5",
                "deployment_env": "AZURE_FOUNDRY_GPT55_DEPLOYMENT",
                "endpoint_env": "AZURE_FOUNDRY_2_ENDPOINT",
                "key_env": "AZURE_FOUNDRY_2_API_KEY",
                "retrieval": "hashing-512",
                "top_k": 5,
                "trace_count": len(traces),
                "step_count": sum(len(trace.steps) for trace in traces),
                "trace_ids_sha256": hashlib.sha256(
                    "\0".join(trace_ids).encode()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("tau2", "routerbench", "terminal_bench_2"),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--allowed", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.benchmark == "tau2":
        if args.allowed is None:
            raise ValueError("Tau2 needs --allowed scenarios_train.jsonl")
        traces, provenance = _tau_traces(args.source, args.allowed)
    else:
        if args.split is None:
            raise ValueError(f"{args.benchmark} needs --split")
        fit_ids = _split_ids(args.split, args.benchmark)
        if args.benchmark == "routerbench":
            traces, provenance = _routerbench_traces(args.source, fit_ids)
        else:
            traces, provenance = _terminal_traces(args.source, fit_ids)
        provenance["split"] = str(args.split)
        provenance["split_sha256"] = _sha256(args.split)
    _write_artifact(args.out_dir, traces, provenance, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

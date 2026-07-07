"use client";

/**
 * Explore a model's recorded traces (grouped by task) and replay any of them OPEN LOOP against
 * the live world model: feed the recorded action sequence to a fresh session and compare, step by
 * step, what the world model produces against what the real environment recorded. A running
 * fidelity score summarizes how faithfully the reconstruction tracks the ground truth. This is
 * the teacher-forced replay `wmh eval` runs, made interactive.
 */

import { useCallback, useEffect, useState } from "react";
import { readableTask } from "@/components/Playground";
import { Spinner } from "@/components/Spinner";
import {
  createSession,
  getTraces,
  startTracesDownload,
  step,
  tracesDownloadProgress,
} from "@/lib/api";
import type {
  DownloadProgress,
  IndexEntry,
  Scenario,
  ScenarioStep,
  TracesResponse,
} from "@/lib/types";

function TaskPrompt({ task }: { task: string | null }) {
  if (!task) return null;
  return (
    <div className="rounded-lg border border-line bg-surface-sunk px-3 py-2">
      <div className="mono-label mb-1">initial task prompt</div>
      <div className="max-h-32 overflow-y-auto whitespace-pre-wrap text-[13px] text-ink-soft">
        {readableTask(task)}
      </div>
    </div>
  );
}

type ReplayRow = {
  label: string;
  recorded: string;
  wm: string | null; // null while that step is still running
  match: boolean | null;
};

const norm = (s: string) => s.replace(/\s+/g, " ").trim();

function fidelity(rows: ReplayRow[]): { done: number; matches: number } {
  const done = rows.filter((r) => r.wm !== null).length;
  const matches = rows.filter((r) => r.match).length;
  return { done, matches };
}

function StepBlock({ step }: { step: ScenarioStep }) {
  return (
    <div className="border-t border-line py-2 first:border-t-0">
      <div className="font-mono text-[13px] text-accent">&rsaquo; {step.action_label}</div>
      <div
        className={`whitespace-pre-wrap font-mono text-[12px] ${
          step.is_error ? "text-accent-red" : "text-ink-soft"
        }`}
      >
        {step.observation}
      </div>
    </div>
  );
}

function ComparisonView({ rows, replaying }: { rows: ReplayRow[]; replaying: boolean }) {
  const { done, matches } = fidelity(rows);
  const pct = done ? Math.round((matches / done) * 100) : null;
  const stoppedEarly = !replaying && done < rows.length;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <span className="mono-label">fidelity this replay</span>
        <span
          className={`text-sm tabular-nums ${
            pct == null ? "text-ink-faint" : pct >= 80 ? "text-live" : "text-accent-amber"
          }`}
        >
          {pct == null ? "..." : `${pct}%`}
        </span>
        <span className="text-xs text-ink-faint">
          {matches}/{done} of {rows.length} steps match the recorded observation
          {stoppedEarly && " (replay stopped early)"}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="mono-label">recorded (ground truth)</div>
        <div className="mono-label">world model</div>
        {rows.map((row, i) => (
          <div key={i} className="contents">
            <div className="col-span-2 mt-1 font-mono text-[12px] text-accent">
              &rsaquo; {row.label}
            </div>
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-md border border-line bg-surface-sunk p-2 font-mono text-[11px] text-ink-soft">
              {row.recorded}
            </pre>
            <pre
              className={`overflow-x-auto whitespace-pre-wrap rounded-md border p-2 font-mono text-[11px] ${
                row.wm === null
                  ? "border-line text-ink-faint"
                  : row.match
                    ? "border-live/40 text-ink"
                    : "border-accent-amber/50 bg-accent-amber/[0.05] text-ink"
              }`}
            >
              {row.wm ?? (replaying ? "..." : "not run")}
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}

function ScenarioCard({ entry, scenario }: { entry: IndexEntry; scenario: Scenario }) {
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<ReplayRow[] | null>(null);
  const [replaying, setReplaying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const replay = useCallback(async () => {
    setReplaying(true);
    setError(null);
    const initial: ReplayRow[] = scenario.steps.map((s) => ({
      label: s.action_label,
      recorded: s.observation,
      wm: null,
      match: null,
    }));
    setRows(initial);
    try {
      const { session_id } = await createSession(entry.card.name, scenario.task);
      for (let i = 0; i < scenario.steps.length; i++) {
        const { observation } = await step(entry.card.name, session_id, scenario.steps[i].action);
        setRows((prev) => {
          if (!prev) return prev;
          const next = [...prev];
          next[i] = {
            ...next[i],
            wm: observation.content,
            match: norm(observation.content) === norm(scenario.steps[i].observation),
          };
          return next;
        });
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setReplaying(false);
    }
  }, [entry.card.name, scenario]);

  return (
    <div className="rounded-xl border border-line">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2">
          <span className={`text-ink-faint transition-transform ${open ? "rotate-90" : ""}`}>
            &rsaquo;
          </span>
          <span className="truncate text-sm text-ink">{scenario.label}</span>
        </span>
        <span className="mono-label shrink-0">{scenario.steps.length} steps</span>
      </button>
      {open && (
        <div className="flex flex-col gap-3 border-t border-line px-4 py-3">
          <TaskPrompt task={scenario.task} />
          <div className="flex items-center justify-between gap-3">
            <span className="mono-label">{rows ? "open-loop replay" : "recorded trace"}</span>
            <button
              onClick={replay}
              disabled={replaying}
              className="rounded-lg bg-ink px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
            >
              {replaying ? "replaying..." : rows ? "replay again" : "Replay open loop"}
            </button>
          </div>
          {rows ? (
            <ComparisonView rows={rows} replaying={replaying} />
          ) : (
            <div>
              {scenario.steps.map((s, i) => (
                <StepBlock key={i} step={s} />
              ))}
            </div>
          )}
          {error && (
            <p className="rounded-lg border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

const INTRO =
  "Recorded agent traces, grouped by task. Replay one open loop to see how faithfully the world model reconstructs each step against what really happened.";

function ScenarioList({ entry, scenarios }: { entry: IndexEntry; scenarios: Scenario[] }) {
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-ink-soft">{INTRO}</p>
      {scenarios.map((s) => (
        <ScenarioCard key={s.id} entry={entry} scenario={s} />
      ))}
    </div>
  );
}

function fmtBytes(n: number): string {
  return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Offers a Hub download for models whose traces are not present locally, and shows progress. */
function DownloadPanel({ entry, onDone }: { entry: IndexEntry; onDone: () => void }) {
  const [progress, setProgress] = useState<DownloadProgress | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const download = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await startTracesDownload(entry.card.name);
      // Poll byte progress until the backend reports a terminal state.
      for (;;) {
        await new Promise((r) => setTimeout(r, 800));
        const { download } = await tracesDownloadProgress(entry.card.name);
        setProgress(download);
        if (!download || download.status === "done") {
          onDone();
          return;
        }
        if (download.status === "failed") {
          setError(download.error ?? "download failed");
          return;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [entry.card.name, onDone]);

  const pct =
    progress?.total && progress.total > 0
      ? Math.min(100, Math.round((progress.downloaded / progress.total) * 100))
      : null;

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-line p-6">
      <div className="mono-label">traces not downloaded</div>
      <p className="text-sm text-ink-soft">
        This model&apos;s traces live on the Hugging Face Hub. Download them to explore and replay
        them here; they land on your local <code className="font-mono">wmh serve</code>, so this is
        a one-time fetch.
      </p>
      <button
        onClick={download}
        disabled={busy}
        className="flex w-fit items-center gap-2 rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
      >
        {busy && <Spinner className="border-white border-t-transparent" />}
        {busy ? "Downloading traces" : "Download traces"}
      </button>
      {progress && progress.status === "running" && (
        <div className="flex items-center gap-2 text-xs text-ink-faint">
          <span className="tabular-nums">
            {fmtBytes(progress.downloaded)}
            {progress.total ? ` / ${fmtBytes(progress.total)}` : ""}
          </span>
          {pct !== null && (
            <span className="h-1.5 w-40 overflow-hidden rounded-full bg-line">
              <span
                className="block h-full rounded-full bg-accent-teal transition-all"
                style={{ width: `${pct}%` }}
              />
            </span>
          )}
          {pct !== null && <span className="tabular-nums">{pct}%</span>}
        </div>
      )}
      {error && (
        <p className="rounded-lg border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}
    </div>
  );
}

export function TracesExplorer({ entry }: { entry: IndexEntry }) {
  // Prefer live traces from the backend (which reflect a local file or a fresh Hub download);
  // fall back to the statically indexed scenarios when the backend is unreachable.
  const [resp, setResp] = useState<TracesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [reachable, setReachable] = useState(true);

  // Async so the effect does no synchronous setState; the first state update lands after the fetch.
  const load = useCallback(async () => {
    try {
      const r = await getTraces(entry.card.name);
      setResp(r);
      setReachable(true);
    } catch {
      setReachable(false);
    } finally {
      setLoading(false);
    }
  }, [entry.card.name]);

  useEffect(() => {
    // load() only setStates after an awaited fetch, so no synchronous cascade despite the rule.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void load();
  }, [load]);

  if (loading) {
    return (
      <div className="flex items-center gap-2 rounded-xl border border-line p-6 text-sm text-ink-faint">
        <Spinner /> checking for traces...
      </div>
    );
  }

  // Backend unreachable: show whatever the static index captured.
  if (!reachable || !resp) {
    return entry.scenarios.length ? (
      <ScenarioList entry={entry} scenarios={entry.scenarios} />
    ) : (
      <div className="rounded-xl border border-line p-6 text-sm text-ink-faint">
        No recorded traces are indexed for this model.
      </div>
    );
  }

  if (resp.scenarios.length) {
    return <ScenarioList entry={entry} scenarios={resp.scenarios} />;
  }
  if (resp.downloadable) {
    return <DownloadPanel entry={entry} onDone={load} />;
  }
  // No live traces and nothing to download: last resort is the static index.
  return entry.scenarios.length ? (
    <ScenarioList entry={entry} scenarios={entry.scenarios} />
  ) : (
    <div className="rounded-xl border border-line p-6 text-sm text-ink-faint">
      No recorded traces are available for this model.
    </div>
  );
}

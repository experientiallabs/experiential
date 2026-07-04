"use client";

/**
 * Explore a model's recorded traces (grouped by task) and replay any of them OPEN LOOP against
 * the live world model: feed the recorded action sequence to a fresh session and compare, step by
 * step, what the world model produces against what the real environment recorded. A running
 * fidelity score summarizes how faithfully the reconstruction tracks the ground truth. This is
 * the teacher-forced replay `wmh eval` runs, made interactive.
 */

import { useCallback, useState } from "react";
import { readableTask } from "@/components/Playground";
import { createSession, step } from "@/lib/api";
import type { IndexEntry, Scenario, ScenarioStep } from "@/lib/types";

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

function ComparisonView({ rows }: { rows: ReplayRow[] }) {
  const { done, matches } = fidelity(rows);
  const pct = done ? Math.round((matches / done) * 100) : null;
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
          {matches}/{done} steps match the recorded observation
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
              {row.wm ?? "..."}
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
            <ComparisonView rows={rows} />
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

export function TracesExplorer({ entry }: { entry: IndexEntry }) {
  if (entry.scenarios.length === 0) {
    return (
      <div className="rounded-xl border border-line p-6 text-sm text-ink-faint">
        No recorded traces are indexed for this model.
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <p className="text-sm text-ink-soft">
        Recorded agent traces, grouped by task. Replay one open loop to see how faithfully the
        world model reconstructs each step against what really happened.
      </p>
      {entry.scenarios.map((s) => (
        <ScenarioCard key={s.id} entry={entry} scenario={s} />
      ))}
    </div>
  );
}

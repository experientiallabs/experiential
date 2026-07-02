"use client";

/**
 * Guided build-your-own flow: traces (local path or upload) + name + provider/model + budget
 * -> POST /world_models/builds -> live SSE progress (BuildReporter stages + rollout ticks)
 * -> link to the fresh model's page. Falls back to the exact `wmh build` command when no
 * backend answers.
 */

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  API_BASE,
  ApiError,
  buildSnapshot,
  isServeUp,
  openBuildEvents,
  startBuild,
  uploadTraces,
} from "@/lib/api";
import type { BuildEvent } from "@/lib/types";

const STAGES = [
  { key: "ingest_done", label: "Ingest traces" },
  { key: "split_done", label: "Split train / held-out" },
  { key: "index_done", label: "Index replay buffer" },
  { key: "optimize_done", label: "Optimize env prompt (GEPA)" },
] as const;

function stageDetail(events: BuildEvent[], key: string): string | null {
  const event = events.find((e) => e.type === key);
  if (!event) return null;
  switch (key) {
    case "ingest_done":
      return `${event.traces} traces, ${event.steps} steps`;
    case "split_done":
      return `${event.train} train / ${event.test} held-out`;
    case "index_done":
      return `${event.steps} steps indexed`;
    case "optimize_done":
      return `held-out accuracy ${((event.held_out_accuracy ?? 0) * 100).toFixed(1)}% after ${event.rollouts} rollouts`;
    default:
      return null;
  }
}

function ServeDownPanel({ serveHint }: { serveHint: string }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface-sunk p-5">
      <div className="mono-label">backend offline</div>
      <p className="text-sm text-ink-soft">
        No <code className="font-mono">wmh serve</code> is answering at{" "}
        <code className="font-mono">{API_BASE}</code>. Start one from the repo root:
      </p>
      <pre className="overflow-x-auto rounded-md border border-line bg-surface p-3 font-mono text-xs">
        {serveHint}
      </pre>
      <p className="text-sm text-ink-soft">
        Or build straight from the terminal, no browser needed:{" "}
        <code className="font-mono">uv run wmh build</code> (interactive wizard).
      </p>
    </div>
  );
}

export function BuildFlow({ serveHint }: { serveHint: string }) {
  const [serveUp, setServeUp] = useState<boolean | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tracesPath, setTracesPath] = useState("");
  const [uploading, setUploading] = useState(false);
  const [model, setModel] = useState("us.anthropic.claude-opus-4-8");
  const [budget, setBudget] = useState(50);
  const [events, setEvents] = useState<BuildEvent[]>([]);
  const [status, setStatus] = useState<"idle" | "running" | "succeeded" | "failed">("idle");
  const [error, setError] = useState<string | null>(null);
  // The name the in-flight build was started with — the success link must use this, not the
  // (still-editable) name input the user may have changed while watching progress.
  const [startedName, setStartedName] = useState("");
  const sourceRef = useRef<EventSource | null>(null);
  const pollRef = useRef<number | null>(null);
  const seenRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    isServeUp().then(setServeUp);
    return () => {
      sourceRef.current?.close();
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, []);

  const onUpload = useCallback(async (file: File) => {
    setUploading(true);
    setError(null);
    try {
      setTracesPath(await uploadTraces(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  }, []);

  const kickOff = useCallback(async () => {
    if (status === "running") return; // guard against a double-click launching two builds
    setError(null);
    setEvents([]);
    setStatus("running"); // disable the button now, before the async POST resolves
    const buildName = name.trim();
    try {
      const buildId = await startBuild({
        name: buildName,
        file: tracesPath.trim(),
        description: description.trim(),
        model,
        gepa_budget: budget,
      });
      setStartedName(buildName);
      seenRef.current = new Set();
      const source = openBuildEvents(buildId);
      sourceRef.current = source;
      source.onmessage = (message) => {
        // Dedup by the frame's event id: on reconnect the server resumes from Last-Event-ID, but
        // if a proxy strips that header the stream replays — the id guard keeps events unique.
        const idx = Number(message.lastEventId);
        if (!Number.isNaN(idx)) {
          if (seenRef.current.has(idx)) return;
          seenRef.current.add(idx);
        }
        const event = JSON.parse(message.data) as BuildEvent;
        setEvents((prev) => [...prev, event]);
        if (event.type === "done") {
          setStatus("succeeded");
          source.close();
        } else if (event.type === "error") {
          setStatus("failed");
          setError(event.error ?? "build failed");
          source.close();
        }
      };
      source.onerror = () => {
        // A transient drop (readyState CONNECTING) auto-reconnects and resumes — not a failure.
        // Only when the browser gives up (CLOSED) do we fall back to polling the snapshot until
        // the build reaches a terminal state, so the UI can never get stuck on "running" while
        // the build finishes server-side out of view.
        if (source.readyState !== EventSource.CLOSED || pollRef.current !== null) return;
        source.close();
        pollRef.current = window.setInterval(async () => {
          try {
            const snap = await buildSnapshot(buildId);
            setEvents(snap.events); // authoritative full log — replaces, never duplicates
            if (snap.status !== "running") {
              window.clearInterval(pollRef.current ?? undefined);
              pollRef.current = null;
              setStatus(snap.status);
              if (snap.status === "failed") setError(snap.error ?? "build failed");
            }
          } catch {
            // transient poll failure; keep trying
          }
        }, 2000);
      };
    } catch (e) {
      setStatus("idle");
      setError(
        e instanceof ApiError && e.status === 409
          ? `${e.message} — pick a different name`
          : e instanceof Error
            ? e.message
            : String(e),
      );
    }
  }, [name, tracesPath, description, model, budget, status]);

  if (serveUp === null) {
    return <div className="rounded-lg border border-line p-5 text-sm text-ink-faint">Checking for a local backend…</div>;
  }
  if (!serveUp) return <ServeDownPanel serveHint={serveHint} />;

  const rollouts = events.filter((e) => e.type === "rollout");
  const lastRollout = rollouts[rollouts.length - 1];
  const running = status === "running";
  const formReady = name.trim() && tracesPath.trim() && budget >= 1 && !running;

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <div className="flex flex-col gap-4 rounded-lg border border-line p-5">
        <label className="flex flex-col gap-1 text-sm">
          <span className="mono-label">model name</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={running}
            placeholder="my-agent-env"
            className="rounded-md border border-line px-3 py-2 outline-none focus:border-accent disabled:opacity-50"
          />
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="mono-label">description (goes on the card)</span>
          <input
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            disabled={running}
            placeholder="What environment do these traces capture?"
            className="rounded-md border border-line px-3 py-2 outline-none focus:border-accent disabled:opacity-50"
          />
        </label>
        <div className="flex flex-col gap-1 text-sm">
          <span className="mono-label">traces (OTel GenAI JSONL)</span>
          <div className="flex gap-2">
            <input
              value={tracesPath}
              onChange={(e) => setTracesPath(e.target.value)}
              disabled={running}
              placeholder="/path/to/traces.otel.jsonl"
              className="flex-1 rounded-md border border-line px-3 py-2 font-mono text-xs outline-none focus:border-accent disabled:opacity-50"
            />
            <label className="cursor-pointer rounded-md border border-line px-3 py-2 text-sm text-ink-soft hover:border-accent">
              {uploading ? "uploading…" : "Upload"}
              <input
                type="file"
                accept=".jsonl,.json"
                className="hidden"
                disabled={running}
                onChange={(e) => e.target.files?.[0] && onUpload(e.target.files[0])}
              />
            </label>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="mono-label">serve LLM (Bedrock)</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={running}
              className="rounded-md border border-line px-3 py-2 outline-none focus:border-accent disabled:opacity-50"
            >
              <option value="us.anthropic.claude-opus-4-8">Opus 4.8</option>
              <option value="us.anthropic.claude-haiku-4-5-20251001-v1:0">Haiku 4.5</option>
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="mono-label">GEPA rollout budget</span>
            <input
              type="number"
              min={1}
              step={1}
              value={budget}
              disabled={running}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                setBudget(Number.isNaN(n) ? 0 : n);
              }}
              className="rounded-md border border-line px-3 py-2 outline-none focus:border-accent disabled:opacity-50"
            />
          </label>
        </div>
        <button
          onClick={kickOff}
          disabled={!formReady}
          className="mt-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {status === "running" ? "Building…" : "Build world model"}
        </button>
        {error && (
          <p className="rounded-md border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
            {error}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-3 rounded-lg border border-line p-5">
        <div className="mono-label">progress</div>
        {status === "idle" && events.length === 0 ? (
          <p className="text-sm text-ink-faint">
            Fill in the form and kick off a build to watch it here.
          </p>
        ) : (
          <ol className="flex flex-col gap-2">
            {STAGES.map((stage) => {
              const detail = stageDetail(events, stage.key);
              const reached = detail !== null;
              const optimizing =
                stage.key === "optimize_done" && !reached && lastRollout !== undefined;
              return (
                <li key={stage.key} className="flex items-start gap-3 text-sm">
                  <span
                    className={`mt-1 inline-block h-2 w-2 rounded-full ${
                      reached ? "bg-accent" : optimizing ? "animate-pulse bg-accent-amber" : "bg-line"
                    }`}
                  />
                  <span>
                    <span className={reached ? "" : "text-ink-faint"}>{stage.label}</span>
                    {detail && <span className="block text-xs text-ink-faint">{detail}</span>}
                    {optimizing && lastRollout && (
                      <span className="block text-xs text-ink-faint">
                        rollout {lastRollout.done}/{lastRollout.budget}
                        {lastRollout.score != null && ` — best ${(lastRollout.score * 100).toFixed(1)}%`}
                      </span>
                    )}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
        {status === "succeeded" && (
          <div className="mt-2 rounded-md border border-line bg-surface-sunk p-3 text-sm">
            Build finished.{" "}
            <Link
              href={`/models/${encodeURIComponent(startedName)}`}
              className="text-accent hover:underline"
            >
              Open {startedName} in the playground →
            </Link>
            <p className="mt-1 text-xs text-ink-faint">
              (It is being served live already; run <code className="font-mono">npm run index</code>{" "}
              to add it to this gallery.)
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

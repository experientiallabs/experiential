"use client";

/**
 * The interactive playground embedded in each model's page: create a session against a locally
 * running `wmh serve`, type actions in the `wmh play` grammar, watch observations, the model's
 * scratchpad state, and live usage. Degrades to the exact serve command when no API answers.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  API_BASE,
  createSession,
  isServeUp,
  sessionUsage,
  step,
} from "@/lib/api";
import { parseAction } from "@/lib/parse-action";
import type { Action, EnvState, Observation, RunRecord } from "@/lib/types";

type Turn = { action: Action; observation: Observation };

function actionLabel(action: Action): string {
  return action.kind === "tool_call"
    ? `${action.name} ${Object.keys(action.arguments).length ? JSON.stringify(action.arguments) : ""}`
    : action.content;
}

function ServeDownPanel({ serveHint }: { serveHint: string }) {
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface-sunk p-5">
      <div className="mono-label">playground offline</div>
      <p className="text-sm text-ink-soft">
        No <code className="font-mono">wmh serve</code> backend is answering at{" "}
        <code className="font-mono">{API_BASE}</code>. From the repo root, run:
      </p>
      <pre className="overflow-x-auto rounded-md border border-line bg-surface p-3 font-mono text-xs">
        {serveHint}
      </pre>
      <p className="text-xs text-ink-faint">
        Then reload this page. Your traces and provider keys never leave your machine.
      </p>
    </div>
  );
}

export function Playground({
  name,
  task,
  serveHint,
}: {
  name: string;
  task: string | null;
  serveHint: string;
}) {
  const [serveUp, setServeUp] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [taskText, setTaskText] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [state, setState] = useState<EnvState | null>(null);
  const [usage, setUsage] = useState<RunRecord | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    isServeUp().then(setServeUp);
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns]);

  const start = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const { session_id, state } = await createSession(name, taskText.trim() || null);
      setSessionId(session_id);
      setTurns([]);
      setUsage(null);
      setState(state);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [name, taskText]);

  const send = useCallback(async () => {
    if (!sessionId || !input.trim()) return;
    let action: Action;
    try {
      action = parseAction(input);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const { observation, state } = await step(name, sessionId, action);
      setTurns((prev) => [...prev, { action, observation }]);
      setInput("");
      setState(state);
      setUsage(await sessionUsage(name, sessionId));
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setError("session expired on the server — start a new one");
        setSessionId(null);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  }, [name, sessionId, input]);

  if (serveUp === null) {
    return <div className="rounded-lg border border-line p-5 text-sm text-ink-faint">Checking for a local backend…</div>;
  }
  if (!serveUp) {
    return <ServeDownPanel serveHint={serveHint} />;
  }

  return (
    <section className="flex flex-col gap-4">
      {sessionId && (
        <button
          onClick={() => {
            setSessionId(null);
            setTurns([]);
            setState(null);
            setUsage(null);
            setError(null);
          }}
          className="self-end text-xs text-ink-faint hover:text-ink"
        >
          reset session
        </button>
      )}

      {!sessionId ? (
        <div className="flex flex-col gap-3 rounded-lg border border-line p-5">
          <label className="mono-label" htmlFor="task">
            task (optional — what is the agent trying to do?)
          </label>
          <div className="flex gap-2">
            <input
              id="task"
              value={taskText}
              onChange={(e) => setTaskText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && start()}
              placeholder={task ? `e.g. a ${task} task` : "e.g. look up user u1 and update their booking"}
              className="flex-1 rounded-md border border-line px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              onClick={start}
              disabled={busy}
              className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
            >
              Start session
            </button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div className="flex flex-col rounded-lg border border-line lg:col-span-2">
            <div
              ref={logRef}
              className="flex h-96 flex-col gap-3 overflow-y-auto p-4"
            >
              {turns.length === 0 && (
                <p className="text-sm text-ink-faint">
                  Type an action below — <code className="font-mono">get_user {"{"}&quot;id&quot;: &quot;u1&quot;{"}"}</code>{" "}
                  calls a tool, <code className="font-mono">say hello</code> sends a message.
                </p>
              )}
              {turns.map((turn, i) => (
                <div key={i} className="flex flex-col gap-1">
                  <div className="self-end rounded-md bg-surface-sunk px-3 py-2 font-mono text-xs">
                    {actionLabel(turn.action)}
                  </div>
                  <pre
                    className={`self-start overflow-x-auto whitespace-pre-wrap rounded-md border px-3 py-2 font-mono text-xs ${
                      turn.observation.is_error
                        ? "border-accent-red/40 text-accent-red"
                        : "border-line"
                    }`}
                  >
                    {turn.observation.content}
                  </pre>
                </div>
              ))}
              {busy && <div className="text-xs text-ink-faint">stepping…</div>}
            </div>
            <div className="flex gap-2 border-t border-line p-3">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !busy && send()}
                placeholder='tool_name {"arg": "value"}  ·  say <message>'
                className="flex-1 rounded-md border border-line px-3 py-2 font-mono text-xs outline-none focus:border-accent"
              />
              <button
                onClick={send}
                disabled={busy || !input.trim()}
                className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
              >
                Step
              </button>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <div className="rounded-lg border border-line p-4">
              <div className="mono-label mb-2">scratchpad (model&apos;s memory)</div>
              <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-xs text-ink-soft">
                {state?.scratchpad || "(empty)"}
              </pre>
            </div>
            <div className="rounded-lg border border-line p-4">
              <div className="mono-label mb-2">session usage</div>
              {usage ? (
                <dl className="grid grid-cols-2 gap-y-1 text-xs">
                  <dt className="text-ink-faint">steps</dt>
                  <dd className="text-right tabular-nums">{usage.total.calls}</dd>
                  <dt className="text-ink-faint">tokens</dt>
                  <dd className="text-right tabular-nums">
                    {(usage.total.input_tokens + usage.total.output_tokens).toLocaleString()}
                  </dd>
                  <dt className="text-ink-faint">cost</dt>
                  <dd className="text-right tabular-nums">${usage.total.cost_usd.toFixed(4)}</dd>
                  <dt className="text-ink-faint">wall clock</dt>
                  <dd className="text-right tabular-nums">{usage.duration_seconds.toFixed(1)}s</dd>
                </dl>
              ) : (
                <p className="text-xs text-ink-faint">Step once to see live cost.</p>
              )}
            </div>
          </div>
        </div>
      )}

      {error && (
        <p className="rounded-md border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}
    </section>
  );
}

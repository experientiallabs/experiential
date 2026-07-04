"use client";

/**
 * The interactive playground: create a session against a locally running `wmh serve`, type
 * actions in the `wmh play` grammar (or click a suggestion / start from a recorded scenario),
 * and watch the world model's observations, scratchpad, and live usage.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, createSession, sessionUsage, step } from "@/lib/api";
import { parseAction } from "@/lib/parse-action";
import type { Action, EnvState, IndexEntry, RunRecord, Scenario } from "@/lib/types";

type Turn = { action: string; observation: string; is_error: boolean };

function actionLabel(action: Action): string {
  if (action.kind === "tool_call") {
    return Object.keys(action.arguments).length
      ? `${action.name} ${JSON.stringify(action.arguments)}`
      : action.name;
  }
  return `say ${action.content ?? ""}`;
}

/** A clickable example action; clicking loads it into the input for review before sending. */
function Chip({ label, onPick }: { label: string; onPick: () => void }) {
  return (
    <button
      onClick={onPick}
      title={label}
      className="max-w-full truncate rounded-full border border-line px-3 py-1 font-mono text-xs text-ink-soft transition-colors hover:border-ink hover:text-ink"
    >
      {label}
    </button>
  );
}

export function Playground({ entry }: { entry: IndexEntry }) {
  const suggestions = entry.suggestions;
  const scenarios = entry.scenarios;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [seededScenario, setSeededScenario] = useState<Scenario | null>(null);
  const [taskText, setTaskText] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [state, setState] = useState<EnvState | null>(null);
  const [usage, setUsage] = useState<RunRecord | null>(null);
  const [input, setInput] = useState(suggestions[0] ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // In a seeded scenario, offer that scenario's recorded actions in order; otherwise the model's
  // generic example actions.
  const chips = useMemo(
    () => (seededScenario ? seededScenario.steps.map((s) => s.action_label) : suggestions),
    [seededScenario, suggestions],
  );

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns, busy]);

  const begin = useCallback(
    async (scenario: Scenario | null) => {
      setBusy(true);
      setError(null);
      const task = scenario ? scenario.task : taskText.trim() || null;
      try {
        const { session_id, state } = await createSession(entry.card.name, task);
        setSessionId(session_id);
        setSeededScenario(scenario);
        setTurns([]);
        setUsage(null);
        setState(state);
        setInput(scenario?.steps[0]?.action_label ?? suggestions[0] ?? "");
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [entry.card.name, taskText, suggestions],
  );

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
      const { observation, state } = await step(entry.card.name, sessionId, action);
      setTurns((prev) => [
        ...prev,
        { action: actionLabel(action), observation: observation.content, is_error: observation.is_error },
      ]);
      setInput("");
      setState(state);
      setUsage(await sessionUsage(entry.card.name, sessionId));
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setError("session expired on the server; start a new one");
        setSessionId(null);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setBusy(false);
    }
  }, [entry.card.name, sessionId, input]);

  if (!sessionId) {
    return (
      <div className="flex flex-col gap-5 rounded-xl border border-line p-6">
        <div className="flex flex-col gap-2">
          <label className="mono-label" htmlFor="task">
            task (optional)
          </label>
          <div className="flex flex-col gap-2 sm:flex-row">
            <input
              id="task"
              value={taskText}
              onChange={(e) => setTaskText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && begin(null)}
              placeholder={
                entry.card.task
                  ? `what should the agent try to do in this ${entry.card.task} environment?`
                  : "what should the agent try to do?"
              }
              className="flex-1 rounded-lg border border-line px-3 py-2 text-sm outline-none focus:border-accent"
            />
            <button
              onClick={() => begin(null)}
              disabled={busy}
              className="rounded-lg bg-ink px-5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
            >
              Start session
            </button>
          </div>
        </div>

        {scenarios.length > 0 && (
          <div className="flex flex-col gap-2 border-t border-line pt-4">
            <span className="mono-label">or replay a recorded scenario, open loop</span>
            <div className="flex flex-col gap-2">
              {scenarios.map((s) => (
                <button
                  key={s.id}
                  onClick={() => begin(s)}
                  disabled={busy}
                  className="flex items-center justify-between gap-3 rounded-lg border border-line px-3 py-2 text-left text-sm transition-colors hover:border-accent disabled:opacity-40"
                >
                  <span className="truncate text-ink-soft">{s.label}</span>
                  <span className="mono-label shrink-0">{s.steps.length} steps</span>
                </button>
              ))}
            </div>
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

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs text-ink-faint">
          <span
            className={`inline-block h-1.5 w-1.5 rounded-full ${busy ? "bg-accent-teal" : "bg-live"}`}
          />
          <span className="mono-label">{busy ? "stepping" : "live session"}</span>
          {seededScenario && <span className="truncate">· replaying: {seededScenario.label}</span>}
        </div>
        <button
          onClick={() => {
            setSessionId(null);
            setSeededScenario(null);
            setTurns([]);
            setState(null);
            setUsage(null);
            setError(null);
            setInput(suggestions[0] ?? "");
          }}
          className="text-xs text-ink-faint hover:text-ink"
        >
          new session
        </button>
      </div>

      {/* Transcript */}
      <div
        ref={logRef}
        className="well h-[26rem] overflow-y-auto rounded-xl border border-line bg-surface px-5 py-4 text-sm leading-7"
      >
        {turns.length === 0 ? (
          <p className="text-ink-faint">
            Type an action below, or click a suggestion. <code className="font-mono">bash</code> /{" "}
            <code className="font-mono">get_user</code>-style calls take JSON args;{" "}
            <code className="font-mono">say hello</code> sends a message.
          </p>
        ) : (
          turns.map((turn, i) => (
            <div key={i} className="mb-3">
              <div className="font-mono text-[13px] text-accent">&rsaquo; {turn.action}</div>
              <div
                className={`whitespace-pre-wrap font-mono text-[13px] ${
                  turn.is_error ? "text-accent-red" : "text-ink"
                }`}
              >
                {turn.observation}
              </div>
            </div>
          ))
        )}
        {busy && <div className="text-xs text-ink-faint">stepping...</div>}
      </div>

      {/* Suggestion chips */}
      {chips.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {chips.map((c, i) => (
            <Chip
              key={`${c}-${i}`}
              label={c}
              onPick={() => {
                setInput(c);
                inputRef.current?.focus();
              }}
            />
          ))}
        </div>
      )}

      {/* Input row */}
      <div className="flex gap-2">
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !busy && send()}
          placeholder={'tool_name {"arg": "value"}   ·   say <message>'}
          className="flex-1 rounded-lg border border-line px-3 py-2 font-mono text-xs outline-none focus:border-accent"
        />
        <button
          onClick={send}
          disabled={busy || !input.trim()}
          className="rounded-lg bg-ink px-5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
        >
          Step
        </button>
      </div>

      {error && (
        <p className="rounded-lg border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}

      {/* Scratchpad shows only when the model actually wrote to it. */}
      {state?.scratchpad?.trim() && (
        <details className="rounded-lg border border-line px-4 py-2">
          <summary className="mono-label cursor-pointer select-none">
            scratchpad (model memory)
          </summary>
          <pre className="mt-2 max-h-40 overflow-y-auto whitespace-pre-wrap font-mono text-xs text-ink-soft">
            {state.scratchpad}
          </pre>
        </details>
      )}

      {/* Session usage, collapsed by default so it never crowds the chat. */}
      {usage && (
        <details className="rounded-lg border border-line px-4 py-2">
          <summary className="mono-label flex cursor-pointer select-none items-center justify-between">
            <span>session usage</span>
            <span className="font-mono text-ink-soft">
              {usage.total.calls} steps · ${usage.total.cost_usd.toFixed(4)}
            </span>
          </summary>
          <dl className="mt-2 grid grid-cols-2 gap-y-1 text-xs">
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
        </details>
      )}
    </section>
  );
}

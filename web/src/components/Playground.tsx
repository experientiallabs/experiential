"use client";

/**
 * The interactive playground. Before a session it is a centered composer (type an action, or pick
 * an example tool call or a scenario to replay). In session it is a single transcript: your action
 * appears immediately, then a spinner while the environment responds. Scratchpad and usage are
 * lifted to the parent so they live in the side panel and never grow the page.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, createSession, sessionUsage, step } from "@/lib/api";
import type { LiveState } from "@/components/LivePanels";
import { Spinner } from "@/components/Spinner";
import { parseAction } from "@/lib/parse-action";
import type { Action, IndexEntry, Scenario } from "@/lib/types";

type Turn = { action: string; observation: string | null; is_error: boolean };

function actionLabel(action: Action): string {
  if (action.kind === "tool_call") {
    return Object.keys(action.arguments).length
      ? `${action.name} ${JSON.stringify(action.arguments)}`
      : action.name;
  }
  return `say ${action.content ?? ""}`;
}

/** Render a recorded task (tau tasks are JSON; terminal/swe are plain) as readable text. */
export function readableTask(task: string): string {
  try {
    const p = JSON.parse(task);
    return [p.reason_for_call, p.task_instructions, p.query].filter(Boolean).join("\n\n") || task;
  } catch {
    return task;
  }
}

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

export function Playground({
  entry,
  onLive,
}: {
  entry: IndexEntry;
  onLive: (live: LiveState | null) => void;
}) {
  const name = entry.card.name;
  const suggestions = entry.suggestions;
  const scenarios = entry.scenarios;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [seeded, setSeeded] = useState<Scenario | null>(null);
  const [sessionTask, setSessionTask] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [scratchpad, setScratchpad] = useState("");
  const [usage, setUsage] = useState<LiveState["usage"]>(null);
  const [input, setInput] = useState(suggestions[0] ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const inFlight = useRef(false); // hard guard: exactly one step in flight at a time

  // Report live session read-outs up so the side panel can show them.
  useEffect(() => {
    onLive(sessionId ? { scratchpad, usage } : null);
  }, [sessionId, scratchpad, usage, onLive]);

  // Clear the side panel when the playground unmounts (e.g. switching to the Traces tab), so it
  // never shows a stale session's scratchpad/usage next to something else.
  useEffect(() => () => onLive(null), [onLive]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [turns]);

  const chips = useMemo(
    () => (seeded ? seeded.steps.map((s) => s.action_label) : suggestions),
    [seeded, suggestions],
  );

  const stepAction = useCallback(
    async (sid: string, raw: string) => {
      const text = raw.trim();
      if (!text || inFlight.current) return; // ignore re-entrant submits (key-repeat, click+Enter)
      let action: Action;
      try {
        action = parseAction(text);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return;
      }
      inFlight.current = true;
      setError(null);
      setInput("");
      // Show the action immediately with a pending observation (the spinner renders for it).
      setTurns((prev) => [...prev, { action: actionLabel(action), observation: null, is_error: false }]);
      setBusy(true);
      try {
        const { observation, state } = await step(name, sid, action);
        setTurns((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            action: next[next.length - 1].action,
            observation: observation.content,
            is_error: observation.is_error,
          };
          return next;
        });
        setScratchpad(state.scratchpad);
        // Usage is best-effort: a failure here must not undo the step the user just saw answered.
        try {
          setUsage(await sessionUsage(name, sid));
        } catch {
          // leave the previous usage in place
        }
      } catch (e) {
        // Only drop the turn if it is still pending (its observation never arrived).
        setTurns((prev) =>
          prev.length && prev[prev.length - 1].observation === null ? prev.slice(0, -1) : prev,
        );
        if (e instanceof ApiError && e.status === 404) {
          setError("session expired on the server; start a new one");
          setSessionId(null);
        } else {
          setError(e instanceof Error ? e.message : String(e));
        }
      } finally {
        inFlight.current = false;
        setBusy(false);
      }
    },
    [name],
  );

  const begin = useCallback(
    async (scenario: Scenario | null, firstAction?: string) => {
      setBusy(true);
      setError(null);
      const task = scenario ? scenario.task : null;
      try {
        const { session_id, state } = await createSession(name, task);
        setSessionId(session_id);
        setSeeded(scenario);
        setSessionTask(task);
        setTurns([]);
        setScratchpad(state.scratchpad);
        setUsage(null);
        setInput(scenario?.steps[0]?.action_label ?? suggestions[0] ?? "");
        if (firstAction) await stepAction(session_id, firstAction);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [name, suggestions, stepAction],
  );

  // Pre-session: a centered composer with example tool calls and scenario replays below it.
  if (!sessionId) {
    return (
      <div className="flex flex-col items-center gap-8 py-4">
        <div className="flex w-full max-w-2xl flex-col gap-3">
          <p className="text-center text-sm text-ink-soft">
            Send a tool call or message to the{" "}
            <span className="text-ink">{entry.card.task ?? "world model"}</span> environment and
            watch how it responds.
          </p>
          <div className="flex gap-2">
            <input
              autoFocus
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && input.trim() && begin(null, input)}
              placeholder={'tool_name {"arg": "value"}   ·   say <message>'}
              className="flex-1 rounded-lg border border-line px-4 py-2.5 font-mono text-xs outline-none focus:border-accent"
            />
            <button
              onClick={() => begin(null, input)}
              disabled={busy || !input.trim()}
              className="rounded-lg bg-ink px-5 py-2.5 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
            >
              Send
            </button>
          </div>
        </div>

        {suggestions.length > 0 && (
          <div className="flex w-full max-w-2xl flex-col items-center gap-2">
            <span className="mono-label">example tool calls</span>
            <div className="flex flex-wrap justify-center gap-2">
              {suggestions.map((s, i) => (
                <Chip key={`${s}-${i}`} label={s} onPick={() => begin(null, s)} />
              ))}
            </div>
          </div>
        )}

        {scenarios.length > 0 && (
          <div className="flex w-full max-w-2xl flex-col gap-2">
            <span className="mono-label text-center">replay a scenario</span>
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
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-live" />
          <span className="mono-label">live session</span>
          {seeded && <span className="truncate">· replaying: {seeded.label}</span>}
        </div>
        <button
          onClick={() => {
            setSessionId(null);
            setSeeded(null);
            setSessionTask(null);
            setTurns([]);
            setScratchpad("");
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
        className="well h-[24rem] overflow-y-auto rounded-xl border border-line bg-surface px-5 py-4 text-sm leading-7"
      >
        {sessionTask && (
          <div className="mb-3 rounded-lg border border-line bg-surface-sunk px-3 py-2">
            <div className="mono-label mb-1">task</div>
            <div className="max-h-28 overflow-y-auto whitespace-pre-wrap text-[13px] text-ink-soft">
              {readableTask(sessionTask)}
            </div>
          </div>
        )}
        {turns.length === 0 && !busy ? (
          <p className="text-ink-faint">
            Type an action below, or click a suggestion. <code className="font-mono">bash</code> /{" "}
            <code className="font-mono">get_user</code>-style calls take JSON args;{" "}
            <code className="font-mono">say hello</code> sends a message.
          </p>
        ) : (
          turns.map((turn, i) => (
            <div key={i} className="mb-3">
              <div className="font-mono text-[13px] text-accent">&rsaquo; {turn.action}</div>
              {turn.observation === null ? (
                <div className="flex items-center gap-2 text-[13px] text-ink-faint">
                  <Spinner /> getting environment response
                </div>
              ) : (
                <div
                  className={`whitespace-pre-wrap font-mono text-[13px] ${
                    turn.is_error ? "text-accent-red" : "text-ink"
                  }`}
                >
                  {turn.observation}
                </div>
              )}
            </div>
          ))
        )}
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
          onKeyDown={(e) => e.key === "Enter" && !busy && sessionId && stepAction(sessionId, input)}
          placeholder={'tool_name {"arg": "value"}   ·   say <message>'}
          className="flex-1 rounded-lg border border-line px-3 py-2 font-mono text-xs outline-none focus:border-accent"
        />
        <button
          onClick={() => sessionId && stepAction(sessionId, input)}
          disabled={busy || !input.trim()}
          className="rounded-lg bg-ink px-5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-85 disabled:opacity-40"
        >
          Send
        </button>
      </div>

      {error && (
        <p className="rounded-lg border border-accent-red/40 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}
    </section>
  );
}

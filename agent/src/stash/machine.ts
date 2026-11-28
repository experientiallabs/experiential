/**
 * The prompt stash as a pure state machine: what Ctrl+S, a submit, and the `/stash` command do to
 * the composer and to the stack of stashed prompts.
 *
 * Cardinal rule: no sequence of operations loses or duplicates composed text. Every function here
 * moves a draft between exactly two places (the composer and the top of the stack) and never
 * writes over a composer that holds text, so a draft is always in one place: the composer, the
 * stack, or sent.
 *
 * The stash is a stack, like `git stash`:
 * - Ctrl+S with text in the composer pushes it and clears the composer, even when something is
 *   already stashed ("double save"): both drafts stay stashed and the composer is free, which is
 *   what pressing Ctrl+S asks for.
 * - Ctrl+S on an empty composer pops the most recent draft back into it, so two quick presses are
 *   a round trip that leaves the draft exactly where it was.
 * - After a submit, the most recent draft pops back automatically, but only into an empty
 *   composer: text typed since is never overwritten (the draft stays stashed and the caller says so).
 */

/** A cursor position in the composer's lines (as the composer shows them, paste markers collapsed). */
export interface Cursor {
  line: number;
  col: number;
}

/** What the composer holds. */
export interface Draft {
  /** The full text exactly as it would be sent: paste markers expanded, newlines and spaces kept. */
  text: string;
  /** Where the cursor was, in the coordinates of `collapsed.text` when present, else of `text`. */
  cursor?: Cursor;
  /**
   * The form the composer showed when large pastes were collapsed to `[paste #N ...]` markers,
   * with each marker's content. Presentation only: `text` is the truth, and a collapsed form
   * that no longer expands to `text` is ignored on restore.
   */
  collapsed?: { text: string; pastes: [number, string][] };
  /**
   * The composer's undo history for this draft, opaque and in memory only (never saved to disk).
   * It travels with the draft, so undo after a stash cannot bring the stashed text back a second
   * time, and undo after a restore walks back through the draft's own edits.
   */
  undo?: readonly unknown[];
  /**
   * Only when reading the composer: the draft hidden while an earlier message from history is
   * shown (Up on an empty prompt). Stashing the shown message brings this draft back.
   */
  hidden?: Draft;
}

/** A stashed draft. */
export interface StashEntry extends Draft {
  /** Unique id, so a draft adopted twice after a crash is kept once. */
  id: string;
  /** When it was stashed (ms since the epoch); orders drafts adopted from several files. */
  stashedAt: number;
}

/** Whether text counts as nothing to stash: empty or only whitespace (Pi never sends it either). */
export function isBlank(text: string): boolean {
  return text.trim().length === 0;
}

export type StashOutcome =
  /** The composer's draft was pushed and the composer cleared; `depth` drafts are now stashed. */
  | { kind: "stashed"; entry: StashEntry; depth: number }
  /** The top draft was popped into the (empty) composer; `depth` drafts remain stashed. */
  | { kind: "restored"; entry: StashEntry; depth: number }
  /** A draft is waiting but the composer has text, so nothing moved. */
  | { kind: "kept"; depth: number }
  /** Nothing stashed and nothing to stash. */
  | { kind: "nothing" };

export interface Transition {
  stack: StashEntry[];
  /** The composer's new content: a draft to show, `null` to clear it, `undefined` to leave it alone. */
  composer: Draft | null | undefined;
  outcome: StashOutcome;
}

/** Builds a stack entry for a draft (id and time come from the caller so tests are deterministic). */
export type EntryFactory = (draft: Draft) => StashEntry;

/**
 * Ctrl+S: stash a non-blank composer, else restore the most recent stash into it. While a message
 * recalled from history is shown, that message is stashed and the draft it hid comes back.
 */
export function pressStash(stack: readonly StashEntry[], composer: Draft, makeEntry: EntryFactory): Transition {
  if (!isBlank(composer.text)) {
    const { hidden, ...shown } = composer;
    if (hidden !== undefined) delete shown.undo; // The undo history belongs to the hidden draft.
    const entry = makeEntry(shown);
    const next = [...stack, entry];
    const rest = hidden === undefined || isBlank(hidden.text) ? null : hidden;
    return { stack: next, composer: rest, outcome: { kind: "stashed", entry, depth: next.length } };
  }
  return popInto(stack, composer);
}

/** After a submit: bring the most recent stash back, but only into an empty composer. */
export function afterSubmit(stack: readonly StashEntry[], composer: Draft): Transition {
  if (stack.length === 0) return { stack: [...stack], composer: undefined, outcome: { kind: "nothing" } };
  return popInto(stack, composer);
}

/** `/stash pop`: the same as Ctrl+S on the composer as it is. */
export function popStash(stack: readonly StashEntry[], composer: Draft): Transition {
  return popInto(stack, composer);
}

/** `/stash <text>`: push a draft without touching the composer. Blank text is not stashed. */
export function pushDraft(stack: readonly StashEntry[], draft: Draft, makeEntry: EntryFactory): Transition {
  if (isBlank(draft.text)) return { stack: [...stack], composer: undefined, outcome: { kind: "nothing" } };
  const entry = makeEntry(draft);
  const next = [...stack, entry];
  return { stack: next, composer: undefined, outcome: { kind: "stashed", entry, depth: next.length } };
}

/** `/stash drop`: discard the most recent draft (an explicit, named discard). */
export function dropStash(stack: readonly StashEntry[]): { stack: StashEntry[]; dropped: StashEntry | undefined } {
  return { stack: stack.slice(0, -1), dropped: stack.at(-1) };
}

/**
 * Merge drafts adopted from earlier sessions into the stack: one copy per id, oldest first, so the
 * draft stashed most recently is on top.
 */
export function mergeStacks(stack: readonly StashEntry[], adopted: readonly StashEntry[]): StashEntry[] {
  const byId = new Map<string, StashEntry>();
  for (const entry of [...stack, ...adopted]) if (!byId.has(entry.id)) byId.set(entry.id, entry);
  return [...byId.values()].sort((a, b) => a.stashedAt - b.stashedAt);
}

function popInto(stack: readonly StashEntry[], composer: Draft): Transition {
  const top = stack.at(-1);
  if (top === undefined) return { stack: [], composer: undefined, outcome: { kind: "nothing" } };
  if (!isBlank(composer.text)) return { stack: [...stack], composer: undefined, outcome: { kind: "kept", depth: stack.length } };
  const next = stack.slice(0, -1);
  return { stack: next, composer: draftOf(top), outcome: { kind: "restored", entry: top, depth: next.length } };
}

/** The draft part of an entry. */
export function draftOf(entry: StashEntry): Draft {
  const draft: Draft = { text: entry.text };
  if (entry.cursor !== undefined) draft.cursor = entry.cursor;
  if (entry.collapsed !== undefined) draft.collapsed = entry.collapsed;
  if (entry.undo !== undefined) draft.undo = entry.undo;
  return draft;
}

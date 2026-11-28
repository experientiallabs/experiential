/**
 * Reading and writing the composer (the prompt editor at the bottom of the terminal UI) as a
 * {@link Draft}.
 *
 * Text goes through the editor's public methods. A few details have no public method, so they are
 * reached on the pinned editor's own fields, each behind a shape check that falls back to plain
 * text when a later Pi release changes it (`composer.test.ts` runs against the real editor):
 *
 * - the cursor position, so a restored draft puts the cursor back where it was;
 * - the contents of collapsed `[paste #N ...]` markers, so a restored large paste shows collapsed
 *   again instead of expanded (the expanded text is stored either way, so nothing depends on it);
 * - the draft hidden while a message recalled from history is shown, so stashing the recalled
 *   message gives the hidden draft back instead of dropping it;
 * - the undo history, which moves with the draft: a stash leaves the composer with no undo
 *   history (undo cannot bring the stashed text back while it is also stashed), and a restore
 *   puts the draft's own history back on top of the composer's.
 */

import type { Cursor, Draft } from "./machine.js";

/** The editor methods the stash uses (Pi's editor, or any custom editor built on it). */
export interface EditorLike {
  getText(): string;
  setText(text: string): void;
  getExpandedText?(): string;
  getCursor?(): Cursor;
}

/** Read and write the composer. */
export interface Composer {
  read(): Draft;
  /** Show `draft`, or clear the composer for `null`. */
  write(draft: Draft | null): void;
}

/** The private editor fields this module reads, each optional because none is a contract. */
interface EditorInternals {
  state?: { lines?: unknown; cursorLine?: unknown; cursorCol?: unknown };
  pastes?: unknown;
  pasteCounter?: unknown;
  undoStack?: { stack?: unknown };
  historyIndex?: unknown;
  historyDraft?: unknown;
  setCursorCol?: unknown;
}

/**
 * Whether a focused component is the main prompt editor: Pi's editor (or a custom editor built on
 * it), not a dialog, picker, or the ask_user answer box.
 */
export function isMainEditor(component: unknown): component is EditorLike {
  if (typeof component !== "object" || component === null) return false;
  const candidate = component as Record<string, unknown>;
  return (
    typeof candidate.getText === "function" &&
    typeof candidate.setText === "function" &&
    candidate.actionHandlers instanceof Map &&
    "onExtensionShortcut" in candidate
  );
}

/** A composer over the editor itself, keeping the cursor, collapsed pastes, and undo history. */
export function editorComposer(editor: EditorLike): Composer {
  const internals = editor as unknown as EditorInternals;
  return {
    read() {
      const shown = editor.getText();
      const pastes = pasteMap(internals);
      const draft = draftFrom(shown, editor.getExpandedText?.() ?? shown, editor.getCursor?.(), pastes);
      const undo = undoSnapshots(internals);
      if (undo !== undefined) draft.undo = [...undo];
      const hidden = hiddenDraft(internals, pastes);
      if (hidden !== undefined) draft.hidden = hidden;
      return draft;
    },
    write(draft) {
      const before = undoSnapshots(internals)?.slice();
      if (draft === null) {
        editor.setText("");
      } else if (draft.collapsed !== undefined && pasteMap(internals) !== undefined && expandMarkers(draft.collapsed.text, draft.collapsed.pastes) === draft.text) {
        editor.setText(draft.collapsed.text);
        // setText cleared the paste table; give the markers their contents back.
        const pastes = pasteMap(internals)!;
        for (const [id, content] of draft.collapsed.pastes) pastes.set(id, content);
        internals.pasteCounter = Math.max(0, ...draft.collapsed.pastes.map(([id]) => id));
        if (editor.getExpandedText !== undefined && editor.getExpandedText() !== draft.text) editor.setText(draft.text);
        else placeCursor(internals, draft.cursor);
      } else {
        editor.setText(draft.text);
        // A cursor recorded over collapsed markers does not fit the expanded text.
        if (draft.collapsed === undefined) placeCursor(internals, draft.cursor);
      }
      // setText recorded the state it replaced as an undo step; that step is the stash itself.
      const stack = undoSnapshots(internals);
      if (stack !== undefined && before !== undefined) {
        stack.length = 0;
        if (draft !== null) stack.push(...before, ...(draft.undo ?? []).map((snapshot) => structuredClone(snapshot)));
      }
    }
  };
}

/** A composer over the extension UI's text methods, for when the editor itself is out of reach. */
export function textComposer(ui: { getEditorText(): string; setEditorText(text: string): void }): Composer {
  return {
    read: () => ({ text: ui.getEditorText() }),
    write: (draft) => ui.setEditorText(draft === null ? "" : draft.text)
  };
}

/** Replace each `[paste #N ...]` marker with its content, as the editor does on submit. */
export function expandMarkers(text: string, pastes: readonly [number, string][]): string {
  let result = text;
  for (const [id, content] of pastes) {
    result = result.replace(new RegExp(`\\[paste #${id}( (\\+\\d+ lines|\\d+ chars))?\\]`, "g"), () => content);
  }
  return result;
}

function draftFrom(shown: string, text: string, cursor: unknown, pastes: Map<number, string> | undefined): Draft {
  const draft: Draft = { text };
  if (isCursor(cursor)) draft.cursor = { line: cursor.line, col: cursor.col };
  if (shown !== text && pastes !== undefined) draft.collapsed = { text: shown, pastes: [...pastes.entries()] };
  return draft;
}

/** The draft the editor keeps aside while it shows a message from history. */
function hiddenDraft(internals: EditorInternals, pastes: Map<number, string> | undefined): Draft | undefined {
  if (typeof internals.historyIndex !== "number" || internals.historyIndex < 0) return undefined;
  const saved = internals.historyDraft as { lines?: unknown; cursorLine?: unknown; cursorCol?: unknown } | null | undefined;
  if (saved === null || saved === undefined || !Array.isArray(saved.lines) || !saved.lines.every((line) => typeof line === "string")) return undefined;
  const shown = (saved.lines as string[]).join("\n");
  const text = pastes === undefined ? shown : expandMarkers(shown, [...pastes.entries()]);
  return draftFrom(shown, text, { line: saved.cursorLine, col: saved.cursorCol }, pastes);
}

function pasteMap(internals: EditorInternals): Map<number, string> | undefined {
  return internals.pastes instanceof Map ? (internals.pastes as Map<number, string>) : undefined;
}

function undoSnapshots(internals: EditorInternals): unknown[] | undefined {
  const stack = internals.undoStack?.stack;
  return Array.isArray(stack) ? stack : undefined;
}

function isCursor(value: unknown): value is Cursor {
  const cursor = value as Cursor | undefined;
  return Number.isInteger(cursor?.line) && Number.isInteger(cursor?.col) && cursor!.line >= 0 && cursor!.col >= 0;
}

/** Put the cursor at `cursor` when it lies inside the current lines; otherwise leave it at the end. */
function placeCursor(internals: EditorInternals, cursor: Cursor | undefined): void {
  const state = internals.state;
  if (cursor === undefined || state === undefined || !Array.isArray(state.lines)) return;
  if (typeof state.cursorLine !== "number" || typeof state.cursorCol !== "number") return;
  const line = state.lines[cursor.line];
  if (typeof line !== "string" || cursor.col > line.length) return;
  state.cursorLine = cursor.line;
  if (typeof internals.setCursorCol === "function") (internals.setCursorCol as (col: number) => void).call(internals, cursor.col);
  else state.cursorCol = cursor.col;
}

/**
 * exp-code's own key bindings, listed by `/hotkeys` beside Pi's (pi-ux-patches.ts reads this
 * process-wide registry). A key an extension handles on raw terminal input, such as the prompt
 * stash's Ctrl+S, is not a Pi shortcut, so Pi's list would otherwise leave it out.
 */

/** The registry key; pi-ux-patches.ts names the same symbol in Pi's source. */
const HOTKEYS = Symbol.for("exp-code.hotkeys");

export interface HotkeyRow {
  /** The key as shown, such as `Ctrl+S`. */
  key: string;
  action: string;
}

function registry(): Map<string, HotkeyRow> {
  const host = process as unknown as Record<symbol, Map<string, HotkeyRow> | undefined>;
  host[HOTKEYS] ??= new Map();
  return host[HOTKEYS];
}

/** List (or, with no rows, unlist) the keys of one exp-code feature in `/hotkeys`. */
export function setHotkeys(feature: string, rows: HotkeyRow[]): void {
  const table = registry();
  for (const id of [...table.keys()]) if (id.startsWith(`${feature}\0`)) table.delete(id);
  rows.forEach((row, index) => table.set(`${feature}\0${index}`, row));
}

/** The rows `/hotkeys` shows, in registration order. */
export function hotkeyRows(): HotkeyRow[] {
  return [...registry().values()];
}

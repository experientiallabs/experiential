/**
 * exp-code's own identity in the terminal UI: the startup header with Experiential's pixel mark and
 * exp-code's version, and the notice when a newer exp-code release exists. (Pi's own name in
 * its help and messages is patched at load time, see pi-patches.ts.)
 *
 * The header replaces the upstream one through the extension header slot, so the key hints and
 * the expand toggle (`ctrl+o`) behave exactly like the built-in header.
 */

import { type ExtensionFactory, keyHint, keyText, rawKeyHint, type Theme } from "@earendil-works/pi-coding-agent";
import { isAppleTerminalSession, Text } from "@earendil-works/pi-tui";

/**
 * Experiential's mark (the "XP" monogram) as a 12 by 12 pixel grid, `#` for ink. Two pixel rows
 * share one terminal line through half blocks, so it draws as 12 columns by 6 lines.
 */
export const EXPERIENTIAL_MARK = [
  "#........###",
  ".##.....#..#",
  "...#...#...#",
  "....#.###.#.",
  "....#.##.#..",
  "....#####...",
  "....###.....",
  "....###.....",
  "....#.#.....",
  "...##..#....",
  "..##....#...",
  "##.......##."
] as const;

/** The mark as half-block text lines (no color). */
export function markLines(grid: readonly string[] = EXPERIENTIAL_MARK): string[] {
  const lines: string[] = [];
  for (let y = 0; y < grid.length; y += 2) {
    const top = grid[y] ?? "";
    const bottom = grid[y + 1] ?? "";
    let line = "";
    for (let x = 0; x < top.length; x++) {
      const upper = top[x] === "#";
      const lower = bottom[x] === "#";
      line += upper && lower ? "█" : upper ? "▀" : lower ? "▄" : " ";
    }
    lines.push(line);
  }
  return lines;
}

/** Minimal theme surface the header paints with. */
export type HeaderTheme = Pick<Theme, "fg" | "bold">;

export interface HeaderContent {
  version: string;
  /** Draw the pixel mark; terminals that misdraw half blocks get a text-only header. */
  showMark: boolean;
}

/** The title lines shown beside the mark (or alone without it). */
function titleLines(theme: HeaderTheme, version: string): string[] {
  return [`${theme.bold(theme.fg("accent", "exp-code"))} ${theme.fg("dim", `v${version}`)}`, theme.fg("muted", "by Experiential")];
}

/** The top of the header: the mark with the title beside its first lines. */
export function headerTop(theme: HeaderTheme, content: HeaderContent): string {
  const title = titleLines(theme, content.version);
  if (!content.showMark) return title.join("\n");
  return markLines()
    .map((line, index) => {
      const mark = theme.fg("accent", line);
      const beside = title[index];
      return beside === undefined ? mark : `${mark}  ${beside}`;
    })
    .join("\n");
}

const ONBOARDING = "exp-code can explain its own features. Ask it how to use it, or how to extend it with your own tools and skills.";

function compactHints(theme: HeaderTheme): string {
  return [
    keyHint("app.interrupt", "interrupt"),
    rawKeyHint(`${keyText("app.clear")}/${keyText("app.exit")}`, "clear/exit"),
    rawKeyHint("/", "commands"),
    rawKeyHint("!", "bash"),
    keyHint("app.tools.expand", "more")
  ].join(theme.fg("muted", " · "));
}

function expandedHints(): string {
  return [
    keyHint("app.interrupt", "to interrupt"),
    keyHint("app.clear", "to clear"),
    rawKeyHint(`${keyText("app.clear")} twice`, "to exit"),
    keyHint("app.exit", "to exit (empty)"),
    keyHint("app.suspend", "to suspend"),
    keyHint("tui.editor.deleteToLineEnd", "to delete to end"),
    keyHint("app.thinking.cycle", "to cycle thinking level"),
    rawKeyHint(`${keyText("app.model.cycleForward")}/${keyText("app.model.cycleBackward")}`, "to cycle models"),
    keyHint("app.model.select", "to select model"),
    keyHint("app.tools.expand", "to expand tools"),
    keyHint("app.thinking.toggle", "to expand thinking"),
    keyHint("app.editor.external", "for external editor"),
    rawKeyHint("/", "for commands"),
    rawKeyHint("!", "to run bash"),
    rawKeyHint("!!", "to run bash (no context)"),
    keyHint("app.message.followUp", "to queue follow-up"),
    keyHint("app.message.dequeue", "to edit all queued messages"),
    keyHint("app.clipboard.pasteImage", "to paste files on macOS, images, or text"),
    rawKeyHint("drop files", "to attach")
  ].join("\n");
}

/** The whole header text, collapsed or expanded. */
export function headerText(theme: HeaderTheme, content: HeaderContent, expanded: boolean): string {
  const top = headerTop(theme, content);
  const onboarding = theme.fg("dim", ONBOARDING);
  if (expanded) return `${top}\n\n${expandedHints()}\n\n${onboarding}`;
  const toggle = theme.fg("dim", `Press ${keyText("app.tools.expand")} to show full startup help and loaded resources.`);
  return `${top}\n\n${compactHints(theme)}\n${toggle}\n\n${onboarding}`;
}

/** The startup header component: rebuilt on theme changes, expandable like the built-in one. */
export class ExpCodeHeader extends Text {
  private expanded = false;
  private stale = true;

  constructor(
    private readonly theme: HeaderTheme,
    private readonly content: HeaderContent
  ) {
    super("", 1, 0);
  }

  setExpanded(expanded: boolean): void {
    this.expanded = expanded;
    this.invalidate();
  }

  override invalidate(): void {
    super.invalidate();
    this.stale = true;
  }

  override render(width: number): string[] {
    if (this.stale) {
      this.stale = false;
      this.setText(headerText(this.theme, this.content, this.expanded));
    }
    return super.render(width);
  }
}

/**
 * The extension that installs exp-code's header in the interactive terminal UI and shows the
 * update notice (once, when `updateNotice` resolves to a line).
 */
export function createBrandingExtension(version: string, updateNotice: Promise<string | undefined> = Promise.resolve(undefined)): ExtensionFactory {
  return (pi) => {
    let noticeShown = false;
    pi.on("session_start", (_event, ctx) => {
      if (!ctx.hasUI) return;
      ctx.ui.setHeader((_tui, theme) => new ExpCodeHeader(theme, { version, showMark: !isAppleTerminalSession() }));
      if (noticeShown) return;
      noticeShown = true;
      void updateNotice.then((line) => {
        if (line !== undefined) ctx.ui.notify(line, "info");
      });
    });
  };
}

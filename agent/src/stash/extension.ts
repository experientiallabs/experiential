/**
 * The prompt stash in the terminal UI: Ctrl+S sets the prompt you are writing aside, so you can
 * send something else first, and brings it back. The same as Claude Code's Ctrl+S.
 *
 * - Ctrl+S with text in the prompt stashes it (text, cursor, collapsed pastes; images and files are
 *   their paths in the text) and clears the prompt. With something already stashed, it stashes
 *   again: drafts stack, the newest on top, and nothing is overwritten.
 * - Ctrl+S on an empty prompt brings the newest stashed draft back.
 * - After you send a message (including a steer while the agent works, a follow-up, or a `!`
 *   command), the newest draft comes back into the prompt by itself, unless you have already
 *   started typing again: then it stays stashed and a hint says so. Built-in commands that open a
 *   picker (/model, /settings, ...) and `/stash` itself do not count as sending.
 * - `/stash` does the same without the key, for terminals where Ctrl+S is taken; see `/stash help`.
 * - The stash belongs to the working directory, not the session: `/new`, `/resume` and `/fork`
 *   keep it. It is saved to disk on every change and survives exit and crashes; the next exp-code
 *   started in that folder puts the draft back in the prompt (see store.ts for how two instances
 *   in one folder stay apart).
 *
 * Ctrl+S reaches exp-code even where the terminal would use it for XOFF flow control: the TUI puts
 * the terminal in raw mode, which turns IXON off while exp-code runs and restores it on exit.
 */

import { randomUUID } from "node:crypto";
import { join } from "node:path";
import type { ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { type Component, isKeyRelease, type KeyId, isKeyRepeat, matchesKey, type TUI, truncateToWidth } from "@earendil-works/pi-tui";
import { setHotkeys } from "../cli/hotkeys.js";
import { type Composer, editorComposer, isMainEditor, textComposer } from "./composer.js";
import { keyLabel, loadStashKeys } from "./keys.js";
import {
  afterSubmit,
  dropStash,
  type EntryFactory,
  mergeStacks,
  popStash,
  pressStash,
  pushDraft,
  type StashEntry,
  type StashOutcome,
  type Transition
} from "./machine.js";
import { StashStore, type StashStoreOptions } from "./store.js";

/** The widget key of the stash indicator below the prompt. */
export const STASH_WIDGET_KEY = "exp-code-prompt-stash";

export interface PromptStashOptions {
  /** exp-code's agent directory: stash files go to `<agentDir>/prompt-stash`, keys come from its keybindings.json. */
  agentDir: string;
  /** Override the stash keys (tests); otherwise read from keybindings.json. */
  keys?: string[];
  /** Store options for tests (pid, host, liveness). */
  store?: Omit<StashStoreOptions, "dir" | "cwd">;
  /** Clock and id source for tests. */
  now?: () => number;
  newId?: () => string;
}

/** One working directory's stash in this process. */
class FolderStash {
  stack: StashEntry[];
  /** Drafts adopted from exited sessions that have not been offered back yet. */
  adoptedPending: number;
  readonly store: StashStore;
  saveError: string | undefined;

  constructor(store: StashStore) {
    this.store = store;
    try {
      const adoption = store.adoptOrphans(store.load(), mergeStacks);
      this.stack = adoption.entries;
      this.adoptedPending = adoption.files > 0 ? adoption.entries.length : 0;
    } catch (error) {
      // Claimed files stay on disk under this process's name and are adopted by a later start.
      this.stack = [];
      this.adoptedPending = 0;
      this.saveError = (error as Error).message;
    }
  }

  /** Apply a transition's stack and persist it. Returns a warning when the disk write failed. */
  commit(stack: StashEntry[]): string | undefined {
    this.stack = stack;
    try {
      this.store.save(stack);
      this.saveError = undefined;
    } catch (error) {
      this.saveError = (error as Error).message;
      return `Could not save the stash to disk (${this.saveError}); it is kept until exp-code exits.`;
    }
    return undefined;
  }
}

/** Each factory's per-folder stacks, for {@link stashedTexts}. */
const stacksByFactory = new WeakMap<ExtensionFactory, Map<string, FolderStash>>();

/** The texts a stash extension holds for a folder, oldest first (tests read the live stack with it). */
export function stashedTexts(factory: ExtensionFactory, cwd: string): string[] {
  return stacksByFactory.get(factory)?.get(cwd)?.stack.map((entry) => entry.text) ?? [];
}

/** Create the stash extension. One instance serves every session the CLI process opens. */
export function createPromptStashExtension(options: PromptStashOptions): ExtensionFactory {
  const folders = new Map<string, FolderStash>();
  const keys = options.keys ?? loadStashKeys(options.agentDir);
  const label = keys[0] === undefined ? undefined : keyLabel(keys[0]);
  const now = options.now ?? Date.now;
  const newId = options.newId ?? randomUUID;
  const makeEntry: EntryFactory = (draft) => ({ ...draft, id: newId(), stashedAt: now() });

  const folderFor = (cwd: string): FolderStash => {
    let folder = folders.get(cwd);
    if (folder === undefined) {
      folder = new FolderStash(new StashStore({ ...options.store, dir: join(options.agentDir, "prompt-stash"), cwd }));
      folders.set(cwd, folder);
    }
    return folder;
  };

  const factory: ExtensionFactory = (pi) => {
    // The live session's view of the UI; set on session_start in the terminal UI only.
    let live: { ctx: ExtensionContext; folder: FolderStash; tui: TUI | undefined; unsubscribe: () => void } | undefined;

    const composerOf = (ctx: ExtensionContext, tui: TUI | undefined): Composer => {
      const focused = focusedComponent(tui);
      return isMainEditor(focused) ? editorComposer(focused) : textComposer(ctx.ui);
    };

    /** Run a transition against the composer and report it. */
    const apply = (ctx: ExtensionContext, folder: FolderStash, composer: Composer, transition: Transition, quiet: StashOutcome["kind"][] = []): void => {
      // Write the composer first: if that throws, the stack is unchanged and nothing is lost.
      if (transition.composer !== undefined) composer.write(transition.composer);
      const warning = folder.commit(transition.stack);
      if (!quiet.includes(transition.outcome.kind)) ctx.ui.notify(describe(transition.outcome, label), "info");
      if (warning !== undefined) ctx.ui.notify(warning, "warning");
      live?.tui?.requestRender();
    };

    const onKey = (data: string): { consume: true } | undefined => {
      if (live === undefined || isKeyRelease(data) || isKeyRepeat(data)) return undefined;
      if (!keys.some((key) => matchesKey(data, key as KeyId))) return undefined;
      const focused = focusedComponent(live.tui);
      // Anywhere but the prompt (a picker, a dialog, the ask_user box) the key keeps its own meaning.
      if (!isMainEditor(focused)) return undefined;
      const composer = editorComposer(focused);
      apply(live.ctx, live.folder, composer, pressStash(live.folder.stack, composer.read(), makeEntry));
      return { consume: true };
    };

    const restoreAfterSubmit = (ctx: ExtensionContext): void => {
      if (live === undefined || live.folder.stack.length === 0) return;
      const composer = composerOf(ctx, live.tui);
      apply(ctx, live.folder, composer, afterSubmit(live.folder.stack, composer.read()), ["nothing"]);
    };

    pi.registerCommand("stash", {
      description: `Set the prompt aside and get it back (${label ?? "no key"}): /stash <text>, /stash pop, /stash list, /stash drop`,
      getArgumentCompletions: (prefix) => {
        const items = STASH_SUBCOMMANDS.filter((item) => item.value.startsWith(prefix.trim()));
        return items.length > 0 ? items : null;
      },
      handler: async (args, ctx) => {
        if (live === undefined || ctx.mode !== "tui") {
          if (ctx.hasUI) ctx.ui.notify("The prompt stash works in the interactive terminal UI.", "warning");
          return;
        }
        const folder = live.folder;
        const composer = composerOf(ctx, live.tui);
        const sub = args.trim();
        switch (sub) {
          case "":
          case "pop":
            apply(ctx, folder, composer, popStash(folder.stack, composer.read()));
            return;
          case "list":
            ctx.ui.notify(listing(folder.stack), "info");
            return;
          case "drop": {
            const { stack, dropped } = dropStash(folder.stack);
            const warning = folder.commit(stack);
            ctx.ui.notify(dropped === undefined ? "Nothing is stashed." : `Dropped the stashed prompt ${quote(dropped.text)}.${remaining(stack.length)}`, "info");
            if (warning !== undefined) ctx.ui.notify(warning, "warning");
            live.tui?.requestRender();
            return;
          }
          case "help":
            ctx.ui.notify(helpText(label), "info");
            return;
          default:
            // Everything after `/stash ` is the draft, exactly as typed.
            apply(ctx, folder, composer, pushDraft(folder.stack, { text: args }, makeEntry));
        }
      }
    });

    pi.on("session_start", (_event, ctx) => {
      if (ctx.mode !== "tui") return;
      setHotkeys(
        "prompt-stash",
        keys.map((key) => ({ key: keyLabel(key), action: "Stash the prompt; on an empty prompt, bring the stashed one back (also /stash)" }))
      );
      live?.unsubscribe();
      const folder = folderFor(ctx.cwd);
      const session: NonNullable<typeof live> = { ctx, folder, tui: undefined, unsubscribe: () => {} };
      live = session;
      // The widget shows what is stashed below the prompt; its factory also hands over the TUI,
      // which tells whether the prompt has focus when the key arrives.
      ctx.ui.setWidget(
        STASH_WIDGET_KEY,
        (tui) => {
          session.tui = tui;
          return indicator(() => folder.stack, label, ctx);
        },
        { placement: "belowEditor" }
      );
      session.unsubscribe = ctx.ui.onTerminalInput(onKey);
      if (folder.adoptedPending > 0) {
        const count = folder.adoptedPending;
        folder.adoptedPending = 0;
        const composer = composerOf(ctx, session.tui);
        const transition = popStash(folder.stack, composer.read());
        if (transition.outcome.kind === "restored") {
          apply(ctx, folder, composer, transition, ["restored"]);
          ctx.ui.notify(`Restored the prompt you stashed in an earlier exp-code session.${remaining(transition.outcome.depth)}`, "info");
        } else if (count > 0) {
          ctx.ui.notify(`${count === 1 ? "A prompt" : `${count} prompts`} stashed in an earlier exp-code session ${count === 1 ? "is" : "are"} waiting: ${label ?? "/stash"} on an empty prompt brings ${count === 1 ? "it" : "the newest"} back.`, "info");
        }
      }
    });

    pi.on("input", (event, ctx) => {
      if (event.source === "interactive") restoreAfterSubmit(ctx);
      return { action: "continue" };
    });

    pi.on("user_bash", (_event, ctx) => {
      restoreAfterSubmit(ctx);
      return undefined;
    });

    pi.on("session_shutdown", () => {
      live?.unsubscribe();
      live = undefined;
    });
  };
  stacksByFactory.set(factory, folders);
  return factory;
}

/** The focused component: every Pi renderer (TuiBase) has the method, the TUI interface does not name it. */
function focusedComponent(tui: TUI | undefined): unknown {
  const candidate = tui as { getFocusedComponent?: () => unknown } | undefined;
  return typeof candidate?.getFocusedComponent === "function" ? candidate.getFocusedComponent() : undefined;
}

const STASH_SUBCOMMANDS = [
  { value: "pop", label: "pop", description: "Bring the newest stashed prompt back" },
  { value: "list", label: "list", description: "Show what is stashed" },
  { value: "drop", label: "drop", description: "Discard the newest stashed prompt" },
  { value: "help", label: "help", description: "How the prompt stash works" }
];

/** The one-line hint for an outcome. */
export function describe(outcome: StashOutcome, label: string | undefined): string {
  const key = label ?? "/stash";
  switch (outcome.kind) {
    case "stashed":
      return `Prompt stashed${outcome.depth > 1 ? ` (${outcome.depth} stashed)` : ""}. It comes back after you send a message, or press ${key} on an empty prompt.`;
    case "restored":
      return `Stashed prompt restored.${remaining(outcome.depth)}`;
    case "kept":
      return `Your prompt is still stashed, since the prompt is not empty. Press ${key} on an empty prompt to bring it back.`;
    case "nothing":
      return `Nothing to stash: type a prompt first, then press ${key} to set it aside.`;
  }
}

function remaining(depth: number): string {
  return depth === 0 ? "" : ` ${depth} more stashed.`;
}

/** The first line of a draft, shortened, in quotes. */
export function quote(text: string, width = 40): string {
  const first = text.trim().split("\n")[0] ?? "";
  const more = text.trim().includes("\n") ? " ..." : "";
  return `"${truncateToWidth(first, width, "...")}${more}"`;
}

function listing(stack: readonly StashEntry[]): string {
  if (stack.length === 0) return "Nothing is stashed.";
  const lines = [...stack].reverse().map((entry, index) => `${index + 1}. ${quote(entry.text, 60)}`);
  return [`Stashed prompts, newest first (the first comes back next):`, ...lines].join("\n");
}

function helpText(label: string | undefined): string {
  return [
    `${label ?? "The stash key"} with text in the prompt stashes it; on an empty prompt it brings the newest stash back.`,
    "After you send a message, the newest stash comes back by itself if the prompt is empty.",
    "/stash <text> stashes text, /stash or /stash pop brings the newest back, /stash list shows them, /stash drop discards the newest.",
    'Rebind the key with "exp-code.prompt.stash" in ~/.exp-code/agent/keybindings.json.'
  ].join("\n");
}

/** The line below the prompt while something is stashed. */
function indicator(stack: () => readonly StashEntry[], label: string | undefined, ctx: ExtensionContext): Component {
  return {
    render(width: number): string[] {
      const entries = stack();
      const top = entries.at(-1);
      if (top === undefined || width <= 0) return [];
      const count = entries.length === 1 ? "Stashed" : `${entries.length} stashed`;
      const hint = label === undefined ? "/stash brings it back" : `${label} brings it back`;
      const line = ` ${count}: ${quote(top.text, 32)} · ${hint}, or it returns after you send`;
      return [ctx.ui.theme.fg("dim", truncateToWidth(line, width, "..."))];
    },
    invalidate() {}
  };
}


/**
 * A terminal UI session for prompt stash tests, built from Pi's real parts: a real TUI on a fake
 * terminal, Pi's real prompt editor, and the stash extension, with keystrokes delivered through
 * the same input path a terminal uses. Submits are routed the way Pi's interactive mode routes
 * them: `/stash` to the extension command, `!` to the `user_bash` event, built-in commands to
 * nothing, and everything else to the `input` event.
 */

import { CustomEditor, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { type Component, KeybindingsManager, setKeybindings, TUI_KEYBINDINGS, type Terminal, TuiMainScreen } from "@earendil-works/pi-tui";
import { createPromptStashExtension, type PromptStashOptions } from "../stash/extension.js";

const APP_KEYS = {
  ...TUI_KEYBINDINGS,
  "app.clear": { defaultKeys: "ctrl+c", description: "Clear editor" },
  "app.exit": { defaultKeys: "ctrl+d", description: "Exit" },
  "app.interrupt": { defaultKeys: "escape", description: "Interrupt" }
} as const;

/** A terminal that records nothing and hands input to the TUI on demand. */
function fakeTerminal(): Terminal & { input: (data: string) => void } {
  let onInput: (data: string) => void = () => {};
  return {
    start(input) {
      onInput = input;
    },
    stop() {},
    async drainInput() {},
    write() {},
    get columns() {
      return 100;
    },
    get rows() {
      return 30;
    },
    get kittyProtocolActive() {
      return false;
    },
    moveBy() {},
    hideCursor() {},
    showCursor() {},
    clearLine() {},
    clearFromCursor() {},
    clearScreen() {},
    setTitle() {},
    setProgress() {},
    setProgramStatus() {},
    input: (data) => onInput(data)
  };
}

type Handler = (event: unknown, ctx: unknown) => unknown;

export interface HarnessOptions extends Omit<PromptStashOptions, "agentDir"> {
  agentDir: string;
  cwd: string;
  /** Reuse a factory across sessions of one "process" (as Pi reuses it on /new and reload). */
  factory?: ExtensionFactory;
}

/** One exp-code terminal session. */
export class StashHarness {
  readonly tui: TuiMainScreen;
  readonly editor: CustomEditor;
  readonly factory: ExtensionFactory;
  readonly notes: string[] = [];
  /** Messages sent to the agent, in order. */
  readonly sent: string[] = [];
  /** `!` commands run, in order. */
  readonly bash: string[] = [];
  /** Built-in commands that were "run" (they do nothing here). */
  readonly builtins: string[] = [];
  /** When set, the agent is streaming: submits are steers, as in Pi. */
  streaming = false;
  readonly streamingBehaviors: (string | undefined)[] = [];
  private readonly terminal: ReturnType<typeof fakeTerminal>;
  private readonly handlers = new Map<string, Handler[]>();
  private readonly commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
  private widget: Component | undefined;
  private readonly ctx: Record<string, unknown>;

  private constructor(options: HarnessOptions) {
    setKeybindings(new KeybindingsManager(APP_KEYS));
    this.terminal = fakeTerminal();
    this.tui = new TuiMainScreen(this.terminal);
    const plain = (text: string) => text;
    const theme = { borderColor: plain, selectList: { selectedPrefix: plain, selectedText: plain, description: plain, scrollInfo: plain, noMatch: plain } };
    this.editor = new CustomEditor(this.tui, theme as never, new KeybindingsManager(APP_KEYS) as never);
    this.editor.onAction("app.clear", () => this.editor.setText(""));
    this.editor.onSubmit = (text) => void this.route(text);
    this.tui.addChild(this.editor);
    this.tui.setFocus(this.editor);
    this.tui.start();
    const { agentDir, cwd, factory, ...rest } = options;
    this.factory = factory ?? createPromptStashExtension({ agentDir, ...rest });
    this.ctx = {
      mode: "tui",
      hasUI: true,
      cwd,
      ui: {
        notify: (message: string) => this.notes.push(message),
        setWidget: (_key: string, content: ((tui: unknown, theme: unknown) => Component) | undefined) => {
          this.widget = content?.(this.tui, {});
        },
        onTerminalInput: (handler: Parameters<TuiMainScreen["addInputListener"]>[0]) => this.tui.addInputListener(handler),
        getEditorText: () => this.editor.getExpandedText(),
        setEditorText: (text: string) => this.editor.setText(text),
        theme: { fg: (_color: string, text: string) => text }
      }
    };
  }

  /** Start a session (runs the extension factory and `session_start`). */
  static async start(options: HarnessOptions): Promise<StashHarness> {
    const harness = new StashHarness(options);
    const api = {
      registerCommand: (name: string, command: { handler: (args: string, ctx: unknown) => Promise<void> }) => harness.commands.set(name, command),
      on: (event: string, handler: Handler) => harness.handlers.set(event, [...(harness.handlers.get(event) ?? []), handler])
    };
    await harness.factory(api as never);
    await harness.emit("session_start", { type: "session_start", reason: "startup" });
    return harness;
  }

  /** Deliver raw terminal input (one key, or a whole bracketed paste). */
  key(data: string): void {
    this.terminal.input(data);
  }

  /** Type text one character at a time; `\n` (Ctrl+J) inserts a new line. */
  type(text: string): void {
    for (const char of text) this.key(char);
  }

  /** Paste text the way a terminal with bracketed paste does. */
  paste(text: string): void {
    this.key(`\x1b[200~${text}\x1b[201~`);
  }

  ctrlS(): void {
    this.key("\x13");
  }

  /** Undo (Ctrl+-, sent by a legacy terminal as 0x1f). */
  undo(): void {
    this.key("\x1f");
  }

  /** Press Enter and let the submit's async routing finish. */
  async submit(): Promise<void> {
    this.key("\r");
    await flush();
  }

  /** Run a `/stash ...` (or any) line as if typed and sent. */
  async command(line: string): Promise<void> {
    this.editor.setText(line);
    await this.submit();
  }

  /** The composer's full text. */
  text(): string {
    return this.editor.getExpandedText();
  }

  /** What the indicator below the prompt shows. */
  indicator(): string[] {
    return this.widget?.render(100) ?? [];
  }

  /** End the session (Ctrl+D, /quit, or a session switch). */
  async shutdown(): Promise<void> {
    await this.emit("session_shutdown", { type: "session_shutdown", reason: "quit" });
    this.tui.stop();
  }

  private async route(raw: string): Promise<void> {
    const text = raw.trim();
    if (!text) return;
    this.editor.addToHistory(text);
    if (text === "/model" || text === "/settings" || text === "/new") {
      this.builtins.push(text);
      return;
    }
    if (text.startsWith("/stash")) {
      const space = text.indexOf(" ");
      if (space === -1 || text.slice(1, space) === "stash") {
        await this.commands.get("stash")!.handler(space === -1 ? "" : text.slice(space + 1), this.ctx);
        return;
      }
    }
    if (text.startsWith("!")) {
      this.bash.push(text.slice(1).trim());
      await this.emit("user_bash", { type: "user_bash", command: text.slice(1).trim(), excludeFromContext: false, cwd: this.ctx.cwd });
      return;
    }
    this.sent.push(text);
    this.streamingBehaviors.push(this.streaming ? "steer" : undefined);
    await this.emit("input", { type: "input", text, source: "interactive", streamingBehavior: this.streaming ? "steer" : undefined });
  }

  /** Fire an extension event directly (for orderings a terminal cannot produce). */
  async emit(event: string, payload: unknown): Promise<void> {
    for (const handler of this.handlers.get(event) ?? []) await handler(payload, this.ctx);
  }
}

/** Let pending promise callbacks run. */
export async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

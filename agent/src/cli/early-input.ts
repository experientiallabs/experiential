/**
 * Keys typed while exp-code starts.
 *
 * Until the terminal UI is up, the terminal is in its normal line mode: what is typed is echoed,
 * and Enter reaches the program as a line feed, which the prompt reads as "new line", so a command
 * typed early (`/hotkeys` Enter) ended up as the first line of a multi-line message. exp-code puts
 * the terminal in raw mode as soon as it starts, keeps what arrives, and hands it to the terminal
 * UI, which reads it before anything typed later. Keys still typed in line mode (the moment before
 * exp-code ran) arrive with Enter as a line feed, so a line feed among the kept keys is read as
 * Enter. Ctrl+C still quits.
 */

/** Where the terminal mode exp-code found is left for the terminal UI to restore (pi-ux-patches.ts). */
export const STDIN_WAS_RAW = Symbol.for("exp-code.stdin-was-raw");

/**
 * Where the remover of the early capture's SIGINT handler is left for the terminal UI, which
 * installs its own when it starts (pi-ux-patches.ts).
 */
export const EARLY_SIGINT = Symbol.for("exp-code.early-sigint");

/** The commands and options that never open the terminal UI. */
const NOT_INTERACTIVE_COMMANDS = new Set(["login", "logout", "whoami", "auth", "update", "install", "uninstall", "remove", "list", "mcp", "help"]);
const NOT_INTERACTIVE_OPTIONS = new Set(["-p", "--print", "--mode", "-h", "--help", "-v", "--version", "--list-models", "--export"]);

/** Whether a run with these arguments may open the terminal UI (a cheap check before Pi loads). */
export function mayOpenTerminalUi(args: readonly string[], stdin: { isTTY?: boolean }, stdout: { isTTY?: boolean }): boolean {
  if (stdin.isTTY !== true || stdout.isTTY !== true) return false;
  if (args[0] !== undefined && NOT_INTERACTIVE_COMMANDS.has(args[0])) return false;
  return !args.some((arg) => NOT_INTERACTIVE_OPTIONS.has(arg) || arg.startsWith("--mode="));
}

export interface EarlyInput {
  /** Stop keeping keys and give them back to stdin in the mode exp-code found (a run without the UI). */
  release(): void;
  /** Give the kept keys to the terminal UI, which starts next; the terminal stays in raw mode. */
  handOff(): void;
}

type RawInput = Pick<NodeJS.ReadStream, "isRaw" | "on" | "removeListener" | "pause" | "unshift"> & { setRawMode(mode: boolean): unknown };

/** Start keeping the keys typed on an interactive terminal. */
export function captureEarlyInput(
  stdin: RawInput = process.stdin as RawInput,
  interrupt: () => void = () => process.kill(process.pid, "SIGINT"),
  signals: Pick<NodeJS.Process, "on" | "removeListener"> = process
): EarlyInput {
  const wasRaw = stdin.isRaw === true;
  const chunks: Buffer[] = [];
  let done = false;
  // A SIGINT from another process (`kill -INT`) before the terminal UI takes over: Node would die
  // with the terminal still in the raw mode set here. Restore the mode, then die of the signal.
  const onSigint = (): void => {
    removeSigint();
    stdin.setRawMode(wasRaw);
    interrupt();
  };
  const removeSigint = (): void => {
    signals.removeListener("SIGINT", onSigint);
    delete (process as unknown as Record<symbol, unknown>)[EARLY_SIGINT];
  };
  const stop = (): void => {
    done = true;
    stdin.removeListener("data", onData);
    stdin.pause();
    const kept = Buffer.concat(chunks);
    if (kept.length > 0) stdin.unshift(Buffer.from(kept.toString("latin1").replaceAll("\n", "\r"), "latin1"));
  };
  const release = (): void => {
    if (done) return;
    stop();
    removeSigint();
    stdin.setRawMode(wasRaw);
  };
  function onData(chunk: Buffer | string): void {
    const bytes = typeof chunk === "string" ? Buffer.from(chunk, "utf8") : chunk;
    // Raw mode turns Ctrl+C into a byte; it quits as it would without exp-code holding the keys.
    if (bytes.includes(0x03)) {
      chunks.length = 0;
      release();
      interrupt();
      return;
    }
    chunks.push(bytes);
  }
  stdin.setRawMode(true);
  stdin.on("data", onData);
  signals.on("SIGINT", onSigint);
  (process as unknown as Record<symbol, unknown>)[EARLY_SIGINT] = removeSigint;
  return {
    release,
    handOff() {
      if (done) return;
      stop();
      // The terminal stays raw until the UI starts; the SIGINT handler stays until it takes over.
      (process as unknown as Record<symbol, boolean>)[STDIN_WAS_RAW] = wasRaw;
    }
  };
}

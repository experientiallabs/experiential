/**
 * Read one line from the terminal without echoing it (for pasting a key).
 *
 * The terminal is put in raw mode for the read and restored afterwards on every path: Enter,
 * Ctrl+C, Ctrl+D, or an abort (the browser approval arriving while the prompt is open).
 */

/** Ctrl+C at the prompt. */
export class LoginCancelled extends Error {
  constructor() {
    super("Login cancelled.");
    this.name = "LoginCancelled";
  }
}

/** The terminal surface the reader needs (a TTY `process.stdin`, or a test double). */
export interface MaskedInput {
  isTTY?: boolean;
  setRawMode?: (mode: boolean) => unknown;
  isRaw?: boolean;
  on(event: "data", listener: (chunk: Buffer | string) => void): unknown;
  off(event: "data", listener: (chunk: Buffer | string) => void): unknown;
  resume(): unknown;
  pause(): unknown;
}

/**
 * Prompt and read a hidden line. Resolves the trimmed line (`null` when empty, at end of input, or
 * when `signal` aborts); rejects with `LoginCancelled` on Ctrl+C.
 */
export function readMaskedLine(input: MaskedInput, write: (text: string) => void, prompt: string, signal?: AbortSignal): Promise<string | null> {
  return new Promise((resolve, reject) => {
    const wasRaw = input.isRaw === true;
    let buffer = "";
    let done = false;
    const finish = (outcome: { value: string | null } | { error: Error }) => {
      if (done) return;
      done = true;
      input.off("data", onData);
      signal?.removeEventListener("abort", onAbort);
      input.setRawMode?.(wasRaw);
      input.pause();
      write("\n");
      if ("error" in outcome) reject(outcome.error);
      else resolve(outcome.value);
    };
    const onAbort = () => finish({ value: null });
    const onData = (chunk: Buffer | string) => {
      // Bracketed-paste markers are terminal framing, not part of the key.
      for (const char of chunk.toString("utf8").replace(/\u001b\[20[01]~/g, "")) {
        if (char === "\u0003") return finish({ error: new LoginCancelled() });
        if (char === "\u0004") return finish({ value: buffer.trim() || null });
        if (char === "\r" || char === "\n") return finish({ value: buffer.trim() || null });
        if (char === "\u007f" || char === "\b") buffer = buffer.slice(0, -1);
        else if (char >= " ") buffer += char;
      }
    };
    write(prompt);
    input.setRawMode?.(true);
    input.on("data", onData);
    input.resume();
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) onAbort();
  });
}

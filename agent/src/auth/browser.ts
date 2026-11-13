/**
 * Open a URL in the default browser without a shell.
 *
 * Each platform's opener gets the URL as one argument in an argument array, so characters in the
 * URL (`&`, quotes, spaces) are never interpreted. On Windows that rules out `cmd /c start`, which
 * re-parses its command line; `rundll32 url.dll,FileProtocolHandler` opens the URL the same way
 * `start` does without going through cmd.
 */

import { spawn as nodeSpawn } from "node:child_process";

/** The program and arguments that open `url` on `platform`. */
export function browserCommand(url: string, platform: NodeJS.Platform = process.platform): { command: string; args: string[] } {
  switch (platform) {
    case "darwin":
      return { command: "open", args: [url] };
    case "win32":
      return { command: "rundll32", args: ["url.dll,FileProtocolHandler", url] };
    default:
      return { command: "xdg-open", args: [url] };
  }
}

export type SpawnLike = (
  command: string,
  args: string[],
  options: { detached: boolean; stdio: "ignore"; shell: false; windowsHide: boolean }
) => { once(event: "spawn" | "error", listener: (...args: unknown[]) => void): unknown; unref(): void };

/** Try to open `url`; resolves `false` when no opener could be started. */
export function openBrowser(url: string, options: { platform?: NodeJS.Platform; spawn?: SpawnLike } = {}): Promise<boolean> {
  const { command, args } = browserCommand(url, options.platform);
  const spawn = options.spawn ?? (nodeSpawn as unknown as SpawnLike);
  return new Promise((resolve) => {
    let child: ReturnType<SpawnLike>;
    try {
      child = spawn(command, args, { detached: true, stdio: "ignore", shell: false, windowsHide: true });
    } catch {
      resolve(false);
      return;
    }
    child.once("error", () => resolve(false));
    child.once("spawn", () => {
      child.unref();
      resolve(true);
    });
  });
}

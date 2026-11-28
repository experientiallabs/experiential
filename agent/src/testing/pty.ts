/**
 * The built exp-code CLI in a real pseudo-terminal, for end-to-end terminal UI tests: Python's
 * standard `pty` module relays keystrokes in and the screen out, so no native Node addon is needed.
 */

import { type ChildProcessWithoutNullStreams, execFileSync, spawn } from "node:child_process";

/** Relay stdin to a child in a pseudo-terminal and its output to stdout; report IXON on fd 3. */
const RELAY = String.raw`
import os, pty, sys, select, termios, fcntl, struct
pid, fd = pty.fork()
if pid == 0:
    os.execvp(sys.argv[1], sys.argv[1:])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
sys.stderr.write("IXON-AT-START %d\n" % (1 if termios.tcgetattr(fd)[0] & termios.IXON else 0)); sys.stderr.flush()
inp = sys.stdin.fileno()
while True:
    r, _, _ = select.select([fd, inp], [], [], 0.5)
    if fd in r:
        try:
            data = os.read(fd, 65536)
        except OSError:
            break
        if not data:
            break
        os.write(1, data)
    if inp in r:
        data = os.read(inp, 65536)
        if data == b"\x00IXON?":
            sys.stderr.write("IXON-NOW %d\n" % (1 if termios.tcgetattr(fd)[0] & termios.IXON else 0)); sys.stderr.flush()
        elif data:
            os.write(fd, data)
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
`;

export function findPython(): string | undefined {
  if (process.platform === "win32") return undefined;
  for (const candidate of ["python3", "python"]) {
    try {
      execFileSync(candidate, ["-c", "import pty, termios"], { stdio: "ignore" });
      return candidate;
    } catch {
      // try the next one
    }
  }
  return undefined;
}

export const strip = (text: string) => text.replace(/\x1b\[[0-9;?<>=]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[@-_]/g, "");

export class Terminal {
  output = "";
  log = "";
  readonly exited: Promise<number>;
  private readonly child: ChildProcessWithoutNullStreams;

  /** Start `node <cli> ...args` in a pseudo-terminal of 30 rows by 110 columns. */
  constructor(cli: string, env: NodeJS.ProcessEnv, cwd: string, args: string[] = []) {
    const python = findPython();
    if (python === undefined) throw new Error("No Python with the pty module.");
    this.child = spawn(python, ["-c", RELAY, process.execPath, cli, ...args], { cwd, env });
    this.child.stdout.on("data", (chunk: Buffer) => (this.output += chunk.toString("utf8")));
    this.child.stderr.on("data", (chunk: Buffer) => (this.log += chunk.toString("utf8")));
    this.exited = new Promise((resolve) => this.child.once("exit", (code) => resolve(code ?? -1)));
  }

  send(keys: string): void {
    this.child.stdin.write(keys);
  }

  /** Type one character at a time, like a person (a burst would arrive as one paste-like chunk). */
  async type(text: string): Promise<void> {
    for (const char of text) {
      this.send(char);
      await pause(4);
    }
  }

  screen(): string {
    return strip(this.output);
  }

  async waitFor(check: () => boolean, what: string, ms = 30_000): Promise<void> {
    const end = Date.now() + ms;
    while (!check()) {
      if (Date.now() > end) throw new Error(`timed out waiting for ${what}\n--- screen ---\n${this.screen().slice(-3000)}\n--- log ---\n${this.log}`);
      await pause(50);
    }
  }

  async waitForText(text: string, after = 0): Promise<number> {
    await this.waitFor(() => this.screen().indexOf(text, after) >= 0, JSON.stringify(text));
    return this.screen().indexOf(text, after) + text.length;
  }

  kill(): void {
    this.child.kill("SIGKILL");
  }
}

export const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));


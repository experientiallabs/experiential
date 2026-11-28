import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import { captureEarlyInput, mayOpenTerminalUi, STDIN_WAS_RAW } from "./early-input.js";

/** A stdin stand-in that records its mode and what is given back to it. */
class FakeStdin extends EventEmitter {
  isRaw = false;
  modes: boolean[] = [];
  unshifted: string[] = [];
  paused = false;
  setRawMode(mode: boolean): this {
    this.isRaw = mode;
    this.modes.push(mode);
    return this;
  }
  pause(): this {
    this.paused = true;
    return this;
  }
  unshift(chunk: Buffer): void {
    this.unshifted.push(chunk.toString("utf8"));
  }
}

const tty = { isTTY: true };

describe("keys typed while exp-code starts", () => {
  it("are kept in raw mode and handed to the terminal UI, a line feed read as Enter", () => {
    const stdin = new FakeStdin();
    const early = captureEarlyInput(stdin as never, () => {});
    expect(stdin.isRaw).toBe(true);
    // Typed in line mode just before exp-code ran: Enter arrived as a line feed.
    stdin.emit("data", Buffer.from("/compact\n"));
    stdin.emit("data", Buffer.from("/usage\r"));
    early.handOff();
    expect(stdin.unshifted).toEqual(["/compact\r/usage\r"]);
    expect(stdin.paused).toBe(true);
    expect(stdin.listenerCount("data")).toBe(0);
    // Raw mode stays on for the UI, which restores the mode exp-code found.
    expect(stdin.isRaw).toBe(true);
    expect((process as unknown as Record<symbol, unknown>)[STDIN_WAS_RAW]).toBe(false);
    delete (process as unknown as Record<symbol, unknown>)[STDIN_WAS_RAW];
    early.release();
    expect(stdin.modes).toEqual([true]);
  });

  it("go back to stdin in the mode it had when no terminal UI opens", () => {
    const stdin = new FakeStdin();
    const early = captureEarlyInput(stdin as never, () => {});
    stdin.emit("data", Buffer.from("y\r"));
    early.release();
    expect(stdin.modes).toEqual([true, false]);
    expect(stdin.unshifted).toEqual(["y\r"]);
  });

  it("let Ctrl+C quit at once", () => {
    const stdin = new FakeStdin();
    let interrupted = 0;
    captureEarlyInput(stdin as never, () => interrupted++);
    stdin.emit("data", Buffer.from("abc\x03"));
    expect(interrupted).toBe(1);
    expect(stdin.isRaw).toBe(false);
    expect(stdin.unshifted).toEqual([]);
  });

  it("are kept only for a run that may open the terminal UI", () => {
    expect(mayOpenTerminalUi([], tty, tty)).toBe(true);
    expect(mayOpenTerminalUi(["--model", "x", "fix it"], tty, tty)).toBe(true);
    expect(mayOpenTerminalUi([], { isTTY: false }, tty)).toBe(false);
    expect(mayOpenTerminalUi(["-p", "hi"], tty, tty)).toBe(false);
    expect(mayOpenTerminalUi(["--mode=json"], tty, tty)).toBe(false);
    expect(mayOpenTerminalUi(["login"], tty, tty)).toBe(false);
    expect(mayOpenTerminalUi(["--version"], tty, tty)).toBe(false);
  });
});

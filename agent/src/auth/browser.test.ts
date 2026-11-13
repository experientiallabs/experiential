import { EventEmitter } from "node:events";
import { describe, expect, it } from "vitest";
import { browserCommand, openBrowser, type SpawnLike } from "./browser.js";

const URL_WITH_METACHARACTERS = "https://platform.example/cli/auth?state=a&port=1&name=exp-code%20(x)\"; rm -rf ~";

describe("browser opener", () => {
  it("uses each platform's opener with the URL as one argument", () => {
    expect(browserCommand(URL_WITH_METACHARACTERS, "darwin")).toEqual({ command: "open", args: [URL_WITH_METACHARACTERS] });
    expect(browserCommand(URL_WITH_METACHARACTERS, "linux")).toEqual({ command: "xdg-open", args: [URL_WITH_METACHARACTERS] });
    // Not cmd /c start: cmd re-parses its line, and & in the URL would split it.
    expect(browserCommand(URL_WITH_METACHARACTERS, "win32")).toEqual({ command: "rundll32", args: ["url.dll,FileProtocolHandler", URL_WITH_METACHARACTERS] });
  });

  it("spawns without a shell and detaches", async () => {
    const calls: Parameters<SpawnLike>[] = [];
    let unrefs = 0;
    const spawn: SpawnLike = (...args) => {
      calls.push(args);
      const child = Object.assign(new EventEmitter(), { unref: () => void unrefs++ });
      queueMicrotask(() => child.emit("spawn"));
      return child;
    };
    await expect(openBrowser(URL_WITH_METACHARACTERS, { platform: "linux", spawn })).resolves.toBe(true);
    expect(calls).toEqual([["xdg-open", [URL_WITH_METACHARACTERS], { detached: true, stdio: "ignore", shell: false, windowsHide: true }]]);
    expect(unrefs).toBe(1);
  });

  it("reports false when the opener is missing", async () => {
    const spawn: SpawnLike = () => {
      const child = Object.assign(new EventEmitter(), { unref: () => undefined });
      queueMicrotask(() => child.emit("error", Object.assign(new Error("spawn xdg-open ENOENT"), { code: "ENOENT" })));
      return child;
    };
    await expect(openBrowser("https://x.example", { platform: "linux", spawn })).resolves.toBe(false);
    const throwing: SpawnLike = () => {
      throw new Error("boom");
    };
    await expect(openBrowser("https://x.example", { spawn: throwing })).resolves.toBe(false);
  });
});

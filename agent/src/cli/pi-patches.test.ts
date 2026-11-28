import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { applyFilePatch, PI_PATCHES, PiPatchError, patchFor, patchSource } from "./pi-patches.js";

const modulesDir = fileURLToPath(new URL("../../node_modules/@earendil-works", import.meta.url));

/** The installed file a patch edits. */
function installedFile(pkg: string, file: string): string {
  return join(modulesDir, pkg, "dist", file);
}

describe("Pi patches", () => {
  it.each(PI_PATCHES.map((patch) => [`${patch.pkg}/${patch.file}`, patch] as const))("apply exactly to the installed %s", (_name, patch) => {
    const path = installedFile(patch.pkg, patch.file);
    const source = readFileSync(path, "utf8");
    const patched = applyFilePatch(patch, source);
    expect(patched).not.toBe(source);
    expect(patchSource(path, source)).toBe(patched);
  });

  it("rename the app, the terminal title and the project folder", () => {
    const patch = patchFor(installedFile("pi-coding-agent", "config.js"))!;
    const patched = applyFilePatch(patch, readFileSync(installedFile("pi-coding-agent", "config.js"), "utf8"));
    expect(patched).toContain('APP_NAME = "exp-code"');
    expect(patched).toContain('CONFIG_DIR_NAME = ".exp-code"');
    expect(patched).not.toContain('"π"');
  });

  it("register none of Pi's direct providers", () => {
    const file = installedFile("pi-coding-agent", "core/model-runtime.js");
    expect(patchSource(file, readFileSync(file, "utf8"))).not.toContain("builtinProviders()");
  });

  it("fail loudly when the text moved", () => {
    expect(() => applyFilePatch({ pkg: "x", file: "y.js", edits: [{ find: "absent", replace: "" }] }, "source")).toThrow(PiPatchError);
    expect(() => applyFilePatch({ pkg: "x", file: "y.js", edits: [{ find: "a", replace: "b" }] }, "a a")).toThrow(/expected 1/);
  });

  it("match module paths and file URLs, on any platform", () => {
    expect(patchFor("file:///x/node_modules/@earendil-works/pi-coding-agent/dist/config.js")?.file).toBe("config.js");
    expect(patchFor("C:\\x\\node_modules\\@earendil-works\\pi-tui\\dist\\tui-main-screen.js")?.pkg).toBe("pi-tui");
    expect(patchFor("/x/node_modules/@earendil-works/pi-coding-agent/dist/core/agent-session.js")).toBeUndefined();
    expect(patchSource("/x/other.js", "same")).toBe("same");
  });
});

describe("Pi as tests load it", () => {
  it("is patched (the test setup installs the load hook)", async () => {
    const config = await import("@earendil-works/pi-coding-agent");
    expect(config.CONFIG_DIR_NAME).toBe(".exp-code");
  });
});

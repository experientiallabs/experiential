import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { EXP_CODE_VERSION } from "./version.js";

describe("version", () => {
  it("matches package.json", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
    expect(EXP_CODE_VERSION).toBe(pkg.version);
  });

  it("pins Pi and every dependency exactly", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as Record<string, Record<string, string>>;
    for (const group of ["dependencies", "devDependencies"]) {
      for (const [name, version] of Object.entries(pkg[group] ?? {})) expect(version, name).toMatch(/^\d+\.\d+\.\d+$/);
    }
    expect(pkg.dependencies?.["@earendil-works/pi-coding-agent"]).toBeDefined();
  });
});

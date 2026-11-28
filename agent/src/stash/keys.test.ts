import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { DEFAULT_STASH_KEYS, keyLabel, loadStashKeys, parseStashKeys, STASH_KEYBINDING_ID } from "./keys.js";

const dirs: string[] = [];
function withBindings(body: string | undefined): string {
  const dir = mkdtempSync(join(tmpdir(), "exp-code-stash-keys-"));
  dirs.push(dir);
  if (body !== undefined) writeFileSync(join(dir, "keybindings.json"), body);
  return dir;
}
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

describe("stash keys", () => {
  it("defaults to Ctrl+S without a keybindings file or entry, or with a broken one", () => {
    expect(DEFAULT_STASH_KEYS).toEqual(["ctrl+s"]);
    expect(loadStashKeys(withBindings(undefined))).toEqual(["ctrl+s"]);
    expect(loadStashKeys(withBindings('{"app.clear":"ctrl+x"}'))).toEqual(["ctrl+s"]);
    expect(loadStashKeys(withBindings("{oops"))).toEqual(["ctrl+s"]);
    expect(loadStashKeys(withBindings(`{"${STASH_KEYBINDING_ID}": 5}`))).toEqual(["ctrl+s"]);
    expect(loadStashKeys(withBindings(`{"${STASH_KEYBINDING_ID}": ["ctrl+s", "has space"]}`))).toEqual(["ctrl+s"]);
  });

  it("rebinds to one or several keys, or to none", () => {
    expect(loadStashKeys(withBindings(`﻿{"${STASH_KEYBINDING_ID}": "Alt+S"}`))).toEqual(["alt+s"]);
    expect(loadStashKeys(withBindings(`{"${STASH_KEYBINDING_ID}": ["ctrl+s", "f2"]}`))).toEqual(["ctrl+s", "f2"]);
    expect(loadStashKeys(withBindings(`{"${STASH_KEYBINDING_ID}": []}`))).toEqual([]);
    expect(parseStashKeys("ctrl+shift+s")).toEqual(["ctrl+shift+s"]);
  });

  it("labels keys the way people write them", () => {
    expect(keyLabel("ctrl+s")).toBe("Ctrl+S");
    expect(keyLabel("alt+shift+x")).toBe("Alt+Shift+X");
    expect(keyLabel("f2")).toBe("F2");
  });
});

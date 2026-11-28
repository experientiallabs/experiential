import { describe, expect, it } from "vitest";
import { hotkeyRows, setHotkeys } from "./hotkeys.js";

describe("exp-code's keys in /hotkeys", () => {
  it("lists each feature's keys once, replacing them when the feature lists them again", () => {
    setHotkeys("test-feature", [{ key: "Ctrl+S", action: "Stash" }]);
    setHotkeys("test-feature", [
      { key: "Alt+S", action: "Stash" },
      { key: "F2", action: "Stash" }
    ]);
    expect(hotkeyRows().filter((row) => row.action === "Stash")).toEqual([
      { key: "Alt+S", action: "Stash" },
      { key: "F2", action: "Stash" }
    ]);
    setHotkeys("test-feature", []);
    expect(hotkeyRows().filter((row) => row.action === "Stash")).toEqual([]);
  });
});

import { spawn } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mergeStacks, type StashEntry } from "./machine.js";
import { parseStashFile, processIsAlive, StashStore, type StashStoreOptions } from "./store.js";

let root: string;
let dir: string;
const cwd = "/work/project";

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "exp-code-stash-store-"));
  dir = join(root, "prompt-stash");
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

const entry = (id: string, text = id, stashedAt = 1): StashEntry => ({ id, text, stashedAt });
const store = (options: Partial<StashStoreOptions> = {}) => new StashStore({ dir, cwd, host: "h", pid: 100, isAlive: () => false, ...options });
const files = () => (existsSync(dir) ? readdirSync(dir).sort() : []);

describe("prompt stash store", () => {
  it("saves atomically with private permissions and removes the file when empty", () => {
    const s = store();
    s.save([entry("a", "line one\nline two  "), { ...entry("b"), cursor: { line: 0, col: 1 }, collapsed: { text: "[paste #1 +11 lines]", pastes: [[1, "x\n".repeat(11)]] } }]);
    expect(files()).toEqual([s.path.slice(dir.length + 1)]);
    expect(statSync(s.path).mode & 0o777).toBe(0o600);
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    expect(s.load().map((e) => e.text)).toEqual(["line one\nline two  ", "b"]);
    expect(s.load()[1]!.collapsed?.pastes[0]![1]).toBe("x\n".repeat(11));
    s.save([]);
    expect(files()).toEqual([]);
    s.save([]); // removing a missing file is fine
  });

  it("never writes the undo history to disk", () => {
    const s = store();
    s.save([{ ...entry("a"), undo: [{ secret: "state" }] }]);
    expect(readFileSync(s.path, "utf8")).not.toContain("secret");
  });

  it("adopts the file of an exited process in the same folder, once", () => {
    store({ pid: 7, token: "old" }).save([entry("a", "A", 1), entry("b", "B", 2)]);
    const s = store({ pid: 8 });
    const adoption = s.adoptOrphans([], mergeStacks);
    expect(adoption).toMatchObject({ files: 1, corrupt: 0 });
    expect(adoption.entries.map((e) => e.text)).toEqual(["A", "B"]);
    expect(files()).toEqual([s.path.slice(dir.length + 1)]);
    // While pid 8 runs, a third instance leaves its (now merged) file alone.
    expect(store({ pid: 9, isAlive: (pid) => pid === 8 }).adoptOrphans([], mergeStacks).files).toBe(0);
  });

  it("leaves running processes, its own process, other folders, and other hosts alone", () => {
    store({ pid: 7, token: "live" }).save([entry("live")]);
    store({ pid: 100, token: "sibling" }).save([entry("same-process")]);
    store({ pid: 5, cwd: "/elsewhere" }).save([entry("other-folder")]);
    store({ pid: 6, host: "other-host" }).save([entry("other-host")]);
    const s = store({ pid: 100, isAlive: (pid) => pid === 7 });
    expect(s.adoptOrphans([], mergeStacks)).toEqual({ entries: [], files: 0, corrupt: 0 });
    expect(files()).toHaveLength(4);
  });

  it("sets unreadable files aside without deleting them or failing", () => {
    const s = store({ pid: 7 });
    s.save([entry("x")]);
    const [folder, host] = s.path.slice(dir.length + 1).split(".");
    for (const [i, body] of ["{not json", "", '{"version":2,"cwd":"/work/project","entries":[]}', '{"version":1,"cwd":"/work/project","entries":[{"id":"","text":"t","stashedAt":1}]}'].entries()) {
      writeFileSync(join(dir, `${folder}.${host}.7.bad${i}.json`), body);
    }
    const adoption = store({ pid: 8 }).adoptOrphans([], mergeStacks);
    expect(adoption.entries.map((e) => e.text)).toEqual(["x"]);
    expect(adoption.corrupt).toBe(4);
    expect(files().filter((f) => f.endsWith(".corrupt"))).toHaveLength(4);
    // Set aside means never read again.
    expect(store({ pid: 9 }).adoptOrphans([], mergeStacks).corrupt).toBe(0);
  });

  it("gives an orphan to exactly one of two instances starting together", () => {
    store({ pid: 7 }).save([entry("only")]);
    const a = store({ pid: 8 }).adoptOrphans([], mergeStacks);
    const b = store({ pid: 9, isAlive: (pid) => pid === 8 }).adoptOrphans([], mergeStacks);
    expect([...a.entries, ...b.entries].map((e) => e.text)).toEqual(["only"]);
  });

  it("keeps one copy of each draft after a crash between saving and deleting a claim", () => {
    store({ pid: 7 }).save([entry("a", "A", 1)]);
    // Pid 8 claimed and saved, then died before deleting its claim: simulate both files.
    const crashed = store({ pid: 8, token: "crashed" });
    crashed.save([entry("a", "A", 1), entry("b", "B", 2)]);
    const adoption = store({ pid: 9 }).adoptOrphans([], mergeStacks);
    expect(adoption.entries.map((e) => e.text)).toEqual(["A", "B"]);
  });

  it("cleans up unfinished writes of exited processes only", () => {
    mkdirSync(dir, { recursive: true });
    const live = store({ pid: 7, token: "a" }).path.replace(/\.json$/, ".abcd.tmp");
    const dead = store({ pid: 6, token: "b" }).path.replace(/\.json$/, ".abcd.tmp");
    writeFileSync(live, "partial");
    writeFileSync(dead, "partial");
    store({ pid: 8, isAlive: (pid) => pid === 7 }).adoptOrphans([], mergeStacks);
    expect(existsSync(live)).toBe(true);
    expect(existsSync(dead)).toBe(false);
  });

  it("merges adopted drafts below the current ones, deduplicated by id", () => {
    store({ pid: 7 }).save([entry("old", "old", 1)]);
    const adoption = store({ pid: 8 }).adoptOrphans([entry("new", "new", 5), entry("old", "old", 1)], mergeStacks);
    expect(adoption.entries.map((e) => e.text)).toEqual(["old", "new"]);
  });

  it("validates file contents strictly", () => {
    expect(parseStashFile('{"version":1,"cwd":"/w","entries":[{"id":"a","text":"t","stashedAt":1,"cursor":{"line":-1,"col":0}}]}', "/w")).toBeUndefined();
    expect(parseStashFile('{"version":1,"cwd":"/w","entries":[{"id":"a","text":"t","stashedAt":1}]}', "/other")).toBeUndefined();
    expect(parseStashFile('{"version":1,"cwd":"/w","entries":[{"id":"a","text":"t","stashedAt":1,"collapsed":{"text":"x","pastes":[[1.5,"y"]]}}]}', "/w")).toBeUndefined();
    expect(parseStashFile('{"version":1,"cwd":"/w","entries":[]}', "/w")).toEqual([]);
  });

  it("tells running processes from exited ones", async () => {
    expect(processIsAlive(process.pid)).toBe(true);
    expect(processIsAlive(0)).toBe(false);
    expect(processIsAlive(-5)).toBe(false);
    const child = spawn(process.execPath, ["-e", "setTimeout(() => {}, 60000)"], { stdio: "ignore" });
    await new Promise((resolve) => child.once("spawn", resolve));
    const pid = child.pid!;
    expect(processIsAlive(pid)).toBe(true);
    // A real second exp-code in this folder: its stash is not taken while it runs.
    new StashStore({ dir, cwd, pid }).save([entry("theirs")]);
    expect(new StashStore({ dir, cwd }).adoptOrphans([], mergeStacks).files).toBe(0);
    child.kill("SIGKILL");
    await new Promise((resolve) => child.once("exit", resolve));
    expect(processIsAlive(pid)).toBe(false);
    expect(new StashStore({ dir, cwd }).adoptOrphans([], mergeStacks).entries.map((e) => e.text)).toEqual(["theirs"]);
  });
});

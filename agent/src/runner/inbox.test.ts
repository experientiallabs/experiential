import { appendFileSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { parseInboxCommand, readInbox } from "./inbox.js";
import { freshState } from "./state.js";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-inbox-"));
});
afterEach(() => rm(dir, { recursive: true, force: true }));

describe("inbox", () => {
  it("parses steer and stop commands and rejects anything else", () => {
    expect(parseInboxCommand('{"type":"steer","text":"hi","requestId":"a"}')).toEqual({ type: "steer", text: "hi", requestId: "a" });
    expect(parseInboxCommand('{"type":"stop"}')).toEqual({ type: "stop" });
    expect(parseInboxCommand('{"type":"steer"}')).toBeNull();
    expect(parseInboxCommand("{torn")).toBeNull();
  });

  it("reads only complete new lines and ignores repeated requestIds", () => {
    const path = join(dir, "inbox.jsonl");
    const state = freshState("r");
    writeFileSync(path, '{"type":"steer","text":"one","requestId":"a"}\n{"type":"steer","text":"one","requestId":"a"}\n{"type":"st');
    expect(readInbox(path, state).commands).toEqual([{ type: "steer", text: "one", requestId: "a" }]);
    appendFileSync(path, 'op","requestId":"b"}\n{"type":"steer","text":"one","requestId":"a"}\n');
    expect(readInbox(path, state).commands).toEqual([{ type: "stop", requestId: "b" }]);
    expect(readInbox(path, state)).toEqual({ commands: [], changed: false });
    expect(state.appliedRequestIds).toEqual(["a", "b"]);
  });
});

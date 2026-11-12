import { chmodSync, writeFileSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { accountEnvironment, accountIds, sandboxBashOperations, sandboxEditOperations, sandboxReadOperations, sandboxWriteOperations } from "./sandbox.js";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "exp-code-sandbox-"));
});
afterEach(() => rm(dir, { recursive: true, force: true }));

const self = { uid: process.getuid?.() ?? 0, gid: process.getgid?.() ?? 0 };

describe("sandbox operations", () => {
  it("reads an account from a passwd file", () => {
    const passwd = join(dir, "passwd");
    writeFileSync(passwd, "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1001::/home/user:/bin/bash\n");
    expect(accountIds("user", passwd)).toEqual({ uid: 1000, gid: 1001, home: "/home/user" });
    expect(accountIds("nobody-here", passwd)).toBeUndefined();
  });

  it("runs commands with only the scrubbed environment and Pi's session variables", async () => {
    process.env.EXP_CODE_TEST_SECRET = "leak";
    const ops = sandboxBashOperations({ ...self, env: accountEnvironment("user", dir) });
    const chunks: Buffer[] = [];
    const { exitCode } = await ops.exec("echo \"secret=${EXP_CODE_TEST_SECRET:-none} home=$HOME model=$PI_MODEL\"", dir, {
      onData: (data) => chunks.push(data),
      env: { ...process.env, PI_MODEL: "m" }
    });
    delete process.env.EXP_CODE_TEST_SECRET;
    expect(exitCode).toBe(0);
    expect(Buffer.concat(chunks).toString()).toBe(`secret=none home=${dir} model=m\n`);
  });

  it("kills a command on timeout and on abort", async () => {
    const ops = sandboxBashOperations({ ...self, env: accountEnvironment("user", dir) });
    await expect(ops.exec("sleep 5", dir, { onData: () => undefined, timeout: 0.2 })).rejects.toThrow(/timed out/);
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 100);
    await expect(ops.exec("sleep 5", dir, { onData: () => undefined, signal: controller.signal })).rejects.toThrow(/aborted/);
  });

  it("writes files and creates parent directories", async () => {
    const ops = sandboxWriteOperations({ ...self, env: {} });
    await ops.writeFile(join(dir, "a", "b", "c.txt"), "hi");
    expect(await readFile(join(dir, "a", "b", "c.txt"), "utf8")).toBe("hi");
  });

  const account = () => ({ ...self, env: accountEnvironment("me", dir) });
  // As root every file is readable, so the refusals below only show for an ordinary account.
  const notRoot = self.uid !== 0;

  it.skipIf(!notRoot)("reads, writes and edits only what the account itself may (the kernel decides)", async () => {
    const read = sandboxReadOperations(account(), dir);
    const edit = sandboxEditOperations(account());
    writeFileSync(join(dir, "mine.txt"), "hello");
    expect((await read.readFile(join(dir, "mine.txt"))).toString()).toBe("hello");
    await read.access(join(dir, "mine.txt"));
    await edit.access(join(dir, "mine.txt"));
    await edit.writeFile(join(dir, "mine.txt"), "changed");
    expect(await readFile(join(dir, "mine.txt"), "utf8")).toBe("changed");

    // A file the account may not read: the runner's privileges never stand in for the account's.
    writeFileSync(join(dir, "secret.json"), '{"apiKey":"xpl_secret"}');
    chmodSync(join(dir, "secret.json"), 0o000);
    await expect(read.readFile(join(dir, "secret.json"))).rejects.toThrow(/Permission denied/);
    await expect(read.access(join(dir, "secret.json"))).rejects.toThrow();
    await expect(edit.access(join(dir, "secret.json"))).rejects.toThrow();
    chmodSync(join(dir, "secret.json"), 0o444);
    await expect(edit.access(join(dir, "secret.json"))).rejects.toThrow();
    await expect(sandboxWriteOperations(account()).writeFile(join(dir, "secret.json"), "x")).rejects.toThrow(/Permission denied/);
  });

  it("detects an image from the header the account can read, and only that", async () => {
    const read = sandboxReadOperations(account(), dir);
    // A 1x1 PNG.
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64");
    writeFileSync(join(dir, "dot.png"), png);
    writeFileSync(join(dir, "note.txt"), "just text");
    expect(await read.detectImageMimeType!(join(dir, "dot.png"))).toBe("image/png");
    expect(await read.detectImageMimeType!(join(dir, "note.txt"))).toBeNull();
    if (notRoot) {
      chmodSync(join(dir, "dot.png"), 0o000);
      await expect(read.detectImageMimeType!(join(dir, "dot.png"))).rejects.toThrow();
    }
  });
});

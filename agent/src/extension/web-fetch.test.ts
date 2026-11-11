import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { execFileSync } from "node:child_process";
import { createLocalBashOperations } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { createWebFetchTool, fetchCommand, shellQuote } from "./web-fetch.js";

function pythonHasBs4(): boolean {
  try {
    execFileSync("python3", ["-c", "import requests, bs4"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

describe("web_fetch", () => {
  it("quotes arguments for the shell", () => {
    expect(shellQuote("it's")).toBe("'it'\\''s'");
    expect(fetchCommand("https://a.example/?q='x'", 1000)).toContain("'https://a.example/?q='\\''x'\\'''");
  });

  it("runs through the given shell operations and refuses non-http URLs", async () => {
    const commands: string[] = [];
    const shell = {
      exec: async (command: string, _cwd: string, options: { onData: (data: Buffer) => void }) => {
        commands.push(command);
        options.onData(Buffer.from("Title: Example\n\n# Hello"));
        return { exitCode: 0 };
      }
    };
    const tool = createWebFetchTool({ shell, cwd: "/tmp" });
    const result = await tool.execute("c", { url: "https://example.com" }, undefined, undefined, undefined as never);
    expect(commands[0]).toMatch(/^python3 -c /);
    expect(result.content[0]).toEqual({ type: "text", text: "Title: Example\n\n# Hello" });
    await expect(tool.execute("c", { url: "file:///etc/passwd" }, undefined, undefined, undefined as never)).rejects.toThrow(/http/);
  });

  it("reports a failed fetch as a tool error", async () => {
    const shell = {
      exec: async (_c: string, _w: string, options: { onData: (data: Buffer) => void }) => {
        options.onData(Buffer.from("Install them with: pip install requests beautifulsoup4"));
        return { exitCode: 3 };
      }
    };
    const result = await createWebFetchTool({ shell, cwd: "/tmp" }).execute("c", { url: "https://x.example" }, undefined, undefined, undefined as never);
    expect((result as { isError?: boolean }).isError).toBe(true);
    expect(JSON.stringify(result.content)).toContain("pip install requests beautifulsoup4");
  });

  it.skipIf(!pythonHasBs4())("extracts readable text from a real page", async () => {
    const server = createServer((_request, response) => {
      response.writeHead(200, { "content-type": "text/html" });
      response.end("<html><head><title>Doc</title><script>x()</script></head><body><h1>Heading</h1><p>Body text.</p><ul><li>Item</li></ul></body></html>");
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    try {
      const result = await createWebFetchTool({ shell: createLocalBashOperations(), cwd: process.cwd() }).execute("c", { url: `http://127.0.0.1:${port}/` }, undefined, undefined, undefined as never);
      const text = JSON.stringify(result.content);
      expect(text).toContain("Title: Doc");
      expect(text).toContain("# Heading");
      expect(text).toContain("- Item");
      expect(text).not.toContain("x()");
    } finally {
      server.close();
    }
  });
});

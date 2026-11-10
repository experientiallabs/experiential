/**
 * `web_fetch`: read a web page's main text.
 *
 * The fetch runs as a short Python program (requests and BeautifulSoup) through the same shell
 * operations as the `bash` tool, so in a sandbox it runs as the sandbox user with the sandbox's
 * network, never in the host process.
 */

import { type BashOperations, defineTool, truncateHead } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export interface WebFetchOptions {
  /** Shell operations the fetch runs through. */
  shell: BashOperations;
  /** Working directory for the fetch process; defaults to the session's working directory. */
  cwd?: string;
}

const fetchSchema = Type.Object({
  url: Type.String({ description: "The http(s) URL to read" }),
  maxChars: Type.Optional(Type.Number({ description: "Most characters of page text to return (default 20000)" }))
});

const FETCH_TIMEOUT_SECONDS = 45;

/** Python that fetches one URL and prints its readable text. */
export const FETCH_SCRIPT = String.raw`
import sys, re
try:
    import requests
    from bs4 import BeautifulSoup
except ModuleNotFoundError as error:
    print(f"web_fetch needs the Python packages requests and beautifulsoup4 ({error}). Install them with: pip install requests beautifulsoup4")
    sys.exit(3)
url, limit = sys.argv[1], int(sys.argv[2])
r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 (compatible; exp-code/1.0)"})
ctype = r.headers.get("content-type", "")
print(f"URL: {r.url}\nStatus: {r.status_code}\nContent-Type: {ctype}\n")
if "html" in ctype or r.text.lstrip().startswith("<"):
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        print(f"Title: {title}\n")
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "pre", "blockquote"]):
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not text:
            continue
        prefix = "#" * int(el.name[1]) + " " if el.name in ("h1", "h2", "h3", "h4") else ("- " if el.name == "li" else "")
        lines.append(prefix + text)
    body = "\n".join(lines) or re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
else:
    body = r.text
print(body[:limit])
if len(body) > limit:
    print(f"\n[... {len(body) - limit} more characters]")
`;

/** Quote a value for a POSIX shell. */
export function shellQuote(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

/** The shell command that runs the fetch script with its arguments. */
export function fetchCommand(url: string, limit: number): string {
  return `python3 -c ${shellQuote(FETCH_SCRIPT)} ${shellQuote(url)} ${limit}`;
}

export function createWebFetchTool(options: WebFetchOptions) {
  return defineTool({
    name: "web_fetch",
    label: "Read page",
    description:
      "Read a web page (or a plain-text/JSON URL) and return its main text. Use it to read sources found with web_search or links the user gives you. To save a file from the web (a PDF, a CSV), use bash with curl instead.",
    promptSnippet: "Read the text of a web page",
    parameters: fetchSchema,
    annotations: { readOnlyHint: true, openWorldHint: true },
    async execute(_toolCallId, { url, maxChars }, signal, _onUpdate, ctx) {
      if (!/^https?:\/\//i.test(url)) throw new Error("web_fetch needs an http(s) URL.");
      const limit = Math.min(Math.max(1000, Math.floor(maxChars ?? 20_000)), 100_000);
      const chunks: Buffer[] = [];
      const { exitCode } = await options.shell.exec(fetchCommand(url, limit), options.cwd ?? ctx?.cwd ?? process.cwd(), {
        onData: (data) => chunks.push(data),
        signal,
        timeout: FETCH_TIMEOUT_SECONDS
      });
      const text = truncateHead(Buffer.concat(chunks).toString("utf8"), { maxBytes: limit * 4 }).content;
      if (exitCode !== 0) {
        return { content: [{ type: "text", text: `Could not read ${url}:\n${text.slice(-2000)}` }], details: { url }, isError: true };
      }
      return { content: [{ type: "text", text }], details: { url } };
    }
  });
}

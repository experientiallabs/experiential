/**
 * `web_search`: a web search through the Experiential gateway.
 *
 * The tool sends one Chat Completions request with `web_search_options`. The gateway runs the
 * search, gives the results to the model in that request, bills the search on the same key, and
 * returns URL citations; the model condenses the results into a source list for the agent.
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { GATEWAY_PROVIDER_ID, type GatewayConfig } from "../gateway/config.js";

export interface WebSearchOptions {
  gateway: GatewayConfig;
  /** Alias the search runs on when the session's model is not a gateway model. */
  model?: string;
  fetch?: typeof globalThis.fetch;
}

export interface Citation {
  url: string;
  title?: string;
}

const webSearchSchema = Type.Object({
  query: Type.String({ description: "What to search the web for. Write it like a search engine query." })
});

const SEARCH_INSTRUCTIONS =
  "You are a web search tool. Using only the web search results provided to you, answer with a numbered list of the most relevant results. For each: the page title, its full URL, and two or three sentences on what the page says that bears on the query. Then one line with the direct answer if the results contain it. Never invent URLs.";

/** Pull the de-duplicated URL citations off a Chat Completions message. */
export function citationsOf(message: { annotations?: unknown } | undefined): Citation[] {
  const annotations = Array.isArray(message?.annotations) ? message.annotations : [];
  const seen = new Set<string>();
  const out: Citation[] = [];
  for (const entry of annotations) {
    const citation = (entry as { url_citation?: { url?: unknown; title?: unknown } }).url_citation;
    if (citation && typeof citation.url === "string" && !seen.has(citation.url)) {
      seen.add(citation.url);
      out.push(typeof citation.title === "string" ? { url: citation.url, title: citation.title } : { url: citation.url });
    }
  }
  return out;
}

function failure(text: string, query: string) {
  return { content: [{ type: "text" as const, text }], details: { query, citations: [] as Citation[] }, isError: true };
}

export function createWebSearchTool(options: WebSearchOptions) {
  return defineTool({
    name: "web_search",
    label: "Web search",
    description:
      "Search the web for current information (news, documentation, prices, facts after your training cutoff). Returns ranked results with titles, URLs and summaries. Cite the URLs you rely on.",
    promptSnippet: "Search the web for current information, with source URLs",
    promptGuidelines: ["Use web_search when the task needs current or external information, read the best sources with web_fetch, and cite the URLs you used."],
    parameters: webSearchSchema,
    annotations: { readOnlyHint: true, openWorldHint: true },
    async execute(_toolCallId, { query }, signal, _onUpdate, ctx) {
      const sessionModel = ctx?.model?.provider === GATEWAY_PROVIDER_ID ? ctx.model.id : undefined;
      const model = sessionModel ?? options.model;
      if (model === undefined) {
        return failure("Web search needs an Experiential gateway model. Switch to one with /model and try again.", query);
      }
      const response = await (options.fetch ?? globalThis.fetch)(`${options.gateway.baseUrl}/chat/completions`, {
        method: "POST",
        signal,
        headers: { "content-type": "application/json", authorization: `Bearer ${options.gateway.apiKey}` },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: SEARCH_INSTRUCTIONS },
            { role: "user", content: query }
          ],
          web_search_options: { search_context_size: "medium" },
          max_tokens: 2000
        })
      });
      const raw = await response.text();
      if (!response.ok) {
        let message = raw.slice(0, 400);
        try {
          message = (JSON.parse(raw) as { error?: { message?: string } }).error?.message ?? message;
        } catch {
          // Keep the raw excerpt.
        }
        return failure(`Web search failed (${response.status}): ${message}`, query);
      }
      if ((response.headers.get("x-experiential-ignored-parameters") ?? "").includes("web_search")) {
        return failure("Web search is unavailable on this gateway right now. Answer from what you know and say so.", query);
      }
      const body = JSON.parse(raw) as { choices?: { message?: { content?: string | null; annotations?: unknown } }[] };
      const message = body.choices?.[0]?.message;
      const answer = (message?.content ?? "").trim();
      const citations = citationsOf(message);
      const sources =
        citations.length > 0 ? `\n\nSources:\n${citations.map((c, i) => `[${i + 1}] ${c.title ? `${c.title} - ` : ""}${c.url}`).join("\n")}` : "";
      return {
        content: [{ type: "text", text: `${answer.length > 0 ? answer : "(no results)"}${sources}` }],
        details: { query, citations }
      };
    }
  });
}

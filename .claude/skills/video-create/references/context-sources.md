# Context: learn the company and find the brand

The film must be one that only this company could post. Capture the **worldview**, not the feature list. Delegate heavy reading (meeting transcripts, big repos) to a subagent and keep only the conclusions. Output: `$ROOT/CONTEXT.md`.

## Sources, in order of value
1. **What the user pasted or linked.** Read all of it first.
2. **Memory** (your persistent memory directory). There may already be a company-vision memory.
3. **The website:** `WebFetch` the homepage, about, manifesto, blog and pricing pages. Pull the hero line, the product sentence and the tone.
4. **The website repo (local),** if it exists (`mdfind -name <company>`, `ls ~/Desktop/Projects | grep -i <company>`):
   - logo component (`app/logo.tsx`, `public/*.svg`, `components/*logo*`)
   - `globals.css` / `tailwind.config` colour tokens
   - `layout.tsx` fonts (`next/font/google` imports)
   - any `prompts/` or `docs/` brand context files
5. **Product repo(s):** README (the one-line pitch), the public GitHub star count (`gh repo view owner/name --json stargazerCount,description,url`) and star history for proof shots.
6. **Meeting notes (Granola):** the public API is at `https://public-api.granola.ai/v1` with header `Authorization: Bearer <grn_ key>`.
   - `GET /notes?page_size=50` (paginate with `&cursor=`)
   - `GET /notes/{id}` (try `?include=transcript`)
   - Find notes with a named co-founder or teammate (check the attendees and the title; names are often misspelled, e.g. Jon/John/Jonn).
   - Pitch calls often state the vision most clearly.
   - Use read-only calls, and save raw data only in /tmp.
7. **Slack / Notion / Drive,** if the tools exist and the user pointed you there.
8. **Downloads:** earlier videos (`ls -t ~/Downloads/*.{mp4,mov}`), decks and brand PDFs. **Watch any earlier company video** so the new one is a step up, not a repeat.

## The subagent brief for heavy sources (template)
> Goal: deeply understand what <company> is building and its vision, to write a cinematic <type> video. I need the essence, not meeting minutiae. [access details]. Return ≤900 words:
> 1. team (names, roles, credible backgrounds);
> 2. products in plain words (who they're for, business model, stage);
> 3. the vision / worldview / "why now", with 5-10 verbatim quotable lines;
> 4. the enemy / status quo and the historical analogy they use;
> 5. brand cues (names, colours, logos, taglines, tone words);
> 6. public milestones and numbers;
> 7. **confidential items that must stay out of a public video**;
> 8. the sources used.

## CONTEXT.md structure
1. **One sentence:** what the company does, plainly.
2. **Worldview:** the belief, why now, the status quo being replaced (a status quo, not a named competitor), and the analogy.
3. **Founder lines:** 5-10 quotable lines, each under 12 words, lightly cleaned.
4. **Proof (public only):** stars, launch date, open source, public users, papers.
5. **KEEP-OUT:** fundraising, investors, valuation, M&A, customer names and sizes, revenue, margins, pipeline, unconfirmed renames, legal or visa matters, internal fraud or security issues.
6. **Naming check:** if different sources disagree on the company name (domain vs. pitch vs. repo), note it; in build mode, pick the one the current website uses.
7. **Brand tokens:** logo paths, colours (hex), fonts, tone words, any existing taglines.
8. **Motif idea:** what the logo *is* (a shape, a metaphor) and how it can become the film's recurring object.

## Brand assets
- Extract inline SVG logos from TSX by pulling out the `<path d>` elements and rebuilding a standalone `<svg>` with the same `viewBox` and transform. Render a PNG with `rsvg-convert -w 1024` or `sips`, and **look at the PNG**.
- Colours: prefer design tokens from CSS. If only a raster exists, sample pixels with numpy (the dominant non-neutral cluster).
- Fonts: use the site's Google fonts. If a font is proprietary and unavailable, pick the closest free match and note it.
- No brand at all: propose a wordmark (font, weight, tracking) plus a 1-accent palette derived from the reference's grade, and log it in `DECISIONS.md`.

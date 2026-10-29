# Interview: question bank

Goal: learn the ~12 things that change the film, in ≤3 rounds of ≤4 questions, using `AskUserQuestion`. Skip anything already answered by pasted text, files, memory, the repo or the website. Every question has a default; if the user skips a question or says "no more questions", use the default and log it in `DECISIONS.md`.

Good questions offer concrete options with trade-offs, not open prompts. Use `preview` for visual choices (ASCII frame mockups, beat-map snippets).

---

## Round 1 (always, unless already answered)

**Q1. Purpose + platform: "What's this video for, and where does it live?"**
- Manifesto / brand film on X or LinkedIn (vision-first, 60-110s)
- Product launch (hook → product reveal → CTA, 30-75s)
- Feature announcement / changelog (15-40s, one feature, one aha)
- Fundraise / recruiting film (vision + team + "join us")
- Explainer / demo for the website (60-120s, clarity over spectacle)
- Social cut (6-20s vertical, loopable)

Default: product launch on X.
*Why it matters:* it decides length, aspect, narrative pattern (see `film-patterns.md`), CTA and pacing.

**Q2. References: "Show me 1-2 videos whose style you want."** Accept paths, URLs or names.
- If they have none, offer 2-3 options from the reference library below with a one-line description each, and recommend one.
- If a URL: download it with `yt-dlp -f "bv*[height<=1080]+ba/b" -o "$ROOT/refs/REF_<name>.%(ext)s" <url>`. For X links use yt-dlp as well. If that fails, ask them to drop the file in `~/Downloads` and pick the newest mp4 there (`ls -t ~/Downloads/*.mp4 | head`).
- "The two most recent mp4s in my downloads" means `ls -t ~/Downloads/*.mp4 | head -2`. Confirm the filenames in your next message rather than asking.

Default: pick from the library by purpose.
If the user gives a reference and says "like this", "very similar" or "replicate this", that is **style-match mode** (`style-match-mode.md`). Don't ask about style, pacing or type: the SPEC answers all of those. Ask only for:
- the subject's footage (and any exclusions, e.g. "no video calls");
- the tone, offered alongside 2–3 spine options built from the subject's own lines;
- the music: the reference's track (Shazam it) or another.

*Why:* without a reference the model falls back to its default look. A reference gives the pacing, the type and the transitions to copy.

**Q3. Voice: "How should it sound?"**
- AI narrator (ElevenLabs) + archival voices answering (the manifesto formula)
- AI narrator only
- Founder records the VO (you give them a script and a recording guide)
- No voice: music + on-screen type + SFX (the launch/one-take formula)

Default: follow the reference. If the reference has narration, use the AI narrator (+ archival if the reference uses it).

**Q4. Visual sources: "What can the film be made of?"** (multi-select)
- Code-only motion graphics (always on)
- Real product screen recordings (you record them with Playwright, or they provide some)
- AI-generated cinematic b-roll (needs a key; see keys.md)
- Archival / public footage (you find and verify it)
- Stock footage (Pexels / Pixabay)
- Their own photos or video

Default: code + product recordings + whatever the reference relies on.

## Round 2 (only what's still unknown)

**Q5. Brand:** "Logo, colours, fonts?"
- I'll provide them
- Find them locally and online (search the repo, website and Downloads; see context-sources.md)
- Design an end-card mark (propose a wordmark treatment)

Default: find them.

**Q6. Company context:** "Where do I learn what you're building and why?" (repo path, website, docs, Granola API key or links, Notion, a pitch deck, "just ask me"). Default: repo + website. This is about vision and worldview, not meeting minutiae.

**Q7. Claims and confidentiality:** "Which numbers can go on screen, and what must stay out?"
- Only public facts (stars, launch date, open source)
- These specific numbers: …
- No numbers

Default: public facts only. Always build a KEEP-OUT list (fundraising, customers, revenue, unannounced deals) even if they don't mention one. If numbers are restricted, UI mock-ups use words (days, "open tickets", "ok"), and facts that come only from the brief are flagged as unverified in the final report.

**Q8. Length + aspect:** "Length and format?"
- Match the reference
- 16:9 1920x1080
- 1:1 1080x1080
- 9:16 1080x1920
- 4:3 1440x1080

Default: match the reference's length and aspect, plus one 1:1 cut if it's going on X.

## Round 3 (optional; only if critical)

**Q9. The one line:** "If viewers remember one sentence, what is it?" Default: derive it from the worldview; propose 3 and pick one.
**Q10. CTA / end card:** URL, handle, "join us", waitlist, none. Default: logo + wordmark + URL.
**Q11. Music taste:** "Cinematic build with a drop / minimal electronic / warm piano / match the reference." Default: match the reference's tempo and arc.
**Q12a. Pacing + captions:** "Match the reference's pacing, or a snappier cut for the feed? Captions on archival quotes?" Default: reference pacing for the body, a snappier intro for X/social, and captions on archival quotes when the film will autoplay muted.
**Q12. Deadline + quality bar:** "Fast draft today or the full pipeline with gates?" Default: full pipeline.

## Things you decide yourself (never ask)
Fonts when not provided (pick from the reference's feel: Geist, Inter Display, Archivo, Schibsted Grotesk, IBM Plex, JetBrains Mono), frame rate (the reference's, or 30/60), grade, the motif, the beat map, transitions, SFX choices, narrator voice (audition 3 and pick), which storyboard wins (unless the user wants to choose), and file layout.

---

## Reference library (suggest these when the user has none)
Ask them to verify the link works. These are descriptions of what to look for. A good source of fresh launch films is whatships.com. Use `WebSearch` to find current examples of each style.

| Style | What it gives you | Good for |
|---|---|---|
| **Manifesto / "civilizational"** (e.g. a16z "It's time to build", Browser Use "let go of the mouse", Apple "Think Different") | History → shift → us → imperative → metallic logo; a calm narrator plus archival voices; one motif; a drop at ~35-40% of runtime | brand films, fundraise, big launches |
| **Apple-keynote one-take morph** | No cuts. Every scene is made from the previous one: a shape floods and contracts, text rises from mask lines, a cursor drives every change, screen-studio zooms | product launches with UI |
| **Screen-studio product demo** | Real UI, smooth zooms to the click, cursor scaled with the camera, captions | feature announcements, website demos |
| **Kinetic typography** | Type as the hero, beat-synced, bold variable fonts | short social, taglines |
| **Linear / Vercel launch** | Dark, precise, glowing edges, 3D-ish UI planes, minimal VO | dev tools |
| **Documentary / founder story** | Interviews, b-roll, warm grade, a slow push-in | recruiting, fundraising |

---
name: video-create
description: Make a pro-level video (launch film, manifesto/brand film, product demo, feature announcement, explainer, social cut, a frame-locked remake of a reference, or a style-matched edit like a reference (sigma, hype or founder edit) with the user's own footage) entirely with code plus optional TTS, AI b-roll and archival footage. It interviews the user, collects brand and company context and API keys, measures reference videos, writes a director's brief, then builds, renders and iterates using director notes. Use it when the user asks to make, direct, plan or remake a video, asks for a video prompt, or points at a video and says "make one like this".
---

# /video-create: from idea to finished film

You are the **director, editor, motion designer and sound designer**. The output is a finished mp4 with audio, rendered from source that can re-render the whole film with one command. Or, in **planner mode**, the output is a self-contained brief that another chat can execute.

The core belief behind this skill: *everyone has the same model; the context you give it is what makes it look pro.* One-prompt videos look mid because the model falls back to its default look: centred text, gradient background, everything fading in. This skill exists to replace defaults with context:
- **references** give you pacing, type and transitions;
- **brand and product assets** make the film recognizably this company's;
- **the company's real worldview** gives it the story;
- **measured numbers** replace impressions;
- **build rules** make every frame exact and every fix a one-line edit;
- **gates** (storyboards → stills → animatic → render) catch problems while fixes are still cheap.

## Files in this skill (read them when the phase needs them)
| File | Read it when |
|---|---|
| `references/interview.md` | Phase 1: the full question bank, defaults and batching rules |
| `references/keys.md` | Phase 2: which API keys unlock what, where to get them, how to store them, and fallbacks |
| `references/context-sources.md` | Phase 3: how to learn the company (repo, website, docs, Granola, Slack, Notion) and find brand assets |
| `references/film-patterns.md` | Phases 4-5: proven structures (manifesto, launch, one-take morph, demo, explainer, social), including measured examples |
| `references/build-rules.md` | Phase 6: engine, render pipeline, gotchas and pitfalls, verbatim. **Non-negotiable.** |
| `references/audio.md` | Phase 6: music, SFX, ElevenLabs VO, mixing and loudness |
| `references/failure-modes.md` | **Every phase.** Real problems from past builds, each written as "to avoid X, do Y". Skim the section for the phase you're entering. |
| `references/remake-mode.md` | Only when the user wants a frame-locked 1:1 remake of a reference |
| `references/style-match-mode.md` | **Whenever the user points at a reference edit and wants "something very similar" with their own subject and footage** (sigma, hype and founder edits). Read it before sourcing anything. |
| `scripts/fa_align.py` | Exact word times by forced alignment, for caption slams and speech cuts. Whisper timestamps drift 0.3–0.8 s. |
| `scripts/uniq_check.py` | Ship gate: the no-repeat rule over the shot table |
| `templates/BRIEF.template.md` | Phase 5: the brief you fill in (and hand off, in planner mode) |
| `examples/style-match-sigma-edit.md` | Phase 5 (style-match mode): a complete, frame-exact brief at the depth expected (every cut, caption, poster and transition measured) |
| `examples/remake-edge-of-the-world.md` | Phase 5 (remake mode): a complete frame-locked remake prompt at the depth expected |
| `scripts/analyze_ref.py` | Phase 4: measures a reference video (cuts per 10s, longest takes, loudness arc, silences, BPM, transcript, contact sheet) |

## Modes
- **Build mode (default):** run all phases yourself.
- **Planner mode:** the user says "just write the prompt", "brief for another chat" or similar. Run phases 1-5 and deliver `BRIEF.md`. The brief must stand alone: absolute paths, every decision made, and build-rules/audio/pitfalls pasted **verbatim**, not summarized. Put the answers to every question another chat would need to ask in the brief's decisions section.
- **Remake mode:** the user wants a frame-locked 1:1 remake of one reference with a brand swap. Follow `references/remake-mode.md`, which replaces phases 4-6.
- **Style-match mode (the default when a reference is given and the content differs):** the user says "make something like this" with their own footage. Follow `references/style-match-mode.md`. In short:
  1. Measure first: a frame-exact SPEC before any sourcing.
  2. Write a source budget: at least the reference's distinct-source count, matching its real-vs-AI mix.
  3. Build the spine on the reference's thesis.
  4. Lock to the reference's frames when the song is the same.
  5. Get word times by forced alignment.
  6. Treat mix QA and the no-repeat rule as ship gates.
  The goal is one pass: pointing at the reference should be enough.

## Phase 0: Set up the project (1 minute)
- Create `~/Desktop/Projects/<slug>-film/` (or wherever the user says) with `brand/ refs/ footage/{archival,ai,product,stock} vo/ music/ sfx/ stills/ storyboards/ out/`. Call this `$ROOT`.
- Check the tools: `ffmpeg`, `ffprobe`, `python3 -c "import numpy"`, `node`, `npx playwright --version`, `yt-dlp`, `whisper-cli`, `gh`. Install what's missing (brew/pip/npm) without asking, and mention it in your report.
- Whisper model: if none is found (`find ~ -name "ggml-*.bin" -size +50M 2>/dev/null | head -1`, bounded), download `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin` into `$ROOT/refs/`.

## Phase 1: Interview (get the right video, not just a video)
Read `references/interview.md`. Rules:
- **Mine before you ask.** Read everything the user already gave you (pasted prompts, files, links, repo, memory) and don't ask what you can answer.
- Ask with `AskUserQuestion`: **at most 4 questions per round and at most 3 rounds.** Put your recommended option first with "(Recommended)". Every question carries a default you'll use if they skip it.
- Round 1 is always: **purpose + platform**, **video type**, **references**, **voice**. Round 2 covers brand/assets, footage sources, claims/confidentiality and length/aspect, unless they're already answered. Round 3 only if something critical is still unknown (usually the one line the film must leave behind).
- If the user says "no more questions", "just do it" or similar: **stop asking immediately**, take every default, record them in `DECISIONS.md`, and never ask again in this project (not even small questions; decide and log them).
- If the user has no reference, propose 2-3 concrete ones (see the "reference library" section in `interview.md`) and pick one by default. **Naming a style works far better than describing one.**

## Phase 2: Keys and accounts
Read `references/keys.md`. Check what's already available (env vars, `$ROOT/.env`, `~/.config`, `gh auth status`) **before** asking. Then ask once, in a single message, for only the keys that unlock something this film actually needs. For each key, give a one-line reason, a signup link and what happens without it (every key has a code-only fallback). Store keys in `$ROOT/.env` with `chmod 600`. Never print, log, commit or paste a key into any file other than `.env`. If a key arrives in chat, suggest rotating it after the project.

## Phase 3: Context (make it a film only this company could post)
Read `references/context-sources.md`. Build `$ROOT/CONTEXT.md` covering:
1. what the company does, in one plain sentence;
2. the worldview (the belief, the "why now", the enemy as a status quo rather than a competitor);
3. 5-10 quotable founder lines;
4. proof points that are public;
5. **a KEEP-OUT list** (fundraising, customers, revenue, unannounced deals, anything confidential);
6. brand tokens: logo files, hex colours, fonts, tone words.

Copy brand assets into `$ROOT/brand/`, convert logos to SVG and PNG, and look at them. **Find the logo's visual idea**: the logo usually suggests the film's signature motif, the way a cursor became Browser Use's motif and an orb became a16z's.

## Phase 4: Measure the references (numbers, not impressions)
For each reference:
`python3 ~/.claude/skills/video-create/scripts/analyze_ref.py <ref.mp4> $ROOT/refs/analysis --label <name> --whisper-model <model>`
Then look at `contact.jpg` and pull dense 1-fps sheets across the key moments: the quiet window, the longest uncut take, the ending. Write `$ROOT/SPEC.md` with:
- the shot table (id, t0-t1, content, transition in/out, camera move);
- cuts per 10s;
- longest takes;
- the loudness arc and silences;
- the drop time;
- the logo-reveal mechanics;
- a VO table (line, start, end, narrator/archival/on-screen);
- colour tokens sampled from pixels;
- type sizes measured from cap height;
- grade traits (grain, vignette, bloom).

**Measure with numpy on the frames; never eyeball.** Where your impression (or the user's brief) and the measurement disagree, the measurement wins. Open SPEC.md with a corrections table (brief said / measured / consequence), and confirm the tempo from the reference's cut spacing, not autocorrelation alone. Then write a "mechanics to copy" list: structure, motif behaviour, pacing curve, transitions, ending. Copy the mechanics, never the content.

## Phase 5: Brief and storyboards
1. Fill in `templates/BRIEF.template.md` → `$ROOT/BRIEF.md`, at the depth of the examples in `examples/` (exact frames, px positions, per-frame arrays, gotchas and a stills gate). A brief that says "fast cuts, bold captions" instead of frame numbers isn't finished. The brief contains:
   - the decisions;
   - the measured reference table;
   - company context and the keep-out list;
   - the motif arc;
   - a beat map with timecodes aligned to the reference structure (silence, drop and ending land where the references land them);
   - the shot sources;
   - the build rules and pitfalls pasted **verbatim** from `references/build-rules.md`;
   - the banned list;
   - the director-notes protocol;
   - the deliverables.
2. Write **3 storyboard variants** (`storyboards/v1-v3.md`). Each has a beat table, the full VO script with a speaker per line, the motif arc, and a cut-density curve compared against the references. Pick the best one (or ask, if the user wants to choose and hasn't said "no more questions"). Three variants means choosing a direction instead of polishing the first idea.
3. Planner mode ends here: deliver `BRIEF.md` (copy it to the clipboard with `pbcopy` and, if asked, send it to the named session).

## Phase 6: Build (follow `references/build-rules.md` exactly)
Gates, in order. Each ends with a short report to the user, including the images.
1. **Engine:** a pure `seek(t)` HTML engine, `core.js` and `render.mjs`, before any shots.
2. **Stills:** one still per scene before anything moves. Compare each one side by side with the closest reference frame (`render.mjs compare`) and fix it. A still takes seconds to change; a render doesn't. Without this step you only find out scene 4 is wrong after the whole thing is animated, and every fix means re-rendering.
3. **Animatic:** stills on the timeline with scratch VO and music, to lock timing to the beat grid.
4. **Parallel build:** 4 contiguous shot groups, one agent each, plus an audio agent (see build-rules). Use fewer agents for films under 45s.
5. **Integrate:** the full render, the mix, contact sheets, seam sheets, the pop scan, the palette scan and the banned-item scan.

**Lead review is not optional.** Before approving any group, sample its frames yourself, compare them with the refs side by side, and run a **rule audit**: legible numbers, logos/brands/real company names, invented data, uncleared people, readability. Agents' "MATCHES" reports have missed all five. Expect agent messages to cross; verify against files and fresh renders, not reports.

## Phase 7: Director-notes loop
The first render is usually 80% there, and the last 20% is what makes it look pro. Take notes in camera words ("slow every zoom to 0.7x", "hard cut here", "push in on the button", "the drop is 3 frames late"). Apply exactly the note, only to the affected shots. Re-render only those frame ranges and show before|after stills. If a note is vague ("make it better"), turn it into 2-3 concrete camera-word options, apply the one closest to the references, and say which one you chose. Vague notes applied literally produce random changes.

## Phase 8: Revisions (v2, v3…)
- Snapshot v1 (`versions/v1/`: source, SCRIPT, mix, cues, crop plan) and render the new version to `out2/` via an `OUTDIR` the whole pipeline respects. Never overwrite the delivered version.
- Map each owner note to what it touches (script, one group, audio, all), and rebuild only that. Re-slot the rest.
- Pacing notes ("snappier", "not on beat"): have the audio agent publish an onset map from music with a clear pulse, snap every cut to a real transient, add a transition vocabulary (match cuts, push-in punches on hits, a stutter, an overscaled flash-cut), and send the locked cut list back to audio for accents.
- Coherence notes ("this line doesn't make sense"): rewrite the through-line sentence first, then re-test every line (archival included) against it.
- Log owner overrides of earlier rules in DECISIONS.md.

## Deliverables (every build)
- `out/<slug>-<W>x<H>.mp4` (h264 + AAC), plus the extra aspect cuts requested (re-framed per shot, never blindly centre-cropped)
- `out/contact-sheet.jpg` (1 fps) and `out/beats.jpg` (one frame per beat)
- `SPEC.md`, `CONTEXT.md`, `DECISIONS.md`, `BRIEF.md`, `SCRIPT.md` (final VO with a speaker per line), `CLIPS.md` (archival: URL, in/out, verbatim line), `MUSIC.md` + `SFX.md` (source and licence), `AI_SHOTS.md` (prompts, plus which shots are still stand-ins)
- The source, re-renderable with one command (`npm run film` or `make film`)
- An honest list of the remaining weaknesses with frame numbers, and what one more pass would fix. It must include anything verified only by analysis ("mixed by numbers, not by ear"), unverified facts, fair-use clips and billing surprises.
- An upload-sized encode next to the master (grain inflates bitrate; e.g. crf ~19, maxrate ~20 Mbps), and captions if requested (see failure-modes.md → Captions).

## Hard rules
- Never invent a quote, statistic, chart or claim. Every archival line is verified by two transcription passes (and forced alignment for caption timing). Every number on screen comes from a public source or was approved by the user. No chart is drawn from interpolated points. When the brief limits numbers, UI copy uses words.
- Never reuse a reference's music, voice or footage in the output.
- Record every third-party asset's licence. Prefer public-domain archival (Prelinger / archive.org). Flag fair-use risk for news or interview clips in `CLIPS.md`.
- Never show a KEEP-OUT item. Never frame named competitors as villains unless the user explicitly asks.
- Never claim a shot matches the references without viewing ref|ours side by side.
- When you say something is done, you have watched the frames: contact sheet, seams and beats.

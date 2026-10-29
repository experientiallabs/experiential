# Build rules (non-negotiable)

These rules come from people who shipped films that looked pro. The **bold / quoted** lines are verbatim from those builds; keep their wording when you paste them into a brief.

## Why code
The whole film is code: every scene is a function of time, rendered straight to mp4, with no video editor. Every frame is exact, and when a change is requested you edit one line and re-render that range instead of starting over. Without this, you can only describe a video or hand over a rough HTML page someone has to screen-record.

Renderer choice:
- **Default: a custom HTML `seek(t)` engine + Playwright** (below). It gives full control over glass, masks and footage, and every rule and pitfall here applies directly.
- **Remotion / HyperFrames** are acceptable if the user asks for them or the project already uses one. The same purity rules apply (no timers, no state between frames, seeded randomness).
- For UI inside the film, use real-quality components (21st.dev-level: real buttons, cards, spacing), never placeholder boxes. Anyone who's used good software can feel fake UI in a second, and the whole video reads as cheap. When the product is real, record it or rebuild its real screens pixel-faithfully from screenshots.

## PHASE 0: ANALYSIS (no building yet)
- ffprobe fps/res/duration. Extract ALL frames 0-based to `ref/full/fNNNN.jpg` + audio to `ref/audio.wav`. Make 1fps contact sheets for an overview. (Run `scripts/analyze_ref.py` first; it does the overview numbers.)
- Detect cuts via per-frame mean-abs-diff spikes and confirm them visually. Write SPEC.md: shot table (id, f0–f1, content, transition in/out), per-shot detail, component inventory, verbatim on-screen text in order, colour tokens sampled from pixels, font sizes from cap height, cursor paths (tip xy per frame), camera keyframes.
- **MEASURE with numpy on ref frames (ink bounding boxes, flood fills, template match), never eyeball.** Store per-frame sample arrays for every major move; **the spec's prose is a guide, the ref frames are the truth.**
- Audio: STT with word timestamps (narration table: line, start, end, text). Music BPM + beat phase (onset autocorrelation), drop time, loudness arc per section, hard-stop time. SFX hit times from onset/spectral analysis. Voice pitch/wpm.

## PHASE 1: ENGINE (you, before any agents)
- A single HTML page with the stage at the target resolution. **`window.seek(t)` renders frame F=t*fps as a PURE function of F: no timers, no Date, no Math.random (use a seeded hash), no CSS transitions/animations.** Shots register `SHOT({id,f0,f1,render(lf,F)})` returning HTML (or canvas draws); seek routes F to its shot. **`window.ready=true` only after document.fonts has loaded + all images are decoded.**
- **"Every style is computed from time inside an async seek(t): no CSS transitions, no timers, no state between frames."**
- **"Springs are closed-form step responses. A value with many targets is the sum of one spring per change, so it stays a pure function of time."**
- `core.js` shared helpers:
  - easing set, `kf(F,keys,ease)`, `samples(F,f0,arr)` (interpolates measured arrays)
  - `camera(inner,scale,tx,ty,origin,blur)` and directional motion blur (SVG feGaussianBlur)
  - cursor (glyph measured from the reference, press-scale curve measured), ripple
  - text reveal (rise out of a mask line), line-draw (stroke-dashoffset from time)
  - brand tokens; the logo component (mask-image of the logo, so any fill, gradient, mosaic or chrome works)
  - the grade overlay (seeded per-frame grain, vignette, bloom, chromatic edge)
  - footage-frame drawing
- **Palette as ONE token set** in core.js. Shots never hard-code hex values. (In a brand-swap remake, use one deterministic filter applied to each rendered HTML string instead, with hex/rgb/rgba re-hued by band and lightness preserved, so hard-coded colours can't leak the old brand. Bitmaps are untouched.)
- **"Footage: re-encode all-intra (ffmpeg -g 1), load it as a blob URL, await 'seeked' before drawing each frame."**
- A proven alternative to blob-URL video: extract each clip to a 30fps JPEG sequence and draw frame N as image N, served by render.mjs's own range-capable HTTP server. It's exact and seek-free.
- For canvas-only engines: do the grade in-page on the final canvas, and average the 4 subframes in-page (a running mean) routed by the frame's integer shot. That's equivalent to tmix, with 4× fewer PNGs, grain applied once per frame, and cuts never smeared.
- `render.mjs` (Playwright Chromium, deviceScaleFactor 1) with modes `stills <frames>` | `compare <frames>` (ref left | ours right, plus a labelled sheet) | `full <f0> <f1>`. **Per-agent OUT dirs so parallel runs don't collide. Print page errors.**
- **"Render with Playwright: 4 subframes per frame blended with ffmpeg tmix, 60fps [or the target fps]. Check one frame per beat, then scan for single-frame pops (frame-difference spikes 3x their neighbours)."**
- `encode.py`: PNG frames → h264 at the target fps, mux the audio. `sync.py` (when there's a reference to match): REF over ours, stacked and frame-locked, as an mp4.

### Advanced effects (when the look needs them)
- **"Liquid glass: each glass element holds its own clone of the scene behind it, filtered with an SVG feImage displacement map (a rounded-rect distance field) through three feDisplacementMaps at slightly different scales for chromatic edges, plus a rim light. Glass letters: a canvas distance field per glyph gives the map, mask and highlights."**
- **"Goo: blur + alpha threshold, then composite the source atop it so the glass stays sharp inside."**
- **"Iris: 6 blades around a hexagonal aperture. Each blade is its two vertices, both edge extensions and the SHORT arc between them."**
- **"Wordmark squeeze: every letter moves toward the dot by the same factor and its drawn width follows (narrow the wdth axis, scale the rest), so the letters stay touching."**
- Mosaic → logo: a grid of N tiles (seeded layout) whose visibility is masked by the logo's SDF; the negative shape cuts through first, then the positive fill, then the fill resolves into the material (chrome/gold) via a per-pixel mix driven by time.

## GATES before building shots
1. **3 storyboard variants.** Pick one (or let the user pick). Three variants means choosing a direction instead of fixing the first idea.
2. **One still frame per scene before anything moves.** Compare each against the closest ref frame. **"Fixing a storyboard is way cheaper than fixing a render."** Without this step you only find out scene 4 is wrong after the whole thing is animated, and every fix means re-rendering. A still frame takes seconds to change.
3. **Animatic:** stills on the timeline with scratch VO and music, to lock timing to the beat grid.

## PHASE 2: PARALLEL BUILD
- Split shots into 4 contiguous groups, one agent each. Each writes ONLY `shots/<G>.js` (an IIFE, helpers prefixed `<G>_`) and **never edits core.js (it requests changes from you)**. Give each: `BRIEF.md` (acceptance bar, file rules, verify loop), its SPEC.md sections, the core.js API, and the look + banned sections of the brief verbatim.
- Verify loop per shot: compare the first and last frame, every keyframe, and 2 frames into each transition; iterate until within tolerance. **≤15 frames per render call; one render at a time per agent.**
- Report table: shot | frames | MATCHES/CLOSE/ROUGH | residual diff | frames that would drift/pop | spec errors found vs ref.
- **5th agent = audio** (see audio.md).
- Films under 45s: 2 shot agents + audio, or build it yourself.

## PHASE 3: INTEGRATE
- Unify shared glyphs and components across groups (cursor, motif, windows, logo, grade): one definition, and everyone uses it.
- Full render of all frames (2–3 parallel chunks max). Encode with the mix. Run sync.py if there's a match target. Build a 1-per-second contact sheet plus sheets at every group seam (last 2 / first 2 frames). **Inspect. Fix drift. Re-render.**
- Scan all frames for: single-frame pops, off-palette colours (and leftover old-brand colour in remakes), unreadable text, KEEP-OUT items and banned items.
- VO: generate per line, trim the silence, place it on its slot, re-mix.

## GOTCHAS (verbatim)
- **backdrop-filter: url() misreads displacement maps in Chromium, so clone the scene instead.**
- **A flood must overscale past the corners and take about 0.3s, or half the screen changes in one frame.**
- **A child with visibility: visible shows through a hidden parent, so use inherit.**
- **Text that swaps inside a morphing shape needs its own mask.**
- **python http.server can't range-seek video, so use the blob URL.**

## PITFALLS (seen in practice, verbatim)
- **Measure text widths only AFTER fonts load (measureText before load caches fallback widths → words collide). Lazy-init any width tables.**
- **Word-length changes from the brand swap shift centred layouts; keep shared elements on ref positions, absorb the delta in the swapped word, report it.**
- **Randomised bursts/particles won't match stacked; drive the visible ones from measured tracks** (or seeded deterministic tracks when there's no match target).
- **Ref whips may be crisp (no blur); check before adding motion blur.**
- **Headless Chromium may need to run outside any OS sandbox; use swiftshader/ANGLE if WebGL is involved.**
- **Never claim a shot matches without viewing ref|ours side by side for it.**

Added from later builds:
- A flash that lasts one frame reads as a glitch, not a cut.
- Archival and stock clips: re-encode to the target fps, all-intra, before use, or seeks land on the wrong frame.
- ElevenLabs: one file per line; trim leading and trailing silence (`silenceremove`); re-check a line's slot whenever you regenerate it.
- Fonts from Google: self-host the woff2 in the project so renders don't depend on the network.
- Long renders: write frames to disk as they finish and make render.mjs resumable (skip existing frames).
- Grain must change every frame (seeded by F) but be identical across re-renders of the same frame. Apply it with `overlay` so pure black stays 0.
- Canvas masks: paint multi-step fills on a scratch canvas, then composite under `source-in` (painting them directly under `source-in` leaves only the last rect). Fill channel buffers black before chromatic-aberration scaling. Reset `filter`/composite/alpha on every reused helper canvas (a leftover filter blacked out a whole session).
- Subframe blur averages within a shot: shots with internal cuts (panel sequences, stutters) need `noBlur`.
- Highlighters need a mode: multiply on paper/light UI, screen on dark UI.
- Archival voice clips: also export a padded `<id>_pic.mp4` (2.000s of handles, phrase at frame 60) so speaker shots never freeze.
- Dump the shot table from the engine (`render.mjs shots`) to drive per-shot reframing: a content-centroid crop plus overrides for text-led shots, then check the square cut for clipped text.
- Everything that writes output reads an `OUTDIR`, so v2 renders beside v1.
- Rule audit before approval (legible digits, logos/brands, invented data, uncleared faces, readability): see failure-modes.md.

## BANNED (default; extend it per film)
Crossfades as the default transition, blur-ins, brightness "developing", everything fading in, centred text on a gradient, 3D flips, generic particles and glows, stock "AI brain", circuit boards, blue holograms, template lower-thirds, holds longer than 1s (except deliberate silences and the final logo), and **anything that looks like a template**. If a shot could be in another company's video, cut it. Transitions are **made from the previous shot** (a shape floods and contracts into the next scene, a line becomes an edge, text rises out of a mask line, icons pop from zero on a spring, bars draw across, pages push) or are hard cuts on the beat.

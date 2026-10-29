# Remake mode: frame-locked 1:1 remake of a reference

Use this only when the user wants the SAME film with a brand swap. It replaces phases 4-6. The text below is the proven prompt, verbatim, with slots filled from the interview.

```
TASK: frame-locked 1:1 remake of REF=[path/to/reference.mp4]. Swap ONLY: brand name→[NAME], logo→[path/to/logo.png|svg], palette→[hex list or "derive from logo"], platform UI→[e.g. LinkedIn→X], faces/screens→[source]. All else identical: layout, sizes, positions, timing, easing, camera, cuts, blur, cursor path, typing cadence, audio beat grid.
ACCEPTANCE: REF and remake stacked + frame-locked → same pose every frame. Cuts 0 frames off. Position/size error ≤1% of frame. Typing char-count, cursor, camera on the same frames. Differences allowed only where the swap forces them (word widths); report each.

PHASE 0 — ANALYSIS (no building yet)
- ffprobe fps/res/duration. Extract ALL frames 0-based to ref/full/fNNNN.jpg + audio to ref/audio.wav. 1fps contact sheets for overview.
- Detect cuts via per-frame mean-abs-diff spikes; confirm visually. Write SPEC.md: shot table (id, f0–f1, content, transition in/out), per-shot detail, component inventory, verbatim on-screen text in order, colour tokens sampled from pixels, font sizes from cap height, cursor paths (tip xy per frame), camera keyframes.
- MEASURE with numpy on ref frames (ink bounding boxes, flood fills, template match), never eyeball. Store per-frame sample arrays for every major move; the spec's prose is a guide, ref/full is truth.
- Audio: STT with word timestamps (narration table: line, start, end, text). Music BPM + beat phase (onset autocorrelation), drop time, loudness arc per section, hard-stop time. SFX hit times from onset/spectral analysis. Voice pitch/wpm.

PHASE 1 — ENGINE (you, before any agents)
- Single HTML page, stage at REF native resolution. window.seek(t) renders frame F=t*fps as a PURE function of F: no timers, no Date, no Math.random (seeded hash), no CSS transitions/animations. Shots register SHOT({id,f0,f1,render(lf,F)}) returning HTML; seek routes F to its shot. window.ready=true only after document.fonts loaded + all images decoded.
- core.js shared helpers: easing set, kf(F,keys,ease), samples(F,f0,arr) interpolating measured arrays, camera(inner,scale,tx,ty,origin,blur), directional motion blur (SVG feGaussianBlur), cursor (glyph MEASURED from ref, press-scale curve measured), ripple, text reveal, brand tokens, logo component (mask-image of logo so any fill/gradient works), avatar/persona pickers.
- Palette swap as ONE deterministic filter applied to each rendered HTML string (hex/rgb/rgba re-hued by band, lightness preserved) so hard-coded colours in shots can't leak the old brand. Bitmaps untouched.
- render.mjs (Playwright Chromium, deviceScaleFactor 1): modes stills <frames> | compare <frames> (ref left | ours right + labelled sheet) | full <f0> <f1>. Per-agent OUT dirs so parallel runs don't collide. Print page errors.
- sync.py: REF over remake stacked, frame-locked mp4. encode.py: PNG frames → h264 at REF fps, mux audio.

PHASE 2 — PARALLEL BUILD
- Split shots into 4 contiguous groups; one agent each; each writes ONLY shots/<G>.js (IIFE, helpers prefixed <G>_), never edits core.js (request changes from you). Give each: BRIEF.md (acceptance bar, swap rules, file rules, verify loop), SPEC.md sections, core.js API.
- Verify loop per shot: compare first/last frame, every keyframe, 2 frames into each transition; iterate until within tolerance. ≤15 frames per render call; one render at a time per agent.
- Report table: shot | frames | MATCHES/CLOSE/ROUGH | residual diff | frames that would drift when stacked | spec errors found vs ref.
- 5th agent = audio: royalty-free commercial-OK track (record URL+licence), time-stretch to REF BPM, cut on beats so drop/breaks/silences land on REF times; SFX on REF hit times; mix script with VO slot table (REF start times, text swapped) that fits each line (≤8% stretch), ducks music under voice, loudness-normalizes (~-14 LUFS, TP ≤ -1). Never reuse REF music or voice.

PHASE 3 — INTEGRATE
- Unify shared glyphs/components across groups (cursor, DM window, logo) — one definition, everyone uses it.
- Full render all frames (2–3 parallel chunks max), encode with mix, run sync.py, build 1-per-second ref|ours contact sheet + sheets at every group seam (last 2 / first 2 frames). Inspect. Fix drift. Re-render. Scan all frames for leftover old-brand colour.
- VO: generate per line (any TTS), trim silence, place on slots, re-mix.

PITFALLS (seen in practice)
- Measure text widths only AFTER fonts load (measureText before load caches fallback widths → words collide). Lazy-init any width tables.
- Word-length changes from the brand swap shift centred layouts; keep shared elements on ref positions, absorb the delta in the swapped word, report it.
- Randomised bursts/particles won't match stacked; drive the visible ones from measured tracks.
- Ref whips may be crisp (no blur); check before adding motion blur.
- Headless Chromium may need to run outside any OS sandbox; use swiftshader/ANGLE if WebGL is involved.
- Never claim a shot matches without viewing ref|ours side by side for it.

DELIVER: remake.mp4 (with audio), sync-check.mp4, SPEC.md, source, and a list of every remaining difference vs REF with frame numbers.
```

Note: the SFX line has been changed from "synthesized (numpy)" to "downloaded, placed by measured peak" (see audio.md). Downloaded SFX sound better; synthesize only when nothing suitable can be downloaded.

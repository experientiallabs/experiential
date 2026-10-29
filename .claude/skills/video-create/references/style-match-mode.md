# Style-match mode: "make this, but about [subject], using [footage]"

**Use this mode when** the user points at a reference edit and asks for something "very similar" with their own subject and footage. Examples: sigma edits, hype reels, founder edits, the "AI Dario Amodei sigma edit".
- It is the default whenever a reference is given and the content differs.
- It differs from `remake-mode.md`, which keeps the same film and only swaps the brand.

**The goal:** one pass, no "it repeats the same clips" round trip.

**Where it came from:** a real sigma-edit build. v1 had to be redone because:
- it used 12 sources against the reference's ~70;
- it used AI b-roll only, where the reference is about a third real footage;
- its spine didn't follow the reference's thesis, and it opened with the same phrase twice;
- its captions were timed from drifting whisper timestamps.

Every rule below closes one of those gaps.

## Order of work (do not reorder)

### 1. MEASURE FIRST, then source
Before downloading or generating anything, spawn a measurement agent with the SPEC prompt below and wait for `SPEC.md`. Sourcing, the script and the timeline all derive from its shot table. If you start sourcing first, you plan for the wrong number and kind of shots.

**SPEC agent prompt (fill in the paths).** Write a frame-exact SPEC of REF so it can be re-made shot for shot without watching it:
1. **Shot table:** f0–f1 with no gaps; content; the material type (speaker / speaker poster / REAL b-roll / AI b-roll / graphic); transitions IN and OUT, named, with per-frame numbers; the camera move measured by tracking; the grade; the beat it lands on.
2. **Caption table:** every state; the font identified by IoU overlay against candidates, system fonts included; widths, cap heights and baselines; the reveal recipe with per-frame numbers; colours sampled from pixels.
3. **Poster and signature-move recipes.**
4. **Transition recipes** reproducible in canvas, with per-frame scale/blur/exposure arrays.
5. **Pacing:** cuts per 5 s and the silent windows.
6. **Audio:** the music edit points found by cross-correlating against the clean track (do not assume an edit), the ducking dB, and the integrated LUFS.

Make stills sheets for every shot. Measure; don't eyeball.

### 2. MATERIAL INVENTORY → SOURCE BUDGET
From the SPEC shot table, write `SOURCES.md` before sourcing:
- **The count:** the number of distinct sources the reference uses, by material type. Our budget is at least that number, by type. A reference with 70 shots needs about 50 distinct sources, not 12 plates.
- **The real-vs-AI ratio:** if the reference uses real footage (launches, microscopy, cities, crowds), source REAL footage for those slots. The public-domain sources:
  - the NASA media API: images-api.nasa.gov, `media_type=video`, then the asset manifest for the mp4. No logos or insignia on screen.
  - Wikimedia Commons, PD/CC0 only;
  - the Prelinger Archives on archive.org.
  Log each clip in CREDITS.md: the ID, in/out, licence and upscale factor.
- **One row per reference shot:** for each, the source that fills it, each source used once. Callbacks are allowed only where the reference itself calls back (its burst montage, PIP grids).
- **Subject footage:** inventory the subject's footage too. Note how many distinct framings exist: close-ups and mediums from different moments, stage versus interview. The reference shows the speaker in many different shots, not one clip.
- **In-person only:** if the user excludes video-call footage, check every candidate with a frame sample before using it.

### 3. SPINE = THE REFERENCE'S THESIS, IN THE SUBJECT'S WORDS
- Write down the reference's argument shape first. For example, a tech-CEO interview edit: a radical view of the future → concrete future benefits → the timeline → "my view hasn't changed" in the silence → a conviction close.
- Fill each role with the subject's verified lines. Match the register: a future-thesis reference needs future-thesis lines, not a personal origin story, unless the user asks for one.
- Transcribe ALL of the subject's footage first, then pick the lines.
- Before showing a spine:
  - check that no two consecutive lines open with the same words (v1 opened "how do I go off the beaten path / how do I take full control");
  - check for self-hype the user may find cringe. Ask about tone ONCE, alongside the spine options.
- The silence or drop line must be the subject's strongest question or claim.

### 4. SAME SONG → LOCK TO THE REFERENCE'S FRAMES
- If the user wants the reference's track, Shazam it. Pull it with yt-dlp, log it as unlicensed, and cross-correlate to confirm the offset.
- With the same song from 0:00, the beat grid is identical, so reuse the reference's cut frames, mute window and slam frame directly.
- With a different song, rebuild the grid from its onsets and map each reference cut to the nearest equivalent beat.

### 5. EXACT WORD TIMES: forced alignment, not whisper timestamps
- whisper.cpp word timestamps (medium, small, even `-dtw`) drift 0.3–0.8 s. That's enough to put caption slams on the wrong word, and to let filler words like "you know" bleed into cuts.
- Use `scripts/fa_align.py`: torchaudio MMS_FA forced alignment of the KNOWN, verified transcript inside a generous window.
- Spot-check disputed boundaries by transcribing the gap on its own, and with an energy envelope (10 ms RMS).
- Cut speech at the aligned word edges, with a 0.04–0.09 s pre-roll and a 0.06 s tail.
- Caption caps slam on the aligned word-start frame. Italic letters type from the word start at about 1 frame per letter.

### 6. MIX QA IS A GATE, NOT A STEP
- Loudness-match every line's active-speech RMS before mixing. Stage and conference audio is quieter and roomier than an interview mic, and lines vanish under the music otherwise.
- Transcribe the FINAL mix with two models. Every line must read back. A missing line means a level or duck problem; an extra word means bleed.
- Run the sync gates (`failure-modes.md` → "Music bed offset…"): the slam onset, matched hits and the mark, each within 1 frame.

### 7. SHIP GATES (run them all before showing the user)
- `scripts/uniq_check.py engine/film.js`: the no-repeat rule. Every allowed repeat must be one the reference also makes.
- A contact sheet every 24–30 frames, checked by you for:
  - brand logos and sponsor slides in conference footage;
  - the source's neighbouring-shot flash on the first frames of a moved clip;
  - type off the frame edge (fit on the DRAWN width, with tracking);
  - Veo device mockups.
- A caption-sync strip: one frame per caps slam and per poster change.
- A REF|OURS side-by-side mp4 in the deliverables.

## Veo plate notes (from this project)
- "anamorphic" in a prompt produces letterbox bars and flare streaks; never use it.
- Veo sometimes frames a shot inside a smartphone mockup, with a bezel and notch. Add "full-frame cinematic shot, no device, no screen frame, no border", and reject any take with a mockup.
- Write each plate for a specific beat (wind tunnel → Navier-Stokes, neurons → AGI), and vary palette and key: some high-key white studio, some daylight. Twelve dark teal plates read as one clip on repeat.
- Background generation agents can stall. Poll the generation log, and pick the takes yourself if the agent stops reporting.

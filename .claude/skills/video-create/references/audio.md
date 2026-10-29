# Audio: music, SFX, voice, mix

Sound is half the film. Run it as its own agent in PHASE 2 (the "5th agent").

## Music
- **A royalty-free, commercial-OK track** (Mixkit, Pixabay Music, Uppbeat, Artlist/Musicbed if the user has an account). Record the URL + licence in `MUSIC.md`. **Never reuse the reference's music.**
- Match the reference's tempo (from `analyze_ref.py`) and its **arc**: build → silence/breath → drop → peak → thin ending. From a proven prompt: *"a royalty-free song around 120 BPM with a drop and a quiet breakdown."*
- **"The song starts on a downbeat: the zoom lands on the drop, the [quiet section] sits in the breakdown, the wordmark returns with the beat."**
- Time-stretch to the target BPM (`rubberband` or ffmpeg `atempo`, ≤8%) and **cut on beats** so the silences, the drop and the ending land on the beat map's times. Crossfade music edits over 1 beat at zero-crossings.
- Build `beats.json` (beat times) and have the engine read it, so visual hits are driven by the same numbers.
- Publish an **onset map** (`onsets.json`: measured kick/snare attack times with strengths, the grid phase, and the 6–8 "hits") for the picture editor, snap every cut to a real transient, then take the locked cut list back for SFX accents. For a snappy intro, choose a section with a clear pulse from frame 0.
- Measure the licensed track's own BPM before stretching; if it's already within 0.05% of the grid, don't stretch (it only smears transients).

## SFX
- **"A downloaded SFX for every event (Mixkit), never synthesized, each placed by its measured peak."** (Measure each file's peak sample offset and align the peak to the event frame, not the file start.) If no download source is reachable, ElevenLabs sound generation is the fallback; numpy synthesis is the last resort, and only for pure tones or sub-hits.
- Typical palette: sub hit on the drop; whooshes on fast cuts and whips; soft ticks when a line-draw completes; UI clicks on cursor presses; paper or CRT texture on archival and screen shots; a low bloom under the logo; a reverse swell into the drop.
- Record every source and licence in `SFX.md`.

## Voice (ElevenLabs)
- Write `SCRIPT.md` as a VO slot table: `slot | start | end | speaker (narrator/archival/founder) | text | max duration`.
- Narrator style for manifesto films: deep, calm, unhurried, lines of 3-12 words with 1-3s of air between them. **Audition 3 voices on the same 2 lines** (`vo/auditions/`) and pick the closest to the reference narrator (or let the user pick, if they want to).
- Generate **one file per line**, trim the silence (`ffmpeg -af silenceremove=start_periods=1:start_threshold=-50dB,areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse`), and fit each line to its slot with **≤8% stretch**. If a line doesn't fit, rewrite it; don't squash it.
- Archival voices: download with yt-dlp, verify the exact words with two whisper sizes (base + small) on the final trimmed file (transcribe the head and tail separately too; whisper hides trailing "um"s and invents leading words), cut to the best 1-4s, denoise lightly (`afftdn`), band-limit slightly for period feel if the reference does, and log each in `CLIPS.md` (URL, in/out, verbatim line, fair-use note, visible logos and their frame ranges). Reject quotes that name companies. For captions, force-align the verified words (wav2vec2 CTC) instead of using whisper's word timestamps.
- No ElevenLabs key: use macOS `say -v Daniel` (or similar) for the animatic, and hand the founder `SCRIPT.md` plus a recording guide (quiet room, phone 20cm away, 3 takes per line).

## Mix
- A mix script (Python + ffmpeg filter graph) builds the whole mix from `SCRIPT.md` slots, `beats.json`, the music edit and the SFX list, so the mix is reproducible like the picture.
- **Duck the music under the voice** (sidechaincompress, or a keyframed volume): about -9 dB under the narrator and about -13 dB under archival, with ~120 ms lookahead so soft onsets aren't masked. Duck SFX under all voice. Never silence-trim hand-cut archival in the mix (it clips soft first words).
- Loudness: normalize to **-14 LUFS integrated** for punchy launch films, or **-16 LUFS** for dynamic cinematic manifesto films (the measured refs sit at -17.5 and -22.4 LUFS). **True peak ≤ -1 dBTP.** Use two-pass `loudnorm`. Keep designed silences truly silent (below -45 LUFS momentary).
- Final check: whisper-check every line cut from the FINAL mix window (not the source file); listen through the whole mix; confirm the drop lands within 1 frame of the visual hit (compare onset times with the event frames); no clipping; VO intelligible on laptop speakers.

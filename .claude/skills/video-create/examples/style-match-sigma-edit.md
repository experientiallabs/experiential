# Worked example: style-match a 48 s vertical "sigma edit" with the user's own footage

This is the depth a style-match brief must reach. Every number below was measured from a real reference: a 48 s 9:16 sigma edit of a tech CEO's interview, set to a slowed Brazilian-funk track. Swap the slots in [brackets] for your subject.

```
<inputs>
Ask me for:
- the reference (path or URL);
- the subject's footage (interviews and stage talks; say whether video-call footage is excluded);
- the music (the reference's own track or another);
- the tone.
Defaults if I skip one:
- music: the reference's track, identified by Shazam on three 12 s samples and pulled with yt-dlp, logged as unlicensed;
- tone: the reference's thesis, in the subject's own words (a future-facing claim → benefits → timeline → a question in the silence → a conviction close), dry rather than hype;
- footage: in-person only.
</inputs>

<direction>
- A 47.85 s vertical edit, 1080x1920, 60 fps, frames f1–f2871 (fN starts at (N-1)/60 s).
- Concept: a founder's take on where a technology is going, cut into short phrases.
  - Each phrase gets kinetic captions over a fast stream of cinematic b-roll.
  - The music cuts out for one bar under the boldest question, then slams back for a captionless montage that ends on a poster triplet carrying the closing line.
- Motion law:
  - The picture changes about 1 frame BEFORE the beat frame.
  - The median shot is exactly 1 beat (32.73 frames).
  - Every shot carries a slow push or pull (2–4 %/s on speakers; ±4–18 % on b-roll).
  - Nothing is static except freeze-frame stabs.
- Look:
  - Speakers are colour, crushed and teal-green, with a strong vignette and letterboxed 160 px top and bottom on the first four talking heads.
  - B-roll is full-frame. About a third is REAL footage (launches, microscopy, cities, crowds) and the rest is AI.
  - No added grain: the reference has none.
- Type:
  - DIN Condensed Bold for caps (free substitute: Bebas Neue at +0.035 em, sized by cap height), tracking +0.015 em.
  - Georgia Italic for lead-ins (free substitute: Gelasio Italic).
  - Off-white #FAFBF3 text, red #D71117 key words, unlit karaoke grey #4B514A.
  - No stroke and no drop shadow; bloom is a Gaussian σ≈18 px at about 0.3 opacity.
- Banned:
  - crossfades as the default;
  - a held shot over 1.7 s outside the mute bar;
  - reusing a clip outside the burst montage;
  - logos, sponsor slides or readable screens in the footage;
  - "anamorphic" in AI prompts (it causes letterboxing);
  - Math.random.
</direction>

<structure>
BEAT GRID (same song from 0:00 = same grid):
- `frame(b) = round((0.0426 + b*0.545503)*60) + 1`, 109.99 BPM, first downbeat f4.
- Phrase starts at f527, f1575, f2098 and f2622.

AUDIO:
- The song plays from 0:00 unedited.
- The MUSIC IS MUTED f1444–f1574 (one bar), with a 12 ms fade-out. The speech reverb tail fills it; true digital silence only at f1571–1574.
- It slams back at f1575 on the song's own drop.
- Speech ducks the music about 8–10 dB, with a 50 ms attack and 300 ms release.
- The reference is mixed at −8.1 LUFS and clips (+1.6 dBTP). Ours: about −9 LUFS, −1 dBTP.

SPEECH SLOTS (the subject's verified lines, with word times from forced alignment):
- First half, f4–f1443: 6–8 lines of 1.5–4.5 s, each cut at aligned word edges with a 0.04–0.09 s pre-roll and a 0.06 s tail.
- f1399–f1570: the strongest question. It starts before the mute and must END before f1575. Remove fillers ("you know") and pauses inside it if needed, and disclose that.
- f2509–f2634: the closing line, with its three key words landing on the three closing posters (≈f2541 / f2575 / f2594).

SHOT TABLE (ours mirrors these frames; types: TH = talking head, P = poster, R = real b-roll, AI = AI b-roll):
- f1–133 TH (letterboxed, push 1.00→1.043)
- f134–231 TH (zoom-whip in)
- f232–329 b-roll
- f330–381 P red
- f382–407 P white
- f408–460 P black
- f461–493 R/AI macro eye (flash in; eye dives through the lashes, scale 1.0→1.58)
- f494–559 TH
- f560–624 b-roll
- f625–689 TH (flash in)
- f690–787 b-roll (zoom-whip)
- f788–854 TH (mirror-kaleidoscope in)
- f855–918 b-roll
- f919–1016 TH (horizontal streak in)
- f1017–1082 b-roll
- f1083–1213 TH (zoom-whip pull in)
- f1214–1278 TH or stage (vertical smear in)
- f1279–1376 b-roll, 1–2 cuts
- f1377–1442 TH
- f1443–1507 b-roll (zoom-whip; the mute begins under it)
- f1508–1540 TH (the question)
- f1541–1573 CRT power-off
- f1574–1638 R launch (CRT power-on + slam)
- f1639–1671 b-roll (zoom-whip)
- f1672–1703 b-roll (vertical smear) + stab
- f1704–1736 P red, no word (flash +131 DN decaying ×0.8/frame)
- f1737–1769 AI robot in a white studio
- f1770–1835 R/AI + inverted stab
- f1836–1908 triptych
- f1909–1932 b-roll (streak)
- f1933–1965 b-roll (zoom-whip) + stab
- f1966–1999 R crowd or reaction (flash to white)
- f2000–2030 b-roll (neon edge-scan in)
- f2031–2064 b-roll (streak)
- f2065–2096 b-roll (3D card flip) + inverted stab
- f2097–2162 PIP grid
- f2163–2190 the PIP's centre clip, full frame
- f2191–2227 b-roll (white wash) + stab
- f2228–2330 speaker cut-out over 3 swapped backgrounds (swirl in; only the background whips) + inverted stab
- f2331–2358 silhouette double exposure
- f2359–2390 b-roll (flash)
- f2391–2424 b-roll (zoom-whip)
- f2425–2456 b-roll
- f2457–2489 b-roll + stab
- f2490–2538 burst montage: 12 clips × 4 frames, zoom alternating +4.6 % / −4.0 %; callbacks allowed here only
- f2539–2558 P red
- f2559/2575–2588 P white
- f2589–2620 P black (zoom out to black)
- f2621–2775 six b-roll shots (zoom-whip, flash, zoom-whip, vertical smear, swirl, zoom-whip pull)
- f2776–2800 TH (no letterbox)
- f2801–2829 the eye clip REVERSED (bookend)
- f2830 dim to 0.38
- f2831–2871 black

STABS (two-level threshold freeze, held, on the "&"):
- f1688–1691
- f1819–1822 inverted
- f1950–1952
- f2081–2083 inverted
- f2212–2214
- f2277–2280 inverted
- f2474–2476
Recipe: σ0–4 px pre-blur, threshold luma ≈105 (raise to ≈150 on blown-out plates), output #050606 / #F4F5F0, hard in, hard out.

TRANSITIONS (per-frame arrays):
- Zoom-whip PUSH:
  - The outgoing shot scales 1.054, 1.150, 1.342, 1.674, 2.355, 3.564 with radial zoom blur k 0.08 → 0.7 and an exposure lift of about +45 %.
  - The incoming shot lands mirror-tiled, scaling 0.38, 0.55, 0.71, 0.84, 0.92, 0.97, 1.00, with blur 0.35 → 0.
- Zoom-whip PULL: the outgoing shot scales 0.99, 0.97, 0.83, 0.67, 0.50 (mirror-tiled border); the incoming shot scales 2.85, 1.92, 1.43, 1.18, 1.07, 1.01, 1.00, with a +20 % lift on its first frame.
- Horizontal streak: the outgoing shot gets a box blur of 14, 29, 47, 53, 47, 31 px and fades about −35 %; the incoming shot gets 57, 45, 31, 18, 8 px.
- Vertical smear: the outgoing shot gets a y-blur of 30, 80, 155, 126, 93, 88 px, plus a 220 px white light bar travelling y 150 → 1060.
- Flash: the incoming shot gets an additive lift of +183, +123, +67, +34, +12 DN.
- 1-frame mix: F = 0.35·A + B + 6 %.
- CRT off:
  - f1541: crop to y 460–1444.
  - f1542: squash to 292 px.
  - f1543: squash to 64 px.
  - f1544–1550: a 2 px line.
  - The line then shrinks 1062 → 232 px wide while its glow grows 12 → 68 px.
- CRT on: a band y 162–1728, then 14–1822, then full, over-exposed with red/cyan edge fringes, decaying to f1581.
- Swirl: the rotation smear grows 2, 5, 10, 20, 35, 50°.
- Neon edge-scan: a scan line sweeps y 0 → 1840 over 20 frames; above the line, the incoming shot as cyan edges.
- Card flip: tilt back about the bottom edge to about 70 % height.
- Triptych: three 1080x640 panels slide in at +0/+3/+6 frames with x-blur.
- PIP grid: tiles pop in about every 5 frames; the centre tile zooms to full frame over the last 12 frames with radial blur.

CAPTIONS (a separate layer that persists across cuts; top-anchored blocks):
- Lines:
  - An italic lead-in of 170–232 px types on letter by letter: each glyph rises from 0.8× ascender below rest, easing ×0.58/frame, alpha 0→1 over 3 frames, with a downward smear of 5 copies; 1 frame/letter from the word's aligned start.
  - Then 1–2 caps lines, fitted to an ink width of 890–960 px with a cap height of 158–302 px.
- Caps slam on the aligned word-start frame:
  - frame L: horizontal ghost copies spanning the frame;
  - then width 1.16, 1.05, 1.02, 1.005, 1 and coverage 0.5, 0.65, 0.82, 0.92, 1;
  - afterwards it grows +0.1 %/frame;
  - older caps lines punch +2.5 % when a new line lands, easing back over about 12 frames.
- Spacing: italic baseline to caps cap-top about 20 px; caps baseline to the next line 30–40 px.
- Exit: a fade of 0.75, 0.45, 0.2, 0.08 timed onto the picture transition.
- Red strike bar: #DD1C1F, 15 % of cap height, overshooting the word by ±40 px, wiping left to right over 2 frames.
- The whole second half is captionless except the poster words.

POSTERS (triplets on consecutive beats):
- Backgrounds: red #D40D16, white #FAFCFA, black #050806.
- Headline: one word or phrase, top at y≈95, fitted to 1000 px of DRAWN width with tracking, cap height ≤ about 350 px. It sits in front of the background and BEHIND the person's head.
- Background texture: the same word as outline-only caps, 160 px, stroke 2.5 px at 5–8 % contrast, brick-offset rows every 242 px.
- Person:
  - a rembg human-seg cut-out (largest component kept), converted to B&W with a hard S-curve;
  - the head top placed at y≈360 by the alpha bounding box;
  - an outline 6–12 px (cream on red, red on white, white on black), with 3 echo contours on the white and black posters;
  - it shrinks 1.0 → 0.875 with an ease-out while drifting 20–55 px.
- Entry: an exposure lift decaying ×0.8 per frame.
</structure>

<build>
1. Measure first: the SPEC agent writes the shot table, captions, posters, transitions and audio before any sourcing.
2. SOURCES.md:
   - at least 50 distinct sources for about 70 shots, matching the reference's real-vs-AI mix;
   - real footage from the NASA images API / Commons PD / Prelinger, credited, with no insignia;
   - AI plates written for specific beats, varied in palette and key;
   - the subject in 8 or more distinct framings (close-ups and mediums from different moments, stage).
   - uniq_check.py must pass.
3. Spine: map the reference's thesis roles to the subject's lines. Two transcription passes per line, no repeated openings, and fa_align.py for the word times.
4. Engine: a pure seek(t) canvas at 1080x1920 / 60 fps. Timeline is one file of S('id', f0, f1, type, {...}) rows; one-line edits.
5. Mix:
   - speech loudness-matched per line (active RMS);
   - −10 dB duck (−13 under the closing line);
   - the mute window;
   - master about −9 LUFS / −1 dBTP;
   - transcribe the final mix with 2 models; every line must read back.
6. Render in 4 parallel lanes (about 90 s). Encode H.264 High 4.2, bt709, CRF 15; an upload copy at CRF 19 / 20 Mbps; and a REF|OURS side-by-side.
</build>

<gotchas>
- whisper.cpp word timestamps drift 0.3–0.8 s: never time captions or speech cuts from them.
- A conference recording mostly shows slides with the speaker in a small inset. Find the full-frame speaker windows with a 3 s frame scan, and crop out the event and sponsor logos.
- Moving a speaker's slot moves the lip-synced source index. Clamp each clip to its clean source span, or the neighbouring shot flashes on the first frames.
- Short poster words overflow when the width fit ignores tracking. Fit on the drawn width.
- A threshold stab on a blown-out plate becomes a white blob. Raise the threshold.
- Veo sometimes frames a shot inside a phone mockup. Reject it, and add "no device, no screen frame, no border" to the prompt.
- Stage audio vanishes under the drop unless it is loudness-matched.
- Re-check the full film's contact sheet after every change. Fixes in one shot have broken another.
</gotchas>

<start>
Ask me for the inputs. Then show me 4 stills next to the reference:
- the first caption slam;
- the first poster;
- a mid-film caption over b-roll;
- the closing poster.
Do this before rendering the full edit.
</start>
```

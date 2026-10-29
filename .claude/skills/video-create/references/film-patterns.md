# Film patterns

Pick the pattern that matches the purpose and the reference, then make its timings match the **measured** reference. Mechanics are copyable; content never is.

---

## 1. Manifesto / civilizational brand film (60-110s)
**Measured examples:**

| | Browser Use "let go of the mouse" | a16z "It's time to build" |
|---|---|---|
| Format | 1440x1080 (4:3), 30fps, 106.05s | 960x720 (4:3), 29.97fps, 105.6s |
| Cuts per 10s (measured: frame-diff, confirmed visually) | `6,4,10*,0,4,3,5,9,10,2,0` (*3 cuts + a 9-cut stutter burst at 21.6–22.8s) | `2,4,10,8,7,7,7,10,8,2,0` |
| Longest uncut take | the motif 27.3–38.0s (glow → blueprint → cursor → rings → explosion; one glow-dissolve at 32.3s) | shorter; the motif lives in fast fly-throughs |
| Silence | audio -47.6 dB RMS 25.5–26.75s; picture pure black 24.0–27.3s | -41 dB at 26–27s |
| Drop / motif ignites | explosion 36.5s, white flash 37.5–37.9s, hard cut to paper at 38.0s | the gold orb bursts ~34s |
| Integrated loudness | **-14.2 LUFS**, TP -1.25, LRA 4.9 | **-18.6 LUFS**, TP -0.66, LRA 8.6 |
| Tempo | **120 BPM**, confirmed by the montage cut grid (cuts exactly 1.0s apart at 76.6–85.4s) | ~107–110 BPM (autocorrelation; unconfirmed) |
| Black point | pure #000 (warmth is in the glow and grade, not lifted blacks) | pure #000 |
| End logo | mark 399px tall (37% of H), wordmark ~50px cap, 64px gap | letters 290px tall |
| Final line | "Now it's time to let go of the mouse." 100.2–105.0s, over the logo | "It is time to build." 101–103s, over the logo |
| Narrator share | ~30% of lines | ~30% of lines |

(Older copies of this table, and briefs written from impressions, said -17.5/-22.4 LUFS, a drop at ~39s and 18–19 cuts per 10s for a16z. Those were wrong; the numbers above are measured.)

**Structure:**
1. **History (0-25%):** civilizational archival footage, desaturated; the narrator frames a grand arc ("Our civilization was built on technology." / "The web is our civilization's library."). Famous voices answer.
2. **Breath:** 1-3s of black and near-silence just before the motif.
3. **Motif ignites (25-40%):** ONE continuous animated take (no cuts). The signature object is born, often as a **line-art blueprint** that draws itself (construction lines, "Fig." labels), then becomes solid, then **multiplies explosively** on the drop, with a flash.
4. **Origin / proof (40-55%):** the humble beginning shot like a physical object (a Show HN post on paper-white with shallow focus and one highlighted line; the repo; the founder at a desk), then a **star-history line drawing itself with a ticking counter**.
5. **New world (55-85%):** accelerating montage, cuts every ~1.2s at the peak. Real product doing real things, plus visionary archival voices. Partner or company wordmarks appear white over darkened footage.
6. **Wall → logo (85-92%):** a mosaic of hundreds of tiles pulls back in perspective and densifies; **the logo's negative shape cuts through it**; the logo is filled with tiles; it resolves into a **metallic logo** (gold in both refs), with the wordmark rising below.
7. **Imperative (last 6s):** one dry line over the held logo, music almost gone. It's always an imperative ("It is time to build." / "Now it's time to let go of the mouse.").

**Voice:** the narrator speaks only ~30-40% of the lines, each 3-12 words with air between them, and never explains the product. Archival voices carry authority, so start the archival search early and collect 2× the candidates (verified lines are scarce).
**Pacing variant:** for X/social, owners often find reference-matched intros slow. Offer a snappier intro (12–18 cuts per 10s, every cut on a kick, match cuts and push-in punches) as an explicit option.
**Look:** warm near-black, heavy vignette, film grain, bloom, one metallic accent, crushed archival, screens shot like objects.

## 2. Apple-keynote one-take morph launch (30-60s, often 1:1)
From a proven prompt:
> "An Apple-keynote launch film, 2D only, one continuous take. Every scene is made out of the previous one: nothing fades, blurs or cuts. Objects change shape instead: text rises out of a mask line, icons pop from zero on a spring, bars draw across, pages push, and a black shape floods the whole frame and contracts into the next scene. … A cursor drives every change with real clicks, drags and long-presses. The camera zooms screen-studio style so each moment fills the square, and the cursor scales with it. Banned: crossfades, blur-ins, brightness 'developing', 3D flips, particles, glows, holds longer than 1s, anything that looks like a template."

- **Beat-locked:** "120 BPM, 54 beats, something happens on every beat." Write a beat map (beat #, time, event, SFX).
- **Loop:** last frame = first frame (the wordmark squeezes into its period at the start and springs back out at the end).
- Structure: open (the wordmark/logo becomes a container) → product moments driven by the cursor → a state machine of one morphing shape (e.g. Ordered ✓ → Printing % → On its way → Delivered ✓) → a flood to black → a real-world payoff → a return to the open.
- Show the **beat map + 4 stills** before writing the full film.

## 3. Screen-studio product demo / feature announcement (15-75s)
- Hook in ≤2s: the result first, then how.
- Real UI (recorded, or rebuilt pixel-faithfully). The camera zooms to each click and the cursor scales with the camera. Captions as kinetic type, not subtitles.
- One feature, one aha, one CTA. 3-6 moments, each ≤5s.
- Sound: soft UI clicks placed on their measured peaks, a light music bed, optional VO.

## 4. Kinetic typography (6-30s, social)
- Type is the hero: variable-font axes animated (wdth/wght), words land on beats, masks and splits. 1-3 words on screen at a time. Loopable.

## 5. Founder / documentary film (60-180s)
- Interview spine (they record it; you write the questions and a lighting and framing guide), b-roll, a warm grade, a slow push-in, lower-thirds designed rather than templated, and a music swell under the thesis line.

## 6. Explainer (60-120s)
- Problem → insight → how it works (3 steps max, each a diagram that builds from the previous one) → proof → CTA. Clarity beats spectacle; the motif is still required.

---

## Signature motif (every pattern)
Every good film has one recurring object that carries the story and becomes the logo at the end: Browser Use's cursor (blueprint → orbit → explosion → logo), a16z's gold orb. **Derive it from the logo's shape or meaning.** Write its arc as 4-6 states and make every transition come out of the previous state.

## Beat-map columns (use for every pattern)
`t | act | picture | transition in | camera | VO/speaker | music/SFX | cuts in this 10s`

## Lengths that work
X/LinkedIn manifesto 90-106s · launch 30-60s · feature 15-40s · vertical social 6-20s · website hero loop 8-15s (silent, loopable).

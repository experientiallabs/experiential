<!--
Fill in every [bracket]. Delete nothing structural. Paste the §7 blocks VERBATIM from references/build-rules.md and references/audio.md; never summarize them.
Depth bar: examples/style-match-sigma-edit.md and examples/remake-edge-of-the-world.md. A brief that doesn't have measured numbers in §2 isn't finished.
Planner mode: this file must stand alone. Use absolute paths, every decision made, no questions left for the executor.
-->

TASK: direct, build and render a [length]s [film type] for **[Company]** to post on [platform]. It uses the same filmmaking language as [REF names], but it is not a copy: new story, new motif, our brand. Work only in `[absolute $ROOT]` (below: `$ROOT`).

You are the director, editor, motion designer and sound designer. Don't ask me questions; every decision is made in §1. When something is ambiguous, pick the option closest to the references and log it in `DECISIONS.md`.

"Everyone has the same model. The context you give it is what makes it look pro." Your context: references with measured numbers, brand assets, the company's real worldview, and build rules and pitfalls learned the hard way. Use all of it.

---

## §0 FILES YOU HAVE
```
$ROOT/
  .env                      [KEY names only, never values] (chmod 600; never print, log or commit)
  brand/[files]             [what the logo is: shape + meaning]
  refs/REF_A_[name].mp4     [WxH, fps, duration]  (original: [absolute path])
  refs/analysis/<ref>/      report.md, contact.jpg, cuts.txt, loudness.txt, transcript.txt
  refs/[dense sheets]       [which moments they cover]
  refs/PRIOR_[...]          [earlier company videos: watch them and don't repeat them]
  CONTEXT.md                company context + KEEP-OUT list
  [whisper model path]
```
Tools installed: [list]. Install anything missing.

## §1 DECISIONS ALREADY MADE
1. End-card name / wordmark: [ ]
2. Format: [WxH, fps], plus extra cuts [ ] (re-framed per shot)
3. Length: [ ]
4. Voice: [narrator via ElevenLabs / archival / founder / none] + how the voice gets chosen
5. Archival: [yes/no, sources, verification rule]
6. AI b-roll: [provider or "stand-ins + AI_SHOTS.md"]
7. Numbers allowed on screen: [exact list]
8. Never on screen or in VO: [KEEP-OUT list]
9. [any other decision]

## §2 WHAT THE REFERENCES DO (measured; copy mechanics, not content)
Naming a style works far better than describing one. Without a reference the model falls back to its default look (centred text, gradient background, everything fading in).
### 2a. Structure table: one column per reference, one row per act (times, what happens, how it transitions)
### 2b. Measured numbers: cuts per 10s, longest takes, silences, drop time, loudness (integrated + arc), BPM, VO speaker split, final-line timing, logo hold
### 2c. Look: base colour, accent, grain, vignette, bloom, how screens are shot, type usage
Verify these observations yourself in PHASE 0. Where I'm wrong, the refs win; note it in SPEC.md.

## §3 THE COMPANY: WHAT THE FILM MUST MAKE PEOPLE FEEL
[One sentence. Founders + credibility. Product, plainly. Worldview bullets. The name's meaning. Tone words. Founder line bank (≤10 words each).]

## §4 THE MOTIF
[What it is (derived from the logo), and its arc in 4-6 states. Which state is the one uncut take, which is the explosion or drop, and how it becomes the logo.]

## §5 BEAT MAP (draft; the storyboards must improve on it)
| t | act | picture | transition in | camera | VO / speaker | music / SFX | cuts in this 10s |
[Silence, drop and ending aligned to the measured reference positions. Include the final imperative line + 3 alternates.]

## §6 SHOT SOURCES
- Motion graphics (code): [list]. UI at real-component quality, never placeholder boxes.
- Real product footage: [what to record, with URLs]
- AI b-roll: [numbered prompts: subject, lens, light, camera move, duration, grade note]
- Archival search targets: [list; verify every word with whisper, never invent quotes, log in CLIPS.md]
- Stock: [queries + licence rule]

## §7 BUILD: FOLLOW THESE RULES
[PASTE VERBATIM from references/build-rules.md: PHASE 0, PHASE 1 (incl. relevant advanced effects), GATES, PHASE 2, PHASE 3, GOTCHAS, PITFALLS]
[PASTE VERBATIM from references/audio.md: Music, SFX, Voice, Mix, with the numbers adjusted for this film]

## §8 BANNED
[Default banned list from build-rules.md + film-specific items. Transitions are made from the previous shot, or are hard cuts on the beat.]

## §9 DIRECTOR NOTES
The first render is usually 80% there; the last 20% is what makes it look pro. Notes come in camera words ("slow every push-in to 0.7x", "hard cut here", "push in on the button", "the drop is 3 frames late"). Apply exactly the note, only to the affected shots, re-render only those ranges, and show before|after stills. A vague note becomes 2-3 concrete options; apply the closest to the refs and say which one you chose.

## §10 DELIVER
- out/[slug]-[WxH].mp4 (+ extra cuts), out/contact-sheet.jpg, out/beats.jpg
- SPEC.md, DECISIONS.md, SCRIPT.md, CLIPS.md, MUSIC.md, SFX.md, AI_SHOTS.md
- Source that re-renders with one command
- Remaining weaknesses with frame numbers

Start with PHASE 0. The first report contains the SPEC summary + 3 storyboards; after that, report at each gate (stills, animatic, full render).

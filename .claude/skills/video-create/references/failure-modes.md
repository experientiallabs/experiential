# Failure modes → how to avoid them

Lessons from real builds (a 100-second manifesto film over many versions, and a 48 s vertical style-match edit). Each entry is a problem that actually happened, and the habit that prevents it. They apply to every film type; read the phase you're in.

## Brief and measurement
- **To avoid designing to wrong numbers**, re-measure everything the brief states about the references before building. In one build the brief (written from impressions) had loudness off by 3–4 LU, the drop 1.2s late, one ref's cut density doubled, "warm near-black" when the pixels were pure #000, and "zero cuts" where there was a glow-dissolve. Put a "corrections: brief said / measured / consequence" table at the top of SPEC.md. Measurements win, and you tell the user.
- **To avoid a wrong tempo grid**, never trust a single onset-autocorrelation BPM (it jumped between 96, 110 and 139 on one track). Cross-check it against the reference's montage cut spacing: equal gaps of 1.0s mean 120 BPM at 2 beats per cut. Then check the music you license with a beat tracker plus your ear.
- **To avoid "matching" the wrong thing**, measure the grade from pixels (black point percentile, vignette radial profile, grain as high-pass std on flat regions only, accent colour from saturated pixels). Don't restate a prose description.

## Story and script
- **To avoid lines that don't make sense in context**, write the through-line as one sentence (A → B → C → imperative) and test every line, including every archival quote, against it: does it advance the argument, and does it pay off a visual? v1's "It will replace the yellow pages" and "a mind that never learns from them" were verified and on-theme, yet the owner flagged both as confusing. v2 put "And everything they learn trains someone else" over light-threads rising into monoliths, so the motif reversal a few seconds later pays it off.
- **To avoid a narrator-heavy film** (the refs run ~30% narrator lines; v1 ended up at 71% because verified archival was scarce), start the archival search in the first hour, collect 2× the candidates you need, and design slots around what verifies. Don't write narrator fallbacks as the plan.
- **To avoid pacing that feels slow to the owner**, remember that reference-matched density isn't always what they want. A manifesto intro at REF_A's 5–6 cuts per 10s read as "not snappy enough" for X. Offer a snappier variant up front: 12–18 cuts per 10s in the intro, still on the beat, with speaker shots holding 1–1.5s so lip-sync reads.
- **To avoid violating a stated rule when the owner changes it**, log every owner override in DECISIONS.md, e.g. "name OpenAI, aspirationally: 'We help any company become its own OpenAI'", against the brief's "no competitors as villains". The next pass then knows it's deliberate.

## Assets: archival, AI, product, data
- **To avoid on-screen logos and brands you didn't choose**, log per clip in CLIPS.md every visible brand or text region and its frame range: event banners, network bugs (TODAY, CBS), conference logos (IAB), equipment plates (IBM), website brands (InsWeb, FedEx pages), channel badges (YC). Plan the crop, blur or start frame before a shot uses the clip. v1 had five such leaks in 18 seconds.
- **To avoid quotes that name companies** (an archival CEO line naming a retailer could read as a customer claim), reject them at search time, not at review.
- **To avoid freezes on short archival clips**, export each voice clip twice: the phrase-trimmed audio/video (for the mix) and a padded picture version `<id>_pic.mp4` with exactly 2.000s of handles, the phrase starting at frame 60. Lip-sync is then `FOOT('<id>_pic', F − slotFrame + 60)`, and cutaways never clamp to a frozen last frame.
- **To avoid trusting one transcript**, verify every archival line with two whisper sizes (base + small) on the final trimmed file, and transcribe the head and tail separately. Whisper hid a trailing "Um." and invented a leading "thing"/"something". A slurred word ("software" → "suffers") can flip with 0.5s of padding; record which evidence verifies it (the larger model plus the platform's captions) and accept or replace it deliberately.
- **To avoid wrong word timings in captions**, don't use whisper's word timestamps (they shifted every word 0.4s and merged two words). Force-align the verified text to the audio (wav2vec2 CTC via torchaudio), and give captions a short hold because alignment ends early on trailing s/f.
- **To avoid invented data on screen**, never draw a chart from guessed or interpolated points. GitHub hides stargazer lists for very large repos (REST 404, GraphQL 0, page cap at 40K), so a star curve can be impossible. Show the real artefact instead (a push-in on the live star count on the real page) and log why.
- **To avoid blurs that miss their target** (the owner's note on v2 and v5: "the blurring is not covering the right place"), build every redaction mask in the same transform as the content it hides, so it moves with the page, the push and the pan, never as a screen-fixed rectangle. Verify it frame by frame across the whole shot, not on one still. Have a separate QA pass hunt for masks that slip before any version is called final.
- **To avoid fake portrait-mode masks on real people**, never hide a background logo behind a moving speaker with a keyed sharp-person/blurred-background mask. Re-keying it still left a visible silhouette and a soft face whenever he moved. Reframe or crop the logo out of frame instead, and redact only static, rigid things (pages, screens, plates) in their own transform.
- **To avoid unverified on-screen facts**, mark facts that come only from the brief ("launched Aug 31") as unverified in DECISIONS and in the final report, even when the brief allows them.
- **To avoid AI b-roll artefacts**, don't say "shot on 35mm film" to Veo: it draws fake film gates, edge codes and sprocket holes. Ask for "full-bleed, no borders, no text, no markings" plus a negative prompt, look at 2 frames of every take, and expect baked letterboxes (measure the picture rows and crop past them).
- **To avoid fake-looking or rule-breaking UI**, write UI copy in words when the brief restricts numbers (days instead of dates, "open tickets" instead of "19 tickets", "ok" instead of latencies). Use fictional company names in mock UIs, and scrub real company names from archival-looking pages.
- **To avoid a repetitive or off-brand mosaic**, build the tile pool deliberately: two tiles per clip, no speakers, slides or branded sources, archival tiles in B&W like their shots, plus product UI. Look at the whole pool as one sheet before the wall renders.

## Keys, providers and cost
- **To avoid billing the wrong account**, check which key a provider call will actually use. A `GEMINI_API_KEY` already in the shell environment silently won over the project key the user supplied, so eleven Veo shots were billed to it. Scripts should read keys from `$ROOT/.env` explicitly. When a key arrives mid-run, message every running agent that uses it.
- **To avoid surprise provider limits**: Veo 3.1 gives 1080p only at 8s (4/6s are 720p), so plan to trim. ElevenLabs shared-library voices must be added to the account before the API will use them (tell the user that happened). eleven_v3 read higher-pitched than v2 for the same voice. YouTube can start refusing HD mid-session (bot check), so pull the key sources early and note any SD fallbacks.

## Engine and rendering
- **To avoid mis-masked logos**, never paint a multi-step fill (gradient + striations + highlight) directly under `source-in`: each later `fillRect` erases everything outside its own rect. Paint the fill `source-over` on a scratch canvas, then `drawImage` it under `source-in`.
- **To avoid coloured borders from the chromatic-aberration pass**, fill each channel buffer black before scaling. Multiply over transparent pixels yields the pure channel colour.
- **To avoid whole renders turning black**, reset `ctx.filter` (and the composite op and alpha) on every helper canvas you reuse. One leftover filter blacked out every later frame of a session. Have the buffer helper reset them.
- **To avoid silent data bugs in layout code**, don't spread objects into records whose keys you then overwrite (`{...word, w: width}` replaced the word text with its pixel width, and the captions printed numbers).
- **To avoid blended "cuts" inside a shot**, remember that subframe motion blur averages everything within the frame's shot. A shot with internal hard cuts (a panel sequence, a stutter) needs `noBlur`, or its sub-cuts must be routed like real cuts.
- **To avoid invisible highlights**, note that a multiply highlighter vanishes on dark UI. Give the highlight a mode: multiply for paper or light UI, screen for dark UI.
- **To avoid lifted blacks from grain**, apply grain with `overlay` (so black stays 0) and leave it off designed silences.
- **To avoid rebuilding an old version from the wrong assets**, give every version its own asset folders (`vo/v2/`, `audio_v2/`) and never re-use an unversioned path. `vo/lines/` held v2's narration until v3 overwrote it, and a later remix of v2's story would have used v3's voice. The same happened with the archival clips, which were re-cut longer after v2. Check each asset's duration against the version's records before building on it.
- **To avoid one version's tool run overwriting another version's data**, give every pipeline tool a required output path, or a default that includes the version. A shot-list dump with a default path (`out/shots.json`) was overwritten by an agent working on a later version.
- **To avoid collisions between parallel renders**, give each agent its own output directory, write each frame as it finishes, and reuse frames when a group's code path hasn't changed since its render (G4's 480 frames were reused between v1 and v2, saving a full re-render).

## Captions (when asked for)
- **To avoid unreadable or orphaned captions**, build them as one engine overlay, not per shot:
  - words appear as spoken, from forced-aligned timings;
  - a balanced two-line wrap, with no orphan word;
  - the speaker's name small beneath;
  - "…" when a quote starts mid-sentence;
  - adaptive contrast: sample the frame's own pixels behind the caption, use dark ink with a light halo on bright shots, and white with a soft scrim on dark shots;
  - brief every shot group to keep the caption band clear and speakers' mouths above it.

## Orchestrating agents
- **To avoid acting on stale reports**, expect messages to cross: an agent's "done" often predates your newest note. Verify against the files and fresh renders, not the report, and re-send a note if the agent goes idle without addressing it.
- **To avoid rule breaks slipping through a MATCHES report** (v1 shipped legible digits that agents called clean, and an agent drew a curve the lead had ruled out), have the lead do a **rule audit of every group** from its own sampled frames, before approving:
  - legible numbers (read the frames, or OCR them);
  - logos, brands and real company names;
  - invented data;
  - identifiable people outside the cleared list;
  - text readability.
- **To avoid wasted spend on regeneration**, when an asset is good but the process had a flaw (the wrong key billed, say), decide explicitly whether to redo it. Usually you don't, and you tell the user.

## Revisions (v2 and later)
- **To avoid losing v1 when revising**, snapshot the source, script, mix, cue list and crop plan into `versions/v1/` first, and render the new version to a new folder (`out2/`) through an `OUTDIR` parameter the whole pipeline respects.
- **To avoid off-beat intros**, have the audio agent choose intro music with a clear pulse from frame 0 and publish an onset map (`onsets.json`: times, strengths and the "hits"). Snap every cut to a real transient, put push-in punches on the hits, and send the final cut list back to audio for accents.
- **To avoid a revision touching everything**, map each owner note to the groups it affects. In v2 only the intro was rebuilt; the other groups got re-slots and rule fixes.

## Voice (TTS)
- **To avoid narrator lines that hang level instead of landing**, only send ElevenLabs the next line's text as context when the line really is the first half of a sentence ("Do you want to rent your intelligence…" → "…or own it?"). With a following line attached, a complete statement is read as a lead-in and stays flat (a 0.8-semitone fall vs 6.2 semitones without it). Score takes on the pitch fall at the end and on median pitch, and keep rejected takes.

## Final render
- **To avoid stale frames after an engine fix**, don't trust file timestamps. A render process that loaded the old engine keeps writing old-engine frames after the fix. Record the start time of every render, re-render anything whose process started before the fix, and verify with a pixel signature of the fix (for example, the edge column's colour), not with file dates.
- **To avoid a final render that mixes old and new frames**, freeze a group's file before its final render (tell the agent "locked, no more edits"). If it must change afterwards, re-render exactly the changed ranges. In v3/v4, agents kept improving code while the final render ran, and two ranges had to be patched.
- **To avoid guessing crop keyframes by eye**, measure each text-led frame's content extent (columns with bright pixels) and derive the square window from it. An eyeballed centre cut a star counter's last digit in v5.1. When content is wider than the square, pan so only one edge is trimmed at a time, favouring the side the narration is about.
- **To avoid a group's timing being overwritten by a re-lock tool**, make table-rewriting scripts idempotent. A slot-layout script matched a 4-column row, and on its second run it duplicated a column, which silently broke the downstream caption parser.
- **To avoid clipped text in the 1:1 cut on long shots**, remember that a per-shot crop centre isn't enough for a 28s one-take: add crop keyframes inside the shot (eased moves) wherever a wide plate or a right-side counter is on screen, and check a square frame of every text-led moment.
- **To avoid "the quote got cut off" notes**, give every archival clip ~100ms of clean pre-roll before its first word and a natural tail (or matched room tone when the next speaker is too close), hold the picture ≥0.7s and the caption ~0.9s after the last word, and release the music duck slowly (≥600ms).

## Mixing
- **To avoid clipped archival onsets**, don't run a relative silence trim on hand-cut archival clips (it ate a soft "every"). Trim only generated narrator lines.
- **To avoid masked quotes**, duck music deeper under archival (about -13 dB) than under the narrator (about -9 dB), use a lookahead of about 120 ms so quiet onsets start clean, duck SFX under all voice, and whisper-check every line cut from the FINAL mix window, not the source file.

## Delivery
- **To avoid platform upload trouble**, remember that grain inflates bitrate: a 104s master was 466 MB. Also export an upload version (crf ~19, maxrate ~20 Mbps, about 150 MB). Check the 1:1 reframe for clipped text lines (an endpoint URL was cut off) and pin text-led shots in the crop overrides.
- **To avoid claiming "done" early**, list anything checked only by analysis ("mixed by numbers, not by ear") in the final report, alongside rights and verification caveats.

## Shared shot files edited while other groups render
A page loads EVERY group's shot file. One group saved a half-edited file (a `//` comment swallowed a `catch`) for about 90 s, and any render whose page booted in that window ran without that group's shots and logged a PAGEERROR. Another group's log piped through `tail -1` dropped the error entirely.
- Stage every edit to a shared shot file in /tmp, run `node --check`, then copy over (an atomic replace).
- Keep FULL render logs and grep them for PAGEERROR after every render. A render with a PAGEERROR is re-rendered, even if its own frames look fine.

## Element pops clip at the frame edge
A scale "pop" about an element's centre pushed a left-aligned table row off-frame ("upport-agent"). Anchor the pop's scale at the element's own edge, or keep its scaled extent ≥24 px inside the frame, and check every pop frame's edges.

## A render chain that "finished" without rendering
A group chained three renders with `> render/v11b/S2/render.log`, but the folder didn't exist. The redirect failed, the render was skipped, and the chain printed ALLDONE. Nobody noticed until the owner asked where the videos were.
- `mkdir -p` every output and log folder before the chain, and `set -e`.
- The lead checks frame COUNTS per group per version (`ls render/<film>/S*/full | wc -l`) before believing any "done", and polls running renders rather than waiting for reports.

## pkill by pattern kills other groups' renders
One group restarted its own chain with `pkill -f "render.mjs full"`, which matched every group's renders. Two other groups' chains died silently mid-render.
- Stop renders by PID only (save `$!` when launching).
- The lead checks `pgrep -fl "render.mjs full"` and frame counts after any reported kill or restart.

## Two render chains sharing one log file
A re-render chain opened the same log path with `>` as a running lane, which truncated it and wiped that lane's earlier
PAGEERROR record for frames that were never re-rendered. Rule: every render chain gets its own log file named by
span and timestamp. If a log is lost, cover the gap with a blank/uniform-frame scan plus spot stills before assembly.

## Music bed offset hidden behind good loudness/whisper numbers
A re-cut music edit (bars removed pre-slam + a drop extended) put the bed ~0.83 s late from the slam to ~0:43
while the picture followed the slot table exactly. LUFS, dBTP and whisper all passed; only QA's bass-onset vs
hits_film cross-correlation caught it. Rule: after ANY music re-edit, the lead runs a picture-vs-music onset
correlation (bass onsets vs hits_film, ≤1 frame everywhere) before encoding, and never reports a cut "checked"
on loudness + transcription alone. Also: moving a speaker's slot shifts lip-synced FOOT() source indices, so
re-check the first frames of every moved archival shot for neighbouring-shot flashes.

## Style-match v1 rejected: "you replay the same clips" and "plays twice"
- **12 sources for a 70-shot reference.** Rule: measure first, then write SOURCES.md with at least the reference's distinct-source count by material type (real/AI/speaker), and run uniq_check.py before showing anything. See style-match-mode.md.
- **All-AI b-roll where a third of the reference is real footage.** Rule: fill the real slots with public-domain real footage (NASA API, Commons PD, Prelinger) and credit it.
- **A personal origin story where the reference argues about the future, with two consecutive lines opening "how do I…".** Rule: map the reference's thesis roles, fill them with the subject's lines in the same register, and check for repeated openings.
- **Captions slammed up to 0.8 s early on whisper.cpp word timestamps, and "you know" bled into a cut.** Rule: fa_align.py (MMS_FA) on the verified text, plus a transcribed-gap check on disputed boundaries.
- **Stage-talk lines vanished under the music.** Rule: loudness-match the active-speech RMS per line, and transcribe the final mix with two models as a gate.
- **Conference footage carried sponsor logos and slides.** A moved clip's first frames showed the source's previous shot. Short poster words overflowed, because the width fit ignored tracking. Rule: contact-sheet review at 24–30-frame spacing before every delivery.

# Keys and accounts

Check first, ask second. Ask for everything in ONE message, and only for keys that this film needs. Every key has a fallback, so a missing key never blocks the film.

## How to check (without printing values)
```bash
for k in ELEVENLABS_API_KEY FAL_KEY REPLICATE_API_TOKEN RUNWAYML_API_SECRET LUMAAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY OPENAI_API_KEY PEXELS_API_KEY PIXABAY_API_KEY FREESOUND_API_KEY GITHUB_TOKEN; do
  v="${!k:-$(grep -h "^$k=" "$ROOT/.env" ~/.env 2>/dev/null | head -1 | cut -d= -f2-)}"
  printf "%-22s %s\n" "$k" "$([ -n "$v" ] && echo present || echo missing)"
done
gh auth status 2>&1 | head -2
```
(zsh: use `${(P)k}` instead of `${!k}`, or run under `bash -c`.)

## What each key unlocks
| Key | Unlocks | Get it | Fallback without it |
|---|---|---|---|
| `ELEVENLABS_API_KEY` | Narrator VO, voice auditions, optional SFX generation | elevenlabs.io → Profile → API keys | macOS `say -v` scratch VO for the animatic only; ask the founder to record the final VO from `SCRIPT.md`; or switch to a no-VO film |
| `FAL_KEY` | One key for many video models (Veo, Kling, Luma, Minimax, etc.) plus image models for stills and textures | fal.ai → Dashboard → Keys | write `AI_SHOTS.md` prompts, build code-made stand-ins, and the user drops the clips in later with the same filenames |
| `REPLICATE_API_TOKEN` | An alternative hub for video and image models | replicate.com/account/api-tokens | same as above |
| `RUNWAYML_API_SECRET` / `LUMAAI_API_KEY` / `GEMINI_API_KEY` (Veo) / `OPENAI_API_KEY` (Sora, images) | Direct access to a specific video model | provider dashboards | same as above |
| `PEXELS_API_KEY` / `PIXABAY_API_KEY` | Stock footage search and download (free, commercial OK; record the licence) | pexels.com/api, pixabay.com/api/docs | manual download links in `STOCK.md` for the user; or code-only |
| `FREESOUND_API_KEY` | SFX search (check each file's licence: CC0 or CC-BY with credit) | freesound.org/apiv2/apply | Mixkit / Pixabay SFX pages, downloaded manually; ElevenLabs sound generation if that key exists |
| `GITHUB_TOKEN` or `gh auth` | Star history and repo stats for proof shots | `gh auth login` | public unauthenticated API (rate-limited) |
| Granola / Notion / Slack access | Company context (see context-sources.md) | the user provides it | ask the user for a 5-line braindump plus the website |

Model names and endpoints change often. **Before calling any provider, check its current docs with WebFetch** and pick its best current model. Don't rely on remembered model ids.

## API sketches (verify against the current docs)
- **ElevenLabs:**
  - List voices: `GET https://api.elevenlabs.io/v1/voices` (header `xi-api-key`).
  - List models: `GET /v1/models`.
  - TTS: `POST /v1/text-to-speech/{voice_id}?output_format=mp3_44100_192` with body `{"text": "...", "model_id": "<best current model>", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.2}}`.
  - Generate **one file per line**.
  - For a calm, cinematic narrator, lower the style value and raise stability.
- **fal:** queue API, `POST https://queue.fal.run/<model-path>` with header `Authorization: Key $FAL_KEY`, then poll the `status_url` / `response_url`. Download the mp4 to `footage/ai/<shot-id>.mp4`.
- **Pexels video:** `GET https://api.pexels.com/videos/search?query=...&orientation=landscape` with header `Authorization: $PEXELS_API_KEY`.

## The ask message (template)
> To make this film I can use a few optional services. Here's what each one adds. Paste any you have, or say "skip" and I'll use the fallback:
> - **ElevenLabs** (narrator voice). Without it: scratch voice now, and you record the final one. elevenlabs.io → API keys
> - **fal.ai** (AI cinematic shots, e.g. Veo/Kling). Without it: I write the shot prompts and build code stand-ins. fal.ai → Keys
> - **Pexels** (free stock clips). Without it: I list download links for you.

## Which key actually gets used
- A key already exported in the shell (e.g. `GEMINI_API_KEY`) can silently win over the one the user gives you for this project: eleven Veo shots were once billed to the shell key. Scripts read keys from `$ROOT/.env` explicitly. Before spending, confirm which key a call uses, and when a key arrives mid-run, message every running agent that uses it.
- Provider quirks seen: Veo 3.1 does 1080p only at 8s (4/6s = 720p), and "shot on 35mm film" makes it draw fake film gates, so ask for full-bleed with no markings. ElevenLabs shared-library voices must be added to the account before API use (tell the user). YouTube may block HD downloads mid-session, so fetch key archival early.

## Storage rules
- Write keys to `$ROOT/.env` (`KEY=value`, one per line) and `chmod 600 $ROOT/.env`. Add `.env` to `.gitignore` if the folder is a repo.
- Load keys in scripts with `python-dotenv` or `set -a; . ./.env; set +a`. Never echo a key, never put one in a URL that gets logged, never write one into BRIEF.md (write "in `.env` as `ELEVENLABS_API_KEY`").
- If the user pasted a key into chat, suggest rotating it once the project is done.
- Planner mode: the brief references `.env` by path, and you write the keys into that `.env` yourself.

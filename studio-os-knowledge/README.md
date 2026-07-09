# Studio OS Knowledge Pipeline — YouTube → Obsidian

Watch a set of YouTube channels, and whenever one publishes a new video, fetch
its transcript, distill it with **Claude** into a structured Markdown note, tag
it against your **agents / skills taxonomy**, and drop it into an **Obsidian
vault**. The notes become reusable *context* for your Studio OS agents (user
acquisition, monetization, analytics, market intelligence, editorial, …) and
their skills (ASO, creative hooks, ROAS, paywalls, icon/screenshot makers, …).

Runs unattended on a **GitHub Actions cron** and commits new notes back to the
repo — no server needed.

```
RSS poll  →  new video ids  →  transcript (captions → whisper fallback)
          →  Claude structured note (classified vs taxonomy)  →  vault/*.md  →  state
```

## How it works

| Concern | Approach |
| --- | --- |
| New-video detection | Each channel's public RSS feed (`.../feeds/videos.xml?channel_id=…`). No API key/quota. |
| Transcript | `youtube-transcript-api` (captions) first; `yt-dlp` + `faster-whisper` fallback when a video has no captions. |
| Structuring | Claude (default `claude-sonnet-5`) via forced tool use → schema-valid JSON, classified against `config/taxonomy.yaml`. |
| Output | Obsidian note: YAML frontmatter + `[[Agent/…]]` wikilinks and `#skill/…` tags. |
| Idempotency | `state/processed.json` records processed video ids; re-runs do nothing new. |
| Runtime | `.github/workflows/poll.yml` cron; commits `vault/` + `state/` back. |

## Setup

1. **Create the repo & add the secret.** Push this project to a GitHub repo, then
   add a repository secret **`ANTHROPIC_API_KEY`** (Settings → Secrets and
   variables → Actions).
2. **Add your channels.** Edit `config/channels.yaml`. For each channel supply a
   `channel_id` (`UC…`), a `handle` (`@name`), or a `url`. Handles/urls are
   resolved to a `channel_id` on the first run and cached back into the file.
3. **(Optional) Tune the taxonomy.** Edit `config/taxonomy.yaml` — the `agents`
   and `skills` ids here are exactly what Claude classifies notes into.
4. **Point Obsidian at `vault/`.** Open the `vault/` folder as a vault (or as a
   folder inside an existing vault). Nested tags (`#skill/…`) and wikilinks give
   you graph navigation across the knowledge base.

The cron runs every 6 hours by default — change the schedule in
`.github/workflows/poll.yml`. You can also trigger it manually from the Actions
tab (`workflow_dispatch`).

## Local usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[whisper,dev]"      # omit [whisper] to skip the audio fallback
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in .env

# Verify on a single known video (prints the note, writes nothing):
ytnotes run --video "https://www.youtube.com/watch?v=<id-with-captions>" --dry-run

# Exercise the whisper path on a captions-less video:
ytnotes run --video "<url>" --force-whisper --dry-run

# Full poll of all configured channels:
ytnotes run                 # writes notes + updates state
ytnotes run --dry-run       # print notes, no writes
ytnotes run --force         # ignore state and reprocess
```

## Configuration reference

`config/channels.yaml`:

```yaml
defaults:
  languages: [en]              # preferred transcript languages
  whisper_fallback: true       # transcribe when no captions exist
  first_seen_max_age_days: 14  # first time a channel is seen, ignore older videos
channels:
  - name: "Some Channel"
    handle: "@somehandle"      # or channel_id: "UC..."  or  url: "..."
    agents_hint: []            # optional classification bias
    skills_hint: []
```

Environment overrides (also settable in the workflow):

| Var | Default | Meaning |
| --- | --- | --- |
| `YTNOTES_MODEL` | `claude-sonnet-5` | `claude-opus-4-8` (max quality) or `claude-haiku-4-5` (cheapest) |
| `YTNOTES_MAX_VIDEOS_PER_RUN` | `25` | cap total videos processed per run |
| `YTNOTES_MAX_WHISPER_PER_RUN` | `3` | cap heavy whisper transcriptions per run |
| `YTNOTES_USE_THINKING` | off | let Claude reason before emitting the note |

## Cost & robustness notes

- **Cost:** ~$0.04/note on Sonnet 5 (a 30-min video ≈ 5k in + 1.5k out tokens);
  ~$0.06 on Opus 4.8; Haiku is cheapest. The system/taxonomy prefix is marked
  for prompt caching to cut cost across a back-to-back run.
- **Whisper in CI is best-effort.** YouTube sometimes rate-limits/blocks
  datacenter IPs, so audio downloads can fail in GitHub Actions. Such videos are
  **flagged and skipped** (not recorded as processed) so a later local run can
  pick them up. Captions cover the large majority of videos.
- **First-run flood guard:** a brand-new channel only ingests videos newer than
  `first_seen_max_age_days`, so you don't import an entire back-catalogue.

## Layout

```
config/     channels.yaml, taxonomy.yaml
state/      processed.json          (idempotency; committed by the workflow)
vault/      generated Obsidian notes
src/ytnotes/
  config.py state.py channels.py feeds.py note.py pipeline.py cli.py
  transcript/  captions.py base.py whisper_fallback.py
  llm/         schema.py client.py structure.py
.github/workflows/poll.yml
tests/
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

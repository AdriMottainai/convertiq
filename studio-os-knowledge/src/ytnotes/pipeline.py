"""End-to-end orchestration: channels -> new videos -> notes -> state.

Enforces per-run caps (total videos and Whisper transcriptions) and prints a
run summary. Designed to be safe to run repeatedly (idempotent via the state
store) and to degrade gracefully when a single video or channel fails.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .channels import resolve_all
from .config import Settings, get_api_key
from .feeds import VideoRef, new_videos_for_channel
from .llm.client import make_client
from .llm.structure import structure_transcript
from .note import write_note
from .state import ProcessedRecord, State
from .transcript import get_transcript


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _channel_ever_seen(state: State, channel_name: str) -> bool:
    return any(rec.channel == channel_name for rec in state._records.values())


def process_video(
    client,
    video: VideoRef,
    settings: Settings,
    *,
    channel_cfg=None,
    allow_whisper: bool,
    dry_run: bool,
) -> ProcessedRecord:
    """Fetch transcript, structure it, write a note. Returns a state record."""
    languages = settings.channels.defaults.languages
    result = get_transcript(
        video.video_id,
        video.url,
        languages=languages,
        allow_whisper=allow_whisper,
    )
    captured = _now_iso()

    if not result.ok:
        print(f"[pipeline] SKIP {video.video_id} '{video.title}': {result.reason}")
        return ProcessedRecord(
            title=video.title,
            channel=video.channel,
            processed_at=captured,
            method="skipped",
            reason=result.reason,
        )

    note = structure_transcript(client, video, result.text, settings, channel_cfg=channel_cfg)

    note_path: str | None = None
    if dry_run:
        from .note import render_note

        print("\n" + "=" * 70)
        print(render_note(video, note, method=result.method, language=result.language, captured=captured))
        print("=" * 70 + "\n")
    else:
        path: Path = write_note(
            settings.vault_dir,
            video,
            note,
            method=result.method,
            language=result.language,
            captured=captured,
        )
        note_path = str(path.relative_to(settings.vault_dir.parent))
        print(f"[pipeline] NOTE {video.video_id} -> {note_path}")

    return ProcessedRecord(
        title=video.title,
        channel=video.channel,
        processed_at=captured,
        note_path=note_path,
        method=result.method,
    )


def run(settings: Settings, *, dry_run: bool = False, force: bool = False) -> dict:
    """Poll all channels and process new videos. Returns a summary dict."""
    api_key = get_api_key()
    client = make_client(api_key)
    state = State.load(settings.state_path)
    now = datetime.now(timezone.utc)

    name_to_id = resolve_all(settings, persist=not dry_run)
    cfg_by_name = {c.name: c for c in settings.channels.channels}

    processed = 0
    whisper_used = 0
    skipped = 0
    notes = 0

    for name, channel_id in name_to_id.items():
        if processed >= settings.max_videos_per_run:
            print("[pipeline] reached max_videos_per_run cap; stopping.")
            break

        def _is_processed(vid: str) -> bool:
            return (not force) and state.is_processed(vid)

        try:
            fresh = new_videos_for_channel(
                channel_id,
                name,
                is_processed=_is_processed,
                now=now,
                first_seen_max_age_days=settings.channels.defaults.first_seen_max_age_days,
                channel_ever_seen=_channel_ever_seen(state, name),
            )
        except Exception as exc:  # noqa: BLE001 - one bad channel shouldn't kill the run
            print(f"[pipeline] WARN feed failed for '{name}': {exc}")
            continue

        for video in fresh:
            if processed >= settings.max_videos_per_run:
                break
            allow_whisper = (
                settings.channels.defaults.whisper_fallback
                and whisper_used < settings.max_whisper_per_run
            )
            record = process_video(
                client,
                video,
                settings,
                channel_cfg=cfg_by_name.get(name),
                allow_whisper=allow_whisper,
                dry_run=dry_run,
            )
            processed += 1
            if record.method == "whisper":
                whisper_used += 1
            if record.method == "skipped":
                skipped += 1
            else:
                notes += 1
            # Only record successful notes in state, so a transient skip can be
            # retried on a later run (e.g. locally where whisper works).
            if not dry_run and record.method != "skipped":
                state.mark(video.video_id, record)

    if not dry_run:
        state.save()

    summary = {
        "channels": len(name_to_id),
        "processed": processed,
        "notes": notes,
        "skipped": skipped,
        "whisper_used": whisper_used,
    }
    print(f"[pipeline] done: {summary}")
    return summary


def run_single_video(settings: Settings, url: str, *, dry_run: bool, force_whisper: bool = False) -> None:
    """Process one video by URL — for local verification, no state writes."""
    from .feeds import VideoRef

    video_id = _extract_video_id(url)
    video = VideoRef(
        video_id=video_id,
        title=f"(single-video {video_id})",
        channel="manual",
        channel_id="manual",
        url=f"https://www.youtube.com/watch?v={video_id}",
        published=datetime.now(timezone.utc),
    )
    client = make_client(get_api_key())

    if force_whisper:
        from .transcript.whisper_fallback import transcribe_with_whisper

        text = transcribe_with_whisper(video.url, language=settings.channels.defaults.languages[0])
        result_method, result_lang = "whisper", settings.channels.defaults.languages[0]
    else:
        result = get_transcript(
            video.video_id,
            video.url,
            languages=settings.channels.defaults.languages,
            allow_whisper=settings.channels.defaults.whisper_fallback,
        )
        if not result.ok:
            print(f"[pipeline] could not get transcript: {result.reason}")
            return
        text, result_method, result_lang = result.text, result.method, result.language

    note = structure_transcript(client, video, text, settings)
    from .note import render_note, write_note

    captured = _now_iso()
    if dry_run:
        print("\n" + render_note(video, note, method=result_method, language=result_lang, captured=captured))
    else:
        path = write_note(settings.vault_dir, video, note, method=result_method, language=result_lang, captured=captured)
        print(f"[pipeline] wrote {path}")


def _extract_video_id(url: str) -> str:
    import re

    m = re.search(r"(?:v=|youtu\.be/|/shorts/|/embed/)([0-9A-Za-z_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
        return url
    raise ValueError(f"could not extract a video id from: {url}")

"""Poll YouTube channel RSS feeds and yield newly published videos.

Each channel exposes an Atom feed at
``https://www.youtube.com/feeds/videos.xml?channel_id=UC...`` listing its ~15
most recent uploads — no API key or quota required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import feedparser
import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class VideoRef(BaseModel):
    video_id: str
    title: str
    channel: str
    channel_id: str
    url: str
    published: datetime  # timezone-aware UTC


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
def _fetch_feed(channel_id: str) -> str:
    url = FEED_URL.format(channel_id=channel_id)
    with httpx.Client(timeout=20.0, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _parse_published(entry) -> datetime:
    # feedparser exposes published_parsed as a time.struct_time in UTC.
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def fetch_channel_videos(channel_id: str, channel_name: str) -> list[VideoRef]:
    """Return all videos currently listed in a channel's RSS feed (newest first)."""
    raw = _fetch_feed(channel_id)
    parsed = feedparser.parse(raw)
    videos: list[VideoRef] = []
    for entry in parsed.entries:
        vid = getattr(entry, "yt_videoid", None)
        if not vid:
            # Fall back to parsing the id field: "yt:video:VIDEOID"
            eid = getattr(entry, "id", "")
            vid = eid.rsplit(":", 1)[-1] if ":" in eid else None
        if not vid:
            continue
        videos.append(
            VideoRef(
                video_id=vid,
                title=getattr(entry, "title", "(untitled)"),
                channel=channel_name,
                channel_id=channel_id,
                url=f"https://www.youtube.com/watch?v={vid}",
                published=_parse_published(entry),
            )
        )
    return videos


def new_videos_for_channel(
    channel_id: str,
    channel_name: str,
    *,
    is_processed,
    now: datetime,
    first_seen_max_age_days: int | None,
    channel_ever_seen: bool,
) -> list[VideoRef]:
    """Filter a channel's feed down to unprocessed (and recent-enough) videos.

    ``is_processed`` is a callable ``video_id -> bool`` (from the state store).
    ``channel_ever_seen`` indicates whether we've processed anything from this
    channel before; when False we apply the ``first_seen_max_age_days`` guard so
    a brand-new channel doesn't dump its whole feed into the vault.
    """
    videos = fetch_channel_videos(channel_id, channel_name)
    fresh = [v for v in videos if not is_processed(v.video_id)]

    if not channel_ever_seen and first_seen_max_age_days is not None:
        cutoff = now - timedelta(days=first_seen_max_age_days)
        fresh = [v for v in fresh if v.published >= cutoff]

    fresh.sort(key=lambda v: v.published)  # oldest-first so notes accrue chronologically
    return fresh

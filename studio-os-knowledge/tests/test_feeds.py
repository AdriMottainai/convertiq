from datetime import datetime, timezone
from pathlib import Path

import feedparser

from ytnotes import feeds

FIXTURE = Path(__file__).parent / "fixtures" / "sample_feed.xml"


def _parse_fixture(monkeypatch):
    raw = FIXTURE.read_text(encoding="utf-8")
    monkeypatch.setattr(feeds, "_fetch_feed", lambda channel_id: raw)


def test_fetch_channel_videos_parses_entries(monkeypatch):
    _parse_fixture(monkeypatch)
    videos = feeds.fetch_channel_videos("UC1234567890abcdefghABCD", "Sample Channel")
    ids = {v.video_id for v in videos}
    assert ids == {"aaaaaaaaaaa", "bbbbbbbbbbb"}
    v = next(v for v in videos if v.video_id == "aaaaaaaaaaa")
    assert v.url == "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    assert v.channel == "Sample Channel"
    assert v.published.tzinfo is not None


def test_new_videos_filters_processed(monkeypatch):
    _parse_fixture(monkeypatch)
    processed = {"bbbbbbbbbbb"}
    fresh = feeds.new_videos_for_channel(
        "UC1234567890abcdefghABCD",
        "Sample Channel",
        is_processed=lambda vid: vid in processed,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
        first_seen_max_age_days=None,
        channel_ever_seen=True,
    )
    assert [v.video_id for v in fresh] == ["aaaaaaaaaaa"]


def test_first_seen_age_guard_drops_old_videos(monkeypatch):
    _parse_fixture(monkeypatch)
    # Brand-new channel: only videos within 14 days of "now" should survive.
    fresh = feeds.new_videos_for_channel(
        "UC1234567890abcdefghABCD",
        "Sample Channel",
        is_processed=lambda vid: False,
        now=datetime(2026, 7, 9, tzinfo=timezone.utc),
        first_seen_max_age_days=14,
        channel_ever_seen=False,
    )
    assert [v.video_id for v in fresh] == ["aaaaaaaaaaa"]  # older ROAS video dropped


def test_feedparser_reads_yt_videoid():
    # Guards our assumption that feedparser exposes yt_videoid for this schema.
    parsed = feedparser.parse(FIXTURE.read_text(encoding="utf-8"))
    assert parsed.entries[0].yt_videoid == "aaaaaaaaaaa"

"""Resolve configured channels to canonical YouTube channel_ids (``UC...``).

A channel may be configured by raw ``channel_id``, ``@handle``, or ``url``.
Only ``channel_id`` works directly with the RSS feed endpoint, so handles/urls
are resolved by fetching the channel page once and extracting the id. Resolved
ids are written back into ``channels.yaml`` so resolution happens only once.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ChannelConfig, Settings

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Matches "UC" + 22 url-safe base64 chars — the canonical channel id shape.
_CHANNEL_ID_RE = re.compile(r"UC[0-9A-Za-z_-]{22}")


class ChannelResolutionError(RuntimeError):
    pass


def _extract_channel_id_from_url(url: str) -> str | None:
    m = re.search(r"/channel/(UC[0-9A-Za-z_-]{22})", url)
    return m.group(1) if m else None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=16))
def _fetch(url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=20.0, headers={"User-Agent": _UA}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _resolve_from_page(target_url: str) -> str:
    """Fetch a channel page and pull the canonical channel_id out of its HTML."""
    html = _fetch(target_url)
    # The canonical id appears in several stable places in the page markup.
    for pattern in (
        r'"channelId":"(UC[0-9A-Za-z_-]{22})"',
        r'"externalId":"(UC[0-9A-Za-z_-]{22})"',
        r'<meta itemprop="(?:identifier|channelId)" content="(UC[0-9A-Za-z_-]{22})">',
        r'/channel/(UC[0-9A-Za-z_-]{22})',
    ):
        m = re.search(pattern, html)
        if m:
            return m.group(1)
    raise ChannelResolutionError(f"could not extract channel_id from {target_url}")


def resolve_channel_id(channel: ChannelConfig) -> str:
    """Return the canonical channel_id for a channel, fetching if needed."""
    if channel.channel_id and _CHANNEL_ID_RE.fullmatch(channel.channel_id):
        return channel.channel_id

    if channel.url:
        direct = _extract_channel_id_from_url(channel.url)
        if direct:
            return direct
        return _resolve_from_page(channel.url)

    if channel.handle:
        handle = channel.handle if channel.handle.startswith("@") else f"@{channel.handle}"
        return _resolve_from_page(f"https://www.youtube.com/{handle}")

    raise ChannelResolutionError(
        f"channel '{channel.name}' has no channel_id, handle, or url to resolve"
    )


def resolve_all(settings: Settings, *, persist: bool = True) -> dict[str, str]:
    """Resolve every configured channel; optionally cache ids back to channels.yaml.

    Returns a mapping of channel name -> channel_id. Channels that fail to
    resolve are logged (via exception message re-raised by the caller) — here we
    skip them so one bad entry doesn't abort the whole run.
    """
    resolved: dict[str, str] = {}
    newly_resolved = False

    for ch in settings.channels.channels:
        if not ch.has_target():
            continue
        try:
            cid = resolve_channel_id(ch)
        except (ChannelResolutionError, httpx.HTTPError) as exc:  # noqa: PERF203
            print(f"[channels] WARN could not resolve '{ch.name}': {exc}")
            continue
        resolved[ch.name] = cid
        if ch.channel_id != cid:
            ch.channel_id = cid
            newly_resolved = True

    if persist and newly_resolved:
        _persist_ids(settings.config_dir / "channels.yaml", resolved)

    return resolved


def _persist_ids(path: Path, name_to_id: dict[str, str]) -> None:
    """Write resolved channel_ids back into channels.yaml (best-effort).

    Uses a plain load/dump. Comments in the file are not preserved by PyYAML, so
    we only rewrite when something actually changed and keep the structure intact.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("channels", []):
            name = entry.get("name")
            if name in name_to_id:
                entry["channel_id"] = name_to_id[name]
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - persistence is best-effort
        print(f"[channels] WARN failed to persist resolved ids: {exc}")

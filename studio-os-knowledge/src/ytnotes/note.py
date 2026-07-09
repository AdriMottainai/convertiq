"""Render a StructuredNote into an Obsidian-compatible Markdown file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml
from slugify import slugify

from .feeds import VideoRef
from .llm.structure import StructuredNote


def note_filename(video: VideoRef) -> str:
    date = video.published.strftime("%Y-%m-%d")
    channel_slug = slugify(video.channel)[:40] or "channel"
    title_slug = slugify(video.title)[:60] or "video"
    return f"{date}-{channel_slug}-{title_slug}-{video.video_id}.md"


def _frontmatter(video: VideoRef, note: StructuredNote, method: str, language: str, captured: str) -> str:
    fm = {
        "title": video.title,
        "channel": video.channel,
        "video_id": video.video_id,
        "video_url": video.url,
        "published": video.published.strftime("%Y-%m-%d"),
        "captured": captured,
        "transcript_method": method,
        "language": language or "",
        "agents": note.agents,
        "skills": note.skills,
        "tags": note.tags,
        "source": "youtube",
    }
    # default_flow_style=False keeps lists block-style; sort_keys=False preserves order.
    body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"---\n{body}---\n"


def _bullets(items: list[str]) -> str:
    if not items:
        return "_None extracted._\n"
    return "".join(f"- {i.strip()}\n" for i in items)


def _quotes(items: list[str]) -> str:
    if not items:
        return "_None extracted._\n"
    return "".join(f"> {i.strip()}\n\n" for i in items)


def render_note(
    video: VideoRef,
    note: StructuredNote,
    *,
    method: str,
    language: str,
    captured: str,
) -> str:
    """Return the full Markdown document for a video's knowledge note."""
    agent_links = " · ".join(f"[[Agent/{a}]]" for a in note.agents) or "_unclassified_"
    skill_tags = " ".join(f"#skill/{s}" for s in note.skills)

    return (
        _frontmatter(video, note, method, language, captured)
        + f"\n# {video.title}\n\n"
        + f"**Channel:** {video.channel} · **Published:** {video.published:%Y-%m-%d} · "
        + f"**Source:** [YouTube]({video.url})\n\n"
        + "## Summary\n"
        + f"{note.summary.strip() or '_No summary._'}\n\n"
        + "## Key Takeaways\n"
        + _bullets(note.key_takeaways)
        + "\n## Actionable Insights\n"
        + _bullets(note.actionable_insights)
        + "\n## Notable Quotes\n"
        + _quotes(note.notable_quotes)
        + "## Agents & Skills\n"
        + f"{agent_links}\n\n"
        + (f"{skill_tags}\n\n" if skill_tags else "")
        + f"[Watch on YouTube]({video.url})\n"
    )


def write_note(
    vault_dir: Path,
    video: VideoRef,
    note: StructuredNote,
    *,
    method: str,
    language: str,
    captured: str,
) -> Path:
    vault_dir.mkdir(parents=True, exist_ok=True)
    path = vault_dir / note_filename(video)
    content = render_note(video, note, method=method, language=language, captured=captured)
    path.write_text(content, encoding="utf-8")
    return path

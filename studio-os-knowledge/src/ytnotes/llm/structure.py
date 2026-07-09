"""Turn a raw transcript into a validated, classified StructuredNote."""

from __future__ import annotations

import anthropic
from pydantic import BaseModel, Field

from ..config import Settings, Taxonomy
from ..feeds import VideoRef
from .client import call_structured
from .schema import build_note_schema


class StructuredNote(BaseModel):
    summary: str = ""
    key_takeaways: list[str] = Field(default_factory=list)
    actionable_insights: list[str] = Field(default_factory=list)
    notable_quotes: list[str] = Field(default_factory=list)
    agents: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


def _taxonomy_block(taxonomy: Taxonomy) -> str:
    lines = ["AGENTS (id — description):"]
    for a in taxonomy.agents:
        lines.append(f"- {a.id} — {a.description}")
    lines.append("")
    lines.append("SKILLS (id — description):")
    for s in taxonomy.skills:
        lines.append(f"- {s.id} — {s.description}")
    return "\n".join(lines)


def build_system_prompt(taxonomy: Taxonomy) -> str:
    """Stable, cacheable prefix: role + full taxonomy + classification rules."""
    return (
        "You are a knowledge editor for a mobile-apps studio 'operating system'. "
        "You read transcripts of YouTube videos about app marketing, user acquisition, "
        "monetization, ASO, analytics, and product, and distill them into dense, reusable "
        "knowledge notes that will serve as CONTEXT for specialized agents and skills.\n\n"
        "Extract only substantive, non-obvious signal. Prefer specific numbers, tactics, "
        "tools, and frameworks over generic advice. Write takeaways and insights as crisp, "
        "self-contained bullets that make sense without watching the video.\n\n"
        "Classify each note against the taxonomy below. Choose the agent ids and skill ids "
        "the content genuinely helps — usually 1-4 of each. It is fine to leave skills empty "
        "if none clearly apply, but always assign at least one agent. Only use ids that appear "
        "in the taxonomy.\n\n"
        f"{_taxonomy_block(taxonomy)}"
    )


def build_user_content(video: VideoRef, transcript: str, settings: Settings, channel_cfg=None) -> str:
    transcript = transcript[: settings.max_transcript_chars]
    hint = ""
    if channel_cfg is not None and (channel_cfg.agents_hint or channel_cfg.skills_hint):
        hint = (
            "\nChannel classification hints (bias, not a constraint): "
            f"agents={channel_cfg.agents_hint} skills={channel_cfg.skills_hint}\n"
        )
    return (
        f"Video title: {video.title}\n"
        f"Channel: {video.channel}\n"
        f"URL: {video.url}\n"
        f"{hint}\n"
        "Transcript:\n"
        f"{transcript}\n\n"
        "Produce the structured knowledge note by calling the emit_note tool."
    )


def _clean(note: StructuredNote, taxonomy: Taxonomy) -> StructuredNote:
    """Dedup and drop any ids not in the taxonomy; guarantee >=1 agent."""
    valid_agents = set(taxonomy.agent_ids())
    valid_skills = set(taxonomy.skill_ids())

    def dedup(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    note.agents = dedup([a for a in note.agents if a in valid_agents])
    note.skills = dedup([s for s in note.skills if s in valid_skills])
    note.tags = dedup([t.strip().lower().replace(" ", "-") for t in note.tags if t.strip()])

    if not note.agents and taxonomy.agent_ids():
        # Never leave a note unclassified; editorial is the safe catch-all.
        fallback = "editorial" if "editorial" in valid_agents else taxonomy.agent_ids()[0]
        note.agents = [fallback]
    return note


def structure_transcript(
    client: anthropic.Anthropic,
    video: VideoRef,
    transcript: str,
    settings: Settings,
    *,
    channel_cfg=None,
) -> StructuredNote:
    schema = build_note_schema(settings.taxonomy)
    system_prompt = build_system_prompt(settings.taxonomy)
    user_content = build_user_content(video, transcript, settings, channel_cfg)

    data = call_structured(
        client,
        model=settings.model,
        system_prompt=system_prompt,
        note_schema=schema,
        user_content=user_content,
        use_thinking=settings.use_thinking,
    )
    note = StructuredNote.model_validate(data)
    return _clean(note, settings.taxonomy)

"""Build the JSON schema for a structured note from the taxonomy at runtime.

``agents`` and ``skills`` are enum-constrained to the taxonomy ids so Claude can
only classify against known values. Note: Anthropic structured outputs do NOT
support array-length / minLength constraints, so "at least one agent", dedup,
and unknown-id filtering are enforced in Python after parsing (see structure.py).
"""

from __future__ import annotations

from ..config import Taxonomy


def build_note_schema(taxonomy: Taxonomy) -> dict:
    agent_ids = taxonomy.agent_ids()
    skill_ids = taxonomy.skill_ids()

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {
                "type": "string",
                "description": "A 1-2 sentence distillation of the video's core message.",
            },
            "key_takeaways": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The most important points, each a concise standalone bullet.",
            },
            "actionable_insights": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concrete, applicable tactics or steps framed so a Studio OS "
                    "agent or skill could act on them."
                ),
            },
            "notable_quotes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verbatim or lightly-trimmed memorable quotes from the speaker.",
            },
            "agents": {
                "type": "array",
                "items": {"type": "string", "enum": agent_ids},
                "description": "The agent ids this content is most relevant to.",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string", "enum": skill_ids},
                "description": "The skill ids this content is most relevant to.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Free-form lowercase topic tags (no spaces; use hyphens).",
            },
        },
        "required": [
            "summary",
            "key_takeaways",
            "actionable_insights",
            "notable_quotes",
            "agents",
            "skills",
            "tags",
        ],
    }

from datetime import datetime, timezone

import yaml

from ytnotes.feeds import VideoRef
from ytnotes.llm.structure import StructuredNote
from ytnotes.note import note_filename, render_note


def _video():
    return VideoRef(
        video_id="dQw4w9WgXcQ",
        title="How we cut CPI 40% with hook testing",
        channel="Example Channel",
        channel_id="UC1234567890abcdefghABCD",
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        published=datetime(2026, 7, 3, tzinfo=timezone.utc),
    )


def _note():
    return StructuredNote(
        summary="A short distillation.",
        key_takeaways=["Test hooks early", "Kill losers fast"],
        actionable_insights=["Launch 5 hook variants per concept"],
        notable_quotes=['"The hook is 80% of the creative."'],
        agents=["user-acquisition-manager", "data-analytics"],
        skills=["creative-hook-manager", "roas-optimizer"],
        tags=["ua", "creatives", "ab-testing"],
    )


def test_filename_is_deterministic_and_safe():
    fn = note_filename(_video())
    assert fn == "2026-07-03-example-channel-how-we-cut-cpi-40-with-hook-testing-dQw4w9WgXcQ.md"


def test_frontmatter_is_valid_yaml_with_expected_fields():
    md = render_note(_video(), _note(), method="captions", language="en", captured="2026-07-04T00:00:00Z")
    assert md.startswith("---\n")
    fm_block = md.split("---\n", 2)[1]
    fm = yaml.safe_load(fm_block)
    assert fm["video_id"] == "dQw4w9WgXcQ"
    assert fm["agents"] == ["user-acquisition-manager", "data-analytics"]
    assert fm["skills"] == ["creative-hook-manager", "roas-optimizer"]
    assert fm["transcript_method"] == "captions"
    assert fm["source"] == "youtube"


def test_body_contains_obsidian_links_and_tags():
    md = render_note(_video(), _note(), method="captions", language="en", captured="2026-07-04T00:00:00Z")
    assert "[[Agent/user-acquisition-manager]]" in md
    assert "#skill/creative-hook-manager" in md
    assert "## Key Takeaways" in md
    assert "Test hooks early" in md

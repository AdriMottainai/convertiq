from ytnotes.config import Taxonomy, TaxonomyEntry
from ytnotes.llm.schema import build_note_schema
from ytnotes.llm.structure import StructuredNote, _clean


def _tax():
    return Taxonomy(
        agents=[TaxonomyEntry(id="editorial"), TaxonomyEntry(id="data-analytics")],
        skills=[TaxonomyEntry(id="roas-optimizer"), TaxonomyEntry(id="icon-maker")],
    )


def test_schema_enums_come_from_taxonomy():
    schema = build_note_schema(_tax())
    assert schema["properties"]["agents"]["items"]["enum"] == ["editorial", "data-analytics"]
    assert schema["properties"]["skills"]["items"]["enum"] == ["roas-optimizer", "icon-maker"]
    assert schema["additionalProperties"] is False


def test_clean_drops_unknown_ids_and_dedups():
    note = StructuredNote(
        agents=["data-analytics", "data-analytics", "nonexistent"],
        skills=["roas-optimizer", "unknown-skill"],
        tags=["UA Stuff", "ua-stuff", "  Retention "],
    )
    cleaned = _clean(note, _tax())
    assert cleaned.agents == ["data-analytics"]
    assert cleaned.skills == ["roas-optimizer"]
    assert cleaned.tags == ["ua-stuff", "retention"]


def test_clean_guarantees_at_least_one_agent():
    note = StructuredNote(agents=[], skills=[])
    cleaned = _clean(note, _tax())
    assert cleaned.agents == ["editorial"]  # safe catch-all

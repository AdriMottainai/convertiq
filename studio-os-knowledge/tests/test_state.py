from ytnotes.state import ProcessedRecord, State


def test_state_roundtrip(tmp_path):
    path = tmp_path / "processed.json"
    state = State.load(path)
    assert not state.is_processed("vid1")

    state.mark(
        "vid1",
        ProcessedRecord(
            title="T", channel="C", processed_at="2026-07-09T00:00:00Z",
            note_path="vault/x.md", method="captions",
        ),
    )
    state.save()

    reloaded = State.load(path)
    assert reloaded.is_processed("vid1")
    assert "vid1" in reloaded
    rec = reloaded.get("vid1")
    assert rec.method == "captions"
    assert rec.note_path == "vault/x.md"


def test_skipped_record_excludes_none(tmp_path):
    path = tmp_path / "processed.json"
    state = State.load(path)
    state.mark(
        "vid2",
        ProcessedRecord(
            title="T", channel="C", processed_at="2026-07-09T00:00:00Z",
            method="skipped", reason="no captions",
        ),
    )
    state.save()
    text = path.read_text(encoding="utf-8")
    assert "note_path" not in text  # None fields excluded
    assert "no captions" in text

"""Processed-video state store.

A tiny JSON file mapping ``video_id -> record`` so the pipeline never
reprocesses a video. The workflow commits this file back to the repo, making
cloud runs idempotent across schedules.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class ProcessedRecord(BaseModel):
    title: str
    channel: str
    processed_at: str  # ISO8601, supplied by caller (no wall-clock in this module)
    note_path: str | None = None
    method: str  # "captions" | "whisper" | "skipped"
    reason: str | None = None  # populated when method == "skipped"


class State:
    def __init__(self, path: Path, records: dict[str, ProcessedRecord]):
        self.path = path
        self._records = records

    @classmethod
    def load(cls, path: Path | str) -> "State":
        path = Path(path)
        records: dict[str, ProcessedRecord] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8") or "{}")
            for vid, rec in raw.items():
                records[vid] = ProcessedRecord.model_validate(rec)
        return cls(path, records)

    def is_processed(self, video_id: str) -> bool:
        return video_id in self._records

    def mark(self, video_id: str, record: ProcessedRecord) -> None:
        self._records[video_id] = record

    def get(self, video_id: str) -> ProcessedRecord | None:
        return self._records.get(video_id)

    def __contains__(self, video_id: str) -> bool:  # convenience
        return self.is_processed(video_id)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            vid: rec.model_dump(exclude_none=True) for vid, rec in sorted(self._records.items())
        }
        self.path.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

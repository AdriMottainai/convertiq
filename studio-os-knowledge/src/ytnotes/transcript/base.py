"""Orchestrate transcript acquisition: captions first, Whisper fallback.

Never raises for an expected "no transcript" outcome — returns a
:class:`TranscriptResult` with ``ok=False`` so the pipeline can flag-and-skip.
"""

from __future__ import annotations

from pydantic import BaseModel

from .captions import NoCaptionsError, fetch_captions
from .whisper_fallback import (
    WhisperFetchError,
    WhisperUnavailableError,
    transcribe_with_whisper,
)


class TranscriptResult(BaseModel):
    ok: bool
    text: str = ""
    language: str = ""
    method: str = ""  # "captions" | "whisper"
    reason: str = ""  # populated when ok is False


def get_transcript(
    video_id: str,
    video_url: str,
    *,
    languages: list[str],
    allow_whisper: bool,
) -> TranscriptResult:
    """Return the best transcript we can obtain for a video."""
    # 1) Captions (fast, free, works from any IP).
    try:
        text, lang = fetch_captions(video_id, languages)
        return TranscriptResult(ok=True, text=text, language=lang, method="captions")
    except NoCaptionsError as exc:
        captions_reason = str(exc)

    # 2) Whisper fallback (heavy; may be blocked in CI).
    if not allow_whisper:
        return TranscriptResult(
            ok=False, reason=f"no captions and whisper disabled ({captions_reason})"
        )

    try:
        text = transcribe_with_whisper(video_url, language=languages[0] if languages else None)
        return TranscriptResult(
            ok=True, text=text, language=languages[0] if languages else "", method="whisper"
        )
    except WhisperUnavailableError as exc:
        return TranscriptResult(
            ok=False, reason=f"no captions; whisper unavailable ({exc})"
        )
    except WhisperFetchError as exc:
        return TranscriptResult(
            ok=False, reason=f"no captions; whisper failed ({exc})"
        )

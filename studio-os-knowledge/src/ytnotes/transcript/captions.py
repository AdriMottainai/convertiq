"""Fetch existing YouTube captions via youtube-transcript-api (1.x API).

Prefers manually-created transcripts, then auto-generated ones, honoring the
caller's preferred language order; falls back to any available transcript,
translated to the first preferred language when possible.
"""

from __future__ import annotations

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)


class NoCaptionsError(Exception):
    """Raised when a video has no usable captions."""


def fetch_captions(video_id: str, languages: list[str]) -> tuple[str, str]:
    """Return ``(text, language_code)`` for the best available captions.

    Raises :class:`NoCaptionsError` when nothing usable exists so the caller can
    decide whether to fall back to Whisper.
    """
    languages = languages or ["en"]
    api = YouTubeTranscriptApi()

    try:
        transcript_list = api.list(video_id)
    except (TranscriptsDisabled, NoTranscriptFound) as exc:
        raise NoCaptionsError(str(exc)) from exc
    except Exception as exc:  # network / IP-block / parsing errors
        raise NoCaptionsError(f"caption lookup failed: {exc}") from exc

    transcript = None
    # 1) Manually created in a preferred language.
    try:
        transcript = transcript_list.find_manually_created_transcript(languages)
    except Exception:
        pass
    # 2) Auto-generated in a preferred language.
    if transcript is None:
        try:
            transcript = transcript_list.find_generated_transcript(languages)
        except Exception:
            pass
    # 3) Anything at all, translated to the first preferred language if possible.
    if transcript is None:
        try:
            any_t = next(iter(transcript_list))
        except StopIteration:
            raise NoCaptionsError("no transcripts listed for video")
        transcript = any_t
        if languages and getattr(any_t, "is_translatable", False):
            try:
                transcript = any_t.translate(languages[0])
            except Exception:
                transcript = any_t

    try:
        fetched = transcript.fetch()  # FetchedTranscript (iterable of snippets)
    except Exception as exc:
        raise NoCaptionsError(f"failed to fetch caption text: {exc}") from exc

    text = " ".join(
        snippet.text.strip() for snippet in fetched if getattr(snippet, "text", "").strip()
    ).strip()
    if not text:
        raise NoCaptionsError("captions were empty")

    lang = getattr(transcript, "language_code", None) or (languages[0] if languages else "en")
    return text, lang

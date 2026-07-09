"""Whisper fallback: download audio with yt-dlp, transcribe with faster-whisper.

This path is heavier (network download + local inference) and, in cloud CI, is
best-effort: YouTube frequently rate-limits or blocks datacenter IPs, so audio
downloads can fail. Callers should treat a raised exception as "skip this video"
rather than a fatal error.

Dependencies (yt-dlp, faster-whisper) are imported lazily so the core captions
path works even if the optional ``[whisper]`` extra isn't installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

WHISPER_MODEL_SIZE = "small"  # good accuracy/speed trade-off on CPU


class WhisperUnavailableError(RuntimeError):
    """Raised when the optional whisper dependencies are not installed."""


class WhisperFetchError(RuntimeError):
    """Raised when audio download or transcription fails."""


def _download_audio(video_url: str, out_dir: Path) -> Path:
    try:
        import yt_dlp  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise WhisperUnavailableError(
            "yt-dlp not installed; install the '[whisper]' extra"
        ) from exc

    outtmpl = str(out_dir / "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # keep the download light — no playlist expansion
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            downloaded = ydl.prepare_filename(info)
    except Exception as exc:
        raise WhisperFetchError(f"audio download failed: {exc}") from exc

    path = Path(downloaded)
    if not path.exists():
        # yt-dlp may have chosen a different extension; grab whatever landed.
        candidates = list(out_dir.iterdir())
        if not candidates:
            raise WhisperFetchError("audio download produced no file")
        path = candidates[0]
    return path


def _transcribe(audio_path: Path, language: str | None) -> str:
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise WhisperUnavailableError(
            "faster-whisper not installed; install the '[whisper]' extra"
        ) from exc

    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), language=language)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        raise WhisperFetchError(f"transcription failed: {exc}") from exc


def transcribe_with_whisper(video_url: str, language: str | None = None) -> str:
    """Download and transcribe a video's audio. Returns the transcript text.

    Raises :class:`WhisperUnavailableError` or :class:`WhisperFetchError` on
    failure so the pipeline can flag-and-skip.
    """
    with tempfile.TemporaryDirectory(prefix="ytnotes_audio_") as tmp:
        out_dir = Path(tmp)
        audio_path = _download_audio(video_url, out_dir)
        text = _transcribe(audio_path, language)
    if not text:
        raise WhisperFetchError("whisper produced empty transcript")
    return text

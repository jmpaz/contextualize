from __future__ import annotations

from pathlib import Path

import pytest
import requests

from contextualize.references.audio_transcription import transcribe_audio_bytes
from contextualize.references.file import FileReference
from contextualize.references.url import URLReference
from contextualize.references.youtube import YouTubeReference


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        content: bytes,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.url = url
        self._text = content.decode("utf-8", errors="ignore")

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        raise ValueError("no json payload")


def test_transcribe_audio_bytes_requires_whisper_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WHISPER_API_BASE", raising=False)
    monkeypatch.delenv("WHISPER_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="WHISPER_API_BASE or WHISPER_API_KEY"):
        transcribe_audio_bytes(b"audio", filename="sample.mp3")


def test_file_reference_uses_media_transcription_for_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")

    calls: list[str] = []

    def _transcribe(
        path: str | Path,
        *,
        timeout: float = 600,
        use_cache: bool = True,
        refresh_cache: bool | None = None,
    ) -> str:
        calls.append(str(path))
        assert timeout == 600
        assert use_cache is True
        assert refresh_cache is False
        return "video transcript"

    monkeypatch.setattr(
        "contextualize.references.file.transcribe_media_file",
        _transcribe,
    )

    ref = FileReference(str(media_path), format="raw")
    assert ref.read() == "video transcript"
    assert ref.output == "video transcript"
    assert calls == [str(media_path)]


def test_url_reference_uses_media_transcription_for_video_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/video"
    captured: dict[str, str] = {}

    def _head_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.RequestException("head failed")

    def _get_video(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"video-bytes",
            url=url,
        )

    def _transcribe(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float = 600,
    ) -> str:
        assert data == b"video-bytes"
        assert timeout == 600
        captured["filename"] = filename
        captured["content_type"] = content_type or ""
        return "video transcript"

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr(
        "contextualize.references.url.transcribe_media_bytes",
        _transcribe,
    )

    ref = URLReference(url=url, format="raw", use_cache=False)
    assert ref.read() == "video transcript"
    assert ref.output == "video transcript"
    assert captured["filename"] == "video.mp4"
    assert captured["content_type"] == "video/mp4"


def test_url_reference_media_transcription_failure_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    url = "https://example.com/video"

    def _head_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.RequestException("head failed")

    def _get_video(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"video-bytes",
            url=url,
        )

    def _fail(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float = 600,
    ) -> str:
        raise RuntimeError("broken")

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr("contextualize.references.url.transcribe_media_bytes", _fail)

    with pytest.raises(ValueError, match="Media transcription failed for"):
        URLReference(url=url, format="raw", use_cache=False)


def test_youtube_reference_uses_shared_media_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "yt-audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "audio.mp3"
    audio_path.write_bytes(b"audio")

    ref = object.__new__(YouTubeReference)

    def _extract_audio(self: YouTubeReference) -> Path:
        return audio_path

    def _transcribe(path: str | Path, *, timeout: float = 600) -> str:
        assert str(path) == str(audio_path)
        assert timeout == 600
        return "yt transcript"

    monkeypatch.setattr(YouTubeReference, "_extract_audio", _extract_audio)
    monkeypatch.setattr(
        "contextualize.references.youtube.transcribe_media_file",
        _transcribe,
    )

    transcript, source = YouTubeReference._get_transcript(ref, 120)
    assert transcript == "yt transcript"
    assert source == "whisper"
    assert not audio_dir.exists()

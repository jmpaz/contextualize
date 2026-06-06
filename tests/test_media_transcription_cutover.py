from __future__ import annotations

from pathlib import Path

import pytest
import requests

from contextualize.references.audio_transcription import transcribe_audio_bytes
from contextualize.references.file import FileReference
from contextualize.references.url import URLReference

try:
    from contextualize.references.youtube import YouTubeReference
except ModuleNotFoundError:
    YouTubeReference = None


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


def test_transcribe_audio_bytes_requires_loaded_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_URL", raising=False)
    monkeypatch.delenv("OPENAI_TRANSCRIPTION_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="No transcription providers are loaded"):
        transcribe_audio_bytes(b"audio", filename="sample.mp3")


def test_file_reference_uses_video_context_for_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    media_path = tmp_path / "clip.mp4"
    media_path.write_bytes(b"video")

    calls: list[str] = []

    def _render(
        path: str | Path,
        *,
        timeout: float | None = None,
        use_cache: bool = True,
        refresh_cache: bool | None = None,
    ) -> str:
        calls.append(str(path))
        assert timeout is None
        assert use_cache is True
        assert refresh_cache is False
        return "video context"

    monkeypatch.setattr(
        "contextualize.references.file.render_video_file",
        _render,
    )

    ref = FileReference(str(media_path), format="raw")
    assert ref.read() == "video context"
    assert ref.output == "video context"
    assert calls == [str(media_path)]


def test_url_reference_uses_video_context_for_video_content_type(
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

    def _render(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float | None = None,
        source_url: str | None = None,
    ) -> str:
        assert data == b"video-bytes"
        assert timeout is None
        captured["filename"] = filename
        captured["content_type"] = content_type or ""
        captured["source_url"] = source_url or ""
        return "video context"

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr(
        "contextualize.references.url.render_video_bytes",
        _render,
    )

    ref = URLReference(url=url, format="raw", use_cache=False)
    assert ref.read() == "video context"
    assert ref.output == "video context"
    assert captured["filename"] == "video.mp4"
    assert captured["content_type"] == "video/mp4"
    assert captured["source_url"] == url


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
        timeout: float | None = None,
        source_url: str | None = None,
    ) -> str:
        raise RuntimeError("broken")

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr("contextualize.references.url.render_video_bytes", _fail)

    with pytest.raises(ValueError, match="Media transcription failed for"):
        URLReference(url=url, format="raw", use_cache=False)


def test_url_reference_video_cache_varies_by_video_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextualize.cache import url as url_cache

    monkeypatch.setattr(url_cache, "URL_CACHE_ROOT", tmp_path / "url-cache")
    url = "https://example.com/video"
    calls: list[dict[str, object] | None] = []

    def _head_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.RequestException("head failed")

    def _get_video(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResponse(
            status_code=200,
            headers={"Content-Type": "video/mp4"},
            content=b"video-bytes",
            url=url,
        )

    def _render(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float | None = None,
        source_url: str | None = None,
        plugin_overrides: dict[str, object] | None = None,
    ) -> str:
        calls.append(plugin_overrides)
        video = plugin_overrides.get("video") if plugin_overrides else None
        if isinstance(video, dict) and video.get("frames") is False:
            return "without frames"
        return "with frames"

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr("contextualize.references.url.render_video_bytes", _render)

    first = URLReference(
        url=url,
        format="raw",
        plugin_overrides={"video": {"frames": True}},
    )
    second = URLReference(
        url=url,
        format="raw",
        plugin_overrides={"video": {"frames": False}},
    )
    cached_first = URLReference(
        url=url,
        format="raw",
        plugin_overrides={"video": {"frames": True}},
    )

    assert first.read() == "with frames"
    assert second.read() == "without frames"
    assert cached_first.read() == "with frames"
    assert calls == [{"video": {"frames": True}}, {"video": {"frames": False}}]


def test_url_reference_does_not_cache_unavailable_video_transcript(
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

    def _render(*_args, **_kwargs) -> str:
        return "## Transcript\n\n[transcript unavailable: missing provider]"

    def _store(*_args, **_kwargs) -> None:
        raise AssertionError("unavailable transcript output should not be cached")

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_video)
    monkeypatch.setattr("contextualize.references.url.render_video_bytes", _render)
    monkeypatch.setattr("contextualize.cache.store_cached", _store)

    ref = URLReference(url=url, format="raw")

    assert "[transcript unavailable: missing provider]" in ref.read()


def test_youtube_reference_uses_shared_media_transcription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if YouTubeReference is None:
        pytest.skip("YouTubeReference is provided by the external ytdlp plugin")
    audio_dir = tmp_path / "yt-audio"
    audio_dir.mkdir()
    audio_path = audio_dir / "audio.mp3"
    audio_path.write_bytes(b"audio")

    ref = object.__new__(YouTubeReference)

    def _extract_audio(self: YouTubeReference) -> Path:
        return audio_path

    def _transcribe(path: str | Path, *, timeout: float | None = None) -> str:
        assert str(path) == str(audio_path)
        assert timeout is None
        return "yt transcript"

    monkeypatch.setattr(YouTubeReference, "_extract_audio", _extract_audio)
    monkeypatch.setattr(
        "contextualize.references.youtube.transcribe_media_file",
        _transcribe,
    )

    transcript, source = YouTubeReference._get_transcript(ref, 120)
    assert transcript == "yt transcript"
    assert source == "transcription"
    assert not audio_dir.exists()

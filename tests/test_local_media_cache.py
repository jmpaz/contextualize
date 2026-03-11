from __future__ import annotations

from pathlib import Path

import pytest

from contextualize.cache import local_media as local_media_cache
from contextualize.references.audio_transcription import (
    transcribe_audio_file,
    transcribe_media_file,
)
from contextualize.references.file import FileReference


def _configure_local_media_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    cache_root = tmp_path / "local-media-cache"
    monkeypatch.setattr(local_media_cache, "LOCAL_MEDIA_CACHE_ROOT", cache_root)
    monkeypatch.setattr(
        local_media_cache,
        "TRANSCRIPT_CACHE_ROOT",
        cache_root / "transcript",
    )
    return cache_root


def test_transcribe_audio_file_reuses_cache_for_identical_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    first = tmp_path / "first.m4a"
    second = tmp_path / "second.m4a"
    first.write_bytes(b"same-audio")
    second.write_bytes(b"same-audio")

    calls: list[str] = []

    def _transcribe(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float = 600,
    ) -> str:
        assert data == b"same-audio"
        assert content_type == "audio/mp4"
        assert timeout == 600
        calls.append(filename)
        return f"audio transcript {len(calls)}"

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.transcribe_audio_bytes",
        _transcribe,
    )

    assert transcribe_audio_file(first) == "audio transcript 1"
    assert transcribe_audio_file(second) == "audio transcript 1"
    assert calls == ["first.m4a"]


def test_transcribe_audio_file_refresh_cache_bypasses_cached_result(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[int] = []

    def _transcribe(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float = 600,
    ) -> str:
        assert data == b"audio"
        calls.append(len(calls) + 1)
        return f"audio transcript {calls[-1]}"

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.transcribe_audio_bytes",
        _transcribe,
    )

    assert transcribe_audio_file(audio_path) == "audio transcript 1"
    assert transcribe_audio_file(audio_path, refresh_cache=True) == "audio transcript 2"
    assert calls == [1, 2]


def test_transcribe_audio_file_cache_invalidates_when_bytes_change(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio-v1")

    calls: list[bytes] = []

    def _transcribe(
        data: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        timeout: float = 600,
    ) -> str:
        calls.append(data)
        return f"audio transcript {len(calls)}"

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.transcribe_audio_bytes",
        _transcribe,
    )

    assert transcribe_audio_file(audio_path) == "audio transcript 1"
    audio_path.write_bytes(b"audio-v2")
    assert transcribe_audio_file(audio_path) == "audio transcript 2"
    assert calls == [b"audio-v1", b"audio-v2"]


def test_transcribe_media_file_reuses_video_cache(tmp_path: Path, monkeypatch) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    calls: list[str] = []

    def _transcribe_video(path: Path, *, timeout: float) -> str:
        calls.append(str(path))
        assert timeout == 600
        return f"video transcript {len(calls)}"

    monkeypatch.setattr(
        "contextualize.references.audio_transcription._transcribe_video_file",
        _transcribe_video,
    )

    assert transcribe_media_file(video_path) == "video transcript 1"
    assert transcribe_media_file(video_path) == "video transcript 1"
    assert calls == [str(video_path)]


def test_file_reference_passes_media_cache_controls(
    tmp_path: Path, monkeypatch
) -> None:
    media_path = tmp_path / "clip.mp3"
    media_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    def _transcribe(
        path: str | Path,
        *,
        timeout: float = 600,
        use_cache: bool = True,
        refresh_cache: bool | None = None,
    ) -> str:
        captured["path"] = str(path)
        captured["timeout"] = timeout
        captured["use_cache"] = use_cache
        captured["refresh_cache"] = refresh_cache
        return "cached transcript"

    monkeypatch.setattr(
        "contextualize.references.file.transcribe_media_file",
        _transcribe,
    )

    ref = FileReference(
        str(media_path),
        format="raw",
        use_cache=False,
        refresh_cache=True,
    )
    assert ref.read() == "cached transcript"
    assert ref.output == "cached transcript"
    assert captured == {
        "path": str(media_path),
        "timeout": 600,
        "use_cache": False,
        "refresh_cache": True,
    }

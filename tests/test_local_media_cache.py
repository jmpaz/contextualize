from __future__ import annotations

from pathlib import Path

import pytest

from contextualize.cache import local_media as local_media_cache
from contextualize.plugins.api import (
    TranscriptionProvider,
    TranscriptionRequest,
    TranscriptionResult,
)
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


def _provider(name: str, transcribe_fn) -> TranscriptionProvider:
    return TranscriptionProvider(
        name=name,
        priority=200,
        transcribe=transcribe_fn,
        cache_identity=lambda _request: {"provider": name},
    )


def test_transcribe_audio_file_reuses_cache_for_identical_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    first = tmp_path / "first.m4a"
    second = tmp_path / "second.m4a"
    first.write_bytes(b"same-audio")
    second.write_bytes(b"same-audio")

    calls: list[str] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        assert request.data == b"same-audio"
        assert request.content_type == "audio/mp4"
        assert request.timeout is None
        calls.append(request.filename)
        return TranscriptionResult(
            text=f"audio transcript {len(calls)}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert transcribe_audio_file(first) == "audio transcript 1"
    assert transcribe_audio_file(second) == "audio transcript 1"
    assert calls == ["first.m4a"]


def test_transcribe_audio_file_preserves_structured_cache_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[int] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(1)
        return TranscriptionResult(
            text="audio transcript",
            model="vibevoice",
            provider="openai",
            metadata={
                "segments": [
                    {
                        "text": "audio transcript",
                        "speaker": "Speaker 0",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ],
                "speakers": ["Speaker 0"],
            },
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert transcribe_audio_file(audio_path) == "audio transcript"
    assert transcribe_audio_file(audio_path) == "audio transcript"
    assert calls == [1]

    payloads = [
        path
        for path in (tmp_path / "local-media-cache" / "transcript").glob("*.json")
        if not path.name.endswith(".meta.json")
    ]
    assert len(payloads) == 1
    assert '"Speaker 0"' in payloads[0].read_text(encoding="utf-8")


def test_transcribe_audio_file_still_reads_text_only_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        return TranscriptionResult(
            text="legacy transcript",
            model="openai",
            provider="openai",
        )

    def _fail(request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("cached text should be used")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )
    assert transcribe_audio_file(audio_path) == "legacy transcript"

    for payload in (tmp_path / "local-media-cache" / "transcript").glob("*.json"):
        if not payload.name.endswith(".meta.json"):
            payload.unlink()

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _fail),),
    )
    assert transcribe_audio_file(audio_path) == "legacy transcript"


def test_transcribe_audio_file_refresh_cache_bypasses_cached_result(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[int] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        assert request.data == b"audio"
        calls.append(len(calls) + 1)
        return TranscriptionResult(
            text=f"audio transcript {calls[-1]}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert transcribe_audio_file(audio_path) == "audio transcript 1"
    assert transcribe_audio_file(audio_path, refresh_cache=True) == "audio transcript 2"
    assert calls == [1, 2]


def test_transcribe_audio_file_passes_language_override(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    captured: dict[str, object] = {}

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        captured["language"] = request.language
        return TranscriptionResult(
            text="audio transcript",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"language": "ES"}},
        )
        == "audio transcript"
    )
    assert captured["language"] == "es"


def test_transcribe_audio_file_passes_model_override(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    captured: dict[str, object] = {}

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        captured["model"] = request.model
        return TranscriptionResult(
            text="audio transcript",
            model=request.model or "server-default",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"model": "cohere"}},
        )
        == "audio transcript"
    )
    assert captured["model"] == "cohere"


def test_transcribe_audio_file_cache_varies_by_language(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[str | None] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(request.language)
        return TranscriptionResult(
            text=f"audio transcript {request.language}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"language": "es"}},
        )
        == "audio transcript es"
    )
    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"language": "en"}},
        )
        == "audio transcript en"
    )
    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"language": "es"}},
        )
        == "audio transcript es"
    )
    assert calls == ["es", "en"]


def test_transcribe_audio_file_cache_varies_by_model(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[str | None] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(request.model)
        return TranscriptionResult(
            text=f"audio transcript {request.model}",
            model=request.model or "server-default",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"model": "distilwhisper"}},
        )
        == "audio transcript distilwhisper"
    )
    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"model": "cohere"}},
        )
        == "audio transcript cohere"
    )
    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"model": "distilwhisper"}},
        )
        == "audio transcript distilwhisper"
    )
    assert calls == ["distilwhisper", "cohere"]


def test_transcribe_audio_file_cache_invalidates_when_bytes_change(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio-v1")

    calls: list[bytes] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(request.data)
        return TranscriptionResult(
            text=f"audio transcript {len(calls)}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
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

    def _run_ffmpeg(*args, **kwargs):
        output_path = Path(args[0][-1])
        output_path.write_bytes(b"video-audio")
        calls.append(str(output_path))

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        assert request.data == b"video-audio"
        assert request.timeout is None
        return TranscriptionResult(
            text=f"video transcript {len(calls)}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.subprocess.run",
        _run_ffmpeg,
    )
    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert transcribe_media_file(video_path) == "video transcript 1"
    assert transcribe_media_file(video_path) == "video transcript 1"
    assert len(calls) == 1


def test_file_reference_passes_media_cache_controls(
    tmp_path: Path, monkeypatch
) -> None:
    media_path = tmp_path / "clip.mp3"
    media_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    def _transcribe(
        path: str | Path,
        *,
        timeout: float | None = None,
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
        "timeout": None,
        "use_cache": False,
        "refresh_cache": True,
    }

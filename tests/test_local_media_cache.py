from __future__ import annotations

import json
from pathlib import Path

import pytest

from contextualize.cache import local_media as local_media_cache
from contextualize.plugins.api import (
    TranscriptionProvider,
    TranscriptionProviderError,
    TranscriptionRequest,
    TranscriptionResult,
)
from contextualize.run_metadata import flush_run_metadata, reset_run_metadata
from contextualize.runtime import reset_verbose_logging, set_verbose_logging
from contextualize.transcription import (
    transcribe_audio_file,
    transcribe_media_bytes,
    transcribe_media_file,
    transcription_routing_identity,
    transcription_routing_summary,
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


def _provider(
    name: str,
    transcribe_fn,
    *,
    supports_diarization: bool = False,
    default_model: str | None = None,
) -> TranscriptionProvider:
    return TranscriptionProvider(
        name=name,
        priority=200,
        transcribe=transcribe_fn,
        cache_identity=lambda _request: {"provider": name},
        supports_diarization=supports_diarization,
        default_model=default_model,
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
            model="openai",
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


def test_transcribe_audio_file_refresh_cache_uses_stale_cache_on_transient_error(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    def _transcribe(_request: TranscriptionRequest) -> TranscriptionResult:
        return TranscriptionResult(
            text="audio transcript",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )
    assert transcribe_audio_file(audio_path) == "audio transcript"

    calls: list[int] = []

    def _rate_limited(_request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(1)
        raise TranscriptionProviderError(
            'OpenAI-compatible transcription failed: 429 {"error":{"message":"transcription queue is full"}}'
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _rate_limited),),
    )

    assert transcribe_audio_file(audio_path, refresh_cache=True) == "audio transcript"
    assert calls == [1]


def test_transcribe_media_bytes_reuses_audio_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    calls: list[int] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(1)
        assert request.data == b"audio"
        return TranscriptionResult(
            text=f"audio transcript {len(calls)}",
            model="openai",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )

    assert transcribe_media_bytes(b"audio", filename="clip.mp3") == "audio transcript 1"
    assert transcribe_media_bytes(b"audio", filename="renamed.mp3") == "audio transcript 1"
    assert calls == [1]


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


def test_transcribe_audio_file_routes_diarization_to_mistral(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[str] = []

    def _openai(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("diarized requests should not use local OpenAI ASR")

    def _mistral(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append("mistral")
        assert request.diarize is True
        assert request.speaker_count == 2
        return TranscriptionResult(
            text="[Speaker 1] hello",
            model="voxtral-mini-latest",
            provider="mistral",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (
            _provider("openai", _openai),
            _provider("mistral", _mistral, supports_diarization=True),
        ),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
        )
        == "[Speaker 1] hello"
    )
    assert calls == ["mistral"]


def test_transcribe_audio_file_verbose_logs_provider_progress(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        return TranscriptionResult(
            text="audio transcript",
            model=request.model or "server-default",
            provider="openai",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _transcribe),),
    )
    token = set_verbose_logging(True)
    try:
        result = transcribe_audio_file(audio_path)
    finally:
        reset_verbose_logging(token)

    captured = capsys.readouterr()
    assert result == "audio transcript"
    assert captured.out == ""
    assert "[audio-transcription] transcription routing" in captured.err
    assert "[audio-transcription] transcription request start" in captured.err
    assert "[audio-transcription] transcription request finished" in captured.err


def test_transcribe_audio_file_routes_auto_provider_diarization_to_mistral(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[str] = []

    def _mistral(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append("mistral")
        assert request.diarize is True
        return TranscriptionResult(
            text="[Speaker 1] hello",
            model="voxtral-mini-latest",
            provider="mistral",
        )

    def _openai(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("local ASR should not be used")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (
            _provider("openai", _openai),
            _provider("mistral", _mistral, supports_diarization=True),
        ),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={
                "transcribe": {
                    "provider": "auto",
                    "diarize": True,
                }
            },
        )
        == "[Speaker 1] hello"
    )
    assert calls == ["mistral"]


def test_transcribe_audio_file_diarized_auto_requires_capable_provider(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    def _openai(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("provider without diarization should not run")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _openai),),
    )

    with pytest.raises(RuntimeError, match="diarization support"):
        transcribe_audio_file(
            audio_path,
            plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
        )


def test_transcription_routing_identity_uses_diarization_capability(
    monkeypatch,
) -> None:
    def _openai(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("routing identity should not transcribe")

    def _mistral(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("routing identity should not transcribe")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (
            _provider("openai", _openai),
            _provider("mistral", _mistral, supports_diarization=True),
        ),
    )

    identity = transcription_routing_identity(
        filename="media.mp3",
        content_type="audio/mpeg",
        plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
    )

    assert identity["explicit_provider"] is None
    assert identity["providers"] == [{"provider": "mistral"}]
    assert identity["resolved_config"]["diarize"] is True
    assert identity["resolved_config"]["speakers"] == 2


def test_transcription_routing_summary_routes_diarization_to_mistral(
    monkeypatch,
) -> None:
    def _openai(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("routing summary should not transcribe")

    def _mistral(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("routing summary should not transcribe")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (
            _provider("openai", _openai),
            TranscriptionProvider(
                name="mistral",
                priority=100,
                transcribe=_mistral,
                cache_identity=lambda request: {
                    "provider": "mistral",
                    "model": request.model or "voxtral-mini-latest",
                },
                supports_diarization=True,
                default_model="voxtral-mini-latest",
            ),
        ),
    )

    summary = transcription_routing_summary(
        filename="media.mp3",
        content_type="audio/mpeg",
        plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
    )

    assert summary == {
        "provider": "mistral",
        "model": "voxtral-mini-latest",
        "diarize": True,
        "speakers": 2,
        "language": None,
    }


def test_transcription_run_metadata_records_request_and_cache_events(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")
    metadata_path = tmp_path / "run-metadata.json"
    monkeypatch.setenv("CONTEXTUALIZE_RUN_METADATA_PATH", str(metadata_path))
    reset_run_metadata()

    calls: list[int] = []

    def _transcribe(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append(1)
        return TranscriptionResult(
            text="audio transcript",
            model="voxtral-mini-latest",
            provider="mistral",
        )

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("mistral", _transcribe, supports_diarization=True),),
    )

    try:
        assert (
            transcribe_audio_file(
                audio_path,
                plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
            )
            == "audio transcript"
        )
        assert (
            transcribe_audio_file(
                audio_path,
                plugin_overrides={"transcribe": {"diarize": True, "speakers": 2}},
            )
            == "audio transcript"
        )
        flush_run_metadata()
    finally:
        reset_run_metadata()

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert calls == [1]
    assert payload == {
        "version": 1,
        "transcriptions": [
            {
                "provider": "mistral",
                "model": "voxtral-mini-latest",
                "diarize": True,
                "speakers": 2,
                "language": None,
                "filename": "clip.mp3",
                "source": "request",
            },
            {
                "provider": "mistral",
                "model": "voxtral-mini-latest",
                "diarize": True,
                "speakers": 2,
                "language": None,
                "filename": "clip.mp3",
                "source": "transcript-cache",
            },
        ],
    }


def test_transcribe_audio_file_keeps_explicit_openai_diarization_provider(
    tmp_path: Path, monkeypatch
) -> None:
    _configure_local_media_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "clip.mp3"
    audio_path.write_bytes(b"audio")

    calls: list[str] = []

    def _openai(request: TranscriptionRequest) -> TranscriptionResult:
        calls.append("openai")
        assert request.diarize is True
        return TranscriptionResult(
            text="local transcript",
            model="cohere",
            provider="openai",
        )

    def _mistral(_request: TranscriptionRequest) -> TranscriptionResult:
        raise AssertionError("explicit OpenAI provider should win")

    monkeypatch.setattr(
        "contextualize.references.audio_transcription.loaded_transcription_providers",
        lambda: (_provider("openai", _openai), _provider("mistral", _mistral)),
    )

    assert (
        transcribe_audio_file(
            audio_path,
            plugin_overrides={
                "transcribe": {
                    "provider": "openai",
                    "diarize": True,
                },
            },
        )
        == "local transcript"
    )
    assert calls == ["openai"]


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

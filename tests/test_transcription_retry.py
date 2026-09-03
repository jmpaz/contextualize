from __future__ import annotations

import pytest

from contextualize.plugins.api import (
    TranscriptionProvider,
    TranscriptionProviderAuthError,
    TranscriptionProviderError,
    TranscriptionRequest,
    TranscriptionResult,
)
from contextualize.references import audio_transcription
from contextualize.references.audio_transcription import (
    TransientTranscriptionError,
    _transcribe_with_retry,
    transcribe_audio_bytes,
)


def _request() -> TranscriptionRequest:
    return TranscriptionRequest(
        data=b"audio",
        filename="clip.mp3",
        content_type="audio/mpeg",
        timeout=None,
        language=None,
        prompt="",
        bias_terms=(),
        diarize=False,
        speaker_count=None,
    )


def _result(text: str = "ok") -> TranscriptionResult:
    return TranscriptionResult(
        text=text, model="test-model", provider="fake", metadata={}
    )


def _provider(name: str, transcribe) -> TranscriptionProvider:
    return TranscriptionProvider(
        name=name,
        priority=100,
        transcribe=transcribe,
        cache_identity=lambda _request: {"provider": name},
    )


def test_retries_retryable_failures_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(audio_transcription.time, "sleep", slept.append)

    attempts = 0

    def _transcribe(_request: TranscriptionRequest) -> TranscriptionResult:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TranscriptionProviderError(
                "queue is full",
                retryable=True,
                status_code=429,
                retry_after=1.0,
            )
        return _result()

    result = _transcribe_with_retry(
        _provider("fake", _transcribe), _request(), filename="clip.mp3"
    )

    assert attempts == 3
    assert result.text == "ok"
    assert slept == [1.0, 1.0]


def test_exhausted_retries_raise_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_transcription.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("CONTEXTUALIZE_TRANSCRIPTION_MAX_ATTEMPTS", "2")

    attempts = 0

    def _transcribe(_request: TranscriptionRequest) -> TranscriptionResult:
        nonlocal attempts
        attempts += 1
        raise TranscriptionProviderError(
            "queue is full", retryable=True, status_code=429
        )

    with pytest.raises(TransientTranscriptionError):
        _transcribe_with_retry(
            _provider("fake", _transcribe), _request(), filename="clip.mp3"
        )

    assert attempts == 2


def test_non_retryable_failure_is_not_retried() -> None:
    attempts = 0

    def _transcribe(_request: TranscriptionRequest) -> TranscriptionResult:
        nonlocal attempts
        attempts += 1
        raise TranscriptionProviderError("bad request", status_code=400)

    with pytest.raises(TranscriptionProviderError):
        _transcribe_with_retry(
            _provider("fake", _transcribe), _request(), filename="clip.mp3"
        )

    assert attempts == 1


def test_auth_failure_is_not_retried() -> None:
    attempts = 0

    def _transcribe(_request: TranscriptionRequest) -> TranscriptionResult:
        nonlocal attempts
        attempts += 1
        raise TranscriptionProviderAuthError("unauthorized", status_code=401)

    with pytest.raises(TranscriptionProviderAuthError):
        _transcribe_with_retry(
            _provider("fake", _transcribe), _request(), filename="clip.mp3"
        )

    assert attempts == 1


def test_exhausted_provider_falls_through_to_next_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_transcription.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("CONTEXTUALIZE_TRANSCRIPTION_MAX_ATTEMPTS", "1")

    seen: list[str] = []

    def _busy(_request: TranscriptionRequest) -> TranscriptionResult:
        seen.append("busy")
        raise TranscriptionProviderError(
            "queue is full", retryable=True, status_code=429
        )

    def _healthy(_request: TranscriptionRequest) -> TranscriptionResult:
        seen.append("healthy")
        return _result("from-second")

    monkeypatch.setattr(
        audio_transcription,
        "loaded_transcription_providers",
        lambda: [_provider("busy", _busy), _provider("healthy", _healthy)],
    )

    result = transcribe_audio_bytes(
        b"audio", filename="clip.mp3", content_type="audio/mpeg"
    )

    assert seen == ["busy", "healthy"]
    assert result == "from-second"


def test_all_candidates_transient_raises_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_transcription.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("CONTEXTUALIZE_TRANSCRIPTION_MAX_ATTEMPTS", "1")

    def _busy(_request: TranscriptionRequest) -> TranscriptionResult:
        raise TranscriptionProviderError(
            "queue is full", retryable=True, status_code=429
        )

    monkeypatch.setattr(
        audio_transcription,
        "loaded_transcription_providers",
        lambda: [_provider("busy", _busy), _provider("busier", _busy)],
    )

    with pytest.raises(TransientTranscriptionError):
        transcribe_audio_bytes(
            b"audio", filename="clip.mp3", content_type="audio/mpeg"
        )


def test_later_hard_failure_keeps_transient_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_transcription.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("CONTEXTUALIZE_TRANSCRIPTION_MAX_ATTEMPTS", "1")

    def _busy(_request: TranscriptionRequest) -> TranscriptionResult:
        raise TranscriptionProviderError(
            "queue is full", retryable=True, status_code=429
        )

    def _broken(_request: TranscriptionRequest) -> TranscriptionResult:
        raise TranscriptionProviderError("returned no text")

    monkeypatch.setattr(
        audio_transcription,
        "loaded_transcription_providers",
        lambda: [_provider("busy", _busy), _provider("broken", _broken)],
    )

    with pytest.raises(TransientTranscriptionError):
        transcribe_audio_bytes(
            b"audio", filename="clip.mp3", content_type="audio/mpeg"
        )


def test_all_hard_failures_stay_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    def _broken(_request: TranscriptionRequest) -> TranscriptionResult:
        raise TranscriptionProviderError("returned no text")

    monkeypatch.setattr(
        audio_transcription,
        "loaded_transcription_providers",
        lambda: [_provider("broken", _broken), _provider("also-broken", _broken)],
    )

    with pytest.raises(RuntimeError) as excinfo:
        transcribe_audio_bytes(
            b"audio", filename="clip.mp3", content_type="audio/mpeg"
        )
    assert not isinstance(excinfo.value, TransientTranscriptionError)

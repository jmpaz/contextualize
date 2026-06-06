from __future__ import annotations

from contextualize.plugins.api import TranscriptionResult
from contextualize.references.video_context import (
    VideoFrame,
    VideoFrameSettings,
    VideoSegment,
    VideoWord,
    format_video_frame_section,
    render_video_file,
    resolve_video_frame_settings,
    select_frame_timestamps,
    speech_anchor_for_timestamp,
    transcript_segments,
    transcript_speech_units,
    transcript_words,
    video_transcription_overrides,
)


def test_duration_sampling_uses_proportional_defaults() -> None:
    settings = VideoFrameSettings(frame_max=10)

    assert select_frame_timestamps(
        duration_seconds=1.5,
        segments=[],
        settings=settings,
    ) == [0.75]
    assert select_frame_timestamps(
        duration_seconds=20,
        segments=[],
        settings=settings,
    ) == [5.0, 10.0, 15.0]


def test_speech_sampling_downsamples_segment_starts() -> None:
    settings = VideoFrameSettings(frame_mode="speech", frame_max=3)
    segments = [
        VideoSegment(start=float(index), end=float(index) + 0.5, text=str(index))
        for index in range(5)
    ]

    assert select_frame_timestamps(
        duration_seconds=120,
        segments=segments,
        settings=settings,
    ) == [0.0, 2.0, 4.0]


def test_speech_sampling_prefers_sentence_units() -> None:
    settings = VideoFrameSettings(frame_mode="speech", frame_max=10)
    segments = [
        VideoSegment(
            start=0.0,
            end=12.0,
            text="first sentence. second sentence. third sentence.",
        )
    ]
    speech_units = [
        VideoSegment(start=0.0, end=2.0, text="first sentence."),
        VideoSegment(start=3.0, end=5.0, text="second sentence."),
        VideoSegment(start=6.0, end=9.0, text="third sentence."),
    ]

    assert select_frame_timestamps(
        duration_seconds=12,
        segments=segments,
        speech_units=speech_units,
        settings=settings,
    ) == [0.0, 3.0, 6.0]


def test_transcript_segments_and_speech_anchor() -> None:
    result = TranscriptionResult(
        text="hello",
        model="test",
        provider="test",
        metadata={
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "first segment"},
                {
                    "startTime": 3.0,
                    "endTime": 5.0,
                    "text": "second segment",
                },
            ]
        },
    )

    segments = transcript_segments(result)

    assert segments == [
        VideoSegment(start=0.0, end=1.0, text="first segment"),
        VideoSegment(start=3.0, end=5.0, text="second segment"),
    ]
    assert speech_anchor_for_timestamp(4.0, segments) == "second segment"


def test_speech_anchor_prefers_unit_start_over_previous_end() -> None:
    segments = [
        VideoSegment(start=0.0, end=2.0, text="first sentence"),
        VideoSegment(start=2.0, end=4.0, text="second sentence"),
    ]

    assert speech_anchor_for_timestamp(2.0, segments) == "second sentence"


def test_transcript_words_derive_sentence_units() -> None:
    result = TranscriptionResult(
        text="First sentence. Second sentence.",
        model="test",
        provider="test",
        metadata={
            "words": [
                {"word": "First", "start": 0.0, "end": 0.2},
                {"word": "sentence.", "start": 0.2, "end": 0.7},
                {"word": "Second", "start": 1.0, "end": 1.3},
                {"word": "sentence.", "start": 1.3, "end": 1.8},
            ]
        },
    )

    assert transcript_words(result) == [
        VideoWord(start=0.0, end=0.2, text="First"),
        VideoWord(start=0.2, end=0.7, text="sentence."),
        VideoWord(start=1.0, end=1.3, text="Second"),
        VideoWord(start=1.3, end=1.8, text="sentence."),
    ]
    assert transcript_speech_units(result) == [
        VideoSegment(start=0.0, end=0.7, text="First sentence."),
        VideoSegment(start=1.0, end=1.8, text="Second sentence."),
    ]


def test_transcript_speech_units_split_timed_segments() -> None:
    result = TranscriptionResult(
        text="First sentence. Second sentence.",
        model="test",
        provider="test",
        metadata={
            "segments": [
                {
                    "start": 10.0,
                    "end": 14.0,
                    "text": "First sentence. Second sentence.",
                }
            ]
        },
    )

    assert transcript_speech_units(result) == [
        VideoSegment(start=10.0, end=12.0, text="First sentence."),
        VideoSegment(start=12.0, end=14.0, text="Second sentence."),
    ]


def test_format_video_frame_section_renders_timestamped_image_tags() -> None:
    rendered = format_video_frame_section(
        [
            VideoFrame(
                index=1,
                timestamp=12.4,
                speech='say "hello"',
                description="A person enters the room.",
            )
        ]
    )

    assert rendered == (
        "## Video Frames\n\n"
        '<image frame="1" timestamp="00:12.400" speech="say &quot;hello&quot;">\n'
        "A person enters the room.\n"
        "</image>"
    )


def test_video_settings_accept_env_and_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_VIDEO_FRAME_MODE", "speech")
    monkeypatch.setenv("CONTEXTUALIZE_VIDEO_FRAME_MAX", "8")

    settings = resolve_video_frame_settings(
        {"video": {"frame-max": 3, "frame-descriptions": False}}
    )

    assert settings == VideoFrameSettings(
        frames=True,
        frame_mode="speech",
        frame_descriptions=False,
        frame_max=3,
    )


def test_video_transcription_overrides_request_segments() -> None:
    overrides = video_transcription_overrides(
        {"transcribe": {"timestamp_granularities": ("word",)}}
    )

    assert overrides == {
        "transcribe": {"timestamp_granularities": ["word", "segment"]}
    }


def test_video_transcription_overrides_request_segments_for_speech_mode() -> None:
    overrides = video_transcription_overrides(
        {"video": {"frame-mode": "speech"}},
    )

    assert overrides == {
        "video": {"frame-mode": "speech"},
        "transcribe": {"timestamp_granularities": ["segment"]},
    }


def test_render_video_file_without_frames_reports_unavailable_transcript(
    tmp_path,
    monkeypatch,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    def _fail_transcription(*_args, **_kwargs):
        raise RuntimeError("missing provider")

    monkeypatch.setattr(
        "contextualize.references.video_context.transcribe_media_file_result",
        _fail_transcription,
    )
    monkeypatch.setattr(
        "contextualize.references.video_context.probe_video_duration",
        lambda _path: 12.0,
    )

    rendered = render_video_file(
        video_path,
        plugin_overrides={"video": {"frames": False}},
    )

    assert "## Video" in rendered
    assert "duration_seconds: 12.000" in rendered
    assert "[transcript unavailable: missing provider]" in rendered

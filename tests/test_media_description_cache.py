from pathlib import Path

from contextualize.render import markitdown
from contextualize.runtime import reset_refresh_videos, set_refresh_videos


def test_video_description_cache_reuses_content_and_descriptor_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-content")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(markitdown, "_video_prompt", lambda _path: "describe")
    calls = 0

    def _describe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return "# Video (auto-generated):\n\ndescription"

    monkeypatch.setattr(markitdown, "_llm_video_markdown", _describe)

    first = markitdown.convert_path_to_markdown(video)
    second = markitdown.convert_path_to_markdown(video)

    assert first.markdown == second.markdown
    assert calls == 1


def test_video_description_cache_varies_by_prompt_and_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-content")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_video_prompt", lambda _path: "describe")
    calls = 0

    def _describe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"# Video (auto-generated):\n\ndescription {calls}"

    monkeypatch.setattr(markitdown, "_llm_video_markdown", _describe)
    monkeypatch.setenv("OPENAI_MODEL", "model-a")
    markitdown.convert_path_to_markdown(video, prompt_append="one")
    markitdown.convert_path_to_markdown(video, prompt_append="two")
    monkeypatch.setenv("OPENAI_MODEL", "model-b")
    markitdown.convert_path_to_markdown(video, prompt_append="two")

    assert calls == 3


def test_failed_video_description_does_not_poison_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-content")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_video_prompt", lambda _path: "describe")
    calls = 0

    def _describe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise markitdown.MarkItDownConversionError("transient")
        return "# Video (auto-generated):\n\nrecovered"

    monkeypatch.setattr(markitdown, "_llm_video_markdown", _describe)
    monkeypatch.setattr(
        markitdown,
        "_format_auto_generated_video_fallback",
        lambda _path: "fallback",
    )

    assert markitdown.convert_path_to_markdown(video).markdown == "fallback"
    assert "recovered" in markitdown.convert_path_to_markdown(video).markdown
    assert calls == 2


def test_video_description_refresh_bypasses_persistent_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video-content")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_video_prompt", lambda _path: "describe")
    calls = 0

    def _describe(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"# Video (auto-generated):\n\ndescription {calls}"

    monkeypatch.setattr(markitdown, "_llm_video_markdown", _describe)
    markitdown.convert_path_to_markdown(video)
    token = set_refresh_videos(True)
    try:
        refreshed = markitdown.convert_path_to_markdown(video)
    finally:
        reset_refresh_videos(token)

    assert "description 2" in refreshed.markdown
    assert calls == 2

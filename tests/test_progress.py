from __future__ import annotations

import re

from contextualize.progress import (
    set_live_progress,
    progress_summary_lines,
    record_progress,
    reset_progress,
    reset_progress_context,
    log_progress,
    set_progress_context,
    write_progress_log,
)
from contextualize.runtime import (
    get_payload_media_jobs,
    get_payload_spec_jobs,
    reset_payload_media_jobs,
    reset_payload_spec_jobs,
    reset_verbose_logging,
    set_payload_media_jobs,
    set_payload_spec_jobs,
    set_verbose_logging,
)


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", value)


def test_progress_summary_groups_by_context_provider_operation() -> None:
    reset_progress()
    token = set_progress_context("gamedev")
    try:
        record_progress("arena", "channel", "cache_hit")
        record_progress("arena", "channel", "cache_hit")
        record_progress("arena", "media", "processed")
    finally:
        reset_progress_context(token)

    lines = progress_summary_lines()

    assert "Progress summary:" in lines
    assert "  context gamedev:" in lines
    assert "    arena channel: cache_hit=2" in lines
    assert "    arena media: processed=1" in lines
    reset_progress()


def test_payload_job_overrides_take_precedence_over_env(monkeypatch) -> None:
    monkeypatch.setenv("CONTEXTUALIZE_PAYLOAD_SPEC_JOBS", "3")
    monkeypatch.setenv("CONTEXTUALIZE_PAYLOAD_MEDIA_JOBS", "2")
    spec_token = set_payload_spec_jobs(9)
    media_token = set_payload_media_jobs(6)
    try:
        assert get_payload_spec_jobs() == 9
        assert get_payload_media_jobs() == 6
    finally:
        reset_payload_spec_jobs(spec_token)
        reset_payload_media_jobs(media_token)

    assert get_payload_spec_jobs() == 3
    assert get_payload_media_jobs() == 2


def test_live_progress_uses_rich_task_rows(capsys) -> None:
    reset_progress()
    token = set_verbose_logging(True)
    try:
        set_live_progress(True, force_terminal=True, transient=False)
        log_progress("hydrate", "component", "total", count=2)
        log_progress("hydrate", "component", "start", target="arena")
        log_progress("arena", "channel", "cache_hit", target="Cached Channel")
        log_progress(
            "plugins",
            "embedded-list",
            "start",
            detail="depth=1/2 targets=3 jobs=2",
        )
        log_progress("plugins", "embedded-list", "done", detail="depth=1/2")
        write_progress_log("[audio-transcription] request start")
        log_progress("hydrate", "component", "done", target="arena")
    finally:
        set_live_progress(False)
        reset_verbose_logging(token)

    captured = _strip_ansi(capsys.readouterr().err)
    assert "[progress]" not in captured
    assert "progress:" not in captured
    assert "components" in captured
    assert "arena channel" in captured
    assert "plugins embedded list" in captured
    assert "3/3" in captured
    assert "[audio-transcription] request start" in captured
    assert "1/2" in captured
    assert "hit=1" in captured
    assert "  hydrate component: total=2 done=1" in progress_summary_lines()
    assert "  arena channel: cache_hit=1" in progress_summary_lines()
    reset_progress()

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from contextualize.render import codex, markitdown


def test_default_media_models_are_provider_specific(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "openrouter-only")
    monkeypatch.delenv("CONTEXTUALIZE_CODEX_APP_SERVER_MODEL", raising=False)

    assert markitdown._DEFAULT_OPENROUTER_MODEL == "google/gemini-3.1-flash-lite"
    assert markitdown._resolve_app_server_request_model("") == "gpt-5.6-luna"
    assert markitdown._DEFAULT_CODEX_APP_SERVER_EFFORT == "medium"

    monkeypatch.setenv("CONTEXTUALIZE_CODEX_APP_SERVER_MODEL", "app-server-only")
    assert markitdown._resolve_app_server_request_model("openrouter-only") == "app-server-only"


def test_describe_image_starts_ephemeral_app_server_thread(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    requests: list[tuple[str, dict[str, Any] | None]] = []

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self._closed = False
            self._events = [
                {
                    "method": "item/completed",
                    "params": {
                        "item": {
                            "type": "agentMessage",
                            "text": "A small test image.",
                        }
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "turn": {
                            "id": "turn-1",
                            "status": "completed",
                        }
                    },
                },
            ]

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            self._closed = True
            return None

        def initialize(self) -> None:
            return None

        def request(
            self,
            method: str,
            *,
            params: dict[str, Any] | None = None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        def next_event(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
            if self._closed:
                raise codex.CodexAppServerError(
                    "client closed before event collection"
                )
            return self._events.pop(0)

    monkeypatch.setattr(codex, "_CodexAppServerClient", _FakeClient)

    result = codex.describe_image_with_codex_app_server(
        image_path,
        prompt="Write detailed alt text for this image.",
        command="codex app-server --listen stdio://",
    )

    assert result.text == "A small test image."
    assert requests[0] == (
        "thread/start",
        {"cwd": str(Path.cwd()), "ephemeral": True},
    )


def test_shared_app_server_reuses_one_process_for_multiple_images(
    tmp_path: Path, monkeypatch
) -> None:
    codex.close_shared_codex_app_server_sessions()
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    instances: list[object] = []
    requests: list[tuple[str, dict[str, Any] | None]] = []

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            instances.append(self)
            self._turn_count = 0
            self._events: list[dict[str, Any]] = []

        def start(self) -> None:
            return None

        def initialize(self) -> None:
            requests.append(("initialize", None))

        def close(self) -> None:
            return None

        def request(
            self,
            method: str,
            *,
            params: dict[str, Any] | None = None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            requests.append((method, params))
            if method == "model/list":
                return {}
            if method == "thread/start":
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                self._turn_count += 1
                turn_id = f"turn-{self._turn_count}"
                self._events.extend(
                    [
                        {
                            "method": "item/completed",
                            "params": {
                                "item": {
                                    "type": "agentMessage",
                                    "text": f"Image {self._turn_count}",
                                }
                            },
                        },
                        {
                            "method": "turn/completed",
                            "params": {
                                "turn": {"id": turn_id, "status": "completed"}
                            },
                        },
                    ]
                )
                return {"turn": {"id": turn_id}}
            return {}

        def next_event(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
            return self._events.pop(0)

    monkeypatch.setattr(codex, "_CodexAppServerClient", _FakeClient)

    try:
        assert codex.is_shared_codex_app_server_live(
            "codex app-server --listen stdio://"
        ) == (
            True,
            None,
        )
        first = codex.describe_image_with_shared_codex_app_server(
            image_path,
            prompt="describe",
            command="codex app-server --listen stdio://",
        )
        second = codex.describe_image_with_shared_codex_app_server(
            image_path,
            prompt="describe",
            command="codex app-server --listen stdio://",
        )
    finally:
        codex.close_shared_codex_app_server_sessions()

    assert [first.text, second.text] == ["Image 1", "Image 2"]
    assert len(instances) == 1
    assert [method for method, _params in requests].count("initialize") == 1
    assert [method for method, _params in requests].count("thread/start") == 2


def test_transcribe_image_batches_reuses_one_app_server_thread(
    tmp_path: Path, monkeypatch
) -> None:
    image_1 = tmp_path / "page-1.png"
    image_2 = tmp_path / "page-2.png"
    image_3 = tmp_path / "page-3.png"
    for image_path in (image_1, image_2, image_3):
        image_path.write_bytes(b"png")
    requests: list[tuple[str, dict[str, Any] | None]] = []

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self._turn_count = 0
            self._events: list[dict[str, Any]] = []

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def initialize(self) -> None:
            return None

        def request(
            self,
            method: str,
            *,
            params: dict[str, Any] | None = None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            requests.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                self._turn_count += 1
                turn_id = f"turn-{self._turn_count}"
                self._events.extend(
                    [
                        {
                            "method": "item/completed",
                            "params": {
                                "item": {
                                    "type": "agentMessage",
                                    "text": f"Batch {self._turn_count}",
                                }
                            },
                        },
                        {
                            "method": "turn/completed",
                            "params": {
                                "turn": {"id": turn_id, "status": "completed"}
                            },
                        },
                    ]
                )
                return {"turn": {"id": turn_id}}
            return {}

        def next_event(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
            return self._events.pop(0)

    monkeypatch.setattr(codex, "_CodexAppServerClient", _FakeClient)

    results = codex.transcribe_image_batches_with_codex_app_server(
        [[image_1, image_2], [image_3]],
        prompts=["pages 1-2", "page 3"],
        command="codex app-server --listen stdio://",
        model="gpt-5.4",
        effort="medium",
        timeout_seconds=42,
    )

    assert [result.text for result in results] == ["Batch 1", "Batch 2"]
    assert [method for method, _params in requests].count("thread/start") == 1
    turn_requests = [params for method, params in requests if method == "turn/start"]
    assert len(turn_requests) == 2
    assert turn_requests[0]["threadId"] == "thread-1"
    assert turn_requests[0]["input"] == [
        {"type": "text", "text": "pages 1-2"},
        {"type": "localImage", "path": str(image_1)},
        {"type": "localImage", "path": str(image_2)},
    ]
    assert turn_requests[1]["input"] == [
        {"type": "text", "text": "page 3"},
        {"type": "localImage", "path": str(image_3)},
    ]


def test_scanned_pdf_falls_back_to_app_server_page_ocr(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")
    written_cache: list[tuple[str, str]] = []
    rendered: list[Path] = []

    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown_with_normalization",
        lambda _path: ("", None),
    )
    monkeypatch.setattr(
        markitdown,
        "_image_context",
        lambda: (
            False,
            "https://openrouter.ai/api/v1",
            "gpt-5.4",
            "",
            None,
            "auto",
            "codex app-server --listen stdio://",
        ),
    )
    monkeypatch.setattr(
        markitdown,
        "_resolve_image_provider",
        lambda *_args, **_kwargs: markitdown._ImageProviderSelection(
            requested_mode="auto",
            effective_provider="app-server",
            app_server_live=True,
            app_server_error=None,
        ),
    )
    monkeypatch.setattr(markitdown, "_file_md5", lambda _path: "pdf-md5")
    monkeypatch.setattr(
        markitdown,
        "_markdown_cache_lookup",
        lambda _payload: ("key", None),
    )

    def fake_render(_path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
        page_1 = output_dir / "page-1.png"
        page_2 = output_dir / "page-2.png"
        page_1.write_bytes(b"page 1")
        page_2.write_bytes(b"page 2")
        rendered.extend([page_1, page_2])
        return [page_1, page_2]

    monkeypatch.setattr(markitdown, "_render_pdf_pages_to_png", fake_render)
    seen_batch: dict[str, Any] = {}

    def fake_batch_texts(page_paths: list[Path], **kwargs: Any) -> list[str]:
        seen_batch["page_names"] = [path.name for path in page_paths]
        seen_batch["batch_size"] = kwargs["batch_size"]
        return ["First page text.\n\nSecond page text."]

    monkeypatch.setattr(
        markitdown,
        "_app_server_pdf_batch_texts_from_pages",
        fake_batch_texts,
    )
    monkeypatch.setattr(
        markitdown,
        "_write_cache_entry",
        lambda key, *, payload, markdown, title: written_cache.append(
            (key, markdown)
        ),
    )

    result = markitdown.convert_path_to_markdown(pdf_path)

    assert result.markdown == "First page text.\n\nSecond page text."
    assert [path.name for path in rendered] == ["page-1.png", "page-2.png"]
    assert seen_batch == {
        "page_names": ["page-1.png", "page-2.png"],
        "batch_size": 5,
    }
    assert written_cache == [("key", result.markdown)]
    prompt = markitdown._pdf_batch_prompt([Path("page-1.png")], total_pages=2)
    assert "Reflow wrapped lines into paragraphs" in prompt
    assert "Ignore running headers, footers, page numbers" in prompt


def test_scanned_pdf_requires_image_ocr_provider(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")

    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown_with_normalization",
        lambda _path: ("\f", None),
    )
    monkeypatch.setattr(
        markitdown,
        "_image_context",
        lambda: (
            False,
            "https://openrouter.ai/api/v1",
            "gpt-5.4",
            "",
            None,
            "auto",
            "codex app-server --listen stdio://",
        ),
    )
    monkeypatch.setattr(
        markitdown,
        "_resolve_image_provider",
        lambda *_args, **_kwargs: markitdown._ImageProviderSelection(
            requested_mode="auto",
            effective_provider="openrouter",
            app_server_live=False,
            app_server_error="not running",
        ),
    )

    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="no embedded text",
    ):
        markitdown.convert_path_to_markdown(pdf_path)


def test_scanned_pdf_app_server_mode_fails_closed_when_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")

    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown_with_normalization",
        lambda _path: ("", None),
    )
    monkeypatch.setattr(
        markitdown,
        "_image_context",
        lambda: (
            True,
            "https://openrouter.ai/api/v1",
            "gpt-5.4",
            "",
            None,
            "app-server",
            "codex app-server --listen stdio://",
        ),
    )
    monkeypatch.setattr(
        markitdown,
        "_resolve_image_provider",
        lambda *_args, **_kwargs: markitdown._ImageProviderSelection(
            requested_mode="app-server",
            effective_provider="openrouter",
            app_server_live=False,
            app_server_error="not running",
        ),
    )
    monkeypatch.setattr(
        markitdown,
        "_render_pdf_pages_to_png",
        lambda *_args, **_kwargs: pytest.fail("should not render pages"),
    )
    monkeypatch.setattr(
        markitdown,
        "_build_llm_config",
        lambda: pytest.fail("OpenRouter client must not be built"),
    )
    monkeypatch.setattr(
        markitdown,
        "_openrouter_image_text_from_path",
        lambda *_args, **_kwargs: pytest.fail("OpenRouter image OCR must not run"),
    )

    with pytest.raises(markitdown.MarkItDownConversionError, match="configured"):
        markitdown.convert_path_to_markdown(pdf_path)


def test_scanned_pdf_wraps_app_server_batch_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    page_1 = tmp_path / "page-1.png"
    page_2 = tmp_path / "page-2.png"
    page_1.write_bytes(b"page 1")
    page_2.write_bytes(b"page 2")
    captured: dict[str, Any] = {}

    def fail_batches(image_batches: list[list[Path]], **kwargs: Any) -> list[Any]:
        captured["image_batches"] = image_batches
        captured["timeout_seconds"] = kwargs["timeout_seconds"]
        raise codex.CodexAppServerError(
            "Timed out waiting for app-server event response"
        )

    monkeypatch.setattr(
        codex,
        "transcribe_image_batches_with_shared_codex_app_server",
        fail_batches,
    )

    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="rendered page batch pages 1-2",
    ):
        markitdown._app_server_pdf_batch_texts_from_pages(
            [page_1, page_2],
            model="gpt-5.4",
            app_server_command="codex app-server --listen stdio://",
            per_page_timeout_seconds=42,
            batch_size=2,
        )

    assert captured["image_batches"] == [[page_1, page_2]]
    assert captured["timeout_seconds"] == 84


def test_app_server_turn_usage_is_reported_with_the_effective_model(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            self._events = [
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "tokenUsage": {
                            "last": {
                                "totalTokens": 16148,
                                "inputTokens": 16117,
                                "cachedInputTokens": 9984,
                                "outputTokens": 31,
                                "reasoningOutputTokens": 0,
                                "unknownField": "ignored",
                            }
                        }
                    },
                },
                {
                    "method": "model/rerouted",
                    "params": {
                        "fromModel": "gpt-5.6-luna",
                        "toModel": "gpt-5.6-luna-mini",
                        "reason": "capacity",
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "item": {"type": "agentMessage", "text": "A small test image."}
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {"turn": {"id": "turn-1", "status": "completed"}},
                },
            ]

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def initialize(self) -> None:
            return None

        def request(
            self,
            method: str,
            *,
            params: dict[str, Any] | None = None,
            timeout_seconds: float | None = None,
        ) -> dict[str, Any]:
            if method == "thread/start":
                return {"thread": {"id": "thread-1"}}
            if method == "turn/start":
                return {"turn": {"id": "turn-1"}}
            return {}

        def next_event(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
            return self._events.pop(0)

    monkeypatch.setattr(codex, "_CodexAppServerClient", _FakeClient)

    result = codex.describe_image_with_codex_app_server(
        image_path,
        prompt="describe",
        command="codex app-server --listen stdio://",
    )

    assert result.usage == {
        "totalTokens": 16148,
        "inputTokens": 16117,
        "cachedInputTokens": 9984,
        "outputTokens": 31,
        "reasoningOutputTokens": 0,
    }
    assert result.rerouted_to_model == "gpt-5.6-luna-mini"


def test_app_server_image_description_records_provider_model_and_usage(
    tmp_path: Path, monkeypatch
) -> None:
    from contextualize.progress import progress_events, reset_progress

    image_path = tmp_path / "shot.png"
    image_path.write_bytes(b"png")
    monkeypatch.setattr(
        markitdown,
        "_resolve_app_server_request_model",
        lambda _model: "gpt-5.6-luna",
    )
    monkeypatch.setattr(
        codex,
        "describe_image_with_shared_codex_app_server",
        lambda *_args, **_kwargs: codex.CodexImageDescriptionResult(
            text="A small test image.",
            requested_model="gpt-5.6-luna",
            rerouted_from_model="gpt-5.6-luna",
            rerouted_to_model="gpt-5.6-luna-mini",
            reroute_reason="capacity",
            usage={"totalTokens": 16148, "outputTokens": 31},
        ),
    )

    reset_progress()
    text = markitdown._app_server_image_text_from_path(
        image_path,
        prompt="describe",
        model="google/gemini-3.1-flash-lite",
        app_server_command="codex app-server --listen stdio://",
    )

    assert text == "A small test image."
    recorded = [
        event
        for event in progress_events()
        if event.operation == "image-description"
    ]
    assert len(recorded) == 1
    assert recorded[0].provider == "codex-app-server"
    assert recorded[0].count == 16148
    assert json.loads(recorded[0].detail or "{}") == {
        "model": "gpt-5.6-luna-mini",
        "usage": {"totalTokens": 16148, "outputTokens": 31},
    }
    reset_progress()


def test_openrouter_image_description_records_usage_from_the_response() -> None:
    from contextualize.progress import progress_events, reset_progress

    class _Usage:
        total_tokens = 812
        prompt_tokens = 780
        completion_tokens = 32
        prompt_tokens_details = types.SimpleNamespace(cached_tokens=256)
        completion_tokens_details = types.SimpleNamespace(reasoning_tokens=0)

    class _Completions:
        def create(self, **_kwargs: Any) -> Any:
            return types.SimpleNamespace(usage=_Usage())

    proxy = markitdown._OpenRouterCompletionsProxy(
        _Completions(),
        provider="openrouter",
        add_openrouter_defaults=False,
    )

    reset_progress()
    proxy.create(
        model="google/gemini-3.1-flash-lite",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                ],
            }
        ],
    )
    proxy.create(
        model="google/gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "plain text only"}],
    )

    recorded = [
        event
        for event in progress_events()
        if event.operation == "image-description"
    ]
    assert len(recorded) == 1
    assert recorded[0].provider == "openrouter"
    assert recorded[0].count == 812
    assert json.loads(recorded[0].detail or "{}") == {
        "model": "google/gemini-3.1-flash-lite",
        "usage": {
            "totalTokens": 812,
            "inputTokens": 780,
            "cachedInputTokens": 256,
            "outputTokens": 32,
            "reasoningOutputTokens": 0,
        },
    }
    reset_progress()


def test_openrouter_image_description_records_the_model_without_usage() -> None:
    from contextualize.progress import progress_events, reset_progress

    class _Completions:
        def create(self, **_kwargs: Any) -> Any:
            return types.SimpleNamespace()

    proxy = markitdown._OpenRouterCompletionsProxy(
        _Completions(),
        provider="openrouter",
        add_openrouter_defaults=False,
    )

    reset_progress()
    proxy.create(
        model="google/gemini-3.1-flash-lite",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}
                ],
            }
        ],
    )

    recorded = [
        event
        for event in progress_events()
        if event.operation == "image-description"
    ]
    assert len(recorded) == 1
    assert recorded[0].count is None
    assert json.loads(recorded[0].detail or "{}") == {
        "model": "google/gemini-3.1-flash-lite"
    }
    reset_progress()


def _image_context_for(mode: str, *, llm_enabled: bool = True):
    return lambda: (
        llm_enabled,
        "https://openrouter.ai/api/v1",
        "gpt-5.4",
        "describe",
        None,
        mode,
        "codex app-server --listen stdio://",
    )


def _selection(mode: str, *, live: bool, error: str | None = None):
    return markitdown._ImageProviderSelection(
        requested_mode=mode,
        effective_provider=(
            "app-server" if live or mode == "app-server" else "openrouter"
        ),
        app_server_live=live,
        app_server_error=error,
    )


def _pin_selection(monkeypatch, selection) -> None:
    monkeypatch.setattr(
        markitdown, "_resolve_image_provider", lambda *_args, **_kwargs: selection
    )


def _forbid_openrouter(monkeypatch) -> None:
    for name in (
        "_build_llm_config",
        "_get_converter",
        "_convert_markitdown",
        "_convert_markitdown_with_normalization",
        "_openrouter_image_text_from_path",
    ):
        monkeypatch.setattr(
            markitdown,
            name,
            lambda *_args, _name=name, **_kwargs: pytest.fail(
                f"OpenRouter path must not run: {_name}"
            ),
        )


def _failed_description_events() -> list[Any]:
    from contextualize.progress import progress_events

    return [
        event
        for event in progress_events()
        if event.outcome == "failed" and event.provider == "codex-app-server"
    ]


class _FakeImageResponse:
    def __init__(self, url: str, content: bytes) -> None:
        self.url = url
        self.content = content
        self.headers = {"Content-Type": "image/png"}


def test_pinned_app_server_image_turn_failure_never_reaches_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    from contextualize.progress import reset_progress

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("app-server"))
    _pin_selection(monkeypatch, _selection("app-server", live=True))
    _forbid_openrouter(monkeypatch)

    def fail_turn(*_args: Any, **_kwargs: Any) -> str:
        raise markitdown.MarkItDownConversionError("turn failed")

    monkeypatch.setattr(
        markitdown, "_app_server_image_markdown_from_path", fail_turn
    )

    reset_progress()
    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="configured image provider and failed",
    ):
        markitdown.convert_path_to_markdown(image_path)

    recorded = _failed_description_events()
    assert [(event.operation, event.target) for event in recorded] == [
        ("image-description", "image.png")
    ]
    assert recorded[0].detail == "turn failed"
    reset_progress()


def test_pinned_app_server_image_probe_failure_never_reaches_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    from contextualize.progress import reset_progress

    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("app-server"))
    _pin_selection(
        monkeypatch, _selection("app-server", live=False, error="not running")
    )
    _forbid_openrouter(monkeypatch)
    monkeypatch.setattr(
        markitdown,
        "_app_server_image_markdown_from_path",
        lambda *_args, **_kwargs: pytest.fail("app-server must not be called"),
    )

    reset_progress()
    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="configured image provider and is unavailable",
    ):
        markitdown.convert_path_to_markdown(image_path)

    assert [event.detail for event in _failed_description_events()] == ["not running"]
    reset_progress()


def test_auto_image_turn_failure_still_falls_back_to_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"png")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("auto"))
    _pin_selection(monkeypatch, _selection("auto", live=True))

    def fail_turn(*_args: Any, **_kwargs: Any) -> str:
        raise markitdown.MarkItDownConversionError("turn failed")

    monkeypatch.setattr(
        markitdown, "_app_server_image_markdown_from_path", fail_turn
    )
    monkeypatch.setattr(markitdown, "_image_text_tools_available", lambda: True)
    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown_with_normalization",
        lambda _path: ("# Description:\nopenrouter description", None),
    )

    result = markitdown.convert_path_to_markdown(image_path)

    assert "openrouter description" in result.markdown


def test_pinned_app_server_response_turn_failure_never_reaches_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    from contextualize.progress import reset_progress

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("app-server"))
    _pin_selection(monkeypatch, _selection("app-server", live=True))
    _forbid_openrouter(monkeypatch)

    def fail_turn(*_args: Any, **_kwargs: Any) -> str:
        raise markitdown.MarkItDownConversionError("turn failed")

    monkeypatch.setattr(
        markitdown, "_app_server_image_markdown_from_bytes", fail_turn
    )

    reset_progress()
    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="configured image provider and failed",
    ):
        markitdown.convert_response_to_markdown(
            _FakeImageResponse("https://example.test/image.png", b"png")
        )

    assert [event.target for event in _failed_description_events()] == [
        "https://example.test/image.png"
    ]
    reset_progress()


def test_pinned_app_server_response_probe_failure_never_reaches_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    from contextualize.progress import reset_progress

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("app-server"))
    _pin_selection(
        monkeypatch, _selection("app-server", live=False, error="not running")
    )
    _forbid_openrouter(monkeypatch)
    monkeypatch.setattr(
        markitdown,
        "_app_server_image_markdown_from_bytes",
        lambda *_args, **_kwargs: pytest.fail("app-server must not be called"),
    )

    reset_progress()
    with pytest.raises(
        markitdown.MarkItDownConversionError,
        match="configured image provider and is unavailable",
    ):
        markitdown.convert_response_to_markdown(
            _FakeImageResponse("https://example.test/image.png", b"png")
        )

    assert [event.detail for event in _failed_description_events()] == ["not running"]
    reset_progress()


def test_auto_response_turn_failure_still_falls_back_to_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("auto"))
    _pin_selection(monkeypatch, _selection("auto", live=True))

    def fail_turn(*_args: Any, **_kwargs: Any) -> str:
        raise markitdown.MarkItDownConversionError("turn failed")

    monkeypatch.setattr(
        markitdown, "_app_server_image_markdown_from_bytes", fail_turn
    )
    monkeypatch.setattr(markitdown, "_image_text_tools_available", lambda: True)
    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown",
        lambda _source, **_kwargs: ("# Description:\nopenrouter description", None),
    )

    result = markitdown.convert_response_to_markdown(
        _FakeImageResponse("https://example.test/image.png", b"png")
    )

    assert "openrouter description" in result.markdown


def test_pinned_app_server_pdf_ocr_failure_never_reaches_openrouter(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        markitdown,
        "_convert_markitdown_with_normalization",
        lambda _path: ("", None),
    )
    monkeypatch.setattr(markitdown, "_image_context", _image_context_for("app-server"))
    _pin_selection(monkeypatch, _selection("app-server", live=True))
    monkeypatch.setattr(
        markitdown,
        "_build_llm_config",
        lambda: pytest.fail("OpenRouter client must not be built"),
    )
    monkeypatch.setattr(
        markitdown,
        "_openrouter_image_text_from_path",
        lambda *_args, **_kwargs: pytest.fail("OpenRouter image OCR must not run"),
    )

    def fake_render(_path: Path, output_dir: Path, *, dpi: int) -> list[Path]:
        page = output_dir / "page-1.png"
        page.write_bytes(b"page 1")
        return [page]

    monkeypatch.setattr(markitdown, "_render_pdf_pages_to_png", fake_render)

    def fail_batches(*_args: Any, **_kwargs: Any) -> list[str]:
        raise markitdown.MarkItDownConversionError("batch failed")

    monkeypatch.setattr(
        markitdown, "_app_server_pdf_batch_texts_from_pages", fail_batches
    )

    with pytest.raises(markitdown.MarkItDownConversionError, match="batch failed"):
        markitdown.convert_path_to_markdown(pdf_path)

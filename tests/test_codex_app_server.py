from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from contextualize.render import codex, markitdown


def test_default_codex_app_server_image_model_is_gpt_5_4(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert markitdown._resolve_app_server_request_model("") == "gpt-5.4"


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
        "transcribe_image_batches_with_codex_app_server",
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

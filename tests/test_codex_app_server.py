from __future__ import annotations

from pathlib import Path
from typing import Any

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

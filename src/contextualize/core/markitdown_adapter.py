from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import tempfile
from typing import Any


class MarkItDownConversionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MarkItDownResult:
    markdown: str
    title: str | None


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except Exception:
        return

    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)


def _data_home() -> Path:
    base = (os.getenv("XDG_DATA_HOME") or "").strip()
    if base:
        return Path(base)
    return Path.home() / ".local" / "share"


def _llm_cache_dir() -> Path:
    return _data_home() / "contextualize" / "cache" / "llm"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@lru_cache(maxsize=1)
def _markitdown_version() -> str | None:
    try:
        from importlib.metadata import version
    except Exception:
        return None

    try:
        return version("markitdown")
    except Exception:
        return None


_DESCRIPTION_HEADING_RE = re.compile(r"(?m)^# Description:?\s*$")


def _postprocess_image_markdown(markdown: str) -> str:
    return _DESCRIPTION_HEADING_RE.sub(
        "# Description (auto-generated)", markdown, count=1
    )


def _cache_entries_dir() -> Path:
    return _llm_cache_dir() / "v1"


def _cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _cache_entry_path(key: str) -> Path:
    return _cache_entries_dir() / key[:2] / f"{key}.json"


def _read_cache_entry(key: str) -> dict[str, Any] | None:
    path = _cache_entry_path(key)
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache_entry(
    key: str, *, payload: dict[str, Any] | None, markdown: str, title: str | None
) -> None:
    if not isinstance(markdown, str):
        return
    entry_path = _cache_entry_path(key)
    try:
        entry_path.parent.mkdir(parents=True, exist_ok=True)
        cache_entry = {"v": 1, "payload": payload, "markdown": markdown, "title": title}
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=entry_path.parent,
            prefix=f"{key}.",
            suffix=".tmp",
        ) as f:
            f.write(_stable_json(cache_entry))
            tmp_name = f.name
        Path(tmp_name).replace(entry_path)
    except OSError:
        try:
            if "tmp_name" in locals():
                Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _build_llm_config() -> tuple[object | None, str | None]:
    _load_dotenv_once()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or (
        os.getenv("OPENROUTER_API_KEY") or ""
    ).strip()
    if not api_key:
        return None, None

    base_url = (
        os.getenv("OPENAI_BASE_URL") or ""
    ).strip() or "https://openrouter.ai/api/v1"
    model = (os.getenv("OPENAI_MODEL") or "").strip() or "google/gemini-2.5-flash"

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


@lru_cache(maxsize=1)
def _image_text_tools_available() -> bool:
    _load_dotenv_once()
    has_openai_key = bool(
        (os.getenv("OPENAI_API_KEY") or "").strip()
        or (os.getenv("OPENROUTER_API_KEY") or "").strip()
    )
    has_exiftool = shutil.which("exiftool") is not None or bool(
        (os.getenv("EXIFTOOL_PATH") or "").strip()
    )
    return has_openai_key or has_exiftool


@lru_cache(maxsize=1)
def _get_converter():
    from markitdown import MarkItDown

    llm_client, llm_model = _build_llm_config()
    if llm_client is None or llm_model is None:
        return MarkItDown()
    llm_prompt = (os.getenv("OPENAI_PROMPT") or "").strip() or None
    return MarkItDown(llm_client=llm_client, llm_model=llm_model, llm_prompt=llm_prompt)


def convert_path_to_markdown(path: str | Path) -> MarkItDownResult:
    path_obj = Path(path)
    is_image = path_obj.suffix.lower() in {".jpg", ".jpeg", ".png"}
    media_md5: str | None = None
    llm_enabled = False
    exiftool_path: str | None = None
    base_url = ""
    model = ""
    prompt = ""

    if is_image:
        _load_dotenv_once()
        llm_enabled = bool(
            (os.getenv("OPENAI_API_KEY") or "").strip()
            or (os.getenv("OPENROUTER_API_KEY") or "").strip()
        )
        base_url = (
            os.getenv("OPENAI_BASE_URL") or ""
        ).strip() or "https://openrouter.ai/api/v1"
        model = (os.getenv("OPENAI_MODEL") or "").strip() or "google/gemini-2.5-flash"
        prompt = (
            os.getenv("OPENAI_PROMPT") or ""
        ).strip() or "Write a detailed caption for this image."
        exiftool_path = (os.getenv("EXIFTOOL_PATH") or "").strip() or shutil.which(
            "exiftool"
        )
        media_md5 = _file_md5(path_obj)

        cache_key_payload = {
            "v": 1,
            "type": "image",
            "media_md5": media_md5,
            "markitdown_version": _markitdown_version(),
            "llm_enabled": llm_enabled,
            "provider": base_url if llm_enabled else None,
            "model": model if llm_enabled else None,
            "prompt": prompt if llm_enabled else None,
            "exiftool_path": exiftool_path,
            "description_heading": "auto-generated",
        }
        key = _cache_key(cache_key_payload)
        cached = _read_cache_entry(key)
        if isinstance(cached, dict):
            cached_markdown = cached.get("markdown")
            cached_title = cached.get("title")
            if isinstance(cached_markdown, str):
                title = cached_title if isinstance(cached_title, str) else None
                return MarkItDownResult(
                    markdown=_postprocess_image_markdown(cached_markdown), title=title
                )

        if not _image_text_tools_available():
            raise MarkItDownConversionError(
                "Image conversion requires either `exiftool` (or EXIFTOOL_PATH) or "
                "OPENAI_API_KEY/OPENROUTER_API_KEY."
            )

    try:
        result = _get_converter().convert(path_obj)
    except Exception as exc:
        raise MarkItDownConversionError(
            f"MarkItDown failed to convert {path_obj}: {exc}"
        ) from exc

    markdown = getattr(result, "markdown", None)
    if not isinstance(markdown, str):
        markdown = str(result)
    title = getattr(result, "title", None)
    if title is not None and not isinstance(title, str):
        title = str(title)

    out_markdown = _postprocess_image_markdown(markdown) if is_image else markdown
    out = MarkItDownResult(markdown=out_markdown, title=title)
    if is_image and media_md5 is not None:
        _write_cache_entry(
            key, payload=cache_key_payload, markdown=markdown, title=out.title
        )
    return out

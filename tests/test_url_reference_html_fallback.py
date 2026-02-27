from __future__ import annotations

import requests

from contextualize.references.url import (
    URLReference,
    _assess_markdown_quality,
    _normalize_markdown_converter_output,
)


class _DummyResponse:
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        text: str,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self._text = text
        self.url = url
        self.content = text.encode("utf-8")

    @property
    def text(self) -> str:
        return self._text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        raise ValueError("no json payload")


class _DummyJSONResponse(_DummyResponse):
    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        text: str,
        url: str,
        payload: dict,
    ) -> None:
        super().__init__(status_code=status_code, headers=headers, text=text, url=url)
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_html_get_uses_markdown_converter_when_head_probe_fails(monkeypatch) -> None:
    url = "https://example.com/page"

    def _head_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.RequestException("head failed")

    def _get_html(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body>raw html</body></html>",
            url=url,
        )

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_html)
    monkeypatch.setattr(
        URLReference,
        "_fetch_via_markdown_converter",
        lambda self: "# converted markdown",
    )

    ref = URLReference(url=url, format="raw", use_cache=False)

    assert ref.read() == "# converted markdown"
    assert ref.get_contents() == "# converted markdown"


def test_raw_prefix_bypasses_html_markdown_converter(monkeypatch) -> None:
    url = "https://example.com/page"

    def _head_fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise requests.RequestException("head failed")

    def _get_html(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _DummyResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body>raw html</body></html>",
            url=url,
        )

    monkeypatch.setattr(requests, "head", _head_fail)
    monkeypatch.setattr(requests, "get", _get_html)

    def _should_not_run(self) -> str:  # type: ignore[no-untyped-def]
        raise AssertionError("converter should be bypassed for raw: URLs")

    monkeypatch.setattr(URLReference, "_fetch_via_markdown_converter", _should_not_run)

    ref = URLReference(url=f"raw:{url}", format="raw", use_cache=False)

    assert ref.read() == "<html><body>raw html</body></html>"
    assert ref.get_contents() == "<html><body>raw html</body></html>"


def test_normalize_markdown_converter_output_unescapes_overescaped_markdown() -> None:
    raw = "\n".join(
        [
            r"\# Heading",
            r"\* one",
            r"\* two",
            "```json",
            r"\{\"keep\": \"escaped in fence\"\}",
            "```",
            r"\[link\](https://example.com)",
            r"\* three",
            r"\* four",
            r"\* five",
            r"\* six",
            r"\* seven",
        ]
    )

    normalized = _normalize_markdown_converter_output(raw)

    assert "# Heading" in normalized
    assert "* one" in normalized
    assert "[link](https://example.com)" in normalized
    assert r"\# Heading" not in normalized
    assert r"\* one" not in normalized
    assert r"\[link\](https://example.com)" not in normalized
    assert '\\{\\"keep\\": \\"escaped in fence\\"\\}' in normalized


def test_normalize_markdown_converter_output_keeps_small_escapes() -> None:
    raw = r"Use \*literal\* markers and \[brackets\] in plain text."
    normalized = _normalize_markdown_converter_output(raw)
    assert normalized == raw


def test_assess_markdown_quality_detects_single_line_blob() -> None:
    blob = (
        "# Title * [a](https://example.com/a) * [b](https://example.com/b) "
        "* [c](https://example.com/c) * [d](https://example.com/d) "
        "* [e](https://example.com/e) " * 8
    )
    assessment = _assess_markdown_quality(blob)
    assert assessment["requires_fallback"]
    assert "single_line_blob" in assessment["reasons"]


def test_assess_markdown_quality_detects_html_document_blob() -> None:
    html_payload = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
        "<title>Aumann</title><script>console.log('x')</script></head>"
        "<body><div>payload</div></body></html>"
    )
    assessment = _assess_markdown_quality(html_payload)
    assert assessment["requires_fallback"]
    assert "html_document" in assessment["reasons"]


def test_fetch_via_markdown_converter_uses_jina_when_markdown_new_requires_fallback(
    monkeypatch,
) -> None:
    markdown_new = (
        r"\# Title \* \[alpha\](https://example.com/a) \* \[beta\](https://example.com/b) "
        r"\* \[gamma\](https://example.com/c) \* \[delta\](https://example.com/d) "
        r"\* \[epsilon\](https://example.com/e) \* \[zeta\](https://example.com/f) "
        r"\* \[eta\](https://example.com/g) \* \[theta\](https://example.com/h) "
    )
    jina_markdown = "# Title\n\n- alpha\n- beta\n- gamma\n\n[ref](https://example.com)"
    calls: list[str] = []

    def _post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(url)
        if url == "https://markdown.new/":
            return _DummyJSONResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                text="",
                url=url,
                payload={"markdown": markdown_new},
            )
        if url == "https://r.jina.ai/":
            return _DummyJSONResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                text="",
                url=url,
                payload={"code": 200, "data": {"content": jina_markdown}},
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(requests, "post", _post)

    ref = object.__new__(URLReference)
    ref.url = "https://example.com/doc"
    out = URLReference._fetch_via_markdown_converter(ref)

    assert out == jina_markdown
    assert calls == ["https://markdown.new/", "https://r.jina.ai/"]


def test_fetch_via_markdown_converter_short_circuits_on_good_markdown_new(
    monkeypatch,
) -> None:
    markdown_new = "# Title\n\n- alpha\n- beta\n\n[ref](https://example.com)"
    calls: list[str] = []

    def _post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(url)
        if url == "https://markdown.new/":
            return _DummyJSONResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                text="",
                url=url,
                payload={"markdown": markdown_new},
            )
        raise AssertionError("jina should not be called for good markdown.new output")

    monkeypatch.setattr(requests, "post", _post)

    ref = object.__new__(URLReference)
    ref.url = "https://example.com/doc"
    out = URLReference._fetch_via_markdown_converter(ref)

    assert out == markdown_new
    assert calls == ["https://markdown.new/"]


def test_fetch_via_markdown_converter_uses_jina_when_markdown_new_returns_html(
    monkeypatch,
) -> None:
    html_payload = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Bad</title>"
        "<script>window.a=1</script></head><body><div>content</div></body></html>"
    )
    jina_markdown = "# Good title\n\n- item one\n- item two"
    calls: list[str] = []

    def _post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(url)
        if url == "https://markdown.new/":
            return _DummyResponse(
                status_code=200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=html_payload,
                url=url,
            )
        if url == "https://r.jina.ai/":
            return _DummyJSONResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                text="",
                url=url,
                payload={"code": 200, "data": {"content": jina_markdown}},
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(requests, "post", _post)

    ref = object.__new__(URLReference)
    ref.url = "https://example.com/doc"
    out = URLReference._fetch_via_markdown_converter(ref)

    assert out == jina_markdown
    assert calls == ["https://markdown.new/", "https://r.jina.ai/"]


def test_fetch_via_markdown_converter_keeps_markdown_new_when_jina_fails(
    monkeypatch,
) -> None:
    markdown_new = (
        r"\# Title \* \[alpha\](https://example.com/a) \* \[beta\](https://example.com/b) "
        r"\* \[gamma\](https://example.com/c) \* \[delta\](https://example.com/d) "
        r"\* \[epsilon\](https://example.com/e) \* \[zeta\](https://example.com/f) "
        r"\* \[eta\](https://example.com/g) \* \[theta\](https://example.com/h) "
    )

    def _post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        if url == "https://markdown.new/":
            return _DummyJSONResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                text="",
                url=url,
                payload={"markdown": markdown_new},
            )
        if url == "https://r.jina.ai/":
            raise requests.RequestException("jina unavailable")
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(requests, "post", _post)

    ref = object.__new__(URLReference)
    ref.url = "https://example.com/doc"
    out = URLReference._fetch_via_markdown_converter(ref)

    assert "# Title" in out
    assert "* [alpha](https://example.com/a)" in out


def test_fetch_via_markdown_converter_keeps_html_when_jina_fails(
    monkeypatch,
) -> None:
    html_payload = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'/><title>Bad</title>"
        "<script>window.a=1</script></head><body><div>content</div></body></html>"
    )

    def _post(url, *args, **kwargs):  # type: ignore[no-untyped-def]
        if url == "https://markdown.new/":
            return _DummyResponse(
                status_code=200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                text=html_payload,
                url=url,
            )
        if url == "https://r.jina.ai/":
            raise requests.RequestException("jina unavailable")
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(requests, "post", _post)

    ref = object.__new__(URLReference)
    ref.url = "https://example.com/doc"
    out = URLReference._fetch_via_markdown_converter(ref)

    assert out == html_payload

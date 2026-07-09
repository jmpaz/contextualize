"""Mark addresses at the target-resolution seam.

The factory strips `@time` suffixes before plugins see a target; the span
rides in as `PluginContext["span"]`. Claim checks run on the base, so
anchored plugins keep their claims under a mark address.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader
from contextualize.references import create_file_references
from contextualize.serve.resolve import is_external_target


class _RecordingPlugin:
    def __init__(self, *, exact_keys: frozenset[str] | None = None) -> None:
        self.exact_keys = exact_keys
        self.can_resolve_calls: list[str] = []
        self.resolve_calls: list[tuple[str, dict[str, object]]] = []

    def can_resolve(self, target: str, _context: dict[str, object]) -> bool:
        self.can_resolve_calls.append(target)
        if self.exact_keys is not None:
            return target in self.exact_keys
        return target.startswith("fake:")

    def resolve(self, target: str, context: dict[str, object]) -> list[dict]:
        self.resolve_calls.append((target, dict(context)))
        return [
            {
                "source": target,
                "label": "fake/item.md",
                "content": "fake content",
                "metadata": {"context_subpath": "fake/item.md"},
            }
        ]


def _install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recorder: _RecordingPlugin
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    class _FakeEntrypoint:
        name = "fake"
        value = "contextualize_plugins.fake:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "fake"
            plugin.PLUGIN_PRIORITY = 500
            plugin.can_resolve = recorder.can_resolve
            plugin.resolve = recorder.resolve
            return plugin

    monkeypatch.setattr(
        plugin_loader, "_iter_plugin_entrypoints", lambda: [_FakeEntrypoint()]
    )
    clear_loaded_plugins_cache()


def test_factory_strips_mark_and_delivers_span(monkeypatch, tmp_path):
    recorder = _RecordingPlugin()
    _install(monkeypatch, tmp_path, recorder)

    create_file_references(["fake:key@4:12-8:00"], format="raw")

    target, context = recorder.resolve_calls[0]
    assert target == "fake:key"
    assert context["span"] == {"start": 252.0, "end": 480.0, "authored": "4:12-8:00"}
    assert recorder.can_resolve_calls[-1] == "fake:key"


def test_factory_omits_span_without_mark(monkeypatch, tmp_path):
    recorder = _RecordingPlugin()
    _install(monkeypatch, tmp_path, recorder)

    create_file_references(["fake:key"], format="raw")

    target, context = recorder.resolve_calls[0]
    assert target == "fake:key"
    assert "span" not in context


def test_factory_keeps_non_time_at_in_target(monkeypatch, tmp_path):
    recorder = _RecordingPlugin()
    _install(monkeypatch, tmp_path, recorder)

    create_file_references(["fake:key@v2"], format="raw")

    target, context = recorder.resolve_calls[0]
    assert target == "fake:key@v2"
    assert "span" not in context


def test_factory_restores_anchored_claim_under_mark(monkeypatch, tmp_path):
    recorder = _RecordingPlugin(exact_keys=frozenset({"fake:key"}))
    _install(monkeypatch, tmp_path, recorder)

    create_file_references(["fake:key@12:04-13:26"], format="raw")

    target, context = recorder.resolve_calls[0]
    assert target == "fake:key"
    assert context["span"] == {"start": 724.0, "end": 806.0, "authored": "12:04-13:26"}


def test_is_external_target_checks_claims_on_base(monkeypatch, tmp_path):
    recorder = _RecordingPlugin(exact_keys=frozenset({"fake:key"}))
    _install(monkeypatch, tmp_path, recorder)

    assert is_external_target("fake:key@4:12") is True
    assert is_external_target("fake:key@4:12-8:00") is True
    assert is_external_target("fake:key") is True
    assert is_external_target("fake:other@4:12") is False
    assert is_external_target("fake:key@v2") is False

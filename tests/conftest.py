"""Shared fixtures: a fake segments-bearing store plugin.

Mirrors the segment-store span contract — `documents segments` shapes
(camelCase, float seconds), span docs with mark/asr/capture metadata, and
designed-state docs carrying `mark_state` — without shelling out.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from contextualize.plugins import clear_loaded_plugins_cache
from contextualize.plugins import loader as plugin_loader

STORE_KEY = "voice/2026-07-07/12-34-52.m4a"
STORE_TARGET = f"store:{STORE_KEY}"
UNTIMED_KEY = "notes/plain.md"
UNTIMED_TARGET = f"store:{UNTIMED_KEY}"
DURATION_S = 81.0

SEGMENTS = [
    {
        "id": 1,
        "documentId": 7,
        "captureId": 3,
        "segmentIndex": 0,
        "startTime": 0.0,
        "endTime": 20.0,
        "text": "so the reckoning note starts here",
    },
    {
        "id": 2,
        "documentId": 7,
        "captureId": 3,
        "segmentIndex": 1,
        "startTime": 20.0,
        "endTime": 45.0,
        "text": "treat this as a case study for the work itself",
    },
    {
        "id": 3,
        "documentId": 7,
        "captureId": 3,
        "segmentIndex": 2,
        "startTime": 45.0,
        "endTime": 66.0,
        "text": "two prompts and the supplement around them",
    },
    {
        "id": 4,
        "documentId": 7,
        "captureId": 3,
        "segmentIndex": 3,
        "startTime": 66.0,
        "endTime": 81.0,
        "text": "which is where the mood kit day begins",
    },
]

CAPTURE = {
    "id": 3,
    "model": "fake-asr-1",
    "captured_at": "2026-07-07T12:40:00Z",
    "synthetic": False,
}


def flat_transcript(segments: list[dict[str, Any]]) -> str:
    return "\n".join(str(segment["text"]) for segment in segments)


class FakeStore:
    def __init__(self) -> None:
        self.segments = [dict(segment) for segment in SEGMENTS]
        self.capture = dict(CAPTURE)
        self.resolve_calls: list[tuple[str, str | None]] = []
        self.entries: dict[str, str | None] = {
            STORE_KEY: flat_transcript(self.segments),
            UNTIMED_KEY: "just prose, no timing",
        }

    def matching_keys(self, query: str) -> list[str]:
        if query == "all":
            return list(self.entries)
        return [key for key in self.entries if key == query]

    def _segments_for(self, key: str) -> list[dict[str, Any]] | None:
        return self.segments if key == STORE_KEY else None

    def resolve(self, target: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        span = context.get("span")
        authored = span["authored"] if isinstance(span, dict) else None
        self.resolve_calls.append((target, authored))
        remainder = target[len("store:") :]
        query, _, params = remainder.partition("?")
        if span is not None and params:
            return [
                _state_doc(
                    target,
                    "mark-params-unsupported",
                    "Query params do not compose with a mark.",
                    span,
                )
            ]
        keys = self.matching_keys(query)
        if span is not None:
            return self._resolve_span(target, query, keys, span)
        return [self._full_doc(key) for key in keys]

    def _resolve_span(
        self,
        target: str,
        query: str,
        keys: list[str],
        span: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if len(keys) != 1:
            return [
                _state_doc(
                    target,
                    "marks-require-single-document",
                    f"Marks require a single document; {query} matched {len(keys)} documents.",
                    span,
                )
            ]
        key = keys[0]
        authored = span["authored"]
        source = f"store:{key}@{authored}"
        segments = self._segments_for(key)
        if segments is None:
            return [
                _state_doc(
                    source,
                    "marks-on-untimed-target",
                    f"No timed transcript for store:{key}; marks need timed media.",
                    span,
                )
            ]
        if span["start"] >= DURATION_S:
            return [
                _state_doc(
                    source,
                    "mark-beyond-duration",
                    f"Mark {authored} is beyond this recording (duration 1:21).",
                    span,
                    duration_s=DURATION_S,
                )
            ]
        selected = _selected_segments(segments, span)
        asr = "\n".join(str(segment["text"]) for segment in selected)
        return [
            {
                "source": source,
                "label": f"{key}@{authored}",
                "content": asr,
                "metadata": {
                    "provider": "store",
                    "trace_path": source,
                    "context_subpath": f"{key}@{authored}",
                    "store_key": key,
                    "mark": {
                        "authored": authored,
                        "start_s": span["start"],
                        "end_s": span["end"],
                        "covered_start_s": min(
                            float(segment["startTime"]) for segment in selected
                        ),
                        "covered_end_s": max(
                            float(segment["endTime"]) for segment in selected
                        ),
                        "segment_count": len(selected),
                    },
                    "asr": asr,
                    "capture": dict(self.capture),
                    "duration_s": DURATION_S,
                },
            }
        ]

    def _full_doc(self, key: str) -> dict[str, Any]:
        return {
            "source": f"store:{key}",
            "label": key,
            "content": str(self.entries[key]),
            "metadata": {
                "provider": "store",
                "trace_path": f"store:{key}",
                "context_subpath": key,
                "store_key": key,
                "source_ref": "store",
                "source_path": key,
            },
        }


def _selected_segments(
    segments: list[dict[str, Any]], span: dict[str, Any]
) -> list[dict[str, Any]]:
    start = span["start"]
    end = span["end"]
    if end is None:
        return [
            segment
            for segment in segments
            if float(segment["startTime"]) <= start < float(segment["endTime"])
        ][:1]
    return [
        segment
        for segment in segments
        if float(segment["endTime"]) > start and float(segment["startTime"]) < end
    ]


def _state_doc(
    source: str,
    state: str,
    line: str,
    span: dict[str, Any],
    **extra: Any,
) -> dict[str, Any]:
    subpath = source[len("store:") :] if source.startswith("store:") else source
    return {
        "source": source,
        "label": subpath,
        "content": line,
        "metadata": {
            "provider": "store",
            "mark_state": state,
            "trace_path": source,
            "context_subpath": subpath,
            "mark": {
                "authored": span["authored"],
                "start_s": span["start"],
                "end_s": span["end"],
            },
            **extra,
        },
    }


def _store_entrypoint(store: FakeStore):
    class _StoreEntrypoint:
        name = "store"
        value = "contextualize_plugins.store:plugin"

        def load(self):
            plugin = types.SimpleNamespace()
            plugin.PLUGIN_API_VERSION = "1"
            plugin.PLUGIN_NAME = "store"
            plugin.PLUGIN_PRIORITY = 50
            plugin.can_resolve = lambda target, _context: target.startswith("store:")
            plugin.resolve = store.resolve
            return plugin

    return _StoreEntrypoint()


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
    store = FakeStore()
    monkeypatch.setattr(
        plugin_loader,
        "_iter_plugin_entrypoints",
        lambda: [_store_entrypoint(store)],
    )
    clear_loaded_plugins_cache()
    yield store
    clear_loaded_plugins_cache()

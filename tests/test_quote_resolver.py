from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.manifest.quote_resolver import (
    LocalVoiceQuoteResolver,
    default_documents_db_path,
)


def _create_database(path: Path) -> None:
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            store_key TEXT NOT NULL UNIQUE,
            content_hash TEXT NOT NULL,
            source_name TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            title TEXT,
            body TEXT NOT NULL,
            tags TEXT,
            source_created TEXT,
            source_modified TEXT,
            resolved_at TEXT NOT NULL,
            resolved_by TEXT,
            metadata TEXT
        );
        CREATE TABLE captures (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            captured_at TEXT NOT NULL,
            active INTEGER NOT NULL,
            synthetic INTEGER NOT NULL
        );
        CREATE TABLE segments (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            capture_id INTEGER NOT NULL,
            segment_index INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            text TEXT NOT NULL
        );
        """
    )
    database.execute(
        """
        INSERT INTO documents (
            id, store_key, content_hash, source_name, source_kind, title, body,
            resolved_at
        ) VALUES (1, 'voice/resolved.m4a', 'hash-1', 'voice', 'audio',
                  'Resolved', 'body', '2026-07-14T00:00:00Z')
        """
    )
    database.execute(
        """
        INSERT INTO captures
            (id, document_id, model, captured_at, active, synthetic)
        VALUES (1, 1, 'asr-1', '2026-07-14T00:00:00Z', 0, 0),
               (2, 1, 'asr-2', '2026-07-14T01:00:00Z', 1, 0)
        """
    )
    database.execute(
        """
        INSERT INTO segments
            (id, document_id, capture_id, segment_index, start_time, end_time, text)
        VALUES (1, 1, 2, 0, 0, 10, 'resolved quote'),
               (2, 1, 2, 1, 10, 20, 'second segment')
        """
    )
    database.execute(
        """
        INSERT INTO documents (
            id, store_key, content_hash, source_name, source_kind, title, body,
            resolved_at
        ) VALUES (2, 'voice/ambiguous.m4a', 'hash-2', 'voice', 'audio',
                  'Ambiguous', 'body', '2026-07-14T00:00:00Z')
        """
    )
    database.execute(
        """
        INSERT INTO captures
            (id, document_id, model, captured_at, active, synthetic)
        VALUES (3, 2, 'asr-1', '2026-07-14T00:00:00Z', 1, 0),
               (4, 2, 'asr-2', '2026-07-14T01:00:00Z', 1, 0)
        """
    )
    database.commit()
    database.close()


def test_local_voice_quote_resolver_is_canonical_read_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "documents.db"
    _create_database(database_path)

    with LocalVoiceQuoteResolver(database_path) as resolver:
        resolved = resolver("voice/resolved.m4a")
        assert resolved == resolver("store:voice/resolved.m4a@0:05")
        assert resolved["source"] == "voice/resolved.m4a#transcript-capture=2"
        assert resolver._database.execute("PRAGMA query_only").fetchone()[0] == 1
        assert resolved["capture"] == {
            "id": 2,
            "documentId": 1,
            "model": "asr-2",
            "capturedAt": "2026-07-14T01:00:00Z",
            "active": True,
            "synthetic": False,
        }
        assert resolved["segments"] == [
            {
                "id": 1,
                "documentId": 1,
                "captureId": 2,
                "segmentIndex": 0,
                "startSeconds": 0.0,
                "endSeconds": 10.0,
                "text": "resolved quote",
            },
            {
                "id": 2,
                "documentId": 1,
                "captureId": 2,
                "segmentIndex": 1,
                "startSeconds": 10.0,
                "endSeconds": 20.0,
                "text": "second segment",
            },
        ]
        assert resolver("store:voice/missing.m4a") is None
        assert resolver("store:voice/ambiguous.m4a") is None
        assert resolver("store:voice/not-audio.txt") is None
        assert resolver("store:notes/plain.m4a") is None
    assert resolver._database is None


def test_default_documents_db_path_uses_only_declared_environment(monkeypatch, tmp_path):
    for name in ("CONTEXTUALIZE_QUOTE_STORE_DB", "CONTEXTUALIZE_READER_STATE_DIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))

    assert default_documents_db_path() is None

    state_dir = tmp_path / "state"
    monkeypatch.setenv("CONTEXTUALIZE_READER_STATE_DIR", str(state_dir))
    assert default_documents_db_path() == state_dir / "documents.db"

    database_path = tmp_path / "configured-documents.db"
    monkeypatch.setenv("CONTEXTUALIZE_QUOTE_STORE_DB", str(database_path))
    assert default_documents_db_path() == database_path


def test_contexts_compile_cli_passes_local_quote_resolver_for_manifest_and_registry(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "documents.db"
    _create_database(database_path)
    monkeypatch.setenv("CONTEXTUALIZE_QUOTE_STORE_DB", str(database_path))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """config:
  name: demo
components:
  - name: voice
    files:
      - path: store:voice/resolved.m4a
        marks:
          - at: 0:05
            quote: resolved quote
""",
        encoding="utf-8",
    )
    registry = tmp_path / "contexts.json"
    registry.write_text(
        json.dumps(
            {
                "version": 1,
                "contexts": {
                    "demo": {
                        "targetDir": str(tmp_path),
                        "manifest": {"source": str(manifest)},
                        "replace": "guarded",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    for args in (
        ["contexts", "compile", "--manifest", str(manifest), "--name", "demo"],
        ["contexts", "compile", "--registry", str(registry), "demo"],
    ):
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        edition = payload["editions"][0] if "editions" in payload else payload
        assert edition["diagnostics"] == []
        authored_range = edition["portals"][0]["ranges"][0]
        assert authored_range["quoteResolution"]["state"] == "resolved"
        assert (
            authored_range["quoteResolution"]["source"]
            == "voice/resolved.m4a#transcript-capture=2"
        )
        assert authored_range["quoteResolution"]["range"] == {
            "startSeconds": 0.0,
            "endSeconds": 10.0,
        }

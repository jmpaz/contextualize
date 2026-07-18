"""Read-only local transcript resolution for authored quote compilation."""

from __future__ import annotations

import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote


_VOICE_TARGET_RE = re.compile(
    r"^(?:store:)?(?P<key>voice/.+?\.m4a)(?:@[^#]+)?$"
)
_REQUIRED_SCHEMA = {
    "documents": {
        "id",
        "store_key",
        "content_hash",
        "source_name",
        "source_kind",
        "title",
        "source_created",
        "source_modified",
        "resolved_at",
        "resolved_by",
    },
    "captures": {"id", "document_id", "model", "captured_at", "active", "synthetic"},
    "segments": {
        "id",
        "document_id",
        "capture_id",
        "segment_index",
        "start_time",
        "end_time",
        "text",
    },
}


def default_documents_db_path() -> Path | None:
    configured = os.environ.get("CONTEXTUALIZE_QUOTE_STORE_DB")
    if configured:
        return Path(configured).expanduser()
    state_dir = os.environ.get("CONTEXTUALIZE_READER_STATE_DIR")
    if state_dir:
        return Path(state_dir).expanduser() / "documents.db"
    return None


def canonical_voice_key(target: str) -> str | None:
    if not isinstance(target, str):
        return None
    match = _VOICE_TARGET_RE.fullmatch(target.strip())
    return match.group("key") if match else None


class LocalVoiceQuoteResolver:
    def __init__(self, database_path: str | os.PathLike[str] | None = None) -> None:
        self.database_path = (
            Path(database_path).expanduser()
            if database_path
            else default_documents_db_path()
        )
        self._database: sqlite3.Connection | None = None

    def __enter__(self) -> "LocalVoiceQuoteResolver":
        try:
            if self.database_path is not None and self.database_path.is_file():
                uri = (
                    "file:"
                    + quote(str(self.database_path.resolve()), safe="/")
                    + "?mode=ro"
                )
                self._database = sqlite3.connect(uri, uri=True)
                self._database.row_factory = sqlite3.Row
                self._database.execute("PRAGMA query_only = ON")
                self._database.execute("BEGIN")
                if not self._schema_is_supported(self._database):
                    self.close()
        except (OSError, sqlite3.Error):
            self.close()
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        database = self._database
        self._database = None
        if database is not None:
            database.close()

    @staticmethod
    def _schema_is_supported(database: sqlite3.Connection) -> bool:
        tables = {
            row["name"]
            for row in database.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name IN ('documents', 'captures', 'segments')
                """
            ).fetchall()
        }
        if tables != set(_REQUIRED_SCHEMA):
            return False
        return all(
            required <= {
                row["name"]
                for row in database.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table, required in _REQUIRED_SCHEMA.items()
        )

    def __call__(self, target: str) -> dict[str, Any] | None:
        key = canonical_voice_key(target)
        database = self._database
        if key is None or database is None:
            return None
        try:
            document = self._document(database, key)
            if document is None:
                return None
            captures = self._active_captures(database, int(document["id"]))
            if len(captures) != 1:
                return None
            capture = captures[0]
            segments = self._segments(database, int(document["id"]), int(capture["id"]))
            if not segments:
                return None
            return {
                "source": f"{key}#transcript-capture={capture['id']}",
                "storeKey": key,
                "document": {
                    "id": int(document["id"]),
                    "storeKey": document["store_key"],
                    "sourceName": document["source_name"],
                    "sourceKind": document["source_kind"],
                    "title": document["title"],
                    "contentHash": document["content_hash"],
                    "sourceCreated": document["source_created"],
                    "sourceModified": document["source_modified"],
                    "resolvedAt": document["resolved_at"],
                    "resolvedBy": document["resolved_by"],
                },
                "capture": {
                    "id": int(capture["id"]),
                    "documentId": int(capture["document_id"]),
                    "model": capture["model"],
                    "capturedAt": capture["captured_at"],
                    "active": bool(capture["active"]),
                    "synthetic": bool(capture["synthetic"]),
                },
                "segments": segments,
            }
        except (TypeError, ValueError, OverflowError, sqlite3.Error):
            return None

    @staticmethod
    def _document(database: sqlite3.Connection, key: str) -> sqlite3.Row | None:
        rows = database.execute(
            "SELECT * FROM documents WHERE store_key = ? LIMIT 2",
            (key,),
        ).fetchall()
        return rows[0] if len(rows) == 1 else None

    @staticmethod
    def _active_captures(
        database: sqlite3.Connection, document_id: int
    ) -> list[sqlite3.Row]:
        return database.execute(
            """
            SELECT id, document_id, model, captured_at, active, synthetic
            FROM captures
            WHERE document_id = ? AND active = 1
            ORDER BY captured_at DESC, id DESC
            """,
            (document_id,),
        ).fetchall()

    @staticmethod
    def _segments(
        database: sqlite3.Connection,
        document_id: int,
        capture_id: int,
    ) -> list[dict[str, Any]] | None:
        rows = database.execute(
            """
            SELECT id, document_id, capture_id, segment_index, start_time, end_time, text
            FROM segments
            WHERE document_id = ? AND capture_id = ?
            ORDER BY segment_index, id
            """,
            (document_id, capture_id),
        ).fetchall()
        segments: list[dict[str, Any]] = []
        for row in rows:
            index = row["segment_index"]
            start = row["start_time"]
            end = row["end_time"]
            text = row["text"]
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or isinstance(start, bool)
                or not isinstance(start, (int, float))
                or not math.isfinite(start)
                or start < 0
                or isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not math.isfinite(end)
                or end <= start
                or not isinstance(text, str)
            ):
                return None
            segments.append(
                {
                    "id": int(row["id"]),
                    "documentId": int(row["document_id"]),
                    "captureId": int(row["capture_id"]),
                    "segmentIndex": index,
                    "startSeconds": float(start),
                    "endSeconds": float(end),
                    "text": text,
                }
            )
        return segments

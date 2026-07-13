from __future__ import annotations

import hashlib
import json
from pathlib import Path

from contextualize.cache.artifacts import (
    artifact_paths,
    get_source_artifact,
    import_source_artifact,
)


def test_source_artifact_is_content_addressed_and_reused(tmp_path: Path) -> None:
    content = b"a retained source attachment"
    digest = hashlib.sha256(content).hexdigest()
    provenance = {
        "provider": "discord",
        "native_ref": "discord:attachment:11:22:33",
        "source_url": "https://cdn.discordapp.com/attachments/11/33/notes.txt",
    }

    first = import_source_artifact(
        content,
        content_type="text/plain",
        provenance=provenance,
        root=tmp_path,
    )
    second = import_source_artifact(
        content,
        content_type="application/octet-stream",
        provenance={"provider": "other"},
        root=tmp_path,
    )

    content_path, metadata_path = artifact_paths(digest, root=tmp_path)
    assert first == second
    assert first.path == content_path
    assert first.metadata_path == metadata_path
    assert content_path.read_bytes() == content
    assert json.loads(metadata_path.read_text(encoding="utf-8"))["provenance"] == provenance
    assert get_source_artifact(digest, root=tmp_path) == first


def test_source_artifact_rejects_corrupt_content(tmp_path: Path) -> None:
    artifact = import_source_artifact(b"trusted", root=tmp_path)
    artifact.path.write_bytes(b"altered")

    assert get_source_artifact(artifact.sha256, root=tmp_path) is None

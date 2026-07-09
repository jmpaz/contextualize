from __future__ import annotations

import pytest

from contextualize.manifest.manifest import GROUP_BASE_KEY, GROUP_PATH_KEY, SET_KEY, normalize_components


def test_normalize_components_recognizes_top_level_set() -> None:
    normalized = normalize_components(
        [{"set": "jul-7 run", "files": ["a.md", "b.md"]}]
    )

    assert len(normalized) == 1
    comp = normalized[0]
    assert comp["name"] == "jul-7 run"
    assert comp[SET_KEY] is True
    assert comp["files"] == ["a.md", "b.md"]


def test_normalize_components_group_may_contain_set() -> None:
    normalized = normalize_components(
        [
            {
                "group": "g",
                "components": [
                    {"set": "fused", "files": ["a.md"]},
                    {"name": "plain", "files": ["b.md"]},
                ],
            }
        ]
    )

    names = [comp["name"] for comp in normalized]
    assert names == ["g.fused", "g.plain"]
    fused = normalized[0]
    assert fused[SET_KEY] is True
    assert fused[GROUP_PATH_KEY] == ("g",)
    assert fused[GROUP_BASE_KEY] == "fused"


def test_normalize_components_set_cannot_contain_group() -> None:
    with pytest.raises(ValueError, match="invalid keys"):
        normalize_components(
            [{"set": "x", "files": ["a.md"], "group": "y", "components": []}]
        )


def test_normalize_components_set_cannot_also_define_name() -> None:
    with pytest.raises(ValueError, match="cannot also define 'name'"):
        normalize_components([{"set": "x", "name": "y", "files": ["a.md"]}])


def test_normalize_components_set_rejects_repos_and_manifests() -> None:
    with pytest.raises(ValueError, match="does not support 'repos'"):
        normalize_components([{"set": "x", "files": ["a.md"], "repos": ["b"]}])

    with pytest.raises(ValueError, match="does not support 'manifests'"):
        normalize_components(
            [{"set": "x", "files": ["a.md"], "manifests": ["b.yaml"]}]
        )


def test_normalize_components_set_requires_files() -> None:
    with pytest.raises(ValueError, match="must define 'files'"):
        normalize_components([{"set": "x", "text": "hello"}])

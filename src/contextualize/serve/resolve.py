"""Selector parsing and manifest/node resolution for the serving surface.

Grammar: `name-or-path:component.member`. The pre-colon origin resolves
against known external schemes, then the context registry, then the
filesystem (§5.1 of the manifest-grammar spec). The post-colon path walks
the manifest's outline tree by name; once it reaches a leaf (component or
set) any remaining token addresses one raw member within it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..manifest.contexts import ContextEntry, load_context_registry
from ..manifest.manifest import normalize_components
from ..manifest.source import ManifestSource, load_manifest_source

_MEMBER_KEYS = ("files", "repos", "manifests")
_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class Selector:
    raw: str
    origin: str
    tokens: tuple[str, ...]


def parse_selector(raw: str, *, cwd: str | None = None) -> Selector:
    stripped = raw.strip()
    if ":" not in stripped:
        return Selector(raw=raw, origin=stripped, tokens=())
    if _existing_file(stripped, cwd):
        return Selector(raw=raw, origin=stripped, tokens=())
    origin, _, remainder = stripped.partition(":")
    tokens = tuple(token for token in remainder.split(".") if token)
    return Selector(raw=raw, origin=origin, tokens=tokens)


def _existing_file(text: str, cwd: str | None) -> bool:
    candidate = Path(os.path.expanduser(text))
    if not candidate.is_absolute():
        candidate = Path(cwd or os.getcwd()) / candidate
    return candidate.is_file()


def is_external_target(text: str) -> bool:
    from ..git.target import parse_git_target
    from ..plugins import plugin_target_provider
    from ..references.helpers import is_http_url, parse_target_spec

    if is_http_url(text):
        return True
    if parse_git_target(text) is not None:
        return True
    target = parse_target_spec(text).get("target", text)
    if not isinstance(target, str):
        target = text
    return plugin_target_provider(target) is not None


@dataclass
class ManifestHandle:
    origin: str
    registry_name: str | None
    manifest_path: Path | None
    context_dir: Path | None
    registry_entry: ContextEntry | None
    source: ManifestSource | None
    source_error: str | None
    index: dict[str, Any] | None
    index_error: str | None
    _normalized: list[dict[str, Any]] | None = field(default=None, repr=False)

    @property
    def hydrated(self) -> bool:
        return self.index is not None

    @property
    def outline(self) -> list[dict[str, Any]]:
        if self.index is not None and isinstance(self.index.get("outline"), list):
            return self.index["outline"]
        if self.source is not None and self.source.source_format is not None:
            return self.source.source_format.outline
        return []

    @property
    def normalized_components(self) -> list[dict[str, Any]]:
        if self._normalized is not None:
            return self._normalized
        components: list[dict[str, Any]] = []
        if self.source is not None and isinstance(self.source.data, dict):
            raw = self.source.data.get("components")
            if isinstance(raw, list):
                try:
                    components = normalize_components(raw)
                except ValueError:
                    components = []
        self._normalized = components
        return components

    @property
    def references_out(self) -> list[dict[str, Any]]:
        if not self.index:
            return []
        refs = self.index.get("references")
        if not isinstance(refs, dict):
            return []
        out = refs.get("out")
        return out if isinstance(out, list) else []

    @property
    def display_name(self) -> str:
        if self.registry_name:
            return self.registry_name
        if self.manifest_path:
            return str(self.manifest_path)
        return self.origin


def _context_dir_for(cfg: dict[str, Any], base_dir: Path) -> Path:
    context_cfg = cfg.get("context")
    raw_dir = context_cfg.get("dir") if isinstance(context_cfg, dict) else None
    if not isinstance(raw_dir, str) or not raw_dir:
        manifest_name = cfg.get("name")
        raw_dir = (
            f".context/{manifest_name}"
            if isinstance(manifest_name, str) and manifest_name
            else ".context"
        )
    dir_path = Path(os.path.expanduser(raw_dir))
    if not dir_path.is_absolute():
        dir_path = (base_dir / dir_path).resolve()
    return dir_path


def _load_index(context_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    index_path = context_dir / "index.json"
    if not index_path.is_file():
        return None, None
    try:
        return json.loads(index_path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def load_manifest_handle(
    origin: str,
    *,
    registry: dict[str, ContextEntry] | None = None,
    registry_path: str | os.PathLike[str] | None = None,
    cwd: str | None = None,
) -> ManifestHandle | None:
    """Resolve an origin to a manifest. None means it is not a context
    origin at all (external scheme, unknown name, no such path) -- callers
    fall back to treating it as ordinary cat input."""
    if is_external_target(origin):
        return None

    if registry is None:
        try:
            registry = load_context_registry(registry_path)
        except (OSError, ValueError):
            registry = {}

    working_dir = Path(cwd or os.getcwd())
    registry_entry = registry.get(origin)
    registry_name: str | None = None
    manifest_path: Path | None = None
    base_dir = working_dir

    if registry_entry is not None:
        registry_name = origin
        base_dir = registry_entry.target_dir
        source_spec = (
            registry_entry.manifest.get("source")
            if isinstance(registry_entry.manifest, dict)
            else None
        )
        if isinstance(source_spec, str) and source_spec:
            candidate = Path(os.path.expanduser(source_spec))
            if not candidate.is_absolute():
                candidate = registry_entry.target_dir / candidate
            manifest_path = candidate
    else:
        candidate = Path(os.path.expanduser(origin))
        if not candidate.is_absolute():
            candidate = working_dir / candidate
        if candidate.is_file():
            manifest_path = candidate

    if manifest_path is None:
        return None

    source: ManifestSource | None = None
    source_error: str | None = None
    try:
        source = load_manifest_source(manifest_path)
    except (OSError, ValueError) as exc:
        source_error = str(exc)

    cfg: dict[str, Any] = {}
    if source is not None and isinstance(source.data, dict):
        raw_cfg = source.data.get("config")
        if isinstance(raw_cfg, dict):
            cfg = raw_cfg

    context_dir = _context_dir_for(cfg, base_dir)
    index, index_error = _load_index(context_dir)
    if index is None and index_error is None and not context_dir.is_dir():
        context_dir = None

    return ManifestHandle(
        origin=origin,
        registry_name=registry_name,
        manifest_path=manifest_path,
        context_dir=context_dir,
        registry_entry=registry_entry,
        source=source,
        source_error=source_error,
        index=index,
        index_error=index_error,
    )


@dataclass
class NodeLookup:
    node: dict[str, Any] | None
    path: tuple[str, ...]
    remainder: tuple[str, ...]
    found: bool
    blocked_disabled: bool


def resolve_node(outline: list[dict[str, Any]], tokens: tuple[str, ...]) -> NodeLookup:
    if not tokens:
        return NodeLookup(node=None, path=(), remainder=(), found=True, blocked_disabled=False)

    level = outline
    path: list[str] = []
    match: dict[str, Any] | None = None
    for index, token in enumerate(tokens):
        match = next((node for node in level if node.get("name") == token), None)
        if match is None:
            return NodeLookup(
                node=None,
                path=tuple(path),
                remainder=tokens[index:],
                found=False,
                blocked_disabled=False,
            )
        path.append(token)
        if match["disabled"]:
            return NodeLookup(
                node=match,
                path=tuple(path),
                remainder=tokens[index + 1 :],
                found=index == len(tokens) - 1,
                blocked_disabled=True,
            )
        if match["kind"] == "group":
            level = match.get("children", [])
            continue
        return NodeLookup(
            node=match,
            path=tuple(path),
            remainder=tokens[index + 1 :],
            found=True,
            blocked_disabled=False,
        )
    assert match is not None, "non-empty tokens ran the loop"
    return NodeLookup(node=match, path=tuple(path), remainder=(), found=True, blocked_disabled=False)


def component_for_node(handle: ManifestHandle, dotted_path: tuple[str, ...]) -> dict[str, Any] | None:
    full_name = ".".join(dotted_path)
    for comp in handle.normalized_components:
        if comp.get("name") == full_name:
            return comp
    return None


@dataclass
class Target:
    """The single resolution shared by `show` and `cat`: a selector's dotted
    path plus an optional trailing member token both settle here, so the two
    verbs cannot drift apart on what a selector means."""

    kind: str  # "root" | "group" | "leaf" | "member" | "disabled" | "not-found"
    node: dict[str, Any] | None
    dotted_path: tuple[str, ...]
    comp: dict[str, Any] | None
    member: tuple[str, int, Any] | None
    detail: str | None = None
    disabled_member: tuple[str, dict[str, Any]] | None = None

    def require_node(self) -> dict[str, Any]:
        assert self.node is not None, f"kind={self.kind!r} resolves to a node"
        return self.node

    def require_member(self) -> tuple[str, int, Any]:
        assert self.member is not None, f"kind={self.kind!r} carries a member"
        return self.member


def resolve_target(handle: ManifestHandle, tokens: tuple[str, ...]) -> Target:
    if not tokens:
        return Target(kind="root", node=None, dotted_path=(), comp=None, member=None)

    lookup = resolve_node(handle.outline, tokens)
    if not lookup.found or lookup.node is None:
        matched = ".".join(lookup.path) or "(root)"
        return Target(
            kind="not-found",
            node=None,
            dotted_path=lookup.path,
            comp=None,
            member=None,
            detail=f"No component matches '{'.'.join(tokens)}' (matched up to '{matched}').",
        )
    if lookup.blocked_disabled:
        return Target(
            kind="disabled",
            node=lookup.node,
            dotted_path=lookup.path,
            comp=None,
            member=None,
            detail=f"'{'.'.join(lookup.path)}' is disabled in the manifest source.",
        )

    node = lookup.node
    dotted = lookup.path
    if node["kind"] == "group":
        if lookup.remainder:
            return Target(
                kind="not-found",
                node=None,
                dotted_path=dotted,
                comp=None,
                member=None,
                detail=f"'{lookup.remainder[0]}' cannot narrow a group; address a component first.",
            )
        return Target(kind="group", node=node, dotted_path=dotted, comp=None, member=None)

    comp = component_for_node(handle, dotted)
    if not lookup.remainder:
        return Target(kind="leaf", node=node, dotted_path=dotted, comp=comp, member=None)
    if comp is None:
        return Target(
            kind="disabled",
            node=node,
            dotted_path=dotted,
            comp=None,
            member=None,
            detail="This component has no enabled members.",
        )
    token = lookup.remainder[0]
    match = match_member(comp, token)
    if match is None:
        disabled_hit = find_disabled_member(node, token)
        if disabled_hit is not None:
            return Target(
                kind="disabled",
                node=node,
                dotted_path=dotted,
                comp=comp,
                member=None,
                detail=f"'{'.'.join(dotted)}.{token}' is disabled in the manifest source.",
                disabled_member=disabled_hit,
            )
        return Target(
            kind="not-found",
            node=node,
            dotted_path=dotted,
            comp=comp,
            member=None,
            detail=f"No member '{token}' on '{'.'.join(dotted)}'.",
        )
    return Target(kind="member", node=node, dotted_path=dotted, comp=comp, member=match)


def spec_text(entry: Any) -> str:
    if isinstance(entry, dict):
        for key in ("path", "target", "url"):
            value = entry.get(key)
            if isinstance(value, str):
                return value
        return ""
    return str(entry)


def spec_alias(entry: Any) -> str | None:
    if isinstance(entry, dict):
        alias = entry.get("alias") or entry.get("filename")
        if isinstance(alias, str):
            return alias
    return None


def manifest_base_dir(handle: ManifestHandle) -> str:
    cfg: dict[str, Any] = {}
    if handle.source is not None and isinstance(handle.source.data, dict):
        raw_cfg = handle.source.data.get("config")
        if isinstance(raw_cfg, dict):
            cfg = raw_cfg
    root = cfg.get("root")
    if isinstance(root, str) and root:
        return os.path.abspath(os.path.expanduser(root))
    manifest_cwd = handle.source.manifest_cwd if handle.source is not None else None
    return os.path.abspath(manifest_cwd or os.getcwd())


def resolve_live_spec(handle: ManifestHandle, raw_spec: str) -> str:
    if is_external_target(raw_spec):
        return raw_spec
    candidate = Path(os.path.expanduser(raw_spec))
    if candidate.is_absolute():
        return str(candidate)
    return str(Path(manifest_base_dir(handle)) / candidate)


def slugify(text: str) -> str:
    stem = Path(text.split("::", 1)[0]).stem or text
    cleaned = _SLUG_RE.sub("-", stem).strip("-").lower()
    return cleaned or "item"


def component_members(comp: dict[str, Any]) -> list[tuple[str, int, Any]]:
    members: list[tuple[str, int, Any]] = []
    for key in _MEMBER_KEYS:
        raw_list = comp.get(key)
        if not isinstance(raw_list, list):
            continue
        for position, entry in enumerate(raw_list):
            members.append((key, position, entry))
    return members


def match_member(comp: dict[str, Any], token: str) -> tuple[str, int, Any] | None:
    members = component_members(comp)
    for key, position, entry in members:
        if spec_alias(entry) == token:
            return key, position, entry
    for key, position, entry in members:
        if slugify(spec_text(entry)) == token:
            return key, position, entry
    if token.isdigit():
        ordinal = int(token)
        if 1 <= ordinal <= len(members):
            return members[ordinal - 1]
    return None


_DISABLED_HINT_RE = re.compile(r"^-?\s*(?:[A-Za-z_-]+:\s*)?(?P<val>.+)$")


def disabled_member_slug(raw: str) -> str:
    first_line = raw.splitlines()[0].strip() if raw else ""
    match = _DISABLED_HINT_RE.match(first_line)
    val = match.group("val") if match else first_line
    val = val.split(" #", 1)[0].strip().strip("'\"")
    return slugify(val)


def find_disabled_member(node: dict[str, Any], token: str) -> tuple[str, dict[str, Any]] | None:
    for key in _MEMBER_KEYS:
        for item in node.get("members", {}).get(key, []):
            if item.get("disabled") and disabled_member_slug(item.get("raw", "")) == token:
                return key, item
    return None

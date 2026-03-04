from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any

from .api import PluginContext, PluginDocument
from .loader import get_loaded_plugins
from .reference import PluginReference, PluginResolvedDocument


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def _normalize_plugin_document(
    plugin_name: str,
    target: str,
    item: PluginDocument,
    *,
    index: int,
) -> PluginResolvedDocument | None:
    source_raw = item.get("source", target)
    label_raw = item.get("label")
    content_raw = item.get("content")
    metadata_raw = item.get("metadata", {})
    if not isinstance(source_raw, str) or not source_raw:
        _warn(
            f"plugin '{plugin_name}' returned invalid source for target '{target}' "
            f"(item {index})"
        )
        return None
    if not isinstance(label_raw, str) or not label_raw:
        _warn(
            f"plugin '{plugin_name}' returned invalid label for target '{target}' "
            f"(item {index})"
        )
        return None
    if not isinstance(content_raw, str):
        _warn(
            f"plugin '{plugin_name}' returned invalid content for target '{target}' "
            f"(item {index})"
        )
        return None
    metadata: dict[str, Any]
    if isinstance(metadata_raw, dict):
        metadata = dict(metadata_raw)
    else:
        metadata = {}
    metadata.setdefault("plugin_name", plugin_name)
    metadata.setdefault("provider", plugin_name)
    return PluginResolvedDocument(
        source=source_raw,
        label=label_raw,
        content=content_raw,
        metadata=metadata,
    )


def _build_context(
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    overrides: dict[str, Any],
) -> PluginContext:
    return {
        "format": format,
        "label": label,
        "label_suffix": label_suffix,
        "include_token_count": include_token_count,
        "token_target": token_target,
        "inject": inject,
        "depth": depth,
        "use_cache": use_cache,
        "cache_ttl": cache_ttl,
        "refresh_cache": refresh_cache,
        "overrides": overrides,
    }


def resolve_plugin_references(
    target: str,
    *,
    format: str,
    label: str,
    label_suffix: str | None,
    include_token_count: bool,
    token_target: str,
    inject: bool,
    depth: int,
    trace_collector: list | None,
    use_cache: bool,
    cache_ttl: timedelta | None,
    refresh_cache: bool,
    overrides: dict[str, Any],
) -> list[PluginReference]:
    context = _build_context(
        format=format,
        label=label,
        label_suffix=label_suffix,
        include_token_count=include_token_count,
        token_target=token_target,
        inject=inject,
        depth=depth,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        overrides=overrides,
    )
    for plugin in get_loaded_plugins():
        matched = False
        try:
            matched = bool(plugin.can_resolve(target, context))
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' can_resolve failed for '{target}': {exc}")
            continue
        if not matched:
            continue

        try:
            documents = plugin.resolve(target, context)
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' resolve failed for '{target}': {exc}")
            continue
        if not isinstance(documents, list):
            _warn(
                f"plugin '{plugin.name}' returned non-list result for '{target}'; "
                "falling back"
            )
            continue

        normalized_documents: list[PluginResolvedDocument] = []
        for index, document in enumerate(documents):
            if not isinstance(document, dict):
                _warn(
                    f"plugin '{plugin.name}' returned non-mapping document for "
                    f"'{target}' (item {index})"
                )
                normalized_documents = []
                break
            normalized = _normalize_plugin_document(
                plugin.name,
                target,
                document,
                index=index,
            )
            if normalized is None:
                normalized_documents = []
                break
            normalized_documents.append(normalized)
        if not normalized_documents:
            continue

        return [
            PluginReference(
                source=document.source,
                document=document,
                format=format,
                label=label,
                label_suffix=label_suffix,
                include_token_count=include_token_count,
                token_target=token_target,
                inject=inject,
                depth=depth,
                trace_collector=trace_collector,
            )
            for document in normalized_documents
        ]
    return []

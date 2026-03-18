from __future__ import annotations

import sys
from datetime import timedelta
from typing import Any

from .api import PluginContext, PluginDocument, PluginTargetDescriptor
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
    cache_only: bool,
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
        "cache_only": cache_only,
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
) -> tuple[list[PluginReference], bool]:
    from contextualize.runtime import get_cache_only

    cache_only = get_cache_only()
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
        cache_only=cache_only,
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
            return [], True

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
                use_cache=use_cache,
                cache_ttl=cache_ttl,
                refresh_cache=refresh_cache,
                plugin_overrides=overrides or None,
            )
            for document in normalized_documents
        ], True
    return [], False


def _build_inspection_context(overrides: dict[str, Any] | None) -> PluginContext:
    return _build_context(
        format="raw",
        label="relative",
        label_suffix=None,
        include_token_count=False,
        token_target="cl100k_base",
        inject=False,
        depth=5,
        use_cache=True,
        cache_ttl=None,
        refresh_cache=False,
        cache_only=False,
        overrides=overrides or {},
    )


def loaded_plugin_names() -> tuple[str, ...]:
    return tuple(plugin.name for plugin in get_loaded_plugins())


def normalize_manifest_plugin_config(
    plugin_name: str,
    raw_config: dict[str, Any] | None,
) -> dict[str, Any] | None:
    plugin = next((p for p in get_loaded_plugins() if p.name == plugin_name), None)
    if plugin is None:
        return raw_config
    if plugin.normalize_manifest_config is None:
        return raw_config
    try:
        normalized = plugin.normalize_manifest_config(raw_config)
    except Exception as exc:
        raise ValueError(
            f"plugin '{plugin_name}' normalize_manifest_config failed: {exc}"
        ) from exc
    if normalized is None:
        return None
    if not isinstance(normalized, dict):
        raise ValueError(
            f"plugin '{plugin_name}' normalize_manifest_config must return a mapping"
        )
    return dict(normalized)


def classify_plugin_target(
    target: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> PluginTargetDescriptor | None:
    context = _build_inspection_context(overrides)
    for plugin in get_loaded_plugins():
        try:
            matched = bool(plugin.can_resolve(target, context))
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' can_resolve failed for '{target}': {exc}")
            continue
        if not matched:
            continue

        descriptor: PluginTargetDescriptor | None
        if plugin.classify_target is None:
            descriptor = {"provider": plugin.name, "is_external": True}
        else:
            try:
                descriptor = plugin.classify_target(target, context)
            except Exception as exc:
                _warn(
                    f"plugin '{plugin.name}' classify_target failed for '{target}': {exc}"
                )
                continue
            if descriptor is None:
                descriptor = {"provider": plugin.name, "is_external": True}
            elif not isinstance(descriptor, dict):
                _warn(
                    f"plugin '{plugin.name}' classify_target returned non-mapping for '{target}'"
                )
                continue
            else:
                descriptor = dict(descriptor)
                descriptor.setdefault("provider", plugin.name)
                descriptor.setdefault("is_external", True)
        return descriptor
    return None

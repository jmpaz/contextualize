from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from .api import (
    PluginContext,
    PluginDocument,
    PluginListEnvelope,
    PluginListItem,
    PluginMaterializedFile,
    PluginTargetDescriptor,
)
from .loader import get_loaded_plugins
from .reference import PluginReference, PluginResolvedDocument


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class PluginListResult:
    items: tuple[PluginListItem, ...]
    matched: bool
    supported: bool
    plugin_name: str | None = None
    error: str | None = None
    summary: dict[str, Any] | None = None
    pagination: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None


@dataclass(frozen=True)
class PluginMaterializeResult:
    files: tuple[PluginMaterializedFile, ...]
    matched: bool
    supported: bool
    plugin_name: str | None = None


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


def _normalize_plugin_list_item(
    plugin_name: str,
    target: str,
    item: PluginListItem,
    *,
    index: int,
) -> PluginListItem | None:
    target_raw = item.get("target")
    if not isinstance(target_raw, str) or not target_raw:
        _warn(
            f"plugin '{plugin_name}' returned invalid listing target for '{target}' "
            f"(item {index})"
        )
        return None

    normalized: PluginListItem = {"target": target_raw}
    label_raw = item.get("label")
    if isinstance(label_raw, str) and label_raw:
        normalized["label"] = label_raw
    kind_raw = item.get("kind")
    if isinstance(kind_raw, str) and kind_raw:
        normalized["kind"] = kind_raw
    traverse_raw = item.get("traverse")
    if isinstance(traverse_raw, bool):
        normalized["traverse"] = traverse_raw
    metadata_raw = item.get("metadata")
    if isinstance(metadata_raw, dict):
        normalized["metadata"] = dict(metadata_raw)
    return normalized


def _normalize_plugin_list_envelope(
    plugin_name: str,
    target: str,
    envelope: PluginListEnvelope,
) -> PluginListResult:
    if not isinstance(envelope, dict):
        _warn(f"plugin '{plugin_name}' returned non-envelope listing for '{target}'")
        return PluginListResult((), True, False, plugin_name)
    raw_items = envelope.get("targets")
    if not isinstance(raw_items, list):
        _warn(
            f"plugin '{plugin_name}' returned listing envelope without "
            f"targets for '{target}'"
        )
        return PluginListResult((), True, False, plugin_name)

    normalized_items: list[PluginListItem] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            _warn(
                f"plugin '{plugin_name}' returned non-mapping listing item for "
                f"'{target}' (item {index})"
            )
            return PluginListResult((), True, False, plugin_name)
        normalized = _normalize_plugin_list_item(
            plugin_name,
            target,
            item,
            index=index,
        )
        if normalized is None:
            return PluginListResult((), True, False, plugin_name)
        normalized_items.append(normalized)

    summary_raw = envelope.get("summary")
    pagination_raw = envelope.get("pagination")
    metadata_raw = envelope.get("metadata")
    capabilities_raw = envelope.get("capabilities")
    return PluginListResult(
        tuple(normalized_items),
        True,
        True,
        plugin_name,
        summary=dict(summary_raw) if isinstance(summary_raw, dict) else None,
        pagination=dict(pagination_raw) if isinstance(pagination_raw, dict) else None,
        metadata=dict(metadata_raw) if isinstance(metadata_raw, dict) else None,
        capabilities=dict(capabilities_raw)
        if isinstance(capabilities_raw, dict)
        else None,
    )


def _optional_positive_int(context: PluginContext, key: str) -> int | None:
    raw = context.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw <= 0:
        raise ValueError(f"{key} must be greater than 0")
    return raw


def _optional_nonnegative_int(context: PluginContext, key: str) -> int:
    raw = context.get(key)
    if raw is None:
        return 0
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if raw < 0:
        raise ValueError(f"{key} must be zero or greater")
    return raw


def _page_plugin_list_result(
    result: PluginListResult,
    context: PluginContext,
) -> PluginListResult:
    limit = _optional_positive_int(context, "list_limit")
    offset = _optional_nonnegative_int(context, "list_offset")
    if limit is None and offset == 0:
        return result
    if not result.supported:
        return result

    raw_pagination = dict(result.pagination or {})
    if raw_pagination.get("offset") == offset and raw_pagination.get("limit") == limit:
        return result

    items = result.items
    end = None if limit is None else offset + limit
    page_items = items[offset:end]
    raw_total = raw_pagination.get("totalCount")
    total_count = (
        max(raw_total, len(items))
        if isinstance(raw_total, int) and not isinstance(raw_total, bool)
        else len(items)
    )
    next_offset = offset + len(page_items)
    has_more = next_offset < total_count
    pagination = {
        **raw_pagination,
        "offset": offset,
        "returned": len(page_items),
        "totalCount": total_count,
        "hasMore": has_more,
    }
    if limit is not None:
        pagination["limit"] = limit
    if has_more:
        pagination["nextOffset"] = next_offset
    else:
        pagination.pop("nextOffset", None)

    summary = result.summary
    if isinstance(summary, dict) and isinstance(summary.get("sampledItems"), list):
        summary = {**summary, "sampledItems": list(page_items)}

    return PluginListResult(
        tuple(page_items),
        result.matched,
        result.supported,
        result.plugin_name,
        error=result.error,
        summary=summary,
        pagination=pagination,
        metadata=result.metadata,
        capabilities=result.capabilities,
    )


def _normalize_plugin_materialized_file(
    plugin_name: str,
    target: str,
    item: PluginMaterializedFile,
    *,
    index: int,
) -> PluginMaterializedFile | None:
    filename_raw = item.get("filename")
    content_raw = item.get("content")
    if not isinstance(filename_raw, str) or not filename_raw.strip():
        _warn(
            f"plugin '{plugin_name}' returned invalid filename for '{target}' "
            f"(item {index})"
        )
        return None
    if not isinstance(content_raw, bytes):
        _warn(
            f"plugin '{plugin_name}' returned invalid materialized content for "
            f"'{target}' (item {index})"
        )
        return None
    source_raw = item.get("source", target)
    label_raw = item.get("label", filename_raw)
    metadata_raw = item.get("metadata", {})
    content_type_raw = item.get("content_type")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    metadata.setdefault("plugin_name", plugin_name)
    metadata.setdefault("provider", plugin_name)
    return {
        "source": source_raw if isinstance(source_raw, str) and source_raw else target,
        "label": label_raw
        if isinstance(label_raw, str) and label_raw
        else filename_raw,
        "filename": filename_raw.strip(),
        "content": content_raw,
        "content_type": content_type_raw
        if isinstance(content_type_raw, str) and content_type_raw.strip()
        else None,
        "metadata": metadata,
    }


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


def list_plugin_targets(
    target: str,
    *,
    overrides: dict[str, Any] | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
    context_options: dict[str, Any] | None = None,
) -> PluginListResult:
    context = _build_inspection_context(
        overrides,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    if context_options:
        context.update(context_options)
    for plugin in get_loaded_plugins():
        try:
            matched = bool(plugin.can_resolve(target, context))
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' can_resolve failed for '{target}': {exc}")
            continue
        if not matched:
            continue
        if plugin.list_targets is None:
            return PluginListResult((), True, False, plugin.name)

        try:
            envelope = plugin.list_targets(target, context)
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' list_targets failed for '{target}': {exc}")
            return PluginListResult(
                (),
                True,
                False,
                plugin.name,
                error=f"{exc.__class__.__name__}: {exc}",
            )
        result = _normalize_plugin_list_envelope(plugin.name, target, envelope)
        return _page_plugin_list_result(result, context)
    return PluginListResult((), False, False, None)


def materialize_plugin_target(
    target: str,
    *,
    overrides: dict[str, Any] | None = None,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> PluginMaterializeResult:
    context = _build_inspection_context(
        overrides,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
    for plugin in get_loaded_plugins():
        try:
            matched = bool(plugin.can_resolve(target, context))
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' can_resolve failed for '{target}': {exc}")
            continue
        if not matched:
            continue
        if plugin.materialize is None:
            return PluginMaterializeResult((), True, False, plugin.name)

        try:
            files = plugin.materialize(target, context)
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' materialize failed for '{target}': {exc}")
            return PluginMaterializeResult((), True, False, plugin.name)
        if not isinstance(files, list):
            _warn(
                f"plugin '{plugin.name}' returned non-list materialization for '{target}'"
            )
            return PluginMaterializeResult((), True, False, plugin.name)

        normalized_files: list[PluginMaterializedFile] = []
        for index, item in enumerate(files):
            if not isinstance(item, dict):
                _warn(
                    f"plugin '{plugin.name}' returned non-mapping materialized file for "
                    f"'{target}' (item {index})"
                )
                return PluginMaterializeResult((), True, False, plugin.name)
            normalized = _normalize_plugin_materialized_file(
                plugin.name,
                target,
                item,
                index=index,
            )
            if normalized is None:
                return PluginMaterializeResult((), True, False, plugin.name)
            normalized_files.append(normalized)
        return PluginMaterializeResult(tuple(normalized_files), True, True, plugin.name)
    return PluginMaterializeResult((), False, False, None)


def _build_inspection_context(
    overrides: dict[str, Any] | None,
    *,
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> PluginContext:
    from contextualize.runtime import get_cache_only

    return _build_context(
        format="raw",
        label="relative",
        label_suffix=None,
        include_token_count=False,
        token_target="cl100k_base",
        inject=False,
        depth=5,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
        cache_only=get_cache_only(),
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
    use_cache: bool = True,
    cache_ttl: timedelta | None = None,
    refresh_cache: bool = False,
) -> PluginTargetDescriptor | None:
    context = _build_inspection_context(
        overrides,
        use_cache=use_cache,
        cache_ttl=cache_ttl,
        refresh_cache=refresh_cache,
    )
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

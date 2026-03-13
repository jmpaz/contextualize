from __future__ import annotations

from typing import Any

import click

from .loader import get_loaded_plugins

_SYNCED_PLUGINS_ATTR = "_contextualize_synced_cli_plugins"


def _warn(message: str) -> None:
    click.echo(f"Warning: {message}", err=True)


def sync_plugin_cli_commands(root: click.Group) -> None:
    loaded = get_loaded_plugins()
    for command_name in ("cat", "hydrate"):
        command = root.commands.get(command_name)
        if command is None:
            continue
        synced = set(getattr(command, _SYNCED_PLUGINS_ATTR, set()))
        for plugin in loaded:
            hook = plugin.register_cli_options
            if hook is None:
                continue
            marker = f"{plugin.name}:{plugin.origin}:{command_name}"
            if marker in synced:
                continue
            try:
                hook(command_name, command)
            except Exception as exc:
                _warn(
                    f"plugin '{plugin.name}' cli registration failed for '{command_name}': {exc}"
                )
                continue
            synced.add(marker)
        setattr(command, _SYNCED_PLUGINS_ATTR, synced)


def collect_plugin_cli_overrides(
    command_name: str,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    for plugin in get_loaded_plugins():
        hook = plugin.collect_cli_overrides
        if hook is None:
            continue
        try:
            raw_mapping = hook(command_name, params)
        except Exception as exc:
            raise click.ClickException(
                f"plugin '{plugin.name}' cli override collection failed: {exc}"
            ) from exc
        if raw_mapping is None:
            continue
        if not isinstance(raw_mapping, dict):
            raise click.ClickException(
                f"plugin '{plugin.name}' cli override collection must return a mapping"
            )
        if raw_mapping:
            overrides[plugin.name] = dict(raw_mapping)
    return overrides or None


def loaded_transcription_providers() -> tuple[Any, ...]:
    providers: list[Any] = []
    for plugin in get_loaded_plugins():
        providers.extend(plugin.transcription_providers)
    return tuple(providers)


def loaded_transcription_gates() -> tuple[Any, ...]:
    gates: list[Any] = []
    for plugin in get_loaded_plugins():
        gates.extend(plugin.transcription_gates)
    return tuple(gates)

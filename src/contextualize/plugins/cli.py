from __future__ import annotations

from typing import Any

import click

from .loader import get_loaded_plugins

_SYNCED_PLUGINS_ATTR = "_contextualize_synced_cli_plugins"


def _warn(message: str) -> None:
    click.echo(f"Warning: {message}", err=True)


def sync_plugin_cli_commands(root: click.Group) -> None:
    loaded = get_loaded_plugins()
    commands = [
        ("cat", root.commands.get("cat")),
        ("hydrate", root.commands.get("hydrate")),
        ("payload", root.commands.get("payload")),
    ]
    contexts = root.commands.get("contexts")
    if isinstance(contexts, click.Group):
        commands.append(("hydrate", contexts.commands.get("hydrate")))

    for command_name, command in commands:
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
            existing_ids = {id(param) for param in command.params}
            try:
                hook(command_name, command)
            except Exception as exc:
                _warn(
                    f"plugin '{plugin.name}' cli registration failed for '{command_name}': {exc}"
                )
                continue
            added_param = False
            for param in command.params:
                if id(param) in existing_ids:
                    continue
                setattr(param, "_contextualize_plugin_name", plugin.name)
                added_param = True
            if added_param:
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


_ROOT_COMMANDS_ATTR = "_contextualize_root_command_plugins"


def sync_plugin_root_commands(root: click.Group) -> None:
    registered = set(getattr(root, _ROOT_COMMANDS_ATTR, set()))
    for plugin in get_loaded_plugins():
        hook = plugin.register_command
        if hook is None:
            continue
        marker = f"{plugin.name}:{plugin.origin}"
        if marker in registered:
            continue
        try:
            hook(root)
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' register_command failed: {exc}")
            continue
        registered.add(marker)
    setattr(root, _ROOT_COMMANDS_ATTR, registered)

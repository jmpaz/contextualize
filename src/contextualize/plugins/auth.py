from __future__ import annotations

import sys

import click

from .api import LoadedPlugin
from .loader import get_loaded_plugins

_AUTH_GROUP_ATTR = "_contextualize_plugin_auth"


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr, flush=True)


def _build_auth_commands(
    plugins: tuple[LoadedPlugin, ...],
) -> dict[str, click.Command]:
    commands: dict[str, click.Command] = {}
    for plugin in plugins:
        hook = plugin.register_auth_command
        if hook is None:
            continue
        scratch = click.Group(name=f"_{plugin.name}_auth")
        try:
            hook(scratch)
        except Exception as exc:
            _warn(f"plugin '{plugin.name}' auth registration failed: {exc}")
            continue
        if not scratch.commands:
            continue
        for command_name, command in scratch.commands.items():
            if command_name in commands:
                owner = commands[command_name]
                _warn(
                    f"plugin '{plugin.name}' auth command '{command_name}' "
                    f"collides with '{owner.name}'; skipping duplicate"
                )
                continue
            commands[command_name] = command
    return commands


def _format_plugin_source(origin: str) -> str:
    if origin.startswith("entrypoint:"):
        return f"package ({origin.replace('entrypoint:', '', 1)})"
    return f"path ({origin})"


def _render_installed_plugins(plugins: tuple[LoadedPlugin, ...]) -> tuple[str, ...]:
    if not plugins:
        return ("Installed plugins: none",)
    lines = ["Installed plugins:"]
    for plugin in plugins:
        lines.append(f"- {plugin.name}: {_format_plugin_source(plugin.origin)}")
    return tuple(lines)


def _echo_installed_plugins(plugins: tuple[LoadedPlugin, ...]) -> None:
    for line in _render_installed_plugins(plugins):
        click.echo(line)


def _build_missing_auth_command(
    plugin_name: str, plugins: tuple[LoadedPlugin, ...]
) -> click.Command:
    installed_names = {plugin.name for plugin in plugins}

    @click.command(name=plugin_name)
    def _missing_auth_provider() -> None:
        if plugin_name in installed_names:
            click.echo(
                f"Plugin '{plugin_name}' is installed but does not expose an authentication handler."
            )
        else:
            click.echo(f"Plugin '{plugin_name}' is not installed.")
        _echo_installed_plugins(plugins)
        raise click.exceptions.Exit(1)

    return _missing_auth_provider


class PluginAuthGroup(click.Group):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("invoke_without_command", True)
        kwargs.setdefault("no_args_is_help", False)
        super().__init__(*args, **kwargs)
        self._loaded_plugins: tuple[LoadedPlugin, ...] = ()

    def set_loaded_plugins(self, plugins: tuple[LoadedPlugin, ...]) -> None:
        self._loaded_plugins = plugins

    def get_command(self, ctx: click.Context, cmd_name: str):
        command = super().get_command(ctx, cmd_name)
        if command is not None:
            return command
        if cmd_name.startswith("-"):
            return None
        return _build_missing_auth_command(cmd_name, self._loaded_plugins)


@click.pass_context
def _auth_group_callback(ctx: click.Context) -> None:
    command = ctx.command
    if not isinstance(command, PluginAuthGroup):
        return
    if ctx.invoked_subcommand is not None:
        return
    if not command.commands:
        click.echo("No loaded plugins expose authentication handlers.")
        _echo_installed_plugins(command._loaded_plugins)
        return
    click.echo(ctx.get_help())


def sync_plugin_auth_group(root: click.Group) -> None:
    loaded = get_loaded_plugins()
    auth_plugins = tuple(plugin for plugin in loaded if plugin.register_auth_command)
    existing = root.commands.get("auth")
    is_plugin_group = bool(getattr(existing, _AUTH_GROUP_ATTR, False))

    if existing is not None and not is_plugin_group:
        _warn("existing non-plugin 'auth' command found; skipping plugin auth setup")
        return

    if existing is None or not isinstance(existing, PluginAuthGroup):
        if existing is not None and is_plugin_group:
            root.commands.pop("auth", None)
        auth_group = PluginAuthGroup(
            name="auth",
            help="Authentication helpers for external providers.",
            callback=_auth_group_callback,
        )
        setattr(auth_group, _AUTH_GROUP_ATTR, True)
        root.add_command(auth_group, name="auth")
    else:
        auth_group = existing

    auth_group.set_loaded_plugins(loaded)
    auth_group.commands = _build_auth_commands(auth_plugins)

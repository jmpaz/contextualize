from __future__ import annotations

from pathlib import Path
import re


_BANNED_PROVIDER_IMPORT = re.compile(
    r"contextualize\.references\.(?:arena|atproto|discord|soundcloud|youtube|atproto_auth|soundcloud_auth)"
)


def test_provider_plugins_do_not_import_core_provider_references() -> None:
    plugin_dir = Path("plugins/src/cx_plugins/providers")
    targets = sorted(plugin_dir.glob("*/plugin.py"))
    assert targets

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert _BANNED_PROVIDER_IMPORT.search(text) is None, str(path)


def test_provider_auth_commands_do_not_import_core_provider_auth_modules() -> None:
    plugin_dir = Path("plugins/src/cx_plugins/providers")
    targets = sorted(plugin_dir.glob("*/**/*auth.py"))
    assert targets

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert _BANNED_PROVIDER_IMPORT.search(text) is None, str(path)

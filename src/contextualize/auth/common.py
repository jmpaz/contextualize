from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

import click


def load_dotenv_optional() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


def is_tty_session() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def should_confirm_login(provider_label: str) -> bool:
    if not is_tty_session():
        return True
    return click.confirm(
        f"No active {provider_label} user session. Start sign-in now?",
        default=True,
    )


def accent_url(url: str) -> str:
    if not is_tty_session():
        return url
    return click.style(url, fg="cyan", underline=True)


def open_url_in_browser(url: str) -> bool:
    def _run_open_command(command: list[str]) -> bool:
        try:
            subprocess.run(
                command,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True
        except Exception:
            return False

    system = platform.system().lower()
    try:
        if system == "windows":
            os.startfile(url)  # type: ignore[attr-defined]
            return True
        if system == "darwin" and shutil.which("open"):
            return _run_open_command(["open", url])
        if shutil.which("xdg-open"):
            return _run_open_command(["xdg-open", url])
        if shutil.which("gio"):
            return _run_open_command(["gio", "open", url])
    except Exception:
        return False
    return False

from __future__ import annotations

import sys
import types

from contextualize.auth.common import load_dotenv_optional


def test_load_dotenv_optional_can_be_disabled(monkeypatch) -> None:
    calls: list[str] = []
    dotenv = types.ModuleType("dotenv")
    dotenv.find_dotenv = lambda usecwd=True: calls.append("find") or ".env"
    dotenv.load_dotenv = lambda path, override=False: calls.append("load")
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setenv("CONTEXTUALIZE_LOAD_DOTENV", "0")

    load_dotenv_optional()

    assert calls == []


def test_load_dotenv_optional_loads_when_enabled(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    dotenv = types.ModuleType("dotenv")
    dotenv.find_dotenv = lambda usecwd=True: calls.append(("find", usecwd)) or ".env"
    dotenv.load_dotenv = lambda path, override=False: calls.append(("load", path))
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.delenv("CONTEXTUALIZE_LOAD_DOTENV", raising=False)

    load_dotenv_optional()

    assert calls == [("find", True), ("load", ".env")]

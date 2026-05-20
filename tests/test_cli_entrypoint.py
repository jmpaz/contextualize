import sys

import pytest

from contextualize import cli


def test_main_uses_stable_program_name(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [".contextualize-wrapped", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    output = capsys.readouterr().out
    assert excinfo.value.code == 0
    assert "Usage: contextualize " in output
    assert ".contextualize-wrapped" not in output

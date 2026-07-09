from __future__ import annotations

import io
import json
from pathlib import Path

from click.testing import CliRunner

from contextualize import cli
from contextualize.serve import core
from contextualize.serve.mcp import run_stdio_server


def _write_manifest(drive: Path) -> None:
    drive.mkdir(parents=True, exist_ok=True)
    (drive / "manifest.yaml").write_text(
        "config:\n"
        "  name: demo\n"
        "  context:\n"
        "    dir: .context/demo\n"
        "    include-meta: true\n"
        "components:\n"
        "  - name: alpha\n"
        "    files:\n"
        "      - a.md  # first note\n"
        "      - b.md\n"
        "  - name: refs\n"
        "    files:\n"
        "      - a.md\n"
        "      # - d.md  # excluded\n",
        encoding="utf-8",
    )
    (drive / "a.md").write_text("alpha content\n", encoding="utf-8")
    (drive / "b.md").write_text("beta content\n", encoding="utf-8")


def _run(messages: list[dict], *, cwd: str, registry_path: str | None = None) -> list[dict]:
    stdin = io.StringIO("\n".join(json.dumps(m) for m in messages) + "\n")
    stdout = io.StringIO()
    run_stdio_server(cwd=cwd, registry_path=registry_path, stdin=stdin, stdout=stdout)
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


def test_lifecycle_initialize_list_and_call(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ],
        cwd=str(drive),
        registry_path=str(registry_path),
    )

    assert len(responses) == 2
    init, tools = responses
    assert init["result"]["serverInfo"]["name"] == "contextualize"
    assert "instructions" in init["result"]
    assert {tool["name"] for tool in tools["result"]["tools"]} == {"show", "cat", "links", "status"}
    for tool in tools["result"]["tools"]:
        assert "inputSchema" in tool


def test_notification_produces_no_response(tmp_path: Path) -> None:
    responses = _run(
        [{"jsonrpc": "2.0", "method": "notifications/initialized"}],
        cwd=str(tmp_path),
    )
    assert responses == []


def test_show_tool_matches_core_and_cli_json(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    selector = str(drive / "manifest.yaml")
    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "show", "arguments": {"selector": selector}},
            }
        ],
        cwd=str(drive),
        registry_path=str(registry_path),
    )
    mcp_payload = responses[0]["result"]["structuredContent"]
    assert responses[0]["result"]["isError"] is False
    text_payload = json.loads(responses[0]["result"]["content"][0]["text"])
    assert text_payload == mcp_payload

    direct = core.show(selector, registry_path=str(registry_path), cwd=str(drive))
    assert mcp_payload == direct

    runner = CliRunner()
    cli_result = runner.invoke(
        cli.cli,
        ["show", selector, "--registry", str(registry_path), "--json"],
        env={"XDG_CONFIG_HOME": str(tmp_path / "xdg-config")},
    )
    assert json.loads(cli_result.output) == mcp_payload


def test_cat_and_links_and_status_tools_match_core(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cat", "arguments": {"selector": "manifest.yaml:alpha", "around": 1}},
            },
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "links", "arguments": {"selector": "manifest.yaml"}},
            },
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "status", "arguments": {"selector": "manifest.yaml"}},
            },
        ],
        cwd=str(drive),
        registry_path=str(registry_path),
    )

    cat_payload = responses[0]["result"]["structuredContent"]
    assert cat_payload == core.draw_substance(
        core.cat_selector(
            "manifest.yaml:alpha", around=1, registry_path=str(registry_path), cwd=str(drive)
        )
    )
    assert "alpha content" in cat_payload["content"]
    assert "beta content" in cat_payload["content"]

    links_payload = responses[1]["result"]["structuredContent"]
    assert links_payload == core.links("manifest.yaml", registry_path=str(registry_path), cwd=str(drive))

    status_payload = responses[2]["result"]["structuredContent"]
    assert status_payload == core.status(
        "manifest.yaml", registry_path=str(registry_path), cwd=str(drive)
    )


def test_cat_designed_state_carries_no_content(tmp_path: Path) -> None:
    drive = tmp_path / "drive"
    _write_manifest(drive)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cat", "arguments": {"selector": "manifest.yaml:nowhere"}},
            }
        ],
        cwd=str(drive),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["state"] == "not-found"
    assert "content" not in payload


def test_designed_states_are_not_protocol_errors(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "show", "arguments": {"selector": "nowhere:x"}},
            }
        ],
        cwd=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["state"] == "not-found"
    assert result["structuredContent"]["next_steps"]


def _write_marked_manifest(drive: Path, target: str) -> Path:
    drive.mkdir(parents=True, exist_ok=True)
    manifest = drive / "manifest.yaml"
    manifest.write_text(
        "config:\n"
        "  context:\n"
        f"    dir: {drive / '.context' / 'annot'}\n"
        "    include-meta: true\n"
        "components:\n"
        "  - name: reckoning\n"
        "    files:\n"
        f'      - path: "{target}"\n'
        "        marks:\n"
        "          - at: 0:04  # opening\n"
        "          - at: 0:04\n"
        "            quote: |\n"
        "              solo\n",
        encoding="utf-8",
    )
    return manifest


def test_cat_of_bare_address_matches_core(fake_store, tmp_path: Path) -> None:
    from conftest import STORE_TARGET

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")
    address = f"{STORE_TARGET}@0:25-1:00"

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cat", "arguments": {"selector": address}},
            }
        ],
        cwd=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload == core.draw_substance(
        core.cat_selector(address, registry_path=str(registry_path), cwd=str(tmp_path))
    )
    assert payload["state"] == "ok"
    assert "treat this as a case study for the work itself" in payload["content"]
    assert "mood kit day begins" not in payload["content"]


def test_cat_of_a_mark_selector_matches_core(fake_store, tmp_path: Path) -> None:
    from conftest import STORE_TARGET

    manifest = _write_marked_manifest(tmp_path / "drive", STORE_TARGET)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")
    selector = f"{manifest}:reckoning.1.1"

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cat", "arguments": {"selector": selector}},
            }
        ],
        cwd=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload == core.draw_substance(
        core.cat_selector(selector, registry_path=str(registry_path), cwd=str(tmp_path))
    )
    assert "asr:\nso the reckoning note starts here" in payload["content"]


def test_cat_mark_designed_state_carries_no_content(fake_store, tmp_path: Path) -> None:
    from conftest import STORE_TARGET

    manifest = _write_marked_manifest(tmp_path / "drive", STORE_TARGET)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "cat", "arguments": {"selector": f"{manifest}:reckoning.1.2"}},
            }
        ],
        cwd=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload["state"] == "mark-quote-requires-range"
    assert payload["next_steps"]
    assert "content" not in payload


def test_links_target_mode_matches_core(fake_store, tmp_path: Path) -> None:
    from conftest import STORE_TARGET

    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"version": 1, "contexts": {}}), encoding="utf-8")

    responses = _run(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "links", "arguments": {"selector": STORE_TARGET}},
            }
        ],
        cwd=str(tmp_path),
        registry_path=str(registry_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is False
    payload = result["structuredContent"]
    assert payload == core.links(
        STORE_TARGET, registry_path=str(registry_path), cwd=str(tmp_path)
    )
    assert payload["origin"]["kind"] == "target"
    assert payload["coverage"]["tag_scope"]["tags"] == ["ctx/manifest"]


def test_tool_descriptions_name_the_mark_surfaces(tmp_path: Path) -> None:
    responses = _run(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}],
        cwd=str(tmp_path),
    )
    tools = {tool["name"]: tool for tool in responses[0]["result"]["tools"]}
    assert "target/@ address" in tools["cat"]["description"]
    assert "mark" in tools["cat"]["inputSchema"]["properties"]["selector"]["description"]
    assert "aggregate every mark" in tools["links"]["description"]


def test_malformed_tool_call_is_a_protocol_error(tmp_path: Path) -> None:
    responses = _run(
        [{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "cat", "arguments": {}}}],
        cwd=str(tmp_path),
    )
    result = responses[0]["result"]
    assert result["isError"] is True


def test_unknown_method_is_a_json_rpc_error(tmp_path: Path) -> None:
    responses = _run(
        [{"jsonrpc": "2.0", "id": 1, "method": "not/a/method"}],
        cwd=str(tmp_path),
    )
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] == -32601

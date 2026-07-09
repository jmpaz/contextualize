"""MCP server for the serving surface: newline-delimited JSON-RPC over
stdio, matching the initialize/tools-list/tools-call shape. contextualize
has no dependency on the MCP Python SDK (checked pyproject.toml), so this
implements the wire protocol directly rather than depending on it for four
tools.

Designed states (not-found, unhydrated, disabled, ...) are successful tool
calls -- the model asked a legible question and got a legible answer, so
isError stays false and the state lives in the payload. isError is
reserved for malformed tool calls and unexpected exceptions, the cases a
model cannot reason its way out of from the response alone."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Iterable, Iterator, TextIO

from . import core
from .core import shelf_for_cwd

SERVER_NAME = "contextualize"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "show",
        "description": (
            "Render a context/manifest as authored: components, groups, and sets "
            "in order, comments as marginalia, disabled members visibly off, "
            "marks under their members. "
            "Works on any manifest, registered or not, hydrated or not."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "name-or-path[:component[.member[.mark]]]. A member token is its "
                        "alias, its filename slug, or a 1-based ordinal; a mark token is "
                        "its authored time or ordinal."
                    ),
                },
                "depth": {
                    "type": "integer",
                    "description": "Limit nested group expansion to N levels.",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "cat",
        "description": (
            "Draw member substance into working context. selector must narrow to "
            "a component, set, a specific member, a mark within one, or a bare "
            "target/@ address (e.g. store:voice/a.m4a@4:12-5:00). A drawn mark "
            "carries the asr slice beside its authored quote, marginalia, and refs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": (
                        "name-or-path:component[.member[.mark]], or a target/@ address. "
                        "A member token is its alias, its filename slug, or a 1-based "
                        "ordinal; a mark token is its authored time or ordinal."
                    ),
                },
                "around": {
                    "type": "integer",
                    "description": "Include N lines of authored adjacency from the manifest source.",
                },
            },
            "required": ["selector"],
        },
    },
    {
        "name": "links",
        "description": (
            "The connective tissue between contexts: who cites this one (in), what it "
            "cites (out), and which sources it holds in common with other registered "
            "contexts (shared members). Pass a target/@ address "
            "(store:voice/a.m4a[@4:12]) to aggregate every mark addressing it across "
            "registered contexts and tag-discovered notes; the scope in effect is "
            "named in coverage.tag_scope."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "Context name, manifest path, or target/@ address.",
                },
                "direction": {"type": "string", "enum": ["in", "out", "both"], "default": "both"},
            },
            "required": ["selector"],
        },
    },
    {
        "name": "status",
        "description": (
            "Freshness and drift: hydration state, cache age, per-source resolution, "
            "and how the composition has drifted from its territory. Omit selector "
            "for the complete registry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "description": "Omit for registry-wide status."},
            },
        },
    },
]


def _require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{key}' is required and must be a non-empty string")
    return value


def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    cwd: str,
    registry_path: str | None,
    status_path: str | None,
) -> dict[str, Any]:
    if name == "show":
        return core.show(
            _require_str(arguments, "selector"),
            depth=arguments.get("depth"),
            registry_path=registry_path,
            cwd=cwd,
        )
    if name == "cat":
        return core.draw_substance(
            core.cat_selector(
                _require_str(arguments, "selector"),
                around=arguments.get("around"),
                registry_path=registry_path,
                cwd=cwd,
            )
        )
    if name == "links":
        return core.links(
            _require_str(arguments, "selector"),
            direction=arguments.get("direction", "both"),
            registry_path=registry_path,
            cwd=cwd,
        )
    if name == "status":
        selector = arguments.get("selector")
        if selector is not None and not isinstance(selector, str):
            raise ValueError("'selector' must be a string when provided")
        return core.status(selector, registry_path=registry_path, status_path=status_path, cwd=cwd)
    raise ValueError(f"Unknown tool: {name}")


def build_instructions(cwd: str, *, registry_path: str | None = None) -> str:
    from ..manifest.contexts import load_context_registry

    try:
        registry = load_context_registry(registry_path)
    except (OSError, ValueError):
        registry = {}

    shelf = shelf_for_cwd(cwd, registry)
    lines = [
        "contextualize serves authored contexts through four verbs: show, cat, links, status.",
        "A relationship not explicitly queryable is absent -- order, comments, exclusions, "
        "and references are all reachable through these tools; they never require re-deriving "
        "structure from prose.",
    ]
    if shelf:
        lines.append(f"\nContexts relevant to {cwd}:")
        for entry in shelf:
            lines.append(f"  - {entry['name']}: {entry['one_line']}")
    else:
        lines.append(f"\nNo registered context has a target containing {cwd}.")
    lines.append(
        f"\n{len(registry)} context(s) registered in total -- "
        "call status with no selector for the complete list."
    )
    return "\n".join(lines)


def _tool_result(payload: dict[str, Any], *, is_error: bool) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _handle_message(
    message: dict[str, Any],
    *,
    cwd: str,
    registry_path: str | None,
    status_path: str | None,
) -> dict[str, Any] | None:
    is_notification = "id" not in message
    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    def respond(result: dict[str, Any]) -> dict[str, Any] | None:
        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    if method == "initialize":
        return respond(
            {
                "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                "instructions": build_instructions(cwd, registry_path=registry_path),
            }
        )

    if method in ("notifications/initialized", "initialized"):
        return None

    if method == "tools/list":
        return respond({"tools": TOOLS})

    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(tool_name, str):
            return respond(_tool_result({"error": "tools/call requires a string 'name'"}, is_error=True))
        try:
            payload = dispatch_tool(
                tool_name, arguments, cwd=cwd, registry_path=registry_path, status_path=status_path
            )
            return respond(_tool_result(payload, is_error=False))
        except Exception as exc:
            error_payload = {"error": str(exc), "tool": tool_name, "arguments": arguments}
            return respond(_tool_result(error_payload, is_error=True))

    if method == "ping":
        return respond({})

    if is_notification:
        return None
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def _read_messages(stream: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict):
            yield message


def run_stdio_server(
    *,
    cwd: str | None = None,
    registry_path: str | None = None,
    status_path: str | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    input_stream = stdin if stdin is not None else sys.stdin
    output_stream = stdout if stdout is not None else sys.stdout
    effective_cwd = cwd or os.getcwd()

    for message in _read_messages(input_stream):
        response = _handle_message(
            message, cwd=effective_cwd, registry_path=registry_path, status_path=status_path
        )
        if response is not None:
            output_stream.write(json.dumps(response) + "\n")
            output_stream.flush()

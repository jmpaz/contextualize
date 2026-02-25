from __future__ import annotations

from collections import deque
from pathlib import Path
from queue import Empty, Queue
from dataclasses import dataclass
import json
import shlex
import subprocess
import threading
import time
from typing import Any


class CodexAppServerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CodexImageDescriptionResult:
    text: str
    requested_model: str | None
    rerouted_from_model: str | None
    rerouted_to_model: str | None
    reroute_reason: str | None


class _CodexAppServerClient:
    def __init__(
        self,
        *,
        command: str,
        startup_timeout_seconds: float = 8.0,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        self._argv = _split_command(command)
        self._startup_timeout_seconds = startup_timeout_seconds
        self._request_timeout_seconds = request_timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._messages: Queue[dict[str, Any] | None] = Queue()
        self._events: deque[dict[str, Any]] = deque()
        self._pending: deque[dict[str, Any]] = deque()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._threads: list[threading.Thread] = []
        self._next_request_id = 1

    def __enter__(self) -> _CodexAppServerClient:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def start(self) -> None:
        if self._process is not None:
            return
        try:
            self._process = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise CodexAppServerError(
                f"Failed to start codex app-server command {' '.join(self._argv)!r}: {exc}"
            ) from exc
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._process.stdout,),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._process.stderr,),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        self._threads.extend([stdout_thread, stderr_thread])

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.5)
        for thread in self._threads:
            thread.join(timeout=0.2)
        self._threads.clear()

    def initialize(self) -> None:
        self.request(
            "initialize",
            params={
                "clientInfo": {
                    "name": "contextualize",
                    "title": "Contextualize",
                    "version": "0.1.0",
                }
            },
            timeout_seconds=self._startup_timeout_seconds,
        )
        self.notify("initialized", {})

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)

    def request(
        self,
        method: str,
        *,
        params: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request_id = self._next_request_id
        self._next_request_id += 1
        payload: dict[str, Any] = {"id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._send(payload)
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._request_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while True:
            message = self._next_message(deadline=deadline, for_label=method)
            if "method" in message and "id" not in message:
                self._events.append(message)
                continue
            if "method" in message and "id" in message:
                raise CodexAppServerError(
                    f"Unsupported app-server callback {message.get('method')!r}"
                )
            if message.get("id") != request_id:
                self._pending.append(message)
                continue
            error = message.get("error")
            if isinstance(error, dict):
                detail = error.get("message")
                detail_text = str(detail) if detail is not None else _safe_json(error)
                raise CodexAppServerError(
                    f"app-server request {method!r} failed: {detail_text}"
                )
            result = message.get("result")
            if isinstance(result, dict):
                return result
            if result is None:
                return {}
            raise CodexAppServerError(
                f"app-server request {method!r} returned unexpected result payload"
            )

    def next_event(self, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        if self._events:
            return self._events.popleft()
        timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._request_timeout_seconds
        )
        deadline = time.monotonic() + timeout
        while True:
            message = self._next_message(deadline=deadline, for_label="event")
            if "method" in message and "id" not in message:
                return message
            if "method" in message and "id" in message:
                raise CodexAppServerError(
                    f"Unsupported app-server callback {message.get('method')!r}"
                )
            self._pending.append(message)

    def _send(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise CodexAppServerError("app-server process is not running")
        try:
            process.stdin.write(_safe_json(payload))
            process.stdin.write("\n")
            process.stdin.flush()
        except OSError as exc:
            raise CodexAppServerError(
                f"Failed to write request to app-server: {exc}"
            ) from exc

    def _next_message(self, *, deadline: float, for_label: str) -> dict[str, Any]:
        if self._pending:
            return self._pending.popleft()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexAppServerError(
                    f"Timed out waiting for app-server {for_label} response"
                )
            try:
                message = self._messages.get(timeout=remaining)
            except Empty as exc:
                raise CodexAppServerError(
                    f"Timed out waiting for app-server {for_label} response"
                ) from exc
            if message is None:
                tail = "; ".join(self._stderr_tail)
                stderr_suffix = f" stderr: {tail}" if tail else ""
                raise CodexAppServerError(
                    f"app-server process exited before completing {for_label}{stderr_suffix}"
                )
            return message

    def _read_stdout(self, stream: Any) -> None:
        try:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self._messages.put(payload)
        finally:
            self._messages.put(None)

    def _read_stderr(self, stream: Any) -> None:
        try:
            for line in stream:
                raw = line.strip()
                if raw:
                    self._stderr_tail.append(raw)
        finally:
            return


def _split_command(command: str) -> tuple[str, ...]:
    raw = command.strip()
    if not raw:
        raise CodexAppServerError("Empty app-server command")
    try:
        argv = tuple(shlex.split(raw))
    except ValueError as exc:
        raise CodexAppServerError(
            f"Invalid app-server command {command!r}: {exc}"
        ) from exc
    if not argv:
        raise CodexAppServerError("Empty app-server command")
    return argv


def _safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def _thread_id(result: dict[str, Any]) -> str:
    thread = result.get("thread")
    if not isinstance(thread, dict):
        raise CodexAppServerError("app-server thread/start response missing thread")
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise CodexAppServerError("app-server thread/start response missing thread id")
    return thread_id


def _turn_id(result: dict[str, Any]) -> str:
    turn = result.get("turn")
    if not isinstance(turn, dict):
        raise CodexAppServerError("app-server turn/start response missing turn")
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id.strip():
        raise CodexAppServerError("app-server turn/start response missing turn id")
    return turn_id


def _turn_error_message(turn: dict[str, Any]) -> str:
    error = turn.get("error")
    if not isinstance(error, dict):
        return "unknown app-server turn error"
    message = error.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return _safe_json(error)


def is_codex_app_server_live(
    command: str,
    *,
    timeout_seconds: float = 8.0,
) -> tuple[bool, str | None]:
    try:
        with _CodexAppServerClient(
            command=command,
            startup_timeout_seconds=timeout_seconds,
            request_timeout_seconds=timeout_seconds,
        ) as client:
            client.initialize()
            client.request("model/list", params={"includeHidden": False})
    except CodexAppServerError as exc:
        return False, str(exc)
    return True, None


def describe_image_with_codex_app_server(
    image_path: Path,
    *,
    prompt: str,
    command: str,
    model: str | None = None,
    effort: str | None = None,
    timeout_seconds: float = 30.0,
) -> CodexImageDescriptionResult:
    if not image_path.exists():
        raise CodexAppServerError(f"Image path does not exist: {image_path}")
    with _CodexAppServerClient(
        command=command,
        startup_timeout_seconds=min(10.0, timeout_seconds),
        request_timeout_seconds=timeout_seconds,
    ) as client:
        client.initialize()
        start_params: dict[str, Any] = {"cwd": str(Path.cwd())}
        thread_id = _thread_id(client.request("thread/start", params=start_params))
        input_items: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "localImage", "path": str(image_path)},
        ]
        turn_params: dict[str, Any] = {"threadId": thread_id, "input": input_items}
        model_name = (model or "").strip()
        effort_name = (effort or "").strip()
        if model_name:
            turn_params["model"] = model_name
        if effort_name:
            turn_params["effort"] = effort_name
        turn_id = _turn_id(client.request("turn/start", params=turn_params))
        return _collect_turn_text(
            client,
            turn_id=turn_id,
            timeout_seconds=timeout_seconds,
            requested_model=model_name or None,
        )


def _collect_turn_text(
    client: _CodexAppServerClient,
    *,
    turn_id: str,
    timeout_seconds: float,
    requested_model: str | None,
) -> CodexImageDescriptionResult:
    deadline = time.monotonic() + timeout_seconds
    delta_chunks: list[str] = []
    final_text: str | None = None
    rerouted_from_model: str | None = None
    rerouted_to_model: str | None = None
    reroute_reason: str | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CodexAppServerError(
                "Timed out waiting for app-server turn completion"
            )
        event = client.next_event(timeout_seconds=remaining)
        method = event.get("method")
        params = event.get("params")
        if not isinstance(method, str):
            continue
        if method == "item/agentMessage/delta":
            if isinstance(params, dict):
                delta = params.get("delta")
                if isinstance(delta, str):
                    delta_chunks.append(delta)
            continue
        if method == "item/completed":
            if isinstance(params, dict):
                item = params.get("item")
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        final_text = text.strip()
            continue
        if method == "model/rerouted":
            if isinstance(params, dict):
                from_model = params.get("fromModel")
                to_model = params.get("toModel")
                reason = params.get("reason")
                if isinstance(from_model, str) and from_model.strip():
                    rerouted_from_model = from_model.strip()
                if isinstance(to_model, str) and to_model.strip():
                    rerouted_to_model = to_model.strip()
                if isinstance(reason, str) and reason.strip():
                    reroute_reason = reason.strip()
            continue
        if method != "turn/completed" or not isinstance(params, dict):
            continue
        turn = params.get("turn")
        if not isinstance(turn, dict):
            continue
        event_turn_id = turn.get("id")
        if isinstance(event_turn_id, str) and event_turn_id != turn_id:
            continue
        status = turn.get("status")
        if status != "completed":
            raise CodexAppServerError(
                f"app-server turn failed with status {status!r}: {_turn_error_message(turn)}"
            )
        break
    text = final_text or "".join(delta_chunks).strip()
    if not text:
        raise CodexAppServerError("app-server returned empty image description")
    return CodexImageDescriptionResult(
        text=text,
        requested_model=requested_model,
        rerouted_from_model=rerouted_from_model,
        rerouted_to_model=rerouted_to_model,
        reroute_reason=reroute_reason,
    )

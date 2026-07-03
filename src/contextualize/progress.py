from __future__ import annotations

from collections import Counter
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import threading
from typing import Any


_CURRENT_CONTEXT: ContextVar[str | None] = ContextVar(
    "contextualize_progress_context", default=None
)
_EVENTS: list["ProgressEvent"] = []
_LOCK = threading.Lock()
_LIVE_LOCK = threading.Lock()
_LIVE_REPORTER: "_RichProgressReporter | None" = None


@dataclass(frozen=True)
class ProgressEvent:
    provider: str
    operation: str
    outcome: str
    target: str | None = None
    detail: str | None = None
    count: int | None = None
    size_bytes: int | None = None
    context: str | None = None
    timestamp: str = ""


def reset_progress() -> None:
    with _LOCK:
        _EVENTS.clear()
    reporter = _current_reporter()
    if reporter is not None:
        reporter.reset()


def set_live_progress(
    enabled: bool,
    *,
    force_terminal: bool | None = None,
    transient: bool = False,
) -> None:
    global _LIVE_REPORTER
    reporter_to_stop: _RichProgressReporter | None = None
    with _LIVE_LOCK:
        if enabled:
            if _LIVE_REPORTER is not None:
                return
            reporter = _RichProgressReporter(
                force_terminal=force_terminal,
                transient=transient,
            )
            if reporter.start():
                _LIVE_REPORTER = reporter
            return
        reporter_to_stop = _LIVE_REPORTER
        _LIVE_REPORTER = None
    if reporter_to_stop is not None:
        reporter_to_stop.stop()


def write_progress_log(line: str) -> bool:
    reporter = _current_reporter()
    if reporter is None:
        return False
    reporter.write(line)
    return True


def live_progress_active() -> bool:
    return _current_reporter() is not None


def set_progress_context(name: str | None) -> Token[str | None]:
    value = name.strip() if isinstance(name, str) and name.strip() else None
    return _CURRENT_CONTEXT.set(value)


def reset_progress_context(token: Token[str | None]) -> None:
    _CURRENT_CONTEXT.reset(token)


def record_progress(
    provider: str,
    operation: str,
    outcome: str,
    *,
    target: str | None = None,
    detail: str | None = None,
    count: int | None = None,
    size_bytes: int | None = None,
) -> None:
    event = ProgressEvent(
        provider=_normalize(provider, "unknown"),
        operation=_normalize(operation, "operation"),
        outcome=_normalize(outcome, "event"),
        target=_clean_optional(target),
        detail=_clean_optional(detail),
        count=count if isinstance(count, int) else None,
        size_bytes=size_bytes if isinstance(size_bytes, int) else None,
        context=_CURRENT_CONTEXT.get(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    with _LOCK:
        _EVENTS.append(event)
    reporter = _current_reporter()
    if reporter is not None:
        reporter.record(event)


def log_progress(
    provider: str,
    operation: str,
    outcome: str,
    *,
    target: str | None = None,
    detail: str | None = None,
    count: int | None = None,
    size_bytes: int | None = None,
) -> None:
    record_progress(
        provider,
        operation,
        outcome,
        target=target,
        detail=detail,
        count=count,
        size_bytes=size_bytes,
    )


def progress_summary_lines() -> list[str]:
    with _LOCK:
        events = list(_EVENTS)
    if not events:
        return []

    groups: dict[tuple[str | None, str, str], list[ProgressEvent]] = {}
    for event in events:
        key = (event.context, event.provider, event.operation)
        groups.setdefault(key, []).append(event)
    lines = ["Progress summary:"]
    current_context: str | None | object = object()
    for (context, provider, operation), group_events in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0] or "",
            item[0][1],
            item[0][2],
        ),
    ):
        parts = _summary_parts(group_events)
        if not parts:
            continue
        if context != current_context:
            current_context = context
            if context:
                lines.append(f"  context {context}:")
        prefix = "    " if context else "  "
        lines.append(f"{prefix}{provider} {operation}: {' '.join(parts)}")
    return lines


def flush_progress_summary(*, enabled: bool) -> None:
    if not enabled:
        return
    lines = progress_summary_lines()
    if not lines:
        return
    print("\n".join(lines), file=sys.stderr, flush=True)


def _current_reporter() -> "_RichProgressReporter | None":
    with _LIVE_LOCK:
        return _LIVE_REPORTER


@dataclass
class _LiveTaskState:
    task_id: Any
    total: int | None = None
    completed: int = 0
    counts: Counter[str] | None = None
    item_count: int = 0
    byte_count: int = 0
    latest: str = ""
    latest_failure: str = ""

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = Counter()


class _RichProgressReporter:
    def __init__(self, *, force_terminal: bool | None, transient: bool) -> None:
        self.force_terminal = force_terminal
        self.transient = transient
        self._lock = threading.Lock()
        self._progress: Any = None
        self._tasks: dict[tuple[str | None, str, str], _LiveTaskState] = {}

    def start(self) -> bool:
        if self.force_terminal is not True and not sys.stderr.isatty():
            return False
        try:
            from rich.console import Console
            from rich.progress import BarColumn
            from rich.progress import Progress
            from rich.progress import SpinnerColumn
            from rich.progress import TextColumn
            from rich.progress import TimeElapsedColumn
            from rich.progress import TimeRemainingColumn
        except ImportError:
            return False

        console = Console(
            file=sys.stderr,
            stderr=True,
            force_terminal=self.force_terminal,
        )
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.fields[units]}", justify="right"),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=self.transient,
            redirect_stderr=True,
            redirect_stdout=False,
            refresh_per_second=8,
        )
        self._progress.start()
        return True

    def stop(self) -> None:
        progress = self._progress
        if progress is None:
            return
        with self._lock:
            self._progress = None
            self._tasks.clear()
        progress.stop()

    def reset(self) -> None:
        with self._lock:
            if self._progress is not None:
                for task in self._tasks.values():
                    self._progress.remove_task(task.task_id)
            self._tasks.clear()

    def write(self, line: str) -> None:
        progress = self._progress
        if progress is None:
            return
        text = str(line)
        if not text:
            return
        with self._lock:
            for part in text.splitlines() or [text]:
                progress.console.print(part, markup=False)

    def record(self, event: ProgressEvent) -> None:
        progress = self._progress
        if progress is None:
            return
        key = _task_key(event)
        with self._lock:
            state = self._tasks.get(key)
            if state is None:
                state = self._add_task(event)
                self._tasks[key] = state
            self._apply_event(state, event)
            progress.update(
                state.task_id,
                total=state.total,
                completed=state.completed,
                units=_format_units(state),
                stats=_format_task_stats(state),
            )

    def _add_task(self, event: ProgressEvent) -> _LiveTaskState:
        task_id = self._progress.add_task(
            _task_description(event),
            total=None,
            units="events 0",
            stats="",
        )
        return _LiveTaskState(task_id=task_id)

    def _apply_event(self, state: _LiveTaskState, event: ProgressEvent) -> None:
        if event.outcome == "total":
            state.total = max(event.count or 0, 0)
            state.completed = 0
        elif event.outcome == "start":
            target_total = _target_total_from_detail(event.detail)
            if target_total is not None:
                state.total = target_total
                state.completed = 0
                state.counts = Counter()
                state.item_count = 0
                state.byte_count = 0
                state.latest = ""
        elif event.outcome == "done":
            if event.operation in {"embedded-list", "embedded-resolve"}:
                state.completed = state.total or state.completed
            else:
                state.completed += 1
        elif event.outcome in {
            "processed",
            "cache_hit",
            "cache_miss",
            "failed",
            "partial",
        }:
            state.completed += 1
            if event.outcome in {"failed", "partial"}:
                state.latest_failure = _format_failure_detail(event)

        if state.total is not None:
            state.completed = min(state.completed, state.total)

        if event.outcome not in {"total", "start"} and not (
            event.provider == "hydrate"
            and event.operation == "file"
            and event.outcome == "done"
        ):
            assert state.counts is not None
            state.counts[event.outcome] += 1
        if event.count is not None and event.outcome not in {"total", "done"}:
            state.item_count += event.count
        if event.size_bytes is not None:
            state.byte_count += event.size_bytes
        state.latest = _format_event_target(event)


def _task_key(event: ProgressEvent) -> tuple[str | None, str, str]:
    operation = "files" if event.operation == "file" else event.operation
    return (event.context, event.provider, operation)


def _summary_parts(events: list[ProgressEvent]) -> list[str]:
    total_value: int | None = None
    counters: Counter[str] = Counter()
    item_count = 0
    byte_count = 0
    start_count = 0
    latest_failure = ""
    for event in events:
        if event.outcome == "total":
            total_value = event.count
            continue
        if event.outcome == "start":
            start_count += 1
            continue
        if event.outcome == "failed":
            latest_failure = _format_failure_detail(event)
        if not (
            event.provider == "hydrate"
            and event.operation == "file"
            and event.outcome == "done"
        ):
            counters[event.outcome] += 1
        if event.count is not None and event.outcome != "done":
            item_count += event.count
        if event.size_bytes is not None:
            byte_count += event.size_bytes

    parts: list[str] = []
    if total_value is not None:
        parts.append(f"total={total_value}")
    for outcome in _ordered_outcomes(counters):
        parts.append(f"{outcome}={counters[outcome]}")
    if item_count:
        parts.append(f"items={item_count}")
    if byte_count:
        parts.append(f"bytes={_format_bytes(byte_count)}")
    if latest_failure:
        parts.append(f"latest_failure={_truncate(latest_failure, 180)}")
    if not parts and start_count:
        parts.append(f"start={start_count}")
    return parts


def _ordered_outcomes(counters: Counter[str]) -> list[str]:
    preferred = [
        "cache_hit",
        "cache_miss",
        "processed",
        "failed",
        "partial",
        "done",
    ]
    ordered = [outcome for outcome in preferred if counters.get(outcome)]
    ordered.extend(sorted(outcome for outcome in counters if outcome not in preferred))
    return ordered


def _task_description(event: ProgressEvent) -> str:
    if event.provider == "hydrate" and event.operation == "context":
        return "hydrate contexts"
    label = _operation_label(event.provider, event.operation)
    return f"{event.context} {label}" if event.context else label


def _operation_label(provider: str, operation: str) -> str:
    if provider == "hydrate" and operation == "component":
        return "components"
    if provider == "hydrate" and operation == "file":
        return "files"
    return f"{provider} {operation.replace('-', ' ')}"


def _target_total_from_detail(detail: str | None) -> int | None:
    if not detail:
        return None
    marker = "targets="
    if marker not in detail:
        return None
    raw = detail.split(marker, 1)[1].split(None, 1)[0]
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(value, 0)


def _format_units(state: _LiveTaskState) -> str:
    if state.total is not None:
        return f"{state.completed}/{state.total}"
    return f"events {state.completed}"


def _format_task_stats(state: _LiveTaskState) -> str:
    parts: list[str] = []
    counts = state.counts or Counter()
    for outcome in ("cache_hit", "cache_miss", "processed", "failed", "partial", "done"):
        total = counts.get(outcome, 0)
        if total:
            parts.append(f"{_outcome_label(outcome)}={total}")
    if state.item_count:
        parts.append(f"items={state.item_count}")
    if state.byte_count:
        parts.append(f"bytes={_format_bytes(state.byte_count)}")
    if state.latest:
        parts.append(f"latest={state.latest}")
    if state.latest_failure:
        parts.append(f"failure={_truncate(state.latest_failure, 180)}")
    return " ".join(parts)


def _format_event_target(event: ProgressEvent) -> str:
    value = event.target or event.detail or ""
    return _truncate(value, 56) if value else ""


def _format_failure_detail(event: ProgressEvent) -> str:
    target = event.target or ""
    detail = event.detail or ""
    if target and detail:
        return f"{target}: {detail}"
    return target or detail


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)}{unit}"
            return f"{amount:.1f}{unit}"
        amount /= 1024


def _outcome_label(outcome: str) -> str:
    return {
        "cache_hit": "hit",
        "cache_miss": "miss",
        "failed": "fail",
    }.get(outcome, outcome)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return "." * limit
    return value[: limit - 3] + "..."


def _normalize(value: str, fallback: str) -> str:
    cleaned = str(value or "").strip().replace(" ", "-")
    return cleaned or fallback


def _clean_optional(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None

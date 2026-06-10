from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar, copy_context
import threading
from typing import Any, Callable

_MEDIA_SEMAPHORE_LOCK = threading.Lock()
_MEDIA_SEMAPHORE: tuple[int, threading.BoundedSemaphore] | None = None
_MEDIA_SEMAPHORE_DEPTH: ContextVar[int] = ContextVar(
    "contextualize_media_semaphore_depth", default=0
)


def run_indexed_tasks_fail_fast(
    tasks: list[tuple[int, Callable[[], Any]]],
    *,
    max_workers: int,
    semaphore: threading.BoundedSemaphore | None = None,
) -> list[tuple[int, Any]]:
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [(index, _run_with_semaphore(task, semaphore)) for index, task in tasks]

    results: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                copy_context().run,
                _run_with_semaphore,
                task,
                semaphore,
            ): index
            for index, task in tasks
        }
        try:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
        except Exception:
            for future in future_to_index:
                future.cancel()
            raise

    return [(index, results[index]) for index in sorted(results)]


def media_task_semaphore() -> threading.BoundedSemaphore:
    from .runtime import get_payload_media_jobs

    jobs = get_payload_media_jobs()
    global _MEDIA_SEMAPHORE
    with _MEDIA_SEMAPHORE_LOCK:
        if _MEDIA_SEMAPHORE is None or _MEDIA_SEMAPHORE[0] != jobs:
            _MEDIA_SEMAPHORE = (jobs, threading.BoundedSemaphore(jobs))
        return _MEDIA_SEMAPHORE[1]


def _run_with_semaphore(
    task: Callable[[], Any],
    semaphore: threading.BoundedSemaphore | None,
) -> Any:
    if semaphore is None:
        return task()
    depth = _MEDIA_SEMAPHORE_DEPTH.get()
    if depth > 0:
        return task()
    with semaphore:
        token = _MEDIA_SEMAPHORE_DEPTH.set(depth + 1)
        try:
            return task()
        finally:
            _MEDIA_SEMAPHORE_DEPTH.reset(token)

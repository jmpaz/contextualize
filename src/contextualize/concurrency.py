from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, Callable


def run_indexed_tasks_fail_fast(
    tasks: list[tuple[int, Callable[[], Any]]],
    *,
    max_workers: int,
) -> list[tuple[int, Any]]:
    if not tasks:
        return []
    if max_workers <= 1 or len(tasks) == 1:
        return [(index, task()) for index, task in tasks]

    results: dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(copy_context().run, task): index for index, task in tasks
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

from __future__ import annotations

from contextualize.concurrency import run_indexed_tasks_fail_fast


def test_run_indexed_tasks_reports_completion() -> None:
    seen: list[tuple[int, str]] = []

    results = run_indexed_tasks_fail_fast(
        [
            (0, lambda: "a"),
            (1, lambda: "b"),
        ],
        max_workers=1,
        on_complete=lambda index, result: seen.append((index, result)),
    )

    assert seen == [(0, "a"), (1, "b")]
    assert results == [(0, "a"), (1, "b")]

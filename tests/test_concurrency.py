from __future__ import annotations

import threading

from contextualize.concurrency import (
    download_lane,
    media_task_semaphore,
    run_indexed_tasks_fail_fast,
    transcription_lane,
)
from contextualize.runtime import (
    reset_transcription_jobs,
    set_payload_media_jobs,
    reset_payload_media_jobs,
    set_transcription_jobs,
)


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


def test_lane_rebuilds_when_job_count_changes() -> None:
    token = set_transcription_jobs(3)
    try:
        with transcription_lane():
            pass
        first = _lane_object("transcription")
        with transcription_lane():
            pass
        assert _lane_object("transcription") is first
    finally:
        reset_transcription_jobs(token)

    token = set_transcription_jobs(5)
    try:
        with transcription_lane():
            pass
        assert _lane_object("transcription") is not first
    finally:
        reset_transcription_jobs(token)


def test_lanes_are_independent() -> None:
    assert _lane_object_for(transcription_lane) is not _lane_object_for(download_lane)


def test_lane_bounds_peak_occupancy() -> None:
    token = set_transcription_jobs(2)
    lock = threading.Lock()
    active = 0
    peak = 0

    def _work() -> None:
        nonlocal active, peak
        with transcription_lane():
            with lock:
                active += 1
                peak = max(peak, active)
            threading.Event().wait(0.01)
            with lock:
                active -= 1

    try:
        run_indexed_tasks_fail_fast(
            [(index, _work) for index in range(8)],
            max_workers=8,
        )
    finally:
        reset_transcription_jobs(token)

    assert peak <= 2


def test_nested_media_semaphore_does_not_deadlock() -> None:
    token = set_payload_media_jobs(1)
    try:

        def _inner() -> str:
            return "inner"

        def _outer() -> str:
            results = run_indexed_tasks_fail_fast(
                [(0, _inner), (1, _inner)],
                max_workers=2,
                semaphore=media_task_semaphore(),
            )
            return ",".join(value for _, value in results)

        results = run_indexed_tasks_fail_fast(
            [(0, _outer), (1, _outer)],
            max_workers=2,
            semaphore=media_task_semaphore(),
        )
    finally:
        reset_payload_media_jobs(token)

    assert [value for _, value in results] == ["inner,inner", "inner,inner"]


def _lane_object(name: str):
    from contextualize import concurrency

    return concurrency._LANES[name][1]


def _lane_object_for(lane):
    with lane():
        pass
    from contextualize import concurrency

    if lane is transcription_lane:
        return concurrency._LANES["transcription"][1]
    return concurrency._LANES["download"][1]

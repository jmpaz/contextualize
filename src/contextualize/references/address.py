"""Mark addresses: `<target>@<start>[-<end>]` with clock-time suffixes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_CLOCK_TIME_RE = re.compile(
    r"^(?:(?P<hours>\d{1,2}):(?P<h_minutes>[0-5]\d)|(?P<minutes>\d{1,2}))"
    r":(?P<seconds>[0-5]\d(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class MarkAddress:
    base: str
    start_seconds: float
    end_seconds: float | None
    authored: str

    def as_span(self) -> dict[str, Any]:
        return {
            "start": self.start_seconds,
            "end": self.end_seconds,
            "authored": self.authored,
        }


def parse_clock_time(text: str) -> float | None:
    match = _CLOCK_TIME_RE.match(text)
    if match is None:
        return None
    seconds = float(match.group("seconds"))
    if match.group("hours") is not None:
        return (
            int(match.group("hours")) * 3600
            + int(match.group("h_minutes")) * 60
            + seconds
        )
    return int(match.group("minutes")) * 60 + seconds


def parse_time_range(text: str) -> tuple[float, float | None] | None:
    start_text, sep, end_text = text.partition("-")
    if not sep:
        start = parse_clock_time(text)
        return None if start is None else (start, None)
    start = parse_clock_time(start_text)
    end = parse_clock_time(end_text)
    if start is None or end is None:
        return None
    return start, end


def split_mark_address(target: str) -> tuple[str, MarkAddress | None]:
    cursor = len(target)
    while (cursor := target.rfind("@", 0, cursor)) > 0:
        remainder = target[cursor + 1 :]
        span = parse_time_range(remainder)
        if span is None:
            continue
        base = target[:cursor]
        return base, MarkAddress(
            base=base,
            start_seconds=span[0],
            end_seconds=span[1],
            authored=remainder,
        )
    return target, None


def format_clock_time(seconds: float) -> str:
    total = round(float(seconds), 3)
    whole = int(total)
    frac = round(total - whole, 3)
    if frac >= 1.0:
        whole += 1
        frac = 0.0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    sec_text = f"{secs:02d}"
    if frac:
        sec_text += f"{frac:.3f}".rstrip("0")[1:]
    if hours:
        return f"{hours}:{minutes:02d}:{sec_text}"
    return f"{minutes}:{sec_text}"

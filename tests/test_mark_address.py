"""Mark address grammar (spec 2026-07-09-marks §4.2).

Colon-required clock times guard the split: an `@` whose remainder is not
`M:SS` / `MM:SS` / `H:MM:SS` (decimals in seconds only, optional `-end`)
belongs to the target.
"""

import pytest

from contextualize.manifest import coerce_mark_spec
from contextualize.references.address import (
    format_clock_time,
    parse_clock_time,
    parse_time_range,
    split_mark_address,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("4:12", 252.0),
        ("04:12", 252.0),
        ("0:59", 59.0),
        ("12:04", 724.0),
        ("99:59", 5999.0),
        ("108:20", 6500.0),
        ("1:02:03", 3723.0),
        ("10:00:00", 36000.0),
        ("4:12.5", 252.5),
        ("1:02:03.25", 3723.25),
    ],
)
def test_parse_clock_time(text, seconds):
    assert parse_clock_time(text) == seconds


@pytest.mark.parametrize(
    "text",
    [
        "",
        "4",
        "412",
        "4:5",
        "4:60",
        "4:123",
        "004:12",
        "100:00:00",
        "1:60:00",
        "1:2:03",
        "4:12.",
        "4.5:12",
        "-4:12",
        " 4:12",
        "4:12 ",
        "24m49s",
        "2x.png",
        "example.com",
        "host/path",
    ],
)
def test_parse_clock_time_rejects(text):
    assert parse_clock_time(text) is None


def test_parse_time_range_point_and_range():
    assert parse_time_range("4:12") == (252.0, None)
    assert parse_time_range("12:04-13:26") == (724.0, 806.0)
    assert parse_time_range("0:04-0:14.25") == (4.0, 14.25)
    assert parse_time_range("108:20-108:45") == (6500.0, 6525.0)


@pytest.mark.parametrize(
    "text",
    ["4:12-", "-4:12", "4:12-oops", "4:12-13:26-14:00", "banana"],
)
def test_parse_time_range_rejects(text):
    assert parse_time_range(text) is None


def test_split_range_address():
    target = "store:voice/2026-07-07/12-34-52.m4a@12:04-13:26"
    base, mark = split_mark_address(target)
    assert base == "store:voice/2026-07-07/12-34-52.m4a"
    assert mark is not None
    assert mark.base == base
    assert mark.start_seconds == 724.0
    assert mark.end_seconds == 806.0
    assert mark.authored == "12:04-13:26"


def test_split_point_address_preserves_authored_form():
    base, mark = split_mark_address("store:voice/2026-07-06/06-37-21.m4a@8:40")
    assert base == "store:voice/2026-07-06/06-37-21.m4a"
    assert mark.start_seconds == 520.0
    assert mark.end_seconds is None
    assert mark.authored == "8:40"


def test_split_keeps_params_in_base():
    base, mark = split_mark_address("store:voice/a.m4a?after=2w@4:12")
    assert base == "store:voice/a.m4a?after=2w"
    assert mark.authored == "4:12"


def test_split_local_path_address():
    base, mark = split_mark_address("notes/op9f.md@4:12")
    assert base == "notes/op9f.md"
    assert mark.start_seconds == 252.0


def test_split_takes_last_at_that_parses():
    base, mark = split_mark_address("store:voice/a@b.m4a@4:12")
    assert base == "store:voice/a@b.m4a"
    assert mark.authored == "4:12"


@pytest.mark.parametrize(
    "target",
    [
        "someone@example.com",
        "mailto:someone@example.com",
        "foo@2x.png",
        "https://user:pass@host/path",
        "store:voice/a@b.m4a",
        "store:voice/a@42",
        "store:voice/a@4:12b",
        "store:keys/prod@2024",
        "a@4:12@b",
        "@4:12",
    ],
)
def test_split_leaves_non_time_at_alone(target):
    base, mark = split_mark_address(target)
    assert base == target
    assert mark is None


def test_split_url_native_time_params_not_recognized():
    target = "https://youtu.be/X?t=24m49s"
    assert split_mark_address(target) == (target, None)


def test_split_url_with_time_shaped_suffix_splits():
    base, mark = split_mark_address("https://youtu.be/X@4:12")
    assert base == "https://youtu.be/X"
    assert mark.authored == "4:12"


def test_as_span_shape():
    _, mark = split_mark_address("store:a.m4a@0:04-0:14.25")
    assert mark.as_span() == {"start": 4.0, "end": 14.25, "authored": "0:04-0:14.25"}


@pytest.mark.parametrize(
    ("seconds", "text"),
    [
        (0, "0:00"),
        (4.0, "0:04"),
        (59, "0:59"),
        (252, "4:12"),
        (252.5, "4:12.5"),
        (630, "10:30"),
        (3723, "1:02:03"),
        (5400, "1:30:00"),
        (14.25, "0:14.25"),
    ],
)
def test_format_clock_time(seconds, text):
    assert format_clock_time(seconds) == text


@pytest.mark.parametrize("seconds", [0.0, 59.0, 252.5, 3723.25, 5999.0, 6500.0])
def test_format_round_trips_through_parse(seconds):
    assert parse_clock_time(format_clock_time(seconds)) == seconds


def test_coerce_point_mark():
    record = coerce_mark_spec({"at": "4:12"})
    assert record == {
        "authored": "4:12",
        "start_seconds": 252.0,
        "end_seconds": None,
        "quote": None,
        "refs": [],
        "problem": None,
    }


def test_coerce_span_alias():
    record = coerce_mark_spec({"span": "12:04-13:26"})
    assert record["start_seconds"] == 724.0
    assert record["end_seconds"] == 806.0
    assert record["authored"] == "12:04-13:26"
    assert record["problem"] is None


def test_coerce_preserves_authored_string_verbatim():
    assert coerce_mark_spec({"at": "04:12"})["authored"] == "04:12"
    assert coerce_mark_spec({"at": " 0:59 "})["authored"] == "0:59"


def test_coerce_sexagesimal_int_renders_authored_form():
    record = coerce_mark_spec({"at": 252})
    assert record["start_seconds"] == 252.0
    assert record["authored"] == "4:12"
    assert record["problem"] is None


def test_coerce_sexagesimal_float_keeps_decimals():
    record = coerce_mark_spec({"at": 252.5})
    assert record["start_seconds"] == 252.5
    assert record["authored"] == "4:12.5"


def test_coerce_quote_with_range():
    record = coerce_mark_spec(
        {"at": "12:04-13:26", "quote": "Two prompts.\n", "refs": ["notes/op9f.md"]}
    )
    assert record["problem"] is None
    assert record["quote"] == "Two prompts.\n"
    assert record["refs"] == ["notes/op9f.md"]


def test_coerce_quote_with_elapsed_minutes_over_an_hour():
    record = coerce_mark_spec({"at": "108:20-108:45", "quote": "Embodied text."})
    assert record["start_seconds"] == 6500.0
    assert record["end_seconds"] == 6525.0
    assert record["problem"] is None


def test_coerce_quote_on_point_is_a_problem_not_a_raise():
    record = coerce_mark_spec({"at": "4:12", "quote": "solo"})
    assert record["problem"] == "mark-quote-requires-range"
    assert record["start_seconds"] == 252.0
    assert record["quote"] == "solo"


def test_coerce_at_and_span_together():
    record = coerce_mark_spec({"at": "4:12", "span": "5:00-6:00", "quote": "kept"})
    assert record["problem"] == "mark-at-and-span"
    assert record["quote"] == "kept"


def test_coerce_missing_time():
    assert coerce_mark_spec({})["problem"] == "mark-missing-time"
    assert coerce_mark_spec({"at": None})["problem"] == "mark-missing-time"
    assert coerce_mark_spec({"quote": "x"})["problem"] == "mark-missing-time"


@pytest.mark.parametrize(
    "value",
    ["banana", "4:12-", True, -30, ["4:12"], "13:26-12:04"],
)
def test_coerce_unusable_time_values(value):
    record = coerce_mark_spec({"at": value})
    assert record["problem"] == "mark-invalid-time"
    assert record["start_seconds"] is None


def test_coerce_non_mapping_mark():
    record = coerce_mark_spec("4:12")
    assert record["problem"] == "mark-invalid"
    assert record["authored"] == "4:12"


def test_coerce_refs_forms():
    assert coerce_mark_spec({"at": "4:12", "refs": "dev/mood-kit"})["refs"] == [
        "dev/mood-kit"
    ]
    assert coerce_mark_spec({"at": "4:12", "refs": ["a", "", " b "]})["refs"] == [
        "a",
        "b",
    ]
    assert coerce_mark_spec({"at": "4:12"})["refs"] == []

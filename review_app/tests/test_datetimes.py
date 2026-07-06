"""Locks down review_app/display/_datetimes — the single source of truth
for ISO → America/Chicago timestamp formatting across the review app.

Tests cover the exact invariants we don't want to drift:

  * Display TZ is America/Chicago — same string the source-data table
    uses (``source_renderer._DISPLAY_TZ``) and the same one the
    canonical pipeline mirrors (``icu_pause/rendering/citations.py``).
  * Naive ISO inputs are treated as UTC (matches the cite-tag builder
    convention; the pipeline writes ``row.time`` as raw-UTC ISO with no
    tzinfo suffix in some shapes).
  * UTC → display conversion handles BOTH DST (CDT, UTC-5) and standard
    time (CST, UTC-6) — the 5-hour shift the user hit in summer becomes
    a 6-hour shift in winter; both must round-trip cleanly.
  * format_compact and format_short both run off the same parsed
    display-TZ datetime, so the source-table M-DD HH:MM and the
    citation-tooltip Mon DD HH:MM can't disagree about the hour shown.
  * Unparseable / empty / None inputs return None — no exceptions.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_REVIEW_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_REVIEW_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REVIEW_APP_ROOT))

from display._datetimes import (  # noqa: E402
    DISPLAY_TZ,
    format_compact,
    format_short,
    iso_to_compact_display,
    iso_to_short_display,
    parse_iso_to_display,
)


# ---------------------------------------------------------------------------
# DISPLAY_TZ — pinned to America/Chicago; mirrored in source_renderer
# and the canonical citations module. Drift breaks the
# tooltip / source-table parity.
# ---------------------------------------------------------------------------


def test_display_tz_is_america_chicago():
    assert DISPLAY_TZ == ZoneInfo("America/Chicago")


def test_display_tz_matches_source_renderer_constant():
    # Drift guard: source_renderer's display TZ must be the same object
    # as the shared one (it imports the shared constant).
    from display import source_renderer
    assert source_renderer._DISPLAY_TZ is DISPLAY_TZ


# ---------------------------------------------------------------------------
# Naive ISO → assumed-UTC → display TZ.
# This is the convention used by the pipeline's cite-tag builder; if it
# changes here, _compact_dttm in source_renderer would also need to
# change. The constants are co-defined here so both surfaces agree.
# ---------------------------------------------------------------------------


def test_naive_iso_is_treated_as_utc():
    # 2024-07-01T04:00:00 (naive) interpreted as UTC = 2024-06-30 23:00 CDT.
    # This is the exact case the user surfaced in the user-reported bug:
    # vital recorded as "(vital 6-30 23:00)" in the source table, while
    # the pre-fix tooltip rendered "Jul 01 04:00" — same instant.
    dt = parse_iso_to_display("2024-07-01T04:00:00")
    assert dt is not None
    assert dt.tzinfo is not None
    # 2024-06-30 23:00 CDT (UTC-5 in July).
    assert dt.year == 2024
    assert dt.month == 6
    assert dt.day == 30
    assert dt.hour == 23
    assert dt.minute == 0


def test_explicit_utc_suffix_iso_converts_to_display():
    dt = parse_iso_to_display("2024-07-01T04:00:00+00:00")
    assert dt is not None
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2024, 6, 30, 23, 0)


def test_explicit_utc_z_suffix_iso_not_supported_by_fromisoformat_falls_back():
    # Python 3.10's datetime.fromisoformat rejects "Z" suffix; the helper
    # falls back to None. Caller's responsibility to normalize Z → +00:00
    # at the data layer if it ever shows up. Pinned here so the failure
    # mode is documented, not surprising.
    # (Python 3.11+ accepts Z; this assertion is permissive — either
    # parse correctly or return None, but never raise.)
    try:
        dt = parse_iso_to_display("2024-07-01T04:00:00Z")
    except Exception as exc:
        raise AssertionError(
            f"parse_iso_to_display must never raise; got {exc!r}"
        )
    # If parsed, it must convert to display TZ; if not, None.
    if dt is not None:
        assert (dt.month, dt.day, dt.hour) == (6, 30, 23)


def test_space_separated_iso_tolerated():
    # Source data sometimes serializes "2024-07-01 04:00:00" (space, not
    # 'T'). _compact_dttm tolerates this and so must the shared helper.
    dt = parse_iso_to_display("2024-07-01 04:00:00")
    assert dt is not None
    assert (dt.month, dt.day, dt.hour) == (6, 30, 23)


# ---------------------------------------------------------------------------
# DST boundary correctness — CDT (-5) vs CST (-6).
# ---------------------------------------------------------------------------


def test_summer_utc_to_cdt_is_minus_five():
    # July is firmly inside CDT (UTC-5). 12:00 UTC → 07:00 CDT.
    dt = parse_iso_to_display("2024-07-15T12:00:00+00:00")
    assert (dt.month, dt.day, dt.hour) == (7, 15, 7)
    # And the same instant via space-separated naive form.
    dt2 = parse_iso_to_display("2024-07-15 12:00:00")
    assert (dt2.month, dt2.day, dt2.hour) == (7, 15, 7)


def test_winter_utc_to_cst_is_minus_six():
    # January is CST (UTC-6). 12:00 UTC → 06:00 CST.
    dt = parse_iso_to_display("2024-01-15T12:00:00+00:00")
    assert (dt.month, dt.day, dt.hour) == (1, 15, 6)


def test_dst_spring_forward_boundary():
    # 2024 spring-forward: 2024-03-10 02:00 CST jumps to 03:00 CDT.
    # 08:00 UTC = 02:00 CST -> doesn't exist locally; +1h => 03:00 CDT.
    # Just before transition (07:00 UTC = 01:00 CST) and after
    # (09:00 UTC = 04:00 CDT) must convert correctly.
    before = parse_iso_to_display("2024-03-10T07:00:00+00:00")
    after = parse_iso_to_display("2024-03-10T09:00:00+00:00")
    assert (before.month, before.day, before.hour) == (3, 10, 1)
    assert (after.month, after.day, after.hour) == (3, 10, 4)


def test_dst_fall_back_boundary():
    # 2024 fall-back: 2024-11-03 02:00 CDT goes to 01:00 CST.
    # 06:00 UTC = 01:00 CDT (before fold); 07:00 UTC = 01:00 CST (after).
    pre_fold = parse_iso_to_display("2024-11-03T06:00:00+00:00")
    post_fold = parse_iso_to_display("2024-11-03T07:00:00+00:00")
    # Both render as "01:00" locally but with different fold values.
    assert (pre_fold.month, pre_fold.day, pre_fold.hour) == (11, 3, 1)
    assert (post_fold.month, post_fold.day, post_fold.hour) == (11, 3, 1)


# ---------------------------------------------------------------------------
# Format helpers — both run off the same parsed display-TZ datetime so
# they can't disagree about the displayed hour.
# ---------------------------------------------------------------------------


def test_format_compact_emits_m_dash_dd_hh_mm():
    dt = datetime(2024, 6, 30, 23, 0, tzinfo=DISPLAY_TZ)
    assert format_compact(dt) == "6-30 23:00"


def test_format_compact_zero_pads_day_and_hour_not_month():
    # Month is single-digit (no zero pad), day and hour are
    # zero-padded — matches the cite-tag format from data/context.py.
    dt = datetime(2024, 3, 7, 9, 5, tzinfo=DISPLAY_TZ)
    assert format_compact(dt) == "3-07 09:05"


def test_format_short_emits_mon_dd_hh_mm():
    dt = datetime(2024, 6, 30, 23, 0, tzinfo=DISPLAY_TZ)
    assert format_short(dt) == "Jun 30 23:00"


def test_format_short_uses_3_letter_month_names():
    cases = [
        (1, "Jan"), (2, "Feb"), (3, "Mar"), (4, "Apr"),
        (5, "May"), (6, "Jun"), (7, "Jul"), (8, "Aug"),
        (9, "Sep"), (10, "Oct"), (11, "Nov"), (12, "Dec"),
    ]
    for month, expected in cases:
        dt = datetime(2024, month, 15, 12, 30, tzinfo=DISPLAY_TZ)
        assert format_short(dt).startswith(expected + " 15 ")


def test_compact_and_short_agree_on_hour_for_same_instant():
    # The whole point of the shared helper: both formatters reflect the
    # SAME display-TZ hour for a given ISO input, so the source table
    # and citation tooltip can't drift.
    iso = "2024-07-01T04:00:00+00:00"  # UTC -> 2024-06-30 23:00 CDT
    compact = iso_to_compact_display(iso)
    short = iso_to_short_display(iso)
    assert compact == "6-30 23:00"
    assert short == "Jun 30 23:00"
    # Hour and minute components match between the two forms.
    assert compact.split()[-1] == short.split()[-1] == "23:00"


# ---------------------------------------------------------------------------
# Defensive cases — empty, None, garbage inputs must return None
# (callers fall back to passing the raw value through unchanged).
# ---------------------------------------------------------------------------


def test_none_input_returns_none():
    assert parse_iso_to_display(None) is None
    assert iso_to_compact_display(None) is None
    assert iso_to_short_display(None) is None


def test_empty_string_returns_none():
    assert parse_iso_to_display("") is None
    assert iso_to_compact_display("") is None
    assert iso_to_short_display("") is None


def test_unparseable_string_returns_none_does_not_raise():
    # Garbage in, None out; never raises.
    for garbage in ("not-a-date", "yesterday", "2024-13-99", "??:??"):
        try:
            result = parse_iso_to_display(garbage)
        except Exception as exc:
            raise AssertionError(
                f"parse_iso_to_display({garbage!r}) must not raise; got {exc!r}"
            )
        assert result is None, f"expected None for {garbage!r}, got {result!r}"


def test_already_tz_aware_datetime_passes_through_after_conversion():
    # Aware datetime in UTC → converted to display TZ.
    dt_utc = datetime(2024, 7, 1, 4, 0, tzinfo=timezone.utc)
    dt_local = parse_iso_to_display(dt_utc)
    assert dt_local is not None
    assert dt_local.tzinfo == DISPLAY_TZ
    assert (dt_local.month, dt_local.day, dt_local.hour) == (6, 30, 23)


def test_already_display_tz_aware_datetime_passes_through_unchanged():
    dt_local_in = datetime(2024, 7, 1, 12, 0, tzinfo=DISPLAY_TZ)
    dt_local_out = parse_iso_to_display(dt_local_in)
    assert dt_local_out is not None
    # Same wall-clock hour after astimezone(DISPLAY_TZ) — no shift.
    assert (
        dt_local_out.year, dt_local_out.month, dt_local_out.day,
        dt_local_out.hour, dt_local_out.minute,
    ) == (2024, 7, 1, 12, 0)

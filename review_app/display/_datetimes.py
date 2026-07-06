"""Single source of truth for display-tz timestamp formatting in review_app.

All review-app surfaces that render an ISO timestamp to a clinician
(source-data table, citation tooltips, notes-window strap line, etc.)
must go through this module so they agree on the display timezone. The
recurring bug this module exists to prevent: source table renders
``recorded_dttm`` in America/Chicago (CDT/CST) while a tooltip nearby
renders the same instant's raw UTC ISO — clinicians read the 5-hour
gap as a real time discrepancy.

Convention:
  * The display TZ is America/Chicago (CDT/CST). Mirror of the
    ``timezone`` field on ``icu_pause.config.Settings`` (default
    "America/Chicago"). Hardcoded here because review_app is a
    standalone deployment with no ``icu_pause`` package dependency.
  * Naive ISO strings are assumed to be UTC (matches the convention
    used by the pipeline's cite-registry builder and by
    ``source_renderer._compact_dttm``).
  * Both ``M-DD HH:MM`` (compact, used by the source table) and
    ``Mon DD HH:MM`` (short, used by citation tooltips) are emitted
    here so the two surface formats can't drift in TZ handling.

If the display TZ ever needs to become configurable per-deployment,
make ``DISPLAY_TZ`` a function reading from a settings module rather
than a module-level constant — every caller already goes through one
helper, so the indirection lands in one place.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

DISPLAY_TZ = ZoneInfo("America/Chicago")

# Short month names for compact / short formatters. Index 0 unused so
# ``_MONTHS[dt.month]`` works directly.
_MONTHS = [
    "",
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def parse_iso_to_display(value: object) -> Optional[datetime]:
    """Parse an ISO-formatted timestamp and return it in the display TZ.

    Accepts ``str`` (ISO 8601), ``datetime``, or any value with a usable
    ``str()``. Returns ``None`` when the input is empty or cannot be
    parsed. Naive datetimes are assumed UTC (matches the pipeline's
    cite-tag builder + ``source_renderer._compact_dttm``).
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value)
        # Tolerate both "2025-07-01 04:00:00" (space) and ISO ("T")
        # forms, mirroring _compact_dttm's normalization.
        normalized = s.replace(" ", "T", 1) if "T" not in s[:11] else s
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(DISPLAY_TZ)


def format_compact(dt: datetime) -> str:
    """Format a display-TZ datetime as ``M-DD HH:MM`` (source-table form)."""
    return f"{dt.month}-{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"


def format_short(dt: datetime) -> str:
    """Format a display-TZ datetime as ``Mon DD HH:MM`` (tooltip form).

    Day is zero-padded; month is the 3-letter abbreviation; hour:minute
    is 24-hour. Mirrors the legacy ``_short_time_from_tag`` output so
    tag-anchor strings and row-time strings compare cleanly inside
    ``format_tooltip``'s ``row_time != time_short`` elision check.
    """
    mon = _MONTHS[dt.month] if 1 <= dt.month <= 12 else str(dt.month)
    return f"{mon} {dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"


def iso_to_compact_display(value: object) -> Optional[str]:
    """Convenience: parse + format_compact in one call. Returns ``None`` if unparseable."""
    dt = parse_iso_to_display(value)
    return format_compact(dt) if dt is not None else None


def iso_to_short_display(value: object) -> Optional[str]:
    """Convenience: parse + format_short in one call. Returns ``None`` if unparseable."""
    dt = parse_iso_to_display(value)
    return format_short(dt) if dt is not None else None

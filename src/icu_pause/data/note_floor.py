"""Physician-note context floor for note routing.

When the 48-hour lookback window contains only ancillary notes (Pharmacy,
Nutrition Therapy, etc.), progress-note-consuming domain agents (Intensivist,
Respiratory, Pharmacy, Dietitian) lose the clinical narrative that puts the
structured data in context. ``ensure_physician_note_floor`` guarantees that
each such agent sees at least one physician-authored note, pulled from
earlier in the hospitalization when the window itself has none.

Designed to be ICU-type-agnostic: primary-team detection is empirical
(highest-volume named physician specialty for the hospitalization), so the
same logic works for MICU, surgical ICU, neuro ICU, etc., without
hardcoding "Internal Medicine".

The floor runs **after** the 48h lookback filter and **before** per-agent
``AGENT_MAX_NOTES_PER_TYPE`` capping, so the floor note cannot be displaced
by the cap. Floor rows are tagged with a sentinel ``_floor_protected``
column the cap stage consults.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import polars as pl

from icu_pause.data.note_specialties import (
    is_physician_note,
    primary_team_tier,
)

logger = logging.getLogger(__name__)


# Column appended to floor rows so the downstream cap stage can keep them
# even when they would otherwise be sorted out.
FLOOR_PROTECTED_COL = "_floor_protected"


# Agents that consume progress / consults / hp notes and therefore receive
# the physician-note floor when their 48h window lacks one. Nurse, case
# manager and therapist routes are unchanged — they read different note
# types whose physician-narrative requirement does not apply.
FLOOR_ELIGIBLE_AGENTS: frozenset[str] = frozenset({
    "intensivist", "respiratory", "pharmacy", "dietitian",
})


def _row_is_physician(
    row: dict[str, Any],
    specialty_col: str,
    note_type_col: str,
) -> bool:
    """Wrapper around ``is_physician_note`` that reads the right columns."""
    return is_physician_note(row.get(specialty_col), row.get(note_type_col))


def _pick_timestamp(
    row: dict[str, Any],
    timestamp_col: str,
    fallback_timestamp_col: str,
) -> Optional[datetime]:
    """Return the best available timestamp for recency ordering."""
    ts = row.get(timestamp_col)
    if ts is None:
        ts = row.get(fallback_timestamp_col)
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


def _concat_with_type_col(
    by_type: dict[str, pl.DataFrame],
    note_type_col: str,
) -> pl.DataFrame:
    """Concatenate per-type frames into one frame, attaching a note-type column.

    The per-type dict shape is what the rest of the pipeline carries, but
    primary-team detection and floor-note search both need to see all
    candidate types at once. The attached ``note_type_col`` tells us which
    bucket to drop the chosen row back into.

    Uses ``how="diagonal_relaxed"`` so per-type frames with non-overlapping
    columns (which happens because the per-CSV note loader doesn't enforce
    a unified schema) merge cleanly.
    """
    frames: list[pl.DataFrame] = []
    for note_type_key, df in by_type.items():
        if df is None or len(df) == 0:
            continue
        if note_type_col in df.columns:
            df_with_type = df
        else:
            df_with_type = df.with_columns(
                pl.lit(note_type_key).alias(note_type_col)
            )
        frames.append(df_with_type)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _sort_recent_first(
    df: pl.DataFrame,
    timestamp_col: str,
    fallback_timestamp_col: str,
) -> pl.DataFrame:
    """Sort newest-first using ``timestamp_col`` then the fallback."""
    sort_keys = [c for c in (timestamp_col, fallback_timestamp_col) if c in df.columns]
    if not sort_keys:
        return df
    return df.sort(
        by=sort_keys,
        descending=[True] * len(sort_keys),
        nulls_last=True,
    )


def _select_primary_team(
    full_stay: pl.DataFrame,
    specialty_col: str,
    note_type_col: str,
    timestamp_col: str,
    fallback_timestamp_col: str,
) -> Optional[str]:
    """Pick the primary team empirically from full-stay physician notes.

    1. Filter to rows that pass ``is_physician_note``.
    2. Drop null/empty specialty (cannot anchor a search on a nameless team).
    3. Group by specialty, count rows; tie-break on tier rank then most-recent
       note within the specialty.

    Returns ``None`` if no candidate survives — caller should drop straight
    to the broad-physician fallback.
    """
    if full_stay is None or len(full_stay) == 0:
        return None
    if specialty_col not in full_stay.columns:
        return None

    rows = full_stay.to_dicts()

    # Step 1 + 2: keep physician-named candidates only.
    candidates: list[dict[str, Any]] = []
    for row in rows:
        spec = row.get(specialty_col)
        if not isinstance(spec, str) or spec.strip() == "":
            continue
        if not _row_is_physician(row, specialty_col, note_type_col):
            continue
        candidates.append(row)

    if not candidates:
        return None

    # Step 3: aggregate per specialty.
    counts: dict[str, int] = {}
    latest_ts: dict[str, datetime] = {}
    for row in candidates:
        spec = row[specialty_col]
        counts[spec] = counts.get(spec, 0) + 1
        ts = _pick_timestamp(row, timestamp_col, fallback_timestamp_col)
        if ts is not None:
            cur = latest_ts.get(spec)
            if cur is None or ts > cur:
                latest_ts[spec] = ts

    # Step 4: pick winner. Sort by (-count, tier_rank, -latest_ts).
    # ``datetime.min`` makes specialties with no parsable timestamp lose
    # the tie-break, which is what we want.
    def _sort_key(spec: str) -> tuple[int, int, float]:
        ts = latest_ts.get(spec)
        ts_score = -ts.timestamp() if ts is not None else float("inf")
        return (-counts[spec], primary_team_tier(spec), ts_score)

    return min(counts.keys(), key=_sort_key)


def _attach_floor_protected_col(
    row: dict[str, Any],
    template_columns: list[str],
) -> dict[str, Any]:
    """Return a row dict with the floor-protected sentinel and template-aligned columns.

    Every column the in-window frame carries must be present on the floor
    row (filled with ``None`` if missing) so a single ``pl.concat`` call
    succeeds without dtype surprises.
    """
    aligned: dict[str, Any] = {col: row.get(col) for col in template_columns}
    aligned[FLOOR_PROTECTED_COL] = True
    return aligned


def _append_floor_row(
    notes_by_type: dict[str, pl.DataFrame],
    floor_row: dict[str, Any],
    target_note_type: str,
) -> dict[str, pl.DataFrame]:
    """Append the floor row to the right per-type bucket and dedup on note_id.

    Schema invariants:
    - The appended row carries the same columns as the existing bucket
      (filled with None where missing).
    - The floor-protected sentinel column is added to the bucket frame as
      well (default False) so the concat is clean.
    - Dedup on ``note_id`` — if the floor candidate is somehow already in
      the bucket, keep one copy.
    """
    existing = notes_by_type.get(target_note_type)
    if existing is None or len(existing) == 0:
        # Build a one-row frame for this bucket. Use the row's keys so the
        # frame matches the schema the caller expects.
        floor_row = {**floor_row, FLOOR_PROTECTED_COL: True}
        new_df = pl.DataFrame([floor_row])
        notes_by_type[target_note_type] = new_df
        return notes_by_type

    template_cols = list(existing.columns)
    if FLOOR_PROTECTED_COL not in template_cols:
        existing = existing.with_columns(
            pl.lit(False).alias(FLOOR_PROTECTED_COL)
        )
        template_cols = list(existing.columns)

    aligned = _attach_floor_protected_col(floor_row, template_cols)
    floor_df = pl.DataFrame([aligned])

    combined = pl.concat([existing, floor_df], how="diagonal_relaxed")

    # Dedup on note_id (keep first occurrence — the existing row).
    if "note_id" in combined.columns:
        # Polars doesn't have keep="first" in unique with subset that
        # preserves the original row order across duplicates without
        # sorting; use a row-index trick.
        combined = combined.with_row_index("_floor_idx")
        combined = combined.unique(subset=["note_id"], keep="first").sort("_floor_idx")
        combined = combined.drop("_floor_idx")

    notes_by_type[target_note_type] = combined
    return notes_by_type


def _floor_age_hours(
    floor_ts: Optional[datetime],
    reference_dttm: Optional[datetime],
) -> Optional[float]:
    """Hours from ``floor_ts`` to ``reference_dttm`` (positive = before ref)."""
    if floor_ts is None or reference_dttm is None:
        return None
    if floor_ts.tzinfo is None and reference_dttm.tzinfo is not None:
        floor_ts = floor_ts.replace(tzinfo=reference_dttm.tzinfo)
    if reference_dttm.tzinfo is None and floor_ts.tzinfo is not None:
        reference_dttm = reference_dttm.replace(tzinfo=floor_ts.tzinfo)
    delta = reference_dttm - floor_ts
    return delta.total_seconds() / 3600.0


def _is_within_window(
    row: dict[str, Any],
    timestamp_col: str,
    fallback_timestamp_col: str,
    reference_dttm: Optional[datetime],
    lookback_hours: Optional[int],
) -> bool:
    """Return True if the row's timestamp is within ``lookback_hours`` of
    ``reference_dttm``. If either is unknown, falls back to False (the
    "any time in stay" branch will still be exercised below)."""
    if reference_dttm is None or lookback_hours is None:
        return False
    ts = _pick_timestamp(row, timestamp_col, fallback_timestamp_col)
    if ts is None:
        return False
    if ts.tzinfo is None and reference_dttm.tzinfo is not None:
        ts = ts.replace(tzinfo=reference_dttm.tzinfo)
    if reference_dttm.tzinfo is None and ts.tzinfo is not None:
        reference_dttm = reference_dttm.replace(tzinfo=ts.tzinfo)
    return ts >= (reference_dttm - timedelta(hours=lookback_hours))


def _build_metadata(
    floor_applied: bool,
    reason: str,
    primary_team: Optional[str],
    floor_row: Optional[dict[str, Any]],
    target_note_type: Optional[str],
    specialty_col: str,
    timestamp_col: str,
    fallback_timestamp_col: str,
    reference_dttm: Optional[datetime],
) -> dict[str, Any]:
    floor_id = floor_row.get("note_id") if floor_row else None
    floor_specialty = floor_row.get(specialty_col) if floor_row else None
    floor_ts = (
        _pick_timestamp(floor_row, timestamp_col, fallback_timestamp_col)
        if floor_row
        else None
    )
    return {
        "floor_applied": floor_applied,
        "reason": reason,
        "primary_team": primary_team,
        "floor_note_id": str(floor_id) if floor_id is not None else None,
        "floor_note_type": target_note_type,
        "floor_specialty": floor_specialty,
        "floor_age_hours": _floor_age_hours(floor_ts, reference_dttm),
    }


def ensure_physician_note_floor(
    notes_in_window_by_type: dict[str, pl.DataFrame],
    notes_full_stay_by_type: dict[str, pl.DataFrame],
    reference_dttm: Optional[datetime] = None,
    lookback_hours: Optional[int] = 48,
    specialty_col: str = "author_specialty",
    note_type_col: str = "note_category",
    timestamp_col: str = "revision_dttm",
    fallback_timestamp_col: str = "creation_dttm",
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Guarantee at least one physician-authored note in the agent's window.

    Args:
        notes_in_window_by_type: Per-note-type frames already filtered to
            the 48h lookback window. Mutated only via re-assignment of the
            returned dict.
        notes_full_stay_by_type: Per-note-type frames spanning the entire
            hospitalization (still capped at < ``reference_dttm`` by the
            upstream loader's leakage guard).
        reference_dttm: The "current" time. Used for window-membership
            checks and ``floor_age_hours`` metadata.
        lookback_hours: Width of the in-window region. Defaults to 48.
        specialty_col: Column carrying the author specialty/department.
        note_type_col: Column carrying the note type. If not present in the
            source frames, the per-type dict key is used.
        timestamp_col: Primary recency column.
        fallback_timestamp_col: Used when ``timestamp_col`` is missing or null.

    Returns:
        ``(augmented_notes_by_type, metadata)``. ``metadata`` always
        contains keys: floor_applied, reason, primary_team, floor_note_id,
        floor_note_type, floor_specialty, floor_age_hours.
    """
    # ------------------------------------------------------------------
    # Step 1: short-circuit — physician note already in window?
    # ------------------------------------------------------------------
    in_window = _concat_with_type_col(notes_in_window_by_type, note_type_col)
    if len(in_window) > 0:
        for row in in_window.to_dicts():
            if _row_is_physician(row, specialty_col, note_type_col):
                return notes_in_window_by_type, _build_metadata(
                    floor_applied=False,
                    reason="physician_note_present_in_window",
                    primary_team=None,
                    floor_row=None,
                    target_note_type=None,
                    specialty_col=specialty_col,
                    timestamp_col=timestamp_col,
                    fallback_timestamp_col=fallback_timestamp_col,
                    reference_dttm=reference_dttm,
                )

    full_stay = _concat_with_type_col(notes_full_stay_by_type, note_type_col)
    if len(full_stay) == 0:
        return notes_in_window_by_type, _build_metadata(
            floor_applied=False,
            reason="no_physician_note_in_hospitalization",
            primary_team=None,
            floor_row=None,
            target_note_type=None,
            specialty_col=specialty_col,
            timestamp_col=timestamp_col,
            fallback_timestamp_col=fallback_timestamp_col,
            reference_dttm=reference_dttm,
        )

    # ------------------------------------------------------------------
    # Step 2: detect primary team empirically (named specialty only).
    # ------------------------------------------------------------------
    primary_team = _select_primary_team(
        full_stay, specialty_col, note_type_col, timestamp_col, fallback_timestamp_col
    )

    # Reorder full_stay newest-first so "first hit" semantics in the
    # downstream search loop reads as "most recent".
    full_stay_sorted = _sort_recent_first(full_stay, timestamp_col, fallback_timestamp_col)
    full_stay_rows = full_stay_sorted.to_dicts()

    # ------------------------------------------------------------------
    # Step 3: pull floor note from primary team (3 phases, stop on first hit).
    # ------------------------------------------------------------------
    if primary_team is not None:
        # Phase 1: most recent note of any type from primary_team within the window.
        for row in full_stay_rows:
            if (
                row.get(specialty_col) == primary_team
                and _is_within_window(
                    row, timestamp_col, fallback_timestamp_col,
                    reference_dttm, lookback_hours,
                )
            ):
                return _commit_floor(
                    notes_in_window_by_type, row, primary_team,
                    "primary_team_window",
                    specialty_col, note_type_col,
                    timestamp_col, fallback_timestamp_col, reference_dttm,
                )

        # Phase 2: most recent hp_note OR consults_note from primary_team
        # anywhere in the hospitalization.
        for row in full_stay_rows:
            if (
                row.get(specialty_col) == primary_team
                and row.get(note_type_col) in {"hp_note", "consults_note"}
            ):
                return _commit_floor(
                    notes_in_window_by_type, row, primary_team,
                    "primary_team_hp_consult",
                    specialty_col, note_type_col,
                    timestamp_col, fallback_timestamp_col, reference_dttm,
                )

        # Phase 3: most recent note of any type from primary_team anywhere.
        for row in full_stay_rows:
            if row.get(specialty_col) == primary_team:
                return _commit_floor(
                    notes_in_window_by_type, row, primary_team,
                    "primary_team_any",
                    specialty_col, note_type_col,
                    timestamp_col, fallback_timestamp_col, reference_dttm,
                )

    # ------------------------------------------------------------------
    # Step 4: broad physician fallback. Picks up null-specialty
    # hp/consults/progress notes plus any named subspecialty not in the
    # tiers. Same 3 phases.
    # ------------------------------------------------------------------
    # Phase 1: most recent physician note within the window.
    for row in full_stay_rows:
        if (
            _row_is_physician(row, specialty_col, note_type_col)
            and _is_within_window(
                row, timestamp_col, fallback_timestamp_col,
                reference_dttm, lookback_hours,
            )
        ):
            return _commit_floor(
                notes_in_window_by_type, row, primary_team,
                "broad_physician_window",
                specialty_col, note_type_col,
                timestamp_col, fallback_timestamp_col, reference_dttm,
            )

    # Phase 2: most recent hp_note or consults_note from any physician note.
    for row in full_stay_rows:
        if (
            _row_is_physician(row, specialty_col, note_type_col)
            and row.get(note_type_col) in {"hp_note", "consults_note"}
        ):
            return _commit_floor(
                notes_in_window_by_type, row, primary_team,
                "broad_physician_hp_consult",
                specialty_col, note_type_col,
                timestamp_col, fallback_timestamp_col, reference_dttm,
            )

    # Phase 3: most recent physician note of any type.
    for row in full_stay_rows:
        if _row_is_physician(row, specialty_col, note_type_col):
            return _commit_floor(
                notes_in_window_by_type, row, primary_team,
                "broad_physician_any",
                specialty_col, note_type_col,
                timestamp_col, fallback_timestamp_col, reference_dttm,
            )

    # ------------------------------------------------------------------
    # Nothing matched — emit a warning so this can be tracked in metrics.
    # ------------------------------------------------------------------
    logger.warning(
        "ensure_physician_note_floor: no physician note found anywhere in "
        "hospitalization (specialty_col=%s); window left unchanged.",
        specialty_col,
    )
    return notes_in_window_by_type, _build_metadata(
        floor_applied=False,
        reason="no_physician_note_in_hospitalization",
        primary_team=primary_team,
        floor_row=None,
        target_note_type=None,
        specialty_col=specialty_col,
        timestamp_col=timestamp_col,
        fallback_timestamp_col=fallback_timestamp_col,
        reference_dttm=reference_dttm,
    )


def _commit_floor(
    notes_in_window_by_type: dict[str, pl.DataFrame],
    floor_row: dict[str, Any],
    primary_team: Optional[str],
    reason: str,
    specialty_col: str,
    note_type_col: str,
    timestamp_col: str,
    fallback_timestamp_col: str,
    reference_dttm: Optional[datetime],
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Append the floor row, mark protected, and build metadata. Helper
    used by the multiple search-success branches above."""
    target_note_type = floor_row.get(note_type_col) or "progress_note"

    # Strip the synth ``note_type_col`` we attached in
    # ``_concat_with_type_col``: it's plumbing for the search loop, not
    # data the per-type bucket needs (the dict key already encodes type).
    # Leaving it in would introduce a column that the in-window rows
    # don't carry when the source loader didn't populate ``note_category``.
    floor_row_for_append = {
        k: v for k, v in floor_row.items() if k != note_type_col
    }

    augmented = _append_floor_row(
        dict(notes_in_window_by_type), floor_row_for_append, target_note_type
    )
    metadata = _build_metadata(
        floor_applied=True,
        reason=reason,
        primary_team=primary_team,
        floor_row=floor_row,
        target_note_type=target_note_type,
        specialty_col=specialty_col,
        timestamp_col=timestamp_col,
        fallback_timestamp_col=fallback_timestamp_col,
        reference_dttm=reference_dttm,
    )
    return augmented, metadata

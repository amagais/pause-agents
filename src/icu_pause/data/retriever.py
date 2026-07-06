"""Data Retrieval Agent: deterministic extraction of CLIF data for a single hospitalization."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import polars as pl

from icu_pause.config import (
    AGENT_NOTE_ROUTING,
    NOTE_ROUTING_VERSION,
    PER_ADMISSION_STABLE_NOTE_TYPES,
    Settings,
)
from icu_pause.data.context import PatientContext
from icu_pause.data.note_floor import FLOOR_ELIGIBLE_AGENTS

logger = logging.getLogger(__name__)


class DataRetriever:
    """Step 1 of the agentic pipeline. No LLM — pure data extraction via clifpy."""

    # CLIF tables to attempt loading (structured / non-note data)
    TABLE_NAMES = [
        "patient",
        "hospitalization",
        "adt",
        "vitals",
        "labs",
        "medication_admin_continuous",
        "medication_admin_intermittent",
        "respiratory_support",
        "patient_assessments",
        "code_status",
        "hospital_diagnosis",
        "microbiology_culture",
        "patient_procedures",
        "crrt_therapy",
        "ecmo_mcs",
        "position",
    ]

    # Tables that may not exist at all sites — suppress warnings for these
    OPTIONAL_TABLES: set[str] = {
        "clinical_notes_facts",  # metadata, not used by agents
        "crrt_therapy",          # only relevant if patient on dialysis
        "ecmo_mcs",              # only relevant if patient on ECMO
        "position",              # positioning data, sparse at many sites
    }

    # Note types to exclude from agent input (used as gold-standard reference only)
    REFERENCE_NOTE_TYPES = {"transfer_note"}

    def __init__(self, settings: Settings):
        self.data_dir = settings.clif_data_dir
        self.notes_data_dir = settings.resolved_notes_data_dir
        self.timezone = settings.timezone
        self.note_file_map = settings.note_file_map
        self.notes_lookback_hours = settings.notes_lookback_hours
        self.structured_data_enabled = settings.structured_data_enabled
        self.notes_enabled = settings.notes_enabled
        self._tables: dict[str, pl.DataFrame] = {}
        # Cache for note CSVs: note_type_key -> full DataFrame (before hosp filter)
        self._note_tables: dict[str, pl.DataFrame] = {}
        # Debug log: captures detailed events during retrieval for the trace viewer
        self.load_log: list[dict] = []

    # ------------------------------------------------------------------
    # Timezone normalization — all comparisons happen in UTC
    # ------------------------------------------------------------------

    def _to_utc(self, dt: datetime) -> datetime:
        """Normalize a Python datetime to UTC.

        - If tz-aware: convert to UTC
        - If tz-naive: assume ``self.timezone`` (e.g. America/Chicago), then convert
        """
        from zoneinfo import ZoneInfo
        if dt.tzinfo is None:
            local_tz = ZoneInfo(self.timezone)
            dt = dt.replace(tzinfo=local_tz)
        return dt.astimezone(ZoneInfo("UTC"))

    def _normalize_column_to_utc(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Normalize a Polars datetime column to tz-aware UTC.

        - String columns are parsed to datetime first.
        - Tz-aware columns are converted to UTC.
        - Tz-naive columns are assumed to be in ``self.timezone`` and localized.
        """
        if col not in df.columns:
            return df
        col_dtype = df[col].dtype
        # Parse strings to datetime
        if col_dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.to_datetime(strict=False).alias(col)
            )
            col_dtype = df[col].dtype
        # Ensure μs precision (Polars comparison requires matching time units)
        if hasattr(col_dtype, "time_unit") and col_dtype.time_unit == "ns":
            tz = col_dtype.time_zone if hasattr(col_dtype, "time_zone") else None
            df = df.with_columns(
                pl.col(col).cast(pl.Datetime("us", time_zone=tz)).alias(col)
            )
            col_dtype = df[col].dtype
        if hasattr(col_dtype, "time_zone") and col_dtype.time_zone is not None:
            # Already tz-aware — convert to UTC
            df = df.with_columns(
                pl.col(col).dt.convert_time_zone("UTC").alias(col)
            )
        else:
            # Tz-naive — assume local timezone, then convert to UTC.
            # Use non_existent="null" to handle DST spring-forward gaps
            # (e.g., 2:00 AM doesn't exist in America/Chicago on DST day).
            # Use ambiguous="earliest" for fall-back duplicates.
            df = df.with_columns(
                pl.col(col)
                .dt.replace_time_zone(
                    self.timezone,
                    non_existent="null",
                    ambiguous="earliest",
                )
                .dt.convert_time_zone("UTC")
                .alias(col)
            )
        return df

    def _trace(self, event_type: str, node: str, message: str, *,
               level: str = "info", data: dict = None) -> None:
        """Emit a trace event to both the Python logger and the load_log."""
        from datetime import datetime, timezone
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[{node}] {message}")
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "node": f"data_retrieval/{node}",
            "level": level,
            "message": message,
        }
        if data:
            event["data"] = data
        self.load_log.append(event)

    # ------------------------------------------------------------------
    # Structured (Parquet) data loading
    # ------------------------------------------------------------------

    def _find_parquet_path(self, name: str) -> Optional[Path]:
        """Find the parquet file path for a table name."""
        data_path = Path(self.data_dir)
        candidates = [
            data_path / f"{name}.parquet",
            data_path / f"clif_{name}.parquet",
            data_path / f"rclif_{name}.parquet",
        ]
        for path in candidates:
            if path.exists():
                return path
        level = "debug" if name in self.OPTIONAL_TABLES else "warning"
        self._trace("parquet_load", name,
                     f"Table not found (tried {', '.join(c.name for c in candidates)})",
                     level=level)
        return None

    def _load_table(self, name: str) -> Optional[pl.DataFrame]:
        """Load entire CLIF Parquet file. Used for small tables (patient, code_status, etc.)."""
        if name in self._tables:
            return self._tables[name]

        path = self._find_parquet_path(name)
        if path is None:
            return None

        try:
            df = pl.read_parquet(str(path))
            self._tables[name] = df
            self._trace("parquet_load", name,
                        f"Loaded {len(df)} rows from {path.name}",
                        data={"path": str(path), "rows": len(df),
                              "columns": df.columns[:8]})
            return df
        except Exception as e:
            self._trace("parquet_load", name,
                        f"Failed to read {path.name}: {e}",
                        level="warning")
        return None

    def _load_table_filtered(
        self, name: str, hospitalization_id: str
    ) -> Optional[pl.DataFrame]:
        """Load a CLIF Parquet with predicate pushdown for a single hospitalization.

        Uses pl.scan_parquet with lazy filter so Polars only reads matching
        row groups from disk. Handles dtype mismatches: the hospitalization_id
        column may be Int64, Float64, or Utf8 — we try the appropriate cast.
        """
        path = self._find_parquet_path(name)
        if path is None:
            return None

        try:
            lazy = pl.scan_parquet(str(path))
            schema = lazy.collect_schema()

            if "hospitalization_id" not in schema.names():
                self._trace("parquet_load", name,
                             f"No hospitalization_id column in {path.name}",
                             level="warning")
                return None

            col_dtype = schema["hospitalization_id"]

            # Build the filter expression based on column dtype
            if col_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
                # Column is integer — cast search ID to int
                try:
                    search_val = int(hospitalization_id)
                    filter_expr = pl.col("hospitalization_id") == search_val
                except ValueError:
                    filter_expr = pl.col("hospitalization_id").cast(pl.Utf8) == hospitalization_id
            elif col_dtype in (pl.Float64, pl.Float32):
                # Column is float — try as float, also try int comparison
                try:
                    search_val = float(hospitalization_id)
                    filter_expr = pl.col("hospitalization_id") == search_val
                except ValueError:
                    filter_expr = pl.col("hospitalization_id").cast(pl.Utf8) == hospitalization_id
            else:
                # Column is string — try exact, then with ".0" stripped
                filter_expr = (
                    (pl.col("hospitalization_id") == hospitalization_id)
                    | (pl.col("hospitalization_id") == f"{hospitalization_id}.0")
                    | (pl.col("hospitalization_id").str.replace(r"\.0$", "") == hospitalization_id)
                )

            df = lazy.filter(filter_expr).collect()

            self._trace("parquet_load", name,
                        f"Scanned {path.name}: {len(df)} rows for hosp_id={hospitalization_id} "
                        f"(col_dtype={col_dtype})",
                        data={"path": str(path), "rows": len(df),
                              "columns": df.columns[:8],
                              "col_dtype": str(col_dtype),
                              "method": "predicate_pushdown"})
            return df if len(df) > 0 else None
        except Exception as e:
            self._trace("parquet_load", name,
                        f"Scan failed for {path.name}: {e}, falling back to full load",
                        level="warning")
            # Fallback to full load if scan fails
            return self._load_table(name)
        return None

    def _filter_by_hosp(
        self,
        table_name: str,
        hospitalization_id: str,
        time_window: Optional[tuple] = None,
    ) -> Optional[pl.DataFrame]:
        """Load a table and filter by hospitalization_id, optionally by time window.

        Uses predicate pushdown (scan_parquet) to avoid loading entire tables
        into memory. Falls back to full load for edge cases.
        """
        filtered = self._load_table_filtered(table_name, hospitalization_id)
        if filtered is None:
            return None

        # Apply time window filter if provided
        if time_window is not None:
            start, end = time_window
            time_col = self._find_time_column(filtered)
            if time_col and end is not None:
                # Normalize column to UTC for consistent comparison.
                # time_window boundaries are already UTC (from retrieve()).
                filtered = self._normalize_column_to_utc(filtered, time_col)

                # Ensure boundaries are tz-aware UTC for Polars comparison
                from zoneinfo import ZoneInfo
                _utc = ZoneInfo("UTC")
                if end.tzinfo is None:
                    end = end.replace(tzinfo=_utc)
                if start is not None and start.tzinfo is None:
                    start = start.replace(tzinfo=_utc)

                # Apply end cap (data leakage guard) — strict less-than to
                # exclude data recorded at exactly reference_dttm (which may
                # be the human transfer note we are trying to replace)
                filtered = filtered.filter(pl.col(time_col) < end)

                # Apply start floor — only if provided (lookback window)
                if start is not None:
                    filtered = filtered.filter(pl.col(time_col) >= start)

                if len(filtered) == 0:
                    return None

        return filtered

    def _filter_code_status_with_leakage_guard(
        self,
        hospitalization_id: str,
        patient_id: str,
        reference_dttm: Optional[datetime],
    ) -> Optional[pl.DataFrame]:
        """Load code_status with a strict ``< reference_dttm`` cap.

        Prefers hospitalization-id scoping so a re-admit's later code-status
        decisions don't leak into the index admission. Falls back to
        patient-id scoping (with the same time cap applied manually) for
        sites whose code_status parquet lacks ``hospitalization_id``.
        """
        path = self._find_parquet_path("code_status")
        if path is None:
            return None
        try:
            schema_names = pl.scan_parquet(str(path)).collect_schema().names()
        except Exception:
            schema_names = []

        if "hospitalization_id" in schema_names:
            return self._filter_by_hosp(
                "code_status",
                hospitalization_id,
                (None, reference_dttm) if reference_dttm is not None else None,
            )

        # Fallback: patient-id scope + manual leakage guard.
        df = self._filter_by_patient("code_status", patient_id)
        if df is None or reference_dttm is None:
            return df
        time_col = self._find_time_column(df)
        if time_col is None:
            return df
        df = self._normalize_column_to_utc(df, time_col)
        from zoneinfo import ZoneInfo
        end = reference_dttm
        if end.tzinfo is None:
            end = end.replace(tzinfo=ZoneInfo("UTC"))
        df = df.filter(pl.col(time_col) < end)
        return df if len(df) > 0 else None

    # Microbiology fields nulled when a culture has not yet resulted
    # at reference_dttm (pre-result fields like collect_dttm/order_dttm
    # stay visible so the row reads as "BCx <collect_dttm>: pending").
    _MICRO_RESULT_NULL_FIELDS: tuple[str, ...] = (
        "result_dttm",
        "organism_id",
        "result_status",
        "result_status_category",
    )

    # Microbiology fields that name the organism. Set to the literal
    # "pending" (rather than nulled) so downstream consumers — the
    # orchestrator's pending-culture detector and the intensivist's
    # Section P prompt — pick the row up uniformly, regardless of which
    # name field a given site populates.
    _MICRO_ORGANISM_NAME_FIELDS: tuple[str, ...] = (
        "organism",
        "organism_name",
        "organism_category",
    )

    _PENDING_MARKER: str = "pending"

    def _mask_future_microbiology_results(
        self,
        df: Optional[pl.DataFrame],
        reference_dttm: Optional[datetime],
    ) -> Optional[pl.DataFrame]:
        """Mark microbiology rows that have not yet resulted at
        ``reference_dttm`` as "pending".

        ``_filter_by_hosp`` only filters on a single time column
        (``collect_dttm`` for microbiology). A culture collected before
        transfer but resulted after transfer passes that filter — yet the
        organism / sensitivities are from the future. This matches what
        a clinician at transfer time would actually see: "BCx sent,
        pending" rather than the eventual organism.

        Rows are kept (the fact that a culture was *sent* is legitimate
        pre-transfer information). result_dttm and organism_id are nulled;
        organism name fields are set to "pending" so the orchestrator's
        pending-culture detector and the intensivist's Section P prompt
        pick them up.

        Rows that were already pending in the source data
        (``result_dttm IS NULL``) are also marked, so all natively-pending
        cultures land in Section P regardless of how the site exports them.
        """
        if df is None or len(df) == 0 or reference_dttm is None:
            return df
        if "result_dttm" not in df.columns:
            return df

        df = self._normalize_column_to_utc(df, "result_dttm")
        ref_utc = self._to_utc(reference_dttm)

        pending_mask = (
            pl.col("result_dttm").is_null() | (pl.col("result_dttm") >= ref_utc)
        )

        null_fields = [c for c in self._MICRO_RESULT_NULL_FIELDS if c in df.columns]
        name_fields = [c for c in self._MICRO_ORGANISM_NAME_FIELDS if c in df.columns]

        df = df.with_columns(
            [
                pl.when(pending_mask).then(None).otherwise(pl.col(c)).alias(c)
                for c in null_fields
            ]
            + [
                pl.when(pending_mask)
                .then(pl.lit(self._PENDING_MARKER))
                .otherwise(pl.col(c))
                .alias(c)
                for c in name_fields
            ]
        )
        return df

    # Lab fields nulled when a lab has not yet resulted at
    # reference_dttm. lab_collect_dttm / lab_order_dttm stay visible.
    _LAB_RESULT_NULL_FIELDS: tuple[str, ...] = (
        "lab_result_dttm",
        "lab_value_numeric",
        "reference_unit",
        "reference_low",
        "reference_high",
        "lab_loinc_code",
    )

    def _mask_future_lab_results(
        self,
        df: Optional[pl.DataFrame],
        reference_dttm: Optional[datetime],
    ) -> Optional[pl.DataFrame]:
        """Mark lab rows that have not yet resulted at ``reference_dttm``
        as "pending" — parallel to ``_mask_future_microbiology_results``.

        Used after ``_filter_labs_with_pending`` keeps rows that were
        collected before reference but not yet resulted. We null the
        numeric value + result_dttm and set ``lab_value`` (the string
        result column) to "pending" so the row renders as
        "<lab_category> (collected <lab_collect_dttm>): pending".
        """
        if df is None or len(df) == 0 or reference_dttm is None:
            return df
        if "lab_result_dttm" not in df.columns:
            return df

        df = self._normalize_column_to_utc(df, "lab_result_dttm")
        ref_utc = self._to_utc(reference_dttm)

        pending_mask = (
            pl.col("lab_result_dttm").is_null() | (pl.col("lab_result_dttm") >= ref_utc)
        )

        null_fields = [c for c in self._LAB_RESULT_NULL_FIELDS if c in df.columns]
        df = df.with_columns(
            [
                pl.when(pending_mask).then(None).otherwise(pl.col(c)).alias(c)
                for c in null_fields
            ]
        )
        if "lab_value" in df.columns:
            df = df.with_columns(
                pl.when(pending_mask)
                .then(pl.lit(self._PENDING_MARKER))
                .otherwise(pl.col("lab_value"))
                .alias("lab_value")
            )
        return df

    def _filter_labs_with_pending(
        self,
        hospitalization_id: str,
        time_window: Optional[tuple],
        reference_dttm: Optional[datetime],
    ) -> Optional[pl.DataFrame]:
        """Load labs and keep both resolved AND pending rows.

        Without this, ``_filter_by_hosp`` filters labs on
        ``lab_result_dttm < reference_dttm``, which silently DROPS any lab
        whose specimen was collected before transfer but did not result
        until after transfer. Those rows are clinically meaningful — they
        belong in Section P (Pending Tests) and are easy to lose.

        Keeps rows where ``lab_collect_dttm < reference_dttm`` OR
        ``lab_result_dttm < reference_dttm``. The lookback floor (when
        provided) is applied to whichever of the two timestamps is
        present per row. Result fields are then masked for rows that
        haven't resolved at reference_dttm.
        """
        from zoneinfo import ZoneInfo

        filtered = self._load_table_filtered("labs", hospitalization_id)
        if filtered is None:
            return None

        if time_window is None:
            return filtered

        start, end = time_window
        if end is None:
            return filtered

        for col in ("lab_collect_dttm", "lab_result_dttm"):
            if col in filtered.columns:
                filtered = self._normalize_column_to_utc(filtered, col)

        _utc = ZoneInfo("UTC")
        if end.tzinfo is None:
            end = end.replace(tzinfo=_utc)
        if start is not None and start.tzinfo is None:
            start = start.replace(tzinfo=_utc)

        has_collect = "lab_collect_dttm" in filtered.columns
        has_result = "lab_result_dttm" in filtered.columns

        # End cap (data-leakage guard) — at least one of collect/result
        # must be before reference_dttm for the row to be kept. Strict
        # less-than excludes data recorded exactly at reference_dttm.
        if has_collect and has_result:
            end_cap = (pl.col("lab_collect_dttm") < end) | (
                pl.col("lab_result_dttm") < end
            )
        elif has_result:
            end_cap = pl.col("lab_result_dttm") < end
        elif has_collect:
            end_cap = pl.col("lab_collect_dttm") < end
        else:
            return filtered
        filtered = filtered.filter(end_cap)

        if start is not None:
            if has_collect and has_result:
                start_floor = (pl.col("lab_collect_dttm") >= start) | (
                    pl.col("lab_result_dttm") >= start
                )
            elif has_result:
                start_floor = pl.col("lab_result_dttm") >= start
            else:
                start_floor = pl.col("lab_collect_dttm") >= start
            filtered = filtered.filter(start_floor)

        if len(filtered) == 0:
            return None

        return self._mask_future_lab_results(filtered, reference_dttm)

    def _filter_by_patient(
        self, table_name: str, patient_id: str
    ) -> Optional[pl.DataFrame]:
        """Load and filter by patient_id using predicate pushdown."""
        path = self._find_parquet_path(table_name)
        if path is None:
            return None

        try:
            lazy = pl.scan_parquet(str(path))
            schema = lazy.collect_schema()

            if "patient_id" not in schema.names():
                return None

            col_dtype = schema["patient_id"]

            # Build filter based on column dtype
            if col_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
                try:
                    filter_expr = pl.col("patient_id") == int(patient_id)
                except ValueError:
                    filter_expr = pl.col("patient_id").cast(pl.Utf8) == patient_id
            elif col_dtype in (pl.Float64, pl.Float32):
                try:
                    filter_expr = pl.col("patient_id") == float(patient_id)
                except ValueError:
                    filter_expr = pl.col("patient_id").cast(pl.Utf8) == patient_id
            else:
                filter_expr = (
                    (pl.col("patient_id") == patient_id)
                    | (pl.col("patient_id") == f"{patient_id}.0")
                    | (pl.col("patient_id").str.replace(r"\.0$", "") == patient_id)
                )

            filtered = lazy.filter(filter_expr).collect()

            self._trace("parquet_load", table_name,
                        f"Scanned {path.name}: {len(filtered)} rows for patient_id={patient_id}",
                        data={"path": str(path), "rows": len(filtered),
                              "col_dtype": str(col_dtype),
                              "method": "predicate_pushdown"})

            return filtered if len(filtered) > 0 else None
        except Exception as e:
            self._trace("parquet_load", table_name,
                        f"Scan by patient_id failed: {e}, falling back to full load",
                        level="warning")
            df = self._load_table(table_name)
            if df is None or "patient_id" not in df.columns:
                return None
            filtered = df.filter(pl.col("patient_id") == patient_id)
            return filtered if len(filtered) > 0 else None

    @staticmethod
    def _coerce_birth_date(value) -> Optional[date]:
        """Normalize a parquet birth_date cell into a ``date``.

        CLIF sites store birth_date as date, datetime, or ISO string.
        Returns ``None`` if the value is missing or unparseable.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.split("T")[0]).date()
            except ValueError:
                return None
        return None

    @staticmethod
    def _age_years(birth: date, anchor: datetime) -> int:
        """Whole-years age at ``anchor`` (date-aware, no birthday off-by-one)."""
        ad = anchor.date() if isinstance(anchor, datetime) else anchor
        years = ad.year - birth.year
        if (ad.month, ad.day) < (birth.month, birth.day):
            years -= 1
        return years

    @staticmethod
    def _find_time_column(df: pl.DataFrame) -> Optional[str]:
        """Find the primary datetime column in a DataFrame."""
        candidates = [
            "recorded_dttm",
            "admin_dttm",
            "lab_result_dttm",
            "collect_dttm",
            "code_status_dttm",
            "start_dttm",
            "procedure_dttm",
            "in_dttm",
            "creation_dttm",
        ]
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _extract_icu_window(
        self,
        adt_df: Optional[pl.DataFrame],
        reference_dttm: Optional[datetime] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Find ICU admission and discharge times from ADT movements.

        For patients with multiple ICU stays (readmits), pick the stay whose
        in_dttm <= reference_dttm and is the most recent — i.e., the ICU stay
        the agents are reasoning about. Without reference_dttm, fall back to
        the most recent ICU stay overall.
        """
        if adt_df is None or len(adt_df) == 0:
            return (None, None)

        if "location_category" not in adt_df.columns:
            return (None, None)

        icu_rows = adt_df.filter(pl.col("location_category") == "icu")
        if len(icu_rows) == 0 or "in_dttm" not in icu_rows.columns:
            return (None, None)

        if reference_dttm is not None:
            ref_utc = self._to_utc(reference_dttm)
            icu_normalized = self._normalize_column_to_utc(icu_rows, "in_dttm")
            at_or_before = icu_normalized.filter(pl.col("in_dttm") <= ref_utc)
            chosen = at_or_before if len(at_or_before) > 0 else icu_normalized
        else:
            chosen = icu_rows

        latest = chosen.sort("in_dttm", descending=True).head(1)
        start = latest["in_dttm"][0] if len(latest) else None
        end = (
            latest["out_dttm"][0]
            if "out_dttm" in latest.columns and len(latest)
            else None
        )
        return (start, end)

    # ------------------------------------------------------------------
    # Note CSV loading (individual files per note type)
    # ------------------------------------------------------------------

    # Keywords for fuzzy note file matching when exact filename not found
    _NOTE_KEYWORDS: dict[str, list[str]] = {
        "nursing_note": ["nursing"],
        "progress_note": ["progress"],
        "consults_note": ["consult"],
        "plan_of_care_note": ["plan_of_care", "care_plan"],
        "case_management_note": ["case_manage", "case_mgmt"],
        "social_work_note": ["social_work"],
        "therapy_note": ["therapy", "rehab"],
        # hp_note keyword list added 2026-05-26. Without explicit entries
        # the loader would fall back to the bare "hp" substring (key with
        # "_note" stripped), which matches "hp_note_*.csv" by accident but
        # misses "history_and_physical_*.csv" / "h_and_p_*.csv" / similar
        # at sites that use a different naming convention. Defend against
        # all the common shapes.
        "hp_note": [
            "hp_note", "history_and_physical", "history_physical",
            "h_and_p", "h_p_note", "admission_h_p",
        ],
    }

    def _find_note_path(self, note_type_key: str) -> Optional[Path]:
        """Find the CSV file path for a note type.

        First tries the exact filename from note_file_map. If not found,
        searches the notes directory for CSV files matching keywords
        associated with the note type (e.g., 'consult' matches both
        'consults_note_2024.csv' and 'consults_2024.csv').
        """
        notes_dir = Path(self.notes_data_dir)

        # Try exact filename first
        filename = self.note_file_map.get(note_type_key)
        if filename:
            path = notes_dir / filename
            if path.exists():
                return path

        # Fuzzy match: search for CSVs containing the keyword
        keywords = self._NOTE_KEYWORDS.get(note_type_key, [])
        if not keywords:
            # Fall back to using the note_type_key itself (without _note suffix)
            keywords = [note_type_key.replace("_note", "")]

        if notes_dir.exists():
            # Collect all matches, then pick the most recent (by filename sort, descending)
            matches = []
            for csv_file in notes_dir.glob("*.csv"):
                name_lower = csv_file.name.lower()
                for kw in keywords:
                    if kw in name_lower:
                        matches.append(csv_file)
                        break

            if matches:
                # Sort descending so 2024 > 2023 > 2022 etc.
                best = sorted(matches, reverse=True)[0]
                self._trace(
                    "note_load", note_type_key,
                    f"Fuzzy match: '{best.name}' matched keywords {keywords} "
                    f"(exact filename '{filename}' not found, {len(matches)} candidates)",
                    data={"matched_file": best.name, "keywords": keywords,
                           "candidates": [f.name for f in sorted(matches)]},
                )
                return best

        self._trace("note_load", note_type_key,
                     f"File not found: tried '{filename}' and keywords {keywords}",
                     level="warning",
                     data={"path": str(notes_dir / (filename or "")),
                           "keywords": keywords, "exists": False})
        return None

    def _load_notes_for_hospitalization(
        self,
        note_type_key: str,
        hospitalization_id: str,
        reference_dttm: Optional[datetime],
        notes_lookback_hours: Optional[int],
    ) -> Optional[pl.DataFrame]:
        """Load a single note type, filtered for one hospitalization.

        Uses pl.scan_csv with lazy filtering to avoid loading the entire
        CSV into memory. Only rows matching the hospitalization_id are
        materialized. Handles dtype mismatches (Float64, Int64, Utf8).

        Filters applied (in order):
        1. hospitalization_id — applied during scan (lazy filter).
        2. Data-leakage guard — exclude rows after reference_dttm.
        3. Lookback window — only keep notes within time range.
        """
        path = self._find_note_path(note_type_key)
        if path is None:
            return None

        try:
            # Use scan_csv for lazy evaluation — doesn't load entire file
            lazy = pl.scan_csv(str(path), infer_schema_length=10000)
            schema = lazy.collect_schema()

            if "hospitalization_id" not in schema.names():
                self._trace("note_load", note_type_key,
                             f"No hospitalization_id column in {path.name}",
                             level="warning")
                return None

            col_dtype = schema["hospitalization_id"]
            search_id = str(hospitalization_id)

            # Build filter expression based on column dtype
            if col_dtype in (pl.Float64, pl.Float32):
                # Float column: cast to Int64 then Utf8 for clean comparison
                filter_expr = (
                    pl.col("hospitalization_id").cast(pl.Int64, strict=False).cast(pl.Utf8)
                    == search_id
                )
                cast_method = f"numeric({col_dtype})->Int64->Utf8"
            elif col_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
                # Integer column: try int comparison
                try:
                    filter_expr = pl.col("hospitalization_id") == int(search_id)
                    cast_method = f"int({col_dtype})==int"
                except ValueError:
                    filter_expr = pl.col("hospitalization_id").cast(pl.Utf8) == search_id
                    cast_method = f"int({col_dtype})->Utf8"
            else:
                # String column: try exact, then strip .0
                filter_expr = (
                    (pl.col("hospitalization_id") == search_id)
                    | (pl.col("hospitalization_id") == f"{search_id}.0")
                    | (pl.col("hospitalization_id").str.replace(r"\.0$", "") == search_id)
                )
                cast_method = "string: multi-match"

            # Collect only matching rows
            filtered = lazy.filter(filter_expr).collect()

            self._trace("note_load", note_type_key,
                         f"Scanned {path.name}: {len(filtered)} rows for hosp_id={search_id} "
                         f"(col_dtype={col_dtype}, cast={cast_method})",
                         data={"path": str(path), "rows": len(filtered),
                               "col_dtype": str(col_dtype), "cast_method": cast_method,
                               "method": "lazy_scan"})

            if len(filtered) == 0:
                return None

        except Exception as e:
            self._trace("note_load", note_type_key,
                         f"Scan failed for {path.name}: {e}, falling back to full load",
                         level="warning")
            # Fallback: full load + filter
            try:
                df = pl.read_csv(str(path), infer_schema_length=10000)
                if "hospitalization_id" not in df.columns:
                    return None
                # Normalize ID column
                hosp_col = df["hospitalization_id"]
                if hosp_col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
                    df = df.with_columns(
                        pl.col("hospitalization_id").cast(pl.Int64, strict=False).cast(pl.Utf8)
                    )
                elif hosp_col.dtype == pl.Utf8:
                    df = df.with_columns(
                        pl.col("hospitalization_id").str.replace(r"\.0$", "")
                    )
                filtered = df.filter(pl.col("hospitalization_id") == str(hospitalization_id))
                if len(filtered) == 0:
                    return None
                self._trace("note_load", note_type_key,
                             f"Fallback load {path.name}: {len(filtered)} rows",
                             data={"rows": len(filtered), "method": "full_load_fallback"})
            except Exception as e2:
                logger.warning(f"Failed to read note CSV {path}: {e2}")
                return None

        # 2. Data-leakage guard: exclude notes created or revised after reference_dttm
        #    Normalize note timestamps to UTC (tz-naive notes assumed to be in
        #    settings.timezone). reference_dttm is already UTC from retrieve().
        pre_filter_count = len(filtered)
        if reference_dttm is not None:
            ref = reference_dttm  # already UTC from retrieve()
            for col in ("creation_dttm", "revision_dttm"):
                if col in filtered.columns:
                    filtered = self._normalize_column_to_utc(filtered, col)

            if "creation_dttm" in filtered.columns:
                filtered = filtered.filter(
                    pl.col("creation_dttm").is_null() | (pl.col("creation_dttm") < ref)
                )
            if "revision_dttm" in filtered.columns:
                filtered = filtered.filter(
                    pl.col("revision_dttm").is_null() | (pl.col("revision_dttm") < ref)
                )

            excluded = pre_filter_count - len(filtered)
            if excluded > 0:
                logger.info(
                    f"Note leakage guard ({note_type_key}): excluded {excluded} notes "
                    f"at/after ref={ref}"
                )

            if len(filtered) == 0:
                return None

        # Log which notes survived filtering (for debugging leakage)
        if len(filtered) > 0:
            preview_cols = ["note_id", "creation_dttm"]
            if "note_text" in filtered.columns:
                preview_cols.append("note_text")
            available_cols = [c for c in preview_cols if c in filtered.columns]
            note_previews = []
            for row in filtered.select(available_cols).iter_rows(named=True):
                preview = {
                    "note_id": str(row.get("note_id", "?")),
                    "creation_dttm": str(row.get("creation_dttm", "?")),
                }
                if "note_text" in row and row["note_text"]:
                    preview["text_preview"] = row["note_text"][:150] + "..." if len(row["note_text"]) > 150 else row["note_text"]
                note_previews.append(preview)
            self._trace(
                "note_load", note_type_key,
                f"After leakage filter: {len(filtered)} notes kept "
                f"(ref={reference_dttm}, excluded={pre_filter_count - len(filtered)})",
                data={"notes": note_previews},
            )

            # 3. Lookback window: only notes within [ref - lookback, ref].
            #    Per-admission stable note types (``PER_ADMISSION_STABLE_NOTE_TYPES``,
            #    currently only ``hp_note``) are written once and remain
            #    valid for the rest of the stay — exempt them from the
            #    recency floor so a long ICU stay does not silently drop
            #    its H&P. Leakage guard above still applies.
            if (
                notes_lookback_hours is not None
                and "creation_dttm" in filtered.columns
                and note_type_key not in PER_ADMISSION_STABLE_NOTE_TYPES
            ):
                lookback_start = ref - timedelta(hours=notes_lookback_hours)
                filtered = filtered.filter(
                    pl.col("creation_dttm").is_null()
                    | (pl.col("creation_dttm") >= lookback_start)
                )
                if len(filtered) == 0:
                    return None

        # 4. Deduplicate: for each note_id, keep the most recent revision.
        #    If the most recent revision is short (<200 chars in note_length),
        #    it's likely an addendum — also keep the next most recent
        #    comprehensive revision.
        pre_dedup_count = len(filtered)
        filtered = self._deduplicate_notes(filtered, note_type_key)

        self._trace("note_dedup", note_type_key,
                     f"Dedup: {pre_dedup_count} -> {len(filtered)} rows",
                     data={"before": pre_dedup_count, "after": len(filtered)})

        logger.info(
            f"Notes '{note_type_key}' for {hospitalization_id}: {len(filtered)} rows "
            f"(lookback={notes_lookback_hours}h, deduped from {pre_dedup_count})"
        )
        return filtered

    # Minimum note_length to be considered a "comprehensive" note (not an addendum)
    ADDENDUM_THRESHOLD = 200

    def _deduplicate_notes(
        self, df: pl.DataFrame, note_type_key: str
    ) -> pl.DataFrame:
        """Deduplicate notes by note_id, keeping the most recent revision.

        When multiple rows share the same note_id, they are different revisions
        of the same note. We keep the most recent revision (by revision_dttm).

        However, if the most recent revision is short (note_length < 200),
        it's likely just an addendum. In that case, we also keep the next
        most recent revision that is comprehensive (note_length >= 200),
        so the agent sees both the full note and the addendum.
        """
        if "note_id" not in df.columns:
            return df
        if "revision_dttm" not in df.columns:
            # Can't deduplicate without revision timestamps — return as-is
            return df

        # Parse revision_dttm if string
        if df["revision_dttm"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("revision_dttm").str.to_datetime(strict=False)
            )

        # Parse note_length: may be string or numeric
        has_note_length = "note_length" in df.columns
        if has_note_length and df["note_length"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("note_length").cast(pl.Int64, strict=False)
            )

        # Sort by note_id and revision_dttm descending (most recent first)
        df = df.sort(["note_id", "revision_dttm"], descending=[False, True])

        # Get unique note_ids
        note_ids = df["note_id"].unique().to_list()

        if len(note_ids) == len(df):
            # Already unique by note_id — no dedup needed
            return df

        keep_rows = []
        for note_id in note_ids:
            revisions = df.filter(pl.col("note_id") == note_id)
            if len(revisions) == 1:
                keep_rows.append(revisions)
                continue

            # Most recent revision
            latest = revisions.head(1)
            keep_rows.append(latest)

            # Check if the latest is short (addendum)
            if has_note_length:
                latest_length = latest["note_length"][0]
                if latest_length is not None and latest_length < self.ADDENDUM_THRESHOLD:
                    # Find the next most recent comprehensive revision
                    comprehensive = revisions.tail(-1).filter(
                        pl.col("note_length") >= self.ADDENDUM_THRESHOLD
                    )
                    if len(comprehensive) > 0:
                        keep_rows.append(comprehensive.head(1))
                        self._trace(
                            "note_dedup", note_id,
                            f"Kept addendum ({latest_length} chars) + comprehensive "
                            f"revision ({comprehensive['note_length'][0]} chars)",
                            level="debug",
                        )
            elif "note_text" in revisions.columns:
                # Fallback: use note_text length if note_length column missing
                latest_text = latest["note_text"][0]
                if latest_text is not None and len(str(latest_text)) < self.ADDENDUM_THRESHOLD:
                    older = revisions.tail(-1)
                    longer = older.filter(
                        pl.col("note_text").str.len_chars() >= self.ADDENDUM_THRESHOLD
                    )
                    if len(longer) > 0:
                        keep_rows.append(longer.head(1))

        if keep_rows:
            return pl.concat(keep_rows)
        return df

    def load_agent_notes(
        self,
        agent_role: str,
        hospitalization_id: str,
        reference_dttm: Optional[datetime],
        notes_lookback_hours: Optional[int] = None,
        skip_per_admission_stable: bool = False,
    ) -> dict[str, pl.DataFrame]:
        """Load all note types assigned to a specific agent.

        Args:
            agent_role: Agent name (e.g. "nurse", "respiratory").
            hospitalization_id: The hospitalization to load notes for.
            reference_dttm: Cut-off time for data leakage prevention.
                None means no future-filtering (prospective / "now" mode).
            notes_lookback_hours: Override for how far back to look.
                Falls back to self.notes_lookback_hours if None.
            skip_per_admission_stable: When True, skip note types listed in
                ``PER_ADMISSION_STABLE_NOTE_TYPES`` (currently just hp_note).
                Used by the physician-note-floor full-stay pass: the lookback
                pass already retrieves per-admission-stable types via their
                lookback exemption, so re-scanning them with no lookback is
                redundant work whose only effect is to populate
                ``agent_notes_full_stay`` with data the floor logic never
                reads (the lookback pass has already satisfied the
                "physician note present" predicate the floor checks).

        Returns:
            Dict mapping note_type_key -> DataFrame for this agent.
            Only includes types that had data after filtering.
        """
        if notes_lookback_hours is None:
            notes_lookback_hours = self.notes_lookback_hours

        note_types = list(AGENT_NOTE_ROUTING.get(agent_role, []))
        if skip_per_admission_stable:
            note_types = [
                nt for nt in note_types
                if nt not in PER_ADMISSION_STABLE_NOTE_TYPES
            ]
        result: dict[str, pl.DataFrame] = {}

        for note_type_key in note_types:
            df = self._load_notes_for_hospitalization(
                note_type_key, hospitalization_id, reference_dttm, notes_lookback_hours
            )
            if df is not None:
                result[note_type_key] = df

        # Per-agent routing summary — emitted so downstream A/B comparisons
        # across runs can group by ``routing_version`` and ask "for runs at
        # v2, did the new hp_note arrival actually change this agent's
        # brief?". The full-stay variant (called with
        # ``skip_per_admission_stable=True`` for the physician-note floor)
        # is tagged separately so the two passes don't collide.
        retrieved = {nt: len(df) for nt, df in result.items()}
        hp_routed = "hp_note" in note_types
        hp_retrieved = retrieved.get("hp_note", 0)
        self._trace(
            "agent_routing_summary",
            agent_role,
            (
                f"routing_version={NOTE_ROUTING_VERSION} "
                f"lookback_h={notes_lookback_hours} "
                f"skip_per_admission_stable={skip_per_admission_stable} "
                f"hp_note_routed={hp_routed} "
                f"hp_note_retrieved={hp_retrieved} "
                f"types_with_data={list(retrieved)}"
            ),
            data={
                "agent_role": agent_role,
                "routing_version": NOTE_ROUTING_VERSION,
                "lookback_hours": notes_lookback_hours,
                "skip_per_admission_stable": skip_per_admission_stable,
                "routed_note_types": list(note_types),
                "retrieved_note_types": retrieved,
                "hp_note_routed": hp_routed,
                "hp_note_retrieved_count": hp_retrieved,
                "per_admission_stable_exempt": sorted(PER_ADMISSION_STABLE_NOTE_TYPES),
            },
        )

        return result

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        hospitalization_id: str,
        reference_dttm: datetime,
        lookback_hours: Optional[int] = 48,
        notes_lookback_hours: Optional[int] = None,
    ) -> PatientContext:
        """Extract all CLIF data for a single hospitalization.

        Args:
            hospitalization_id: The CLIF hospitalization identifier.
            reference_dttm: The "current" time for the pipeline (required).
                Must be supplied externally — typically the first ICU→ward
                transfer note time from the MICU note IDs CSV. Notes and
                structured data after this timestamp are excluded to prevent
                data leakage. Auto-detection has been removed because for
                multi-stay patients it silently anchored on the readmission.
            lookback_hours: Number of hours of structured data to include before
                reference_dttm. Defaults to 48. Pass None for entire stay.
            notes_lookback_hours: How many hours of notes to include (from
                reference_dttm). Falls back to settings.notes_lookback_hours
                (default 48) if None.

        Returns:
            PatientContext with all available data loaded.

        Raises:
            ValueError: If reference_dttm is None.
        """
        if reference_dttm is None:
            raise ValueError(
                "reference_dttm is required. Auto-detection has been removed — "
                "supply the first ICU→ward transfer note time from the cohort CSV."
            )
        # Load hospitalization record
        hosp_df = self._filter_by_hosp("hospitalization", hospitalization_id)
        if hosp_df is None or len(hosp_df) == 0:
            logger.warning(f"No hospitalization found for {hospitalization_id}")
            return PatientContext(
                hospitalization_id=hospitalization_id,
                patient_id="unknown",
            )

        row = hosp_df.row(0, named=True)
        patient_id = row.get("patient_id", "unknown")

        # Load ADT (ICU window extracted after reference_dttm is resolved,
        # so readmits can be disambiguated to the relevant ICU stay).
        adt_df = self._filter_by_hosp("adt", hospitalization_id)

        # Normalize reference_dttm to UTC for consistent comparisons.
        # If tz-naive, assume settings.timezone (e.g. America/Chicago).
        # If tz-aware (e.g. user passed '2024-01-12T19:35:00Z'), respect it.
        reference_dttm = self._to_utc(reference_dttm)
        logger.info(f"Reference dttm normalized to UTC: {reference_dttm}")

        icu_start, icu_end = self._extract_icu_window(adt_df, reference_dttm)

        # Determine the time window for filtering structured clinical data.
        # Uses reference_dttm as the hard ceiling to prevent data leakage —
        # no structured data recorded after reference_dttm is included.
        if lookback_hours is not None:
            start = reference_dttm - timedelta(hours=lookback_hours)
            time_window = (start, reference_dttm)
            logger.info(
                f"Structured data window: {start} to {reference_dttm} "
                f"(lookback={lookback_hours}h from reference_dttm)"
            )
        else:
            # Entire stay but capped at reference_dttm (no future data)
            time_window = (None, reference_dttm)
            logger.info(
                f"Structured data window: entire stay up to {reference_dttm} "
                f"(capped at reference_dttm to prevent data leakage)"
            )

        # Build context — demographics and ADT are always loaded (minimal cost,
        # needed for ICU window detection regardless of modality toggles)
        ctx = PatientContext(
            hospitalization_id=hospitalization_id,
            patient_id=patient_id,
            age_at_admission=row.get("age_at_admission"),
            sex_category=row.get("sex_category"),
            admission_dttm=str(row.get("admission_dttm", "")),
            discharge_dttm=str(row.get("discharge_dttm", "")),
            admission_type_category=row.get("admission_type_category"),
            discharge_category=row.get("discharge_category"),
            icu_admission_dttm=str(icu_start) if icu_start else None,
            icu_discharge_dttm=str(icu_end) if icu_end else None,
            adt=adt_df,
        )

        # Structured clinical data (conditionally loaded)
        if self.structured_data_enabled:
            ctx.vitals = self._filter_by_hosp("vitals", hospitalization_id, time_window)
            # Labs use a custom filter so a lab collected before transfer
            # but not yet resulted at reference_dttm survives as a
            # "pending" row instead of being silently dropped by a
            # lab_result_dttm-only filter. Result fields are masked.
            ctx.labs = self._filter_labs_with_pending(
                hospitalization_id, time_window, reference_dttm
            )
            ctx.meds_continuous = self._filter_by_hosp(
                "medication_admin_continuous", hospitalization_id, time_window
            )
            ctx.meds_intermittent = self._filter_by_hosp(
                "medication_admin_intermittent", hospitalization_id, time_window
            )
            ctx.respiratory_support = self._filter_by_hosp(
                "respiratory_support", hospitalization_id, time_window
            )
            ctx.patient_assessments = self._filter_by_hosp(
                "patient_assessments", hospitalization_id, time_window
            )
            ctx.code_status = self._filter_code_status_with_leakage_guard(
                hospitalization_id, patient_id, reference_dttm
            )
            ctx.diagnoses = self._filter_by_hosp("hospital_diagnosis", hospitalization_id)
            ctx.microbiology = self._filter_by_hosp(
                "microbiology_culture", hospitalization_id, time_window
            )
            # collect_dttm filter alone leaks future results: a culture
            # collected before reference_dttm but resulted after it would
            # otherwise expose the organism / sensitivities. Mask the
            # result fields while keeping the row visible as "pending".
            ctx.microbiology = self._mask_future_microbiology_results(
                ctx.microbiology, reference_dttm
            )
            ctx.procedures = self._filter_by_hosp(
                "patient_procedures", hospitalization_id, time_window
            )
            ctx.crrt_therapy = self._filter_by_hosp("crrt_therapy", hospitalization_id, time_window)
            ctx.ecmo_mcs = self._filter_by_hosp("ecmo_mcs", hospitalization_id, time_window)
            ctx.position = self._filter_by_hosp("position", hospitalization_id, time_window)
        else:
            logger.info("Structured data disabled — skipping CLIF table loading")

        # Load patient demographics (sex, race, birth_date) — always loaded.
        # birth_date drives age_at_icu_admission, which is preferred over
        # CLIF's age_at_admission for display + agent prompts because CLIF's
        # value is anchored at hospital admit and can be off by 1 around the
        # patient's birthday or for long pre-ICU ward stays.
        patient_df = self._filter_by_patient("patient", patient_id)
        if patient_df is not None and len(patient_df) > 0:
            p_row = patient_df.row(0, named=True)
            ctx.race_category = p_row.get("race_category")
            if ctx.sex_category is None:
                ctx.sex_category = p_row.get("sex_category")
            ctx.birth_date = self._coerce_birth_date(p_row.get("birth_date"))
            if ctx.birth_date is not None and icu_start is not None:
                ctx.age_at_icu_admission = self._age_years(ctx.birth_date, icu_start)
                if (
                    ctx.age_at_admission is not None
                    and abs(ctx.age_at_icu_admission - ctx.age_at_admission) >= 1
                ):
                    logger.warning(
                        "Age mismatch for hosp_id=%s: CLIF age_at_admission=%s, "
                        "computed age_at_icu_admission=%s (birth_date=%s, "
                        "icu_admission_dttm=%s). Using computed value.",
                        hospitalization_id,
                        ctx.age_at_admission,
                        ctx.age_at_icu_admission,
                        ctx.birth_date,
                        icu_start,
                    )

        # Load per-agent notes from individual CSV files (conditionally)
        if self.notes_enabled:
            effective_notes_lookback = (
                notes_lookback_hours if notes_lookback_hours is not None
                else self.notes_lookback_hours
            )
            for agent_role in AGENT_NOTE_ROUTING:
                agent_notes = self.load_agent_notes(
                    agent_role,
                    hospitalization_id,
                    reference_dttm,
                    effective_notes_lookback,
                )
                ctx.agent_notes[agent_role] = agent_notes

            # Physician-note context floor — also load each floor-eligible
            # agent's notes WITHOUT a lookback so the floor can fall back
            # to an earlier-stay physician note when the 48h window has
            # only ancillary notes.  Cap at < reference_dttm is preserved
            # by load_agent_notes' leakage guard.
            #
            # ``skip_per_admission_stable=True``: hp_note (and any future
            # per-admission-stable types) is already retrieved by the
            # lookback pass above via its lookback exemption, so
            # re-scanning it here just to populate ``agent_notes_full_stay``
            # is wasted work — the floor logic checks "does the lookback
            # frame already contain a physician note?", and the
            # exemption-loaded hp_note satisfies that check, so the
            # full-stay copy of hp_note is never consumed.
            for agent_role in FLOOR_ELIGIBLE_AGENTS:
                if agent_role not in AGENT_NOTE_ROUTING:
                    continue
                full_stay_notes = self.load_agent_notes(
                    agent_role,
                    hospitalization_id,
                    reference_dttm,
                    notes_lookback_hours=None,
                    skip_per_admission_stable=True,
                )
                ctx.agent_notes_full_stay[agent_role] = full_stay_notes
        else:
            logger.info("Notes disabled — skipping clinical notes loading")

        # Store reference_dttm on context for downstream use
        ctx.reference_dttm = reference_dttm

        # Deterministic chronic-condition + baseline inference. Runs after all
        # CLIF tables are attached so detection rules can read them directly.
        # Imported here (not at module top) to avoid a circular import:
        # safety.clinical_context imports PatientContext from data.context.
        from icu_pause.safety.clinical_context import infer_clinical_context

        try:
            ctx.clinical_context = infer_clinical_context(ctx)
        except Exception as exc:  # pragma: no cover — defensive only
            logger.warning(
                "infer_clinical_context failed for %s: %s",
                hospitalization_id,
                exc,
            )
            ctx.clinical_context = None

        return ctx

    def get_transfer_note(
        self, hospitalization_id: str
    ) -> Optional[pl.DataFrame]:
        """Load the transfer note for a hospitalization (gold-standard reference).

        This is separate from the agent pipeline — used by the evaluation
        harness to compare generated ICU-PAUSE notes against the physician-
        written transfer summary.
        """
        facts = self._filter_by_hosp("clinical_notes_facts", hospitalization_id)
        if facts is None or "note_type" not in facts.columns:
            return None

        transfer_facts = facts.filter(pl.col("note_type") == "transfer_note")
        if len(transfer_facts) == 0:
            return None

        text = self._filter_by_hosp("clinical_notes_text", hospitalization_id)
        if text is None:
            return transfer_facts

        joined = transfer_facts.join(
            text.select(["note_id", "revision_id", "note_text"]),
            on=["note_id", "revision_id"],
            how="left",
        )
        return joined if len(joined) > 0 else None

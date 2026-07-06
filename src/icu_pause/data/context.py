"""PatientContext dataclass and serialization utilities for CLIF DataFrames."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

import polars as pl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Citation format constants
# ---------------------------------------------------------------------------
# The formatter (_add_cite_fields) and regex (CITE_PATTERN) MUST agree.
# If you change the format in one, update the other.
# Format: (source_type M-DD HH:MM) — hyphen date separator (not slash — LLMs merge "1/09" into "109").
#
# Phase 3: exam-vitals / exam-neuro / exam-resp source types added for the
# deterministic transfer-exam block. Each tag carries a single anchor
# timestamp (the latest in-window row); per-row timestamps live on
# CitationRow.time and the tooltip renders them when divergent from the
# anchor.
_CITE_SOURCE_TYPES = {
    "lab", "vital", "med", "resp", "assess", "code", "proc",
    "exam-vitals", "exam-neuro", "exam-resp",
    # Note types — one per AGENT_NOTE_ROUTING key. Format parity with
    # structured types: tag shape is "(<note_type> M-DD HH:MM)" keyed on
    # the note's revision_dttm (with creation_dttm fallback). cite_registry
    # rows hold the full routed-note row by reference, so a Focus rationale
    # cite resolves through the same build_citation_index path as a lab tag.
    "progress_note", "hp_note", "consults_note", "plan_of_care_note",
    "nursing_note", "case_management_note", "social_work_note", "therapy_note",
}
CITE_PATTERN = re.compile(
    r"\((?:lab|vital|med|resp|assess|code|proc"
    r"|exam-vitals|exam-neuro|exam-resp"
    r"|progress_note|hp_note|consults_note|plan_of_care_note"
    r"|nursing_note|case_management_note|social_work_note|therapy_note) "
    r"\d{1,2}-\d{2} \d{2}:\d{2}\)"
)


# Default display timezone for any timestamp the LLM will see. Matches
# the cite-tag format in _add_cite_fields and the reviewer source-table
# formatter in review_app/display/source_renderer.py — clinicians should
# never see a `+00:00` in agent output.
_DEFAULT_DISPLAY_TZ = ZoneInfo("America/Chicago")


def format_local_dttm(
    value: Any, tz: Optional[ZoneInfo] = None
) -> str:
    """Format a timestamp to ``M-DD HH:MM`` in the display timezone.

    Matches the cite-tag format from ``_add_cite_fields`` so agents,
    cite tags, and the reviewer source table all show the same hour.
    Naive datetimes are assumed to be UTC.

    Returns ``"?"`` for None/empty values, and the original string
    representation when the value cannot be parsed.
    """
    if value is None or value == "":
        return "?"
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value)
        try:
            normalized = s.replace(" ", "T", 1) if "T" not in s[:11] else s
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return s
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_local = dt.astimezone(tz or _DEFAULT_DISPLAY_TZ)
    return (
        f"{dt_local.month}-{dt_local.day:02d} "
        f"{dt_local.hour:02d}:{dt_local.minute:02d}"
    )


@dataclass
class PatientContext:
    """All structured CLIF data for a single hospitalization, scoped to the ICU stay."""

    hospitalization_id: str
    patient_id: str

    # Demographics
    age_at_admission: Optional[int] = None  # CLIF hospitalization.age_at_admission (raw)
    age_at_icu_admission: Optional[int] = None  # computed from birth_date + ICU admit time
    birth_date: Optional[date] = None  # CLIF patient.birth_date; source for age_at_icu_admission
    sex_category: Optional[str] = None
    race_category: Optional[str] = None
    admission_dttm: Optional[str] = None
    discharge_dttm: Optional[str] = None
    admission_type_category: Optional[str] = None
    discharge_category: Optional[str] = None

    # ICU window
    icu_admission_dttm: Optional[str] = None
    icu_discharge_dttm: Optional[str] = None

    # Clinical DataFrames (filtered to ICU window)
    adt: Optional[pl.DataFrame] = None
    vitals: Optional[pl.DataFrame] = None
    labs: Optional[pl.DataFrame] = None
    meds_continuous: Optional[pl.DataFrame] = None
    meds_intermittent: Optional[pl.DataFrame] = None
    respiratory_support: Optional[pl.DataFrame] = None
    patient_assessments: Optional[pl.DataFrame] = None

    # Episodic / status data
    code_status: Optional[pl.DataFrame] = None
    diagnoses: Optional[pl.DataFrame] = None
    microbiology: Optional[pl.DataFrame] = None
    procedures: Optional[pl.DataFrame] = None
    crrt_therapy: Optional[pl.DataFrame] = None
    ecmo_mcs: Optional[pl.DataFrame] = None
    position: Optional[pl.DataFrame] = None

    # Per-agent notes: agent_role -> {note_type_key -> DataFrame}
    # Populated by DataRetriever.load_agent_notes(); replaces the old
    # single ``clinical_notes`` DataFrame.
    agent_notes: dict[str, dict[str, pl.DataFrame]] = field(default_factory=dict)

    # Per-agent full-hospitalization notes (no lookback filter, still capped
    # at < reference_dttm by the upstream leakage guard). Populated only for
    # the floor-eligible agents (intensivist, respiratory, pharmacy,
    # dietitian) so the physician-note context floor in
    # data.note_floor.ensure_physician_note_floor can pull a fallback note
    # from earlier in the stay when the 48h window has only ancillary notes.
    agent_notes_full_stay: dict[str, dict[str, pl.DataFrame]] = field(default_factory=dict)

    # Legacy: kept for backward compatibility with evaluation harness
    clinical_notes: Optional[pl.DataFrame] = None  # joined facts + text

    # Reference timestamp used for data-leakage filtering
    reference_dttm: Optional[datetime] = None

    # Derived scores
    sofa_scores: Optional[pl.DataFrame] = None

    # Deterministic patient-level chronic conditions and baselines, populated
    # at the end of DataRetriever.retrieve(). See safety/clinical_context.py.
    # Typed as Any to avoid a circular import at module load time
    # (safety.clinical_context imports PatientContext).
    clinical_context: Optional[Any] = None


class ContextSerializer:
    """Converts PatientContext DataFrames into text summaries for LLM prompts."""

    @staticmethod
    def _compute_icu_stay_hours(ctx: PatientContext) -> Optional[float]:
        """Compute ICU length of stay in hours from ADT timestamps."""
        if not ctx.icu_admission_dttm or not ctx.icu_discharge_dttm:
            return None
        try:
            # Timestamps stored as strings; parse them
            fmt_candidates = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S%z"]
            icu_in = icu_out = None
            for fmt in fmt_candidates:
                try:
                    icu_in = datetime.strptime(str(ctx.icu_admission_dttm), fmt)
                    icu_out = datetime.strptime(str(ctx.icu_discharge_dttm), fmt)
                    break
                except ValueError:
                    continue
            if icu_in is None or icu_out is None:
                return None
            return (icu_out - icu_in).total_seconds() / 3600
        except Exception:
            return None

    @staticmethod
    def demographics_summary(
        ctx: PatientContext,
        icu_stay_hours: Optional[float] = None,
        effective_window_hours: Optional[float] = None,
    ) -> str:
        parts = []
        age = ctx.age_at_icu_admission if ctx.age_at_icu_admission is not None else ctx.age_at_admission
        if age is not None:
            parts.append(f"Age: {age}")
        if ctx.sex_category:
            parts.append(f"Sex: {ctx.sex_category}")
        if ctx.race_category:
            parts.append(f"Race: {ctx.race_category}")
        if ctx.admission_type_category:
            parts.append(f"Admission type: {ctx.admission_type_category}")
        if ctx.admission_dttm:
            parts.append(f"Hospital admission: {format_local_dttm(ctx.admission_dttm)}")
        if ctx.icu_admission_dttm:
            parts.append(f"ICU admission: {format_local_dttm(ctx.icu_admission_dttm)}")
        if ctx.reference_dttm is not None:
            parts.append(f"Data as of (transfer note time): {format_local_dttm(ctx.reference_dttm)}")
        if icu_stay_hours is not None:
            parts.append(f"ICU length of stay: {icu_stay_hours:.1f}h ({icu_stay_hours / 24:.1f} days)")
        if effective_window_hours is not None:
            parts.append(f"Data window for this report: last {effective_window_hours:.0f}h")
        return "Demographics:\n" + "\n".join(f"  {p}" for p in parts) if parts else "No demographics data available."

    @staticmethod
    def adt_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No ADT data available."
        lines = ["ADT (Admit-Discharge-Transfer) movements:"]
        for row in df.sort("in_dttm").iter_rows(named=True):
            location = row.get("location_category", "unknown")
            in_dt = format_local_dttm(row.get("in_dttm"))
            out_dt = format_local_dttm(row.get("out_dttm"))
            lines.append(f"  {location}: {in_dt} -> {out_dt}")
        return "\n".join(lines)

    @staticmethod
    def vitals_summary(
        df: Optional[pl.DataFrame],
        effective_window_hours: Optional[float] = None,
        bucket_hours: int = 8,
    ) -> str:
        """Summarise vitals with time-bucketed medians so agents can see trends.

        Buckets are anchored to the effective data window (ending at the latest
        data point), so empty early buckets show up as ``-`` rather than being
        silently omitted.

        Args:
            df: Vitals DataFrame with ``recorded_dttm`` and ``vital_value``.
            effective_window_hours: The data window to cover (e.g. 48 or 24).
                When *None* the window is derived from the data span.
            bucket_hours: Width of each time bucket (default 8).
        """
        if df is None or len(df) == 0:
            return "No vitals data available."

        if "recorded_dttm" not in df.columns or "vital_value" not in df.columns:
            return "No vitals data available."

        time_max = df["recorded_dttm"].max()
        if time_max is None:
            return "No vitals data available."

        # Anchor the window: go back effective_window_hours from the latest data point
        if effective_window_hours is not None:
            window_start = time_max - timedelta(hours=effective_window_hours)
            window_hours = effective_window_hours
        else:
            time_min = df["recorded_dttm"].min()
            if time_min is None:
                return "No vitals data available."
            window_start = time_min
            window_hours = (time_max - time_min).total_seconds() / 3600

        # Build fixed bucket boundaries from window_start to time_max
        n_buckets = max(1, int(window_hours // bucket_hours) + (1 if window_hours % bucket_hours else 0))
        bucket_delta = timedelta(hours=bucket_hours)
        boundaries = [window_start + i * bucket_delta for i in range(n_buckets + 1)]
        # Ensure the last boundary covers time_max
        if boundaries[-1] < time_max:
            boundaries.append(time_max + timedelta(seconds=1))
        n_actual_buckets = len(boundaries) - 1

        # Build bucket labels relative to "now" (e.g. "48-40h ago", "8-0h ago")
        bucket_labels = []
        for i in range(n_actual_buckets):
            h_ago_start = int((n_actual_buckets - i) * bucket_hours)
            h_ago_end = int((n_actual_buckets - i - 1) * bucket_hours)
            bucket_labels.append(f"{h_ago_start}-{h_ago_end}h ago")

        lines = [f"Vitals (last {int(window_hours)}h, {bucket_hours}h buckets, oldest->newest: {', '.join(bucket_labels)}):"]

        for category in sorted(df["vital_category"].unique().to_list()):
            subset = df.filter(
                (pl.col("vital_category") == category) & pl.col("vital_value").is_not_null()
            ).sort("recorded_dttm")
            values = subset["vital_value"].drop_nulls()
            if len(values) == 0:
                continue

            # Compute median per bucket
            bucket_medians = []
            for i in range(n_actual_buckets):
                bucket_df = subset.filter(
                    (pl.col("recorded_dttm") >= boundaries[i])
                    & (pl.col("recorded_dttm") < boundaries[i + 1])
                )
                bvals = bucket_df["vital_value"].drop_nulls()
                if len(bvals) > 0:
                    bucket_medians.append(f"{bvals.median():.0f}")
                else:
                    bucket_medians.append("-")

            trend = " -> ".join(bucket_medians)
            vmin = values.min()
            vmax = values.max()
            count = len(values)
            lines.append(f"  {category}: {trend}  (min={vmin:.0f}, max={vmax:.0f}, n={count})")

        return "\n".join(lines)

    @staticmethod
    def labs_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No labs data available."
        lines = ["Labs (ICU stay):"]
        for category in sorted(df["lab_category"].unique().to_list()):
            subset = df.filter(pl.col("lab_category") == category).sort("lab_result_dttm")
            # Try numeric first
            if "lab_value_numeric" in subset.columns:
                values = subset["lab_value_numeric"].drop_nulls()
                if len(values) > 0:
                    latest = values[-1]
                    lines.append(f"  {category}: latest={latest}")
                    continue
            # Fall back to text
            if "lab_value" in subset.columns:
                vals = subset["lab_value"].drop_nulls()
                if len(vals) > 0:
                    lines.append(f"  {category}: latest={vals[-1]}")
        return "\n".join(lines)

    @staticmethod
    def meds_summary(
        cont_df: Optional[pl.DataFrame],
        intermittent_df: Optional[pl.DataFrame],
    ) -> str:
        lines = ["Medications (ICU stay):"]
        has_data = False

        if cont_df is not None and len(cont_df) > 0:
            has_data = True
            lines.append("  Continuous infusions:")
            med_col = "med_category" if "med_category" in cont_df.columns else "medication_name"
            if med_col in cont_df.columns:
                for med in sorted(cont_df[med_col].unique().to_list()):
                    subset = cont_df.filter(pl.col(med_col) == med).sort("admin_dttm")
                    dose_col = next(
                        (c for c in ["med_dose", "dose"] if c in subset.columns), None
                    )
                    if dose_col and len(subset[dose_col].drop_nulls()) > 0:
                        latest_dose = subset[dose_col].drop_nulls()[-1]
                        lines.append(f"    {med}: latest dose={latest_dose}")
                    else:
                        lines.append(f"    {med}: active")

        if intermittent_df is not None and len(intermittent_df) > 0:
            has_data = True
            lines.append("  Intermittent medications:")
            med_col = "med_category" if "med_category" in intermittent_df.columns else "medication_name"
            if med_col in intermittent_df.columns:
                for med in sorted(intermittent_df[med_col].unique().to_list()):
                    subset = intermittent_df.filter(pl.col(med_col) == med)
                    count = len(subset)
                    lines.append(f"    {med}: {count} administrations")

        if not has_data:
            return "No medication data available."
        return "\n".join(lines)

    @staticmethod
    def respiratory_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No respiratory support data available."
        lines = ["Respiratory support (ICU stay):"]
        sorted_df = df.sort("recorded_dttm")

        # Latest device
        if "device_category" in sorted_df.columns:
            devices = sorted_df["device_category"].drop_nulls()
            if len(devices) > 0:
                lines.append(f"  Current device: {devices[-1]}")

        # Latest mode
        if "mode_category" in sorted_df.columns:
            modes = sorted_df["mode_category"].drop_nulls()
            if len(modes) > 0:
                lines.append(f"  Current mode: {modes[-1]}")

        # Key parameters
        param_cols = {
            "fio2_set": "FiO2 set",
            "peep_set": "PEEP set",
            "tidal_volume_set": "TV set",
            "resp_rate_set": "RR set",
            "pressure_support_set": "PS set",
            "flow_rate_set": "Flow rate set",
        }
        for col, label in param_cols.items():
            if col in sorted_df.columns:
                values = sorted_df[col].drop_nulls()
                if len(values) > 0:
                    lines.append(f"  {label}: latest={values[-1]}")

        # Device progression
        if "device_category" in sorted_df.columns:
            device_changes = sorted_df.select(["recorded_dttm", "device_category"]).drop_nulls()
            if len(device_changes) > 1:
                progression = device_changes["device_category"].to_list()
                unique_seq = []
                for d in progression:
                    if not unique_seq or unique_seq[-1] != d:
                        unique_seq.append(d)
                if len(unique_seq) > 1:
                    lines.append(f"  Device progression: {' -> '.join(unique_seq)}")

        return "\n".join(lines)

    @staticmethod
    def assessments_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No patient assessment data available."
        lines = ["Patient assessments (ICU stay):"]
        cat_col = "assessment_category" if "assessment_category" in df.columns else None
        if cat_col is None:
            return "No patient assessment data available."
        for category in sorted(df[cat_col].unique().to_list()):
            subset = df.filter(pl.col(cat_col) == category).sort("recorded_dttm")
            val_col = next(
                (c for c in ["assessment_value", "numerical_value"] if c in subset.columns),
                None,
            )
            if val_col and len(subset[val_col].drop_nulls()) > 0:
                latest = subset[val_col].drop_nulls()[-1]
                lines.append(f"  {category}: latest={latest}")
            else:
                lines.append(f"  {category}: recorded (value unavailable)")
        return "\n".join(lines)

    @staticmethod
    def code_status_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No code status data available."
        lines = ["Code status:"]
        time_col = next(
            (c for c in ["recorded_dttm", "start_dttm"] if c in df.columns), None
        )
        sorted_df = df.sort(time_col) if time_col else df
        code_col = next(
            (c for c in ["code_status_category", "code_status"] if c in sorted_df.columns),
            None,
        )
        if code_col:
            latest = sorted_df[code_col].drop_nulls()
            if len(latest) > 0:
                lines.append(f"  Current: {latest[-1]}")
        return "\n".join(lines)

    @staticmethod
    def diagnoses_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No diagnosis data available."
        lines = ["Hospital diagnoses:"]
        code_col = next(
            (c for c in ["diagnosis_code", "icd_code"] if c in df.columns), None
        )
        name_col = next(
            (c for c in ["diagnosis_name", "icd_name", "description"] if c in df.columns),
            None,
        )
        if code_col:
            for row in df.iter_rows(named=True):
                code = row.get(code_col, "")
                name = row.get(name_col, "") if name_col else ""
                if name:
                    lines.append(f"  {code}: {name}")
                else:
                    lines.append(f"  {code}")
        return "\n".join(lines[:20])  # Limit to 20 diagnoses

    @staticmethod
    def microbiology_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No microbiology data available."
        lines = ["Microbiology cultures:"]
        for row in df.iter_rows(named=True):
            specimen = row.get("specimen_type", row.get("specimen", "unknown"))
            organism = row.get("organism", row.get("organism_name", "pending"))
            collected = format_local_dttm(
                row.get("collect_dttm") or row.get("collected_dttm")
            )
            lines.append(f"  {specimen} ({collected}): {organism}")
        return "\n".join(lines[:15])

    @staticmethod
    def procedures_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No procedure data available."
        lines = ["Procedures:"]
        name_col = next(
            (c for c in ["procedure_name", "procedure_category", "description"] if c in df.columns),
            None,
        )
        date_col = next(
            (c for c in ["procedure_dttm", "performed_dttm"] if c in df.columns), None
        )
        if name_col:
            for row in df.iter_rows(named=True):
                name = row.get(name_col, "unknown")
                date = row.get(date_col, "?") if date_col else ""
                lines.append(f"  {name} ({date})" if date else f"  {name}")
        return "\n".join(lines[:15])

    @staticmethod
    def sofa_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No SOFA score data available."
        lines = ["SOFA scores:"]
        if "sofa_total" in df.columns:
            values = df["sofa_total"].drop_nulls()
            if len(values) > 0:
                lines.append(f"  Latest total SOFA: {values[-1]}")
                lines.append(f"  Range: [{values.min()}-{values.max()}]")
        # Component scores
        components = [
            "cardiovascular", "coagulation", "liver",
            "respiratory", "cns", "renal",
        ]
        for comp in components:
            col = f"sofa_{comp}"
            if col in df.columns:
                vals = df[col].drop_nulls()
                if len(vals) > 0:
                    lines.append(f"  {comp}: latest={vals[-1]}")
        return "\n".join(lines)

    @staticmethod
    def crrt_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No CRRT data available."
        return f"CRRT therapy: {len(df)} records during ICU stay."

    @staticmethod
    def ecmo_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No ECMO/MCS data available."
        return f"ECMO/MCS: {len(df)} records during ICU stay."

    @staticmethod
    def notes_summary(df: Optional[pl.DataFrame]) -> str:
        """Summarize a single DataFrame of notes (legacy / fallback)."""
        if df is None or len(df) == 0:
            return "No clinical notes data available."
        lines = ["Clinical notes:"]
        type_col = "note_type" if "note_type" in df.columns else None
        if type_col is None:
            return f"Clinical notes: {len(df)} notes available (schema unknown)."
        for ntype in sorted(df[type_col].unique().to_list()):
            subset = df.filter(pl.col(type_col) == ntype)
            lines.append(f"\n  === {ntype.upper()} ({len(subset)} notes) ===")
            time_col = "creation_dttm" if "creation_dttm" in subset.columns else None
            sorted_sub = subset.sort(time_col) if time_col else subset
            for row in sorted_sub.iter_rows(named=True):
                note_id = row.get("note_id", "?")
                created = format_local_dttm(row.get("creation_dttm"))
                text = row.get("note_text", "")
                lines.append(f"\n  [{note_id}] {created}")
                for tline in text.split("\n"):
                    lines.append(f"    {tline}")
        return "\n".join(lines)

    @staticmethod
    def agent_notes_summary(notes_by_type: dict[str, pl.DataFrame]) -> str:
        """Summarize per-agent notes (dict of note_type_key -> DataFrame).

        Used when the pipeline loads notes from individual CSV files and
        routes specific note types to specific agents.
        """
        if not notes_by_type:
            return "No clinical notes data available."
        lines = ["Clinical notes:"]
        for note_type_key in sorted(notes_by_type.keys()):
            df = notes_by_type[note_type_key]
            if df is None or len(df) == 0:
                continue
            label = note_type_key.replace("_", " ").upper()
            lines.append(f"\n  === {label} ({len(df)} notes) ===")
            time_col = "creation_dttm" if "creation_dttm" in df.columns else None
            sorted_df = df.sort(time_col) if time_col else df
            for row in sorted_df.iter_rows(named=True):
                note_id = row.get("note_id", "?")
                created = format_local_dttm(row.get("creation_dttm"))
                text = row.get("note_text", "")
                if not isinstance(text, str):
                    text = str(text) if text is not None else ""
                lines.append(f"\n  [{note_id}] {created}")
                for tline in text.split("\n"):
                    lines.append(f"    {tline}")
        return "\n".join(lines)

    @staticmethod
    def position_summary(df: Optional[pl.DataFrame]) -> str:
        if df is None or len(df) == 0:
            return "No position data available."
        lines = ["Patient positioning:"]
        if "position_category" in df.columns:
            positions = df["position_category"].unique().to_list()
            lines.append(f"  Positions used: {', '.join(str(p) for p in positions)}")
        return "\n".join(lines)

    def serialize_all(
        self,
        ctx: PatientContext,
        lookback_hours: Optional[int] = None,
        agent_role: Optional[str] = None,
    ) -> dict[str, str]:
        """Serialize all patient context into a dict of text summaries.

        Args:
            ctx: The patient data context.
            lookback_hours: The user-requested lookback window.  The effective
                window is capped to the actual ICU stay length so that we never
                claim more hours than the patient was actually in the ICU.
            agent_role: If provided, the ``notes`` key will contain only the
                note types routed to this agent (from ctx.agent_notes).
        """
        # Compute ICU stay duration and cap the effective window
        icu_stay_hours = self._compute_icu_stay_hours(ctx)
        if lookback_hours is not None and icu_stay_hours is not None:
            effective_window_hours = min(float(lookback_hours), icu_stay_hours)
        elif lookback_hours is not None:
            effective_window_hours = float(lookback_hours)
        else:
            effective_window_hours = icu_stay_hours  # entire stay; may be None

        # Resolve notes: per-agent routing if available, else legacy
        if agent_role and agent_role in ctx.agent_notes:
            notes_text = self.agent_notes_summary(ctx.agent_notes[agent_role])
        else:
            notes_text = self.notes_summary(ctx.clinical_notes)

        return {
            "demographics": self.demographics_summary(
                ctx,
                icu_stay_hours=icu_stay_hours,
                effective_window_hours=effective_window_hours,
            ),
            "adt": self.adt_summary(ctx.adt),
            "vitals": self.vitals_summary(
                ctx.vitals, effective_window_hours=effective_window_hours
            ),
            "labs": self.labs_summary(ctx.labs),
            "meds": self.meds_summary(ctx.meds_continuous, ctx.meds_intermittent),
            "respiratory": self.respiratory_summary(ctx.respiratory_support),
            "assessments": self.assessments_summary(ctx.patient_assessments),
            "code_status": self.code_status_summary(ctx.code_status),
            "diagnoses": self.diagnoses_summary(ctx.diagnoses),
            "microbiology": self.microbiology_summary(ctx.microbiology),
            "procedures": self.procedures_summary(ctx.procedures),
            "sofa": self.sofa_summary(ctx.sofa_scores),
            "crrt": self.crrt_summary(ctx.crrt_therapy),
            "ecmo": self.ecmo_summary(ctx.ecmo_mcs),
            "position": self.position_summary(ctx.position),
            "notes": notes_text,
        }


# ---------------------------------------------------------------------------
# Vitals aggregation — deterministic, preserves temporal structure
# ---------------------------------------------------------------------------

# Standard ICU nursing vital sign normal ranges (for abnormal flagging)
_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate":       (60, 100),
    "sbp":              (90, 140),
    "dbp":              (60, 90),
    "spo2":             (92, 100),
    "respiratory_rate": (12, 20),
    "temperature":      (36.1, 38.0),
    "map":              (65, 110),
}

# Vitals representation strategy per agent role:
# - raw_only: bedside agent, needs recent raw values, not 48h aggregates
# - both: needs bucketed trends + raw recent for fine-grained recency
# - buckets_only: needs temporal trends, not individual readings
_VITALS_RAW_ONLY = {"nurse"}
_VITALS_BOTH = {"respiratory", "pharmacy", "intensivist"}
# All other agents (case_manager, dietitian, therapist) get buckets only


def aggregate_vitals_for_agent(
    vitals_df: Optional[pl.DataFrame],
    agent_role: Optional[str] = None,
) -> dict[str, Any]:
    """Aggregate vitals into role-appropriate representations.

    - Nurse (raw_only): last 12 raw rows — bedside agent needs recent values
    - Respiratory/pharmacy/intensivist (both): 8h bucketed trends + 24 raw rows
    - Case manager/dietitian/therapist (buckets_only): 8h bucketed trends only

    This is deterministic compaction — NOT LLM summarization. Reduces
    267 rows → ~12-50 records while preserving temporal trend, recency,
    and abnormal event detection.
    """
    if vitals_df is None or len(vitals_df) == 0:
        return {"bucketed_trends": [], "recent_raw": []}

    # Ensure recorded_dttm is datetime
    if "recorded_dttm" in vitals_df.columns:
        col_dtype = vitals_df["recorded_dttm"].dtype
        if col_dtype == pl.Utf8:
            vitals_df = vitals_df.with_columns(
                pl.col("recorded_dttm").str.to_datetime(strict=False)
            )

    # Determine the vital name column
    vital_name_col = "vital_category" if "vital_category" in vitals_df.columns else "vital_name"
    if vital_name_col not in vitals_df.columns:
        return {"bucketed_trends": vitals_df.to_dicts(), "recent_raw": []}

    # Ensure vital_value is numeric for aggregation
    value_col = "vital_value"
    if value_col in vitals_df.columns and vitals_df[value_col].dtype == pl.Utf8:
        vitals_df = vitals_df.with_columns(
            pl.col(value_col).cast(pl.Float64, strict=False)
        )

    # 8-hour bucket aggregation
    try:
        bucketed = (
            vitals_df
            .filter(pl.col(value_col).is_not_null())
            .with_columns(
                (pl.col("recorded_dttm").dt.epoch("s") // (8 * 3600)).alias("bucket_8h")
            )
            .group_by([vital_name_col, "bucket_8h"])
            .agg([
                pl.col(value_col).mean().round(1).alias("mean"),
                pl.col("recorded_dttm").max().alias("bucket_end"),
            ])
            .sort([vital_name_col, "bucket_end"])
        )

        # Flag abnormal buckets
        bucketed_records = []
        for row in bucketed.to_dicts():
            vname = str(row.get(vital_name_col, "")).lower()
            mean_val = row.get("mean")
            if mean_val is not None and vname in _NORMAL_RANGES:
                lo, hi = _NORMAL_RANGES[vname]
                row["abnormal"] = not (lo <= mean_val <= hi)
            bucketed_records.append(row)

    except Exception:
        # Fallback if aggregation fails (e.g., non-numeric values)
        bucketed_records = vitals_df.head(50).to_dicts()

    # Tiered representation based on agent role
    if agent_role in _VITALS_RAW_ONLY:
        # Nurse: bedside agent, needs recent raw values only (no buckets)
        return {
            "recent_raw": (
                vitals_df
                .sort("recorded_dttm", descending=True)
                .head(12)
                .to_dicts()
            )
        }
    elif agent_role in _VITALS_BOTH:
        # Respiratory/pharmacy/intensivist: buckets + raw recent
        return {
            "bucketed_trends": bucketed_records,
            "recent_raw": (
                vitals_df
                .sort("recorded_dttm", descending=True)
                .head(24)
                .to_dicts()
            ),
        }
    else:
        # Case manager/dietitian/therapist: buckets only
        return {"bucketed_trends": bucketed_records}


# ---------------------------------------------------------------------------
# JSON serialization — feeds raw structured data to the LLM
# ---------------------------------------------------------------------------


def _df_to_dicts(df: Optional[pl.DataFrame]) -> list[dict[str, Any]]:
    """Convert a Polars DataFrame to a list of row dicts, or empty list if None/empty."""
    if df is None or len(df) == 0:
        return []
    return df.to_dicts()


# ---------------------------------------------------------------------------
# Citation injection — deterministic cite strings for each data record
# ---------------------------------------------------------------------------


def _parse_cite_timestamp(ts: Any) -> Optional[datetime]:
    """Parse a timestamp value from a Polars row dict into a datetime.

    Handles: Python datetime objects (from Polars to_dicts()), ISO-format
    strings, and None/empty values.
    """
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        ts = ts.strip()
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


def _sort_rows_by_dttm_desc(
    rows: list[dict[str, Any]], *keys: str
) -> list[dict[str, Any]]:
    """Sort row dicts in place, newest-first, by the first parseable timestamp
    in ``keys``. Rows with no parseable value on any key sink to the bottom.

    Reviewer parity: the same row order reaches the agent prompt and the
    review_app source bundle, so sorting here covers both consumers.
    """
    if not rows:
        return rows

    def _sort_key(row: dict[str, Any]) -> tuple[bool, datetime]:
        for k in keys:
            ts = _parse_cite_timestamp(row.get(k))
            if ts is not None:
                if ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                return (True, ts)
        return (False, datetime.min)

    rows.sort(key=_sort_key, reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Per-domain admin/coding column drop list
# ---------------------------------------------------------------------------
#
# These are fields that have NO deterministic consumers in src/ or eval/
# (confirmed by grep, 2026-04-24) and carry no clinical reasoning value at
# the bedside.  Dropping them before the row dict reaches agent prompts
# saves tokens on long stays (LOINC codes alone can be ~15 chars × every
# lab row) and makes metadata.source_data — the reviewer's view — match
# what the agent actually consumed.
#
# INVARIANT: this set MUST be disjoint from every field read by
# citation_index._trim_* — the cite_registry holds references to these same
# row dicts, and shrinking them must not break tooltip resolution.  The only
# current overlap was ``procedure_code`` in _trim_proc, which is being
# removed from the fallback chain in the same commit.
#
# To remove a field from this drop set, first grep src/ and tests/ to
# confirm nothing added a deterministic consumer.
_DOMAIN_ADMIN_COLUMNS: dict[str, frozenset[str]] = {
    "lab": frozenset({
        "lab_loinc_code",
        "lab_order_name",
        "lab_order_category",
        "lab_specimen_name",
        "lab_specimen_category",
        "hospitalization_id",
        # Class A audit (2026-04-30): lab_order_dttm is redundant with
        # lab_result_dttm for resolved labs.
        #
        # lab_collect_dttm is intentionally NOT in this set — it's the
        # only timestamp left on rows whose result was masked by the
        # data-leakage guard (lab collected before reference_dttm but
        # not resulted until after; see retriever._mask_future_lab_results).
        # The slight redundancy on resolved labs is cheap; without it,
        # pending lab rows would have no time anchor for Section P.
        "lab_order_dttm",
    }),
    "med": frozenset({
        "med_order_id",
        "hospitalization_id",
        # Class A audit: med_route_name is verbose ("Intravenous"); the
        # canonical med_route_category ("iv") is what every downstream
        # consumer reads.
        "med_route_name",
    }),
    "resp": frozenset({"hospitalization_id"}),
    "assess": frozenset({"hospitalization_id"}),
    "code": frozenset({"hospitalization_id"}),
    "vital": frozenset({
        "hospitalization_id",
        # Class A audit: bucket_end is the cite anchor for bucketed
        # rows — once the cite tag is injected, the raw bucket_end
        # timestamp is redundant. bucket_8h still encodes the bucket.
        "bucket_end",
    }),
    "proc": frozenset({
        "billing_provider_id",
        "performing_provider_id",
        "procedure_code",
        "hospitalization_id",
    }),
    # Class A audit: microbiology had no admin drops at all. organism_id
    # is an internal CLIF identifier with no clinical use; order_dttm is
    # redundant with result_dttm when the culture has resulted.
    #
    # collect_dttm is intentionally NOT in this set — it's the only
    # timestamp left on rows whose result was masked by the retriever's
    # data-leakage guard (result_dttm >= reference_dttm), and we want
    # those rows to render as "BCx <collect_dttm>: pending" rather than
    # as a row with no time anchor at all. For resulted rows the slight
    # redundancy with result_dttm is cheap.
    "micro": frozenset({
        "organism_id",
        "order_dttm",
        "hospitalization_id",
    }),
    # Non-cite domains — still worth stripping the per-row constant.
    "adt": frozenset({"hospitalization_id"}),
    "note": frozenset({"hospitalization_id"}),
}


# Class A audit (2026-04-30): CBC differential lab_categories with no
# downstream consumer in any agent's section authoring. CLIF data routinely
# emits these q4-q6h, producing dozens of rows per case for zero clinical
# signal. Drop unilaterally for ALL agents (not only the heavy three).
#
# If you add an agent that actually authors hematologic narrative beyond
# what the existing pharmacy/respiratory cytopenia tracking covers, remove
# the relevant entries from this set.
_LAB_CATEGORY_NOISE: frozenset[str] = frozenset({
    "basophils_absolute",
    "basophils_percent",
    "lymphocytes_absolute",
    "lymphocytes_percent",
    "monocytes_absolute",
    "monocytes_percent",
    "neutrophils_absolute",
    "neutrophils_percent",
})

# 2026-04-30 Class B audit, item 5C — non-standardized residual.
#
# CLIF v2.1 categorizes labs that don't map to a standard vocabulary as
# ``lab_category == "other"``. Northwestern's extract puts ~2.6M rows here;
# investigation showed the bucket contains a mix of:
#   1. ABG components mislabeled because ETL didn't apply the canonical
#      bicarbonate/base_excess/fio2/co-oximetry categories
#      → handled by _PROMOTE_FROM_OTHER below (rewrites lab_category)
#   2. High-volume single-name labs (HbA1c, adjusted calcium, circuit
#      ionized calcium) that have clinical content but no canonical category
#      → also handled by _PROMOTE_FROM_OTHER
#   3. Genuinely non-clinical-or-mismapped residual: gram stain results
#      (belong in microbiology), urinalysis components, manual differential
#      redundancies, device-status indicators
#      → dropped here as residual
#
# The promotion map runs FIRST in serialize_to_json. By the time this
# filter runs, any row still tagged "other" failed the name-based promotion
# and is residual. Drop applies to all agents (no role exception).
_LAB_CATEGORY_NON_STANDARDIZED: frozenset[str] = frozenset({"other"})


# 2026-04-30 Class B audit, item 5D — promote-from-other map.
#
# Reconciles inconsistent ETL mapping in Northwestern's CLIF labs table.
# Several arterial blood gas components and high-volume clinical labs
# appear under lab_category == "other" with informative lab_name strings
# that match a canonical category. This map rewrites lab_category for those
# rows so agents can reason over a complete ABG panel and the high-volume
# clinical labs (HbA1c, adjusted calcium) without parsing the noisy "other"
# bucket at LLM time.
#
# Match logic: case-insensitive exact match on lab_name (these are the
# canonical Northwestern variants observed in the labs parquet). Promotion
# happens at the same retrieval-shaping layer as Class A column drops.
#
# Adding entries: confirm the lab_name string against
# scripts/dump_other_lab_names.py output before adding. Avoid substring
# matches — exact-match keeps the audit trail clean and prevents accidental
# captures (e.g., "BICARBONATE, VENOUS BLOOD" if it appeared shouldn't
# silently land in the arterial bicarbonate canonical category).
#
# Manuscript framing: this is methodologically defensible as reconciling
# an internal inconsistency in the source data, NOT adding institution-
# specific scope. The full map ships as Supplementary Table SX (see
# docs/clif_data_gaps_investigation.md).
_PROMOTE_FROM_OTHER: dict[str, tuple[str, ...]] = {
    # ABG bicarbonate variants → canonical bicarbonate category
    "bicarbonate": (
        "NM BKR HCO3 ART", "HCO3 ART",
        "NM BKR HCO3, ARTERIAL BLOOD", "HCO3, ARTERIAL BLOOD",
    ),
    # Base excess
    "base_excess_arterial": (
        "NM BKR BASE EX/DEF ART", "BASE EX/DEF ART",
        "NM BKR BASE EXCESS, ARTERIAL BLOOD", "BASE EXCESS, ARTERIAL BLOOD",
    ),
    # FiO2 from ABG draw
    "fio2_arterial": (
        "FIO2 ART PCT", "FIO2, ARTERIAL BLOOD",
    ),
    # Co-oximetry
    "oxyhemoglobin": ("NM BKR OXYHEMOGLOBIN",),
    "methemoglobin": ("NM BKR METHEMOGLOBIN",),
    "hemoglobin_arterial": (
        "NM BKR HEMOGLOBIN, ARTERIAL BLOOD",
        "HEMOGLOBIN, ARTERIAL BLOOD",
    ),
    "hematocrit_arterial": ("HEMATOCRIT, ARTERIAL BLOOD",),
    "total_co2_arterial": (
        "NM BKR TOTAL CO2 CONTENT, ARTERIAL BLOOD",
        "TOTAL CO2 CONTENT, ARTERIAL BLOOD",
    ),
    # Non-ABG but high-volume clinical content
    "adjusted_calcium": ("NM BKR ADJUSTED CALCIUM",),
    "hba1c": ("NM BKR HEMOGLOBIN A1C",),
    "circuit_ionized_calcium": ("NM BKR CIRCUIT IONIZED CALCIUM",),
}

# Reverse lookup built once at import time: uppercased lab_name → canonical
# category. Pre-uppercased so the row-level loop does one .upper() per row,
# not per-pattern. _PROMOTE_FROM_OTHER stays in canonical→variants form
# because that's the readable / reviewable shape for clinicians + the
# manuscript supplementary table.
_PROMOTION_INDEX: dict[str, str] = {
    name.upper(): canonical
    for canonical, names in _PROMOTE_FROM_OTHER.items()
    for name in names
}


def _filter_noisy_lab_categories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop rows whose ``lab_category`` is noise OR non-standardized residual.

    Two-source filter: (a) ``_LAB_CATEGORY_NOISE`` (CBC differential noise,
    audit item Class A) and (b) ``_LAB_CATEGORY_NON_STANDARDIZED`` (the
    "other" bucket residual after _promote_lab_categories_from_other has
    promoted ABG components + high-volume clinical labs to canonical
    categories).

    Returns a new list (does not mutate). Case-insensitive on category.

    Order-of-operations contract: caller must apply
    _promote_lab_categories_from_other BEFORE this filter, otherwise rows
    that should be promoted will get dropped as residual.
    """
    drop = _LAB_CATEGORY_NOISE | _LAB_CATEGORY_NON_STANDARDIZED
    return [
        r for r in rows
        if str(r.get("lab_category", "")).lower() not in drop
    ]


def _promote_lab_categories_from_other(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rewrite ``lab_category`` for rows in "other" with a known lab_name.

    Reconciles inconsistent ETL mapping in Northwestern's labs table.
    Operates ONLY on rows where ``lab_category == "other"`` and the
    uppercased ``lab_name`` matches an entry in ``_PROMOTION_INDEX``.
    Other rows pass through untouched. Mutates rows in place (matches the
    in-place pattern used by ``_drop_admin_columns``); returns the same
    list for chainability.

    No-op if rows lack a ``lab_name`` field (defensive — keeps the function
    safe against future schema variance).
    """
    for r in rows:
        if str(r.get("lab_category", "")).lower() != "other":
            continue
        name = r.get("lab_name")
        if not name:
            continue
        canonical = _PROMOTION_INDEX.get(str(name).upper())
        if canonical is not None:
            r["lab_category"] = canonical
    return rows


def _drop_lab_value_when_numeric_present(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-row: drop ``lab_value`` (string) when ``lab_value_numeric`` is set.

    citation_index._trim_lab reads lab_value_numeric first, falls back to
    lab_value — so dropping the string when numeric is present preserves
    every existing tooltip while saving ~10-20 chars/row × hundreds of rows.
    """
    for row in rows:
        if row.get("lab_value_numeric") is not None:
            row.pop("lab_value", None)
    return rows


def _drop_med_name_when_category_present(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-row: drop verbose ``med_name`` when canonical ``med_category`` is set.

    med_name is often a multi-line full prescription string
    ("PIPERACILLIN 4 G / TAZOBACTAM 500 MG IN SODIUM CHLORIDE 0.9%
    (ISO-OS) 100 ML IV"); med_category is the canonical drug name
    ("piperacillin-tazobactam"). When the canonical exists, the verbose
    duplicate adds ~80-200 chars/row.

    citation_index._trim_med falls back from med_name to med_category, so
    dropping med_name when category is present preserves tooltip behavior
    (with the canonical, which is preferable clinically).
    """
    for row in rows:
        if row.get("med_category"):
            row.pop("med_name", None)
    return rows


# Assessments use one of these three value columns per row; the other two
# are typically null. Strip nulls at row level so we don't ship empty fields.
_ASSESSMENT_VALUE_COLS = ("numerical_value", "categorical_value", "text_value")


def _strip_null_assessment_value_columns(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-row: pop null entries from numerical/categorical/text_value."""
    for row in rows:
        for col in _ASSESSMENT_VALUE_COLS:
            if row.get(col) is None:
                row.pop(col, None)
    return rows


# ---------------------------------------------------------------------------
# 2026-04-30 Class B audit, item 5A: respiratory microbiology specimen allowlist
# ---------------------------------------------------------------------------
#
# Respiratory authors S/E sections about ventilation, weaning, and pulmonary
# infection. Only specimens that could attribute infection to a pulmonary
# source are useful — Urine, Swab (skin/wound), Stool, etc. are not.
#
# Strings come verbatim from the agent_data_categories_v2 enumeration of
# Northwestern's CLIF microbiology_culture table (2026-04-30): the column
# uses Title Case with spaces. Exact-string match — case sensitive — to
# match what the data actually emits.
#
# Body Fluid is included as a catch-all for pleural / peritoneal / CSF
# fluids per the audit doc clinician review (2026-04-30 sign-off).
#
# Pharmacy and other agents continue to receive the full microbiology
# slice — pharmacy uses all specimens for ABx de-escalation reasoning.
#
# If the broader Northwestern cohort exposes additional pulmonary specimen
# strings (BAL, tracheal_aspirate, pleural_fluid, bronchial_wash,
# endotracheal_tube_tip), add them here after observing them in real data.
RESPIRATORY_SPECIMEN_ALLOWLIST: frozenset[str] = frozenset({
    "Sputum",
    "Sputum, Expectorated",
    "Body Fluid",
    "Blood",
})


def _filter_microbiology_for_respiratory(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only pulmonary-relevant specimens for the respiratory agent.

    Filters on `specimen_category`. Rows with a missing/empty value are
    excluded — if we can't classify the specimen we don't ship it to
    respiratory (the caller, pharmacy, still gets everything and acts as
    the safety net for unclassified specimens).
    """
    return [
        r for r in rows
        if r.get("specimen_category") in RESPIRATORY_SPECIMEN_ALLOWLIST
    ]


# 2026-04-30 Class B audit, item 5B — per-role vital_category allowlists.
#
# Northwestern's CLIF emits vital_category values with explicit unit suffixes
# (temp_c, weight_kg, height_cm). Other CLIF sites may use unsuffixed forms
# ("temp", "weight", "height") or different conventions ("temperature",
# "body_weight"). This allowlist is scoped to Northwestern's naming. Cross-site
# deployment requires updating the value strings; the filter does not normalize
# value variants.
#
# Rationale (per docs/agent_data_audit.md):
#   - Respiratory: vitals are noise outside oxygenation/ventilation/fever.
#     Heart rate, BP, MAP belong to nurse/intensivist hemodynamic reasoning;
#     respiratory's domain is gas exchange + ventilatory effort.
#   - Dietitian: vitals are noise outside weight/temperature/height.
#     Weight is the only nutrition-relevant longitudinal vital; temperature
#     informs metabolic demand; height is needed for BMI/ideal body weight.
#
# Other agents (nurse, pharmacy, intensivist, case_manager, therapist) keep
# the full vital category set — they reason hemodynamically.
RESPIRATORY_VITALS_ALLOWLIST: frozenset[str] = frozenset({
    "spo2",
    "respiratory_rate",
    "temp_c",
})

DIETITIAN_VITALS_ALLOWLIST: frozenset[str] = frozenset({
    "weight_kg",
    "temp_c",
    "height_cm",
})

_AGENT_VITALS_ALLOWLIST: dict[str, frozenset[str]] = {
    "respiratory": RESPIRATORY_VITALS_ALLOWLIST,
    "dietitian": DIETITIAN_VITALS_ALLOWLIST,
}


def _filter_vitals_by_category(
    vitals_data: dict[str, Any], agent_role: Optional[str],
) -> dict[str, Any]:
    """Apply per-role vital_category allowlist post-tiered-compaction.

    Operates on the dict returned by ``aggregate_vitals_for_agent``: filters
    both ``bucketed_trends`` and ``recent_raw`` sub-lists (the tiered shape
    means an agent may have one or both — whichever keys are present get
    filtered). Roles without an allowlist entry get vitals_data unchanged.

    Mutates the sub-lists in place (matches the in-place pattern used by
    ``_drop_admin_columns`` and ``_filter_noisy_lab_categories``) so the cite
    registry sees the filtered shape. Returns the same dict for chainability.

    Falls back to ``vital_name`` when ``vital_category`` is absent — mirrors
    the same fallback ``aggregate_vitals_for_agent`` uses.
    """
    allow = _AGENT_VITALS_ALLOWLIST.get(agent_role) if agent_role else None
    if not allow or not isinstance(vitals_data, dict):
        return vitals_data
    for sub in ("bucketed_trends", "recent_raw"):
        rows = vitals_data.get(sub)
        if isinstance(rows, list):
            vitals_data[sub] = [
                r for r in rows
                if (r.get("vital_category") or r.get("vital_name")) in allow
            ]
    return vitals_data


def _drop_admin_columns(
    rows: list[dict[str, Any]], domain: str,
) -> list[dict[str, Any]]:
    """Strip admin/coding columns from row dicts in place.

    Mutates each row dict so that references held by ``cite_registry`` see
    the trimmed shape too — this is intentional: agent prompt, metadata
    snapshot, and citation tooltips should all converge on the same row
    data.
    """
    cols = _DOMAIN_ADMIN_COLUMNS.get(domain, frozenset())
    if not cols:
        return rows
    for row in rows:
        for col in cols:
            row.pop(col, None)
    return rows


def _add_cite_fields(
    rows: list[dict[str, Any]],
    source_type: str,
    time_col: str,
    tz: ZoneInfo,
    cite_registry: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Inject a pre-formatted ``cite`` string into each row dict.

    Rows without a parseable timestamp in *time_col* are skipped (no cite
    field added) to avoid cluttering the registry with indistinguishable
    entries.

    Args:
        rows: Row dicts from ``_df_to_dicts()`` — mutated in place.
        source_type: Short label for the data domain (must be in
            ``_CITE_SOURCE_TYPES``).
        time_col: Column name holding the timestamp.
        tz: Display timezone for formatting (e.g. America/Chicago).
        cite_registry: Accumulated mapping of cite string → list of source
            rows.  Mutated in place so the caller can collect all emitted
            cites across the full serialization pass.
    """
    assert source_type in _CITE_SOURCE_TYPES, f"Unknown source_type: {source_type}"
    for row in rows:
        dt = _parse_cite_timestamp(row.get(time_col))
        if dt is None:
            continue  # no cite for rows without timestamps
        # Ensure timezone-aware before converting
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_local = dt.astimezone(tz)
        # Portable format — no strftime %-m (breaks on Windows)
        # Use hyphen (not slash) as date separator — LLMs merge "1/09" into "109"
        cite = (
            f"({source_type} "
            f"{dt_local.month}-{dt_local.day:02d} "
            f"{dt_local.hour:02d}:{dt_local.minute:02d})"
        )
        row["cite"] = cite
        cite_registry.setdefault(cite, []).append(row)
    return rows


# ---------------------------------------------------------------------------
# Note deduplication — remove near-identical notes before capping
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.9


def _bag_of_words(text: str) -> dict[str, int]:
    """Tokenize text into a word frequency dict (lowercase, alphanumeric only)."""
    import re
    words = re.findall(r"[a-z0-9]+", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def _cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two bag-of-words dicts."""
    import math
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _best_timestamp_col(df: pl.DataFrame) -> Optional[str]:
    """Return the best available timestamp column for recency ordering."""
    for col in ("revision_dttm", "signed_dttm", "creation_dttm"):
        if col in df.columns:
            return col
    return None


def _cap_with_floor_protection(
    df: pl.DataFrame, cap: int,
) -> pl.DataFrame:
    """Apply ``AGENT_MAX_NOTES_PER_TYPE`` cap while preserving floor rows.

    Floor rows carry the ``_floor_protected`` sentinel column (set by
    ``ensure_physician_note_floor``).  They survive the cap unconditionally;
    the cap drops the oldest non-protected rows to make room.  Total kept
    is min(cap, len(df)) — the floor row consumes one of the ``cap`` slots
    rather than adding to it, which keeps the token-budget contract
    behind ``AGENT_MAX_NOTES_PER_TYPE`` intact.
    """
    if df is None or len(df) == 0:
        return df

    from icu_pause.data.note_floor import FLOOR_PROTECTED_COL

    # Sort newest-first so head(N) keeps recent rows.
    if "revision_dttm" in df.columns:
        df_sorted = df.sort("revision_dttm", descending=True, nulls_last=True)
    elif "creation_dttm" in df.columns:
        df_sorted = df.sort("creation_dttm", descending=True, nulls_last=True)
    else:
        df_sorted = df

    if FLOOR_PROTECTED_COL not in df_sorted.columns:
        return df_sorted.head(cap)

    protected = df_sorted.filter(pl.col(FLOOR_PROTECTED_COL).fill_null(False))
    others = df_sorted.filter(~pl.col(FLOOR_PROTECTED_COL).fill_null(False))

    # Reserve cap slots: protected gets first claim, others fill the rest.
    remaining = max(cap - len(protected), 0)
    others_capped = others.head(remaining)

    if len(protected) == 0:
        return others_capped

    combined = pl.concat([others_capped, protected], how="diagonal_relaxed")
    if "revision_dttm" in combined.columns:
        combined = combined.sort("revision_dttm", descending=True, nulls_last=True)
    elif "creation_dttm" in combined.columns:
        combined = combined.sort("creation_dttm", descending=True, nulls_last=True)
    return combined


def _deduplicate_similar_notes(
    df: pl.DataFrame,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> pl.DataFrame:
    """Remove near-duplicate notes within a single note type.

    When two notes have cosine similarity > threshold (default 0.9), the
    older note is dropped. Keeps the most recent by revision_dttm, then
    signed_dttm, then creation_dttm.

    This runs BEFORE the per-agent max-notes cap, so the cap operates on
    genuinely distinct notes rather than redundant revisions.
    """
    if df is None or len(df) <= 1:
        return df

    text_col = "note_text" if "note_text" in df.columns else None
    if text_col is None:
        return df

    time_col = _best_timestamp_col(df)
    if time_col:
        df = df.sort(time_col, descending=True, nulls_last=True)

    texts = df[text_col].to_list()
    bows = [_bag_of_words(str(t) if t else "") for t in texts]

    # Greedy dedup: iterate newest-first, drop any later row similar to a kept row
    keep_indices: list[int] = []
    for i, bow_i in enumerate(bows):
        is_duplicate = False
        for j in keep_indices:
            if _cosine_similarity(bow_i, bows[j]) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    if len(keep_indices) == len(df):
        return df
    return df[keep_indices]


# ---------------------------------------------------------------------------
# Union of post-cap agent contexts — reviewer-facing source data
# ---------------------------------------------------------------------------


def _dedup_rows(rows_seq: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Concatenate lists of row-dicts with full-row JSON-canonical dedup.

    Same physical record produces byte-identical row dicts across agents
    (deterministic cite tags, identical _drop_admin_columns output), so
    JSON canonical-form comparison is reliable without per-domain stable-ID
    knowledge.
    """
    import json
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for rows in rows_seq:
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                key = json.dumps(row, sort_keys=True, default=str)
            except (TypeError, ValueError):
                key = repr(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
    return out


def union_post_cap_contexts(
    agent_context_text: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the union of per-agent serialized contexts (post-cap, post-routing).

    Each agent's slice in *agent_context_text* is already cap-applied
    (AGENT_MAX_NOTES_PER_TYPE) and routing-filtered (_AGENT_DATA_KEYS in
    workflow.py).  The union across agents is the correct denominator for
    document-level reviewer evaluation: every record in the union was seen
    by at least one agent, and nothing the agents never saw leaks through.

    Cap-then-union semantics: each agent's per-type cap is enforced on its
    own slice first; the union is taken afterward.  For a note type routed
    to N agents with cap=K, the union has at most N*K notes (typically far
    fewer after dedup).

    Returns an empty dict if *agent_context_text* is empty.  Domains absent
    from every agent's slice are absent from the union.
    """
    if not agent_context_text:
        return {}

    # Collect all top-level domain keys present in any agent's slice
    all_domains: set[str] = set()
    for ctx in agent_context_text.values():
        if isinstance(ctx, dict):
            all_domains.update(ctx.keys())

    union: dict[str, Any] = {}
    for domain in all_domains:
        slices = [
            ctx[domain] for ctx in agent_context_text.values()
            if isinstance(ctx, dict) and domain in ctx
        ]
        if not slices:
            continue
        sample = next((s for s in slices if s), slices[0])

        if isinstance(sample, list):
            # Flat list domains: labs, meds_*, respiratory, assessments, etc.
            union[domain] = _dedup_rows(slices)
        elif isinstance(sample, dict):
            # Nested-dict domains:
            #   notes  -> {note_type: [rows...]}
            #   vitals -> {bucketed_trends: [...], recent_raw: [...]}
            #   meds   -> {continuous: [...], intermittent: [...]}
            # demographics is also a dict but its values are scalars, not lists;
            # the sub-key handling falls through to the "first non-empty value"
            # branch and reconstructs it correctly.
            sub_keys: set[str] = set()
            for s in slices:
                if isinstance(s, dict):
                    sub_keys.update(s.keys())
            merged: dict[str, Any] = {}
            for sub in sub_keys:
                sub_lists = [
                    s.get(sub) for s in slices
                    if isinstance(s, dict) and isinstance(s.get(sub), list)
                ]
                if sub_lists:
                    merged[sub] = _dedup_rows(sub_lists)
                    continue
                # Sub-key holds a non-list value (e.g. demographics scalars):
                # take the first non-empty value across agents.
                for s in slices:
                    if isinstance(s, dict) and s.get(sub) not in (None, "", [], {}):
                        merged[sub] = s[sub]
                        break
            union[domain] = merged
        else:
            # Scalar / other: take from first non-empty slice
            union[domain] = sample

    # Per-agent bucketed_trends and recent_raw are individually DESC-sorted
    # at serialize_to_json time, but _dedup_rows above concatenates slices
    # in agent-iteration order without a final re-sort. That broke the
    # newest-first invariant for the reviewer view: vitals from a later
    # agent (e.g. height_cm from the dietitian) landed below vitals from
    # the first slice instead of slotting into its date-sorted position,
    # so an older value could appear above a much newer one. Re-sort here
    # so the merged source_data view matches what each individual agent
    # saw. bucket_8h survives _drop_admin_columns (line 829-832 drops
    # bucket_end but keeps bucket_8h) and is a clean integer time key
    # (epoch_seconds // 8h), so we don't need to parse cite strings.
    vitals_union = union.get("vitals")
    if isinstance(vitals_union, dict):
        bucketed = vitals_union.get("bucketed_trends")
        if isinstance(bucketed, list):
            bucketed.sort(
                key=lambda r: (
                    r.get("bucket_8h") if isinstance(r.get("bucket_8h"), (int, float)) else -1
                ),
                reverse=True,
            )
        recent = vitals_union.get("recent_raw")
        if isinstance(recent, list):
            _sort_rows_by_dttm_desc(recent, "recorded_dttm")

    return union


def _build_med_states_block(
    cont_rows: list[dict[str, Any]],
    intermittent_rows: list[dict[str, Any]],
    reference_dttm: Optional[datetime],
    renal_status: str = "unknown",
    lookback_hours: int = 48,
) -> dict[str, Any]:
    """Run med_state.classify_med_states and shape it for agent context.

    Returns a dict with three lists (active / recently_stopped / historical)
    and a flat ``records`` list with full per-drug detail. Always returns a
    dict with the same keys (possibly empty) so downstream consumers can
    rely on shape.
    """
    if reference_dttm is None or (not cont_rows and not intermittent_rows):
        return {
            "active": [],
            "recently_stopped": [],
            "historical": [],
            "records": [],
            "reference_dttm": None,
            "renal_status": renal_status,
            "lookback_hours": lookback_hours,
        }

    from icu_pause.tools.med_state import classify_med_states, state_summary

    records = classify_med_states(
        {"continuous": cont_rows, "intermittent": intermittent_rows},
        reference_dttm,
        renal_status=renal_status,  # type: ignore[arg-type]
        lookback_hours=lookback_hours,
    )
    buckets = state_summary(records)
    return {
        "active": buckets["active"],
        "recently_stopped": buckets["recently_stopped"],
        "historical": buckets["historical"],
        "records": [
            {
                "drug_name": r.drug_name,
                "state": r.state,
                "is_continuous": r.is_continuous,
                "last_admin_dttm": (
                    r.last_admin_dttm.isoformat() if r.last_admin_dttm else None
                ),
                "last_dose": r.last_dose,
                "admin_count_in_window": r.admin_count_in_window,
                "trending_to_zero": r.trending_to_zero,
                "expected_interval_hours": r.expected_interval_hours,
                "hours_since_last_admin": r.hours_since_last_admin,
            }
            for r in records
        ],
        "reference_dttm": format_local_dttm(reference_dttm),
        "renal_status": renal_status,
        "lookback_hours": lookback_hours,
    }


# ---------------------------------------------------------------------------
# Deterministic transfer-exam block (Phase 3)
# ---------------------------------------------------------------------------
#
# Builds a coherent point-in-time exam snapshot from structured CLIF data
# at ``reference_dttm``. Replaces the freehand stitching the nurse agent
# used to do from ``recent_raw`` (which produced values the reviewer panel
# couldn't audit and caused silent drops on ~25% of cases per the Phase 2
# audit).
#
# Output shape (when all sub-blocks fire):
#
#   TRANSFER EXAM (deterministic — DO NOT paraphrase or duplicate in narrative)
#   Neuro: GCS 15 (E4 V5 M6), RASS 0, CAM-ICU negative (exam-neuro 11-09 08:00)
#   Vitals: BP 119/73, MAP 89, HR 70, RR 23, SpO2 100%, Temp 37.7°C (exam-vitals 11-09 08:00)
#   Respiratory: Nasal cannula, FiO2 35% (exam-resp 11-09 08:00)
#
# Sub-blocks render `—` for a missing single category (e.g. BP 119/—)
# but emit a full fallback sentence when ALL categories in a sub-block
# miss the 4 h window — stale values labeled as a current snapshot is
# the failure mode this whole construct is designed to prevent.

EXAM_WINDOW_HOURS = 4

# Ordered vital categories for the deterministic Vitals sub-block.
# Render order matches clinical convention (BP/MAP first, peripheral last).
EXAM_VITAL_CATEGORIES: tuple[str, ...] = (
    "sbp", "dbp", "map", "heart_rate", "respiratory_rate", "spo2", "temp_c",
)

# Per-Phase-0 grep: production CLIF uses these exact strings.
EXAM_GCS_TOTAL = "gcs_total"
EXAM_GCS_COMPONENTS = ("gcs_eye", "gcs_verbal", "gcs_motor")
EXAM_RASS = "RASS"  # uppercase in production; normalize case-insensitively
EXAM_CAM_TOTAL = "cam_total"

_EXAM_VITAL_UNITS = {
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map": "mmHg",
    "heart_rate": "bpm",
    "respiratory_rate": "/min",
    "spo2": "%",
    "temp_c": "°C",
}


def _normalize_dttm(value: Any) -> Optional[datetime]:
    """Coerce common timestamp shapes to a tz-aware ``datetime``.

    Accepts datetime instances, ISO strings (with or without 'Z' / offset),
    and naive datetimes (assumed UTC). Returns None for unparseable input
    so callers can branch cleanly.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if "T" not in s and len(s) >= 19:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _exam_cite_tag(source_type: str, anchor: datetime, tz: ZoneInfo) -> str:
    """Build a citation tag in the canonical (source_type M-DD HH:MM) format."""
    local = anchor.astimezone(tz)
    return (
        f"({source_type} {local.month}-{local.day:02d} "
        f"{local.hour:02d}:{local.minute:02d})"
    )


def _exam_window(reference_dttm: datetime) -> tuple[datetime, datetime]:
    """Return ``(window_start, window_end_exclusive)`` for the exam block.

    ``window_end_exclusive`` is reference_dttm itself — equal-timestamp
    rows are accepted (closed lower, closed upper).
    """
    return reference_dttm - timedelta(hours=EXAM_WINDOW_HOURS), reference_dttm


def _fmt_exam_value(v: Any, category: Optional[str] = None) -> Optional[str]:
    """Render a numeric value with category-aware rounding.

    CLIF stores some vitals as floats with spurious precision (MAP often
    computed as ``(SBP + 2*DBP)/3`` → 104.333; Temp converted between
    units → 36.667). Round to clinically meaningful precision so the
    deterministic block doesn't look unpolished.

      * BP/MAP/HR/RR/SpO2 → integer
      * Temp → 1 decimal
      * Other → trailing zero trim
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if category in {"sbp", "dbp", "map", "heart_rate", "respiratory_rate", "spo2"}:
        return str(int(round(f)))
    if category == "temp_c":
        return f"{f:.1f}"
    if f.is_integer():
        return str(int(f))
    return f"{f:g}"


def _pick_latest_in_window(
    rows: list[dict[str, Any]],
    category_field: str,
    category_value: str,
    ts_field: str,
    window: tuple[datetime, datetime],
) -> Optional[dict[str, Any]]:
    """Latest row matching *category_value* (case-insensitive) with ts in window.

    Returns the raw row dict (caller extracts numeric / categorical / text
    fields). None when no match.
    """
    win_lo, win_hi = window
    best: Optional[tuple[datetime, dict[str, Any]]] = None
    cat_lower = category_value.lower()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        cat = row.get(category_field)
        if cat is None or str(cat).lower() != cat_lower:
            continue
        ts = _normalize_dttm(row.get(ts_field))
        if ts is None or not (win_lo <= ts <= win_hi):
            continue
        if best is None or ts > best[0]:
            best = (ts, row)
    return best[1] if best else None


def _latest_any_for_category(
    rows: list[dict[str, Any]],
    category_field: str,
    category_value: str,
    ts_field: str,
) -> Optional[datetime]:
    """Used by the fallback sentence to surface "last documented" outside the window."""
    cat_lower = category_value.lower()
    best: Optional[datetime] = None
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        cat = row.get(category_field)
        if cat is None or str(cat).lower() != cat_lower:
            continue
        ts = _normalize_dttm(row.get(ts_field))
        if ts is None:
            continue
        if best is None or ts > best:
            best = ts
    return best


def _build_exam_vitals_subblock(
    vitals_rows: list[dict[str, Any]],
    reference_dttm: datetime,
    cite_registry: Optional[dict[str, list[dict[str, Any]]]],
    tz: ZoneInfo,
) -> Optional[str]:
    """Build the deterministic Vitals line of the transfer-exam block.

    Per-category selection: latest non-null in 4h window. Single-component
    BP renders as ``119/—`` or ``—/73`` rather than dropping the whole row
    (single-component BP is still clinically informative). All-empty case
    returns a fallback sentence; partial-empty returns the assembled line.

    Returns the rendered line (including trailing cite tag) or None when
    *vitals_rows* is structurally absent (caller skips the line entirely).
    """
    if not vitals_rows:
        return None

    window = _exam_window(reference_dttm)
    # Find latest in-window value per category.
    picked: dict[str, dict[str, Any]] = {}
    anchor_dt: Optional[datetime] = None
    for cat in EXAM_VITAL_CATEGORIES:
        row = _pick_latest_in_window(
            vitals_rows, "vital_category", cat, "recorded_dttm", window
        )
        if row is not None:
            picked[cat] = row
            ts = _normalize_dttm(row.get("recorded_dttm"))
            if ts is not None and (anchor_dt is None or ts > anchor_dt):
                anchor_dt = ts

    if not picked:
        # All-empty fallback. Surface the most recent vital timestamp
        # outside the window so the clinician sees how stale the data is.
        latest_overall: Optional[datetime] = None
        for cat in EXAM_VITAL_CATEGORIES:
            lat = _latest_any_for_category(
                vitals_rows, "vital_category", cat, "recorded_dttm"
            )
            if lat is not None and (latest_overall is None or lat > latest_overall):
                latest_overall = lat
        if latest_overall is None:
            return f"Vitals: no vitals data within {EXAM_WINDOW_HOURS}h of transfer."
        local = latest_overall.astimezone(tz)
        return (
            f"Vitals: no vitals data within {EXAM_WINDOW_HOURS}h of transfer "
            f"(last documented {local.month}-{local.day:02d} "
            f"{local.hour:02d}:{local.minute:02d})."
        )

    # Render BP from sbp + dbp. Single-component BP renders with `—`.
    sbp_row = picked.get("sbp")
    dbp_row = picked.get("dbp")
    sbp_v = _fmt_exam_value(sbp_row.get("vital_value"), "sbp") if sbp_row else None
    dbp_v = _fmt_exam_value(dbp_row.get("vital_value"), "dbp") if dbp_row else None

    parts: list[str] = []
    if sbp_row is not None or dbp_row is not None:
        parts.append(f"BP {sbp_v or '—'}/{dbp_v or '—'}")

    # MAP, HR, RR, SpO2, Temp.
    for cat, label in (
        ("map", "MAP"),
        ("heart_rate", "HR"),
        ("respiratory_rate", "RR"),
        ("spo2", "SpO2"),
        ("temp_c", "Temp"),
    ):
        row = picked.get(cat)
        if row is None:
            continue
        v = _fmt_exam_value(row.get("vital_value"), cat)
        if v is None:
            continue
        if cat == "spo2":
            parts.append(f"SpO2 {v}%")
        elif cat == "temp_c":
            parts.append(f"Temp {v}°C")
        else:
            parts.append(f"{label} {v}")

    if not parts:
        # All picked rows had null values somehow — treat as all-empty.
        return f"Vitals: no vitals data within {EXAM_WINDOW_HOURS}h of transfer."

    assert anchor_dt is not None  # at least one row had a timestamp
    tag = _exam_cite_tag("exam-vitals", anchor_dt, tz)

    # Build the normalized cite-registry rows.
    if cite_registry is not None:
        registry_rows: list[dict[str, Any]] = []
        # BP as a single combined row if either sbp or dbp present.
        if sbp_row is not None or dbp_row is not None:
            bp_ts = _normalize_dttm(
                (sbp_row or dbp_row).get("recorded_dttm")
            )
            registry_rows.append(
                {
                    "label": "BP",
                    "value": f"{sbp_v or '—'}/{dbp_v or '—'}",
                    "unit": "mmHg",
                    "time": bp_ts.isoformat() if bp_ts else None,
                }
            )
        for cat in ("map", "heart_rate", "respiratory_rate", "spo2", "temp_c"):
            row = picked.get(cat)
            if row is None:
                continue
            v = _fmt_exam_value(row.get("vital_value"), cat)
            if v is None:
                continue
            ts = _normalize_dttm(row.get("recorded_dttm"))
            registry_rows.append(
                {
                    "label": cat,
                    "value": v,
                    "unit": _EXAM_VITAL_UNITS.get(cat, ""),
                    "time": ts.isoformat() if ts else None,
                }
            )
        cite_registry.setdefault(tag, []).extend(registry_rows)

    return f"Vitals: {', '.join(parts)} {tag}"


_CAM_LABEL_BY_CAT = {"1": "positive", "0": "negative"}


def _camicu_label_from_row(row: Optional[dict[str, Any]]) -> Optional[str]:
    """Map a cam_total row to ``positive`` / ``negative`` / ``unable to assess``.

    Implements the Q6 sign-off rendering — must not collapse UTA into
    negative even when categorical_value is null. Returns None when the
    row carries no usable information (caller renders as `—` or skips).
    """
    if not row:
        return None
    cat = row.get("categorical_value")
    cat_str = str(cat).strip() if cat is not None else ""
    if cat_str in _CAM_LABEL_BY_CAT:
        return _CAM_LABEL_BY_CAT[cat_str]
    txt = row.get("text_value")
    if txt:
        lo = str(txt).strip().lower()
        if lo.startswith("uta"):
            return "unable to assess"
        if lo.startswith("positive"):
            return "positive"
        if lo.startswith("negative"):
            return "negative"
    return None


def _build_exam_neuro_subblock(
    assessment_rows: list[dict[str, Any]],
    reference_dttm: datetime,
    cite_registry: Optional[dict[str, list[dict[str, Any]]]],
    tz: ZoneInfo,
) -> Optional[str]:
    """Build the deterministic Neuro line: GCS total + components, RASS, CAM-ICU."""
    if not assessment_rows:
        return None

    window = _exam_window(reference_dttm)

    def _pick(cat: str) -> Optional[dict[str, Any]]:
        return _pick_latest_in_window(
            assessment_rows, "assessment_category", cat, "recorded_dttm", window
        )

    gcs_total_row = _pick(EXAM_GCS_TOTAL)
    gcs_e_row = _pick(EXAM_GCS_COMPONENTS[0])
    gcs_v_row = _pick(EXAM_GCS_COMPONENTS[1])
    gcs_m_row = _pick(EXAM_GCS_COMPONENTS[2])
    rass_row = _pick(EXAM_RASS)
    cam_row = _pick(EXAM_CAM_TOTAL)

    picked_any = any(
        r is not None
        for r in (gcs_total_row, gcs_e_row, gcs_v_row, gcs_m_row, rass_row, cam_row)
    )

    if not picked_any:
        # All-empty fallback — find most recent neuro assessment timestamp
        # across all relevant categories.
        latest_overall: Optional[datetime] = None
        for cat in (EXAM_GCS_TOTAL, *EXAM_GCS_COMPONENTS, EXAM_RASS, EXAM_CAM_TOTAL):
            lat = _latest_any_for_category(
                assessment_rows, "assessment_category", cat, "recorded_dttm"
            )
            if lat is not None and (latest_overall is None or lat > latest_overall):
                latest_overall = lat
        if latest_overall is None:
            return f"Neuro: no neuro assessment within {EXAM_WINDOW_HOURS}h of transfer."
        local = latest_overall.astimezone(tz)
        return (
            f"Neuro: no neuro assessment within {EXAM_WINDOW_HOURS}h of transfer "
            f"(last documented {local.month}-{local.day:02d} "
            f"{local.hour:02d}:{local.minute:02d})."
        )

    parts: list[str] = []
    anchor_dt: Optional[datetime] = None

    def _num(row: Optional[dict[str, Any]]) -> Optional[int]:
        if row is None:
            return None
        v = row.get("numerical_value")
        if v is None:
            return None
        try:
            return int(round(float(v)))
        except (TypeError, ValueError):
            return None

    def _update_anchor(row: Optional[dict[str, Any]]) -> None:
        nonlocal anchor_dt
        if row is None:
            return
        ts = _normalize_dttm(row.get("recorded_dttm"))
        if ts is not None and (anchor_dt is None or ts > anchor_dt):
            anchor_dt = ts

    gcs_total = _num(gcs_total_row)
    gcs_e = _num(gcs_e_row)
    gcs_v = _num(gcs_v_row)
    gcs_m = _num(gcs_m_row)
    if gcs_total is not None:
        if gcs_e is not None and gcs_v is not None and gcs_m is not None:
            parts.append(f"GCS {gcs_total} (E{gcs_e} V{gcs_v} M{gcs_m})")
        else:
            parts.append(f"GCS {gcs_total}")
        _update_anchor(gcs_total_row)
        _update_anchor(gcs_e_row)
        _update_anchor(gcs_v_row)
        _update_anchor(gcs_m_row)

    rass = _num(rass_row)
    if rass is not None:
        parts.append(f"RASS {rass}")
        _update_anchor(rass_row)

    cam_label = _camicu_label_from_row(cam_row)
    if cam_label is not None:
        parts.append(f"CAM-ICU {cam_label}")
        _update_anchor(cam_row)

    if not parts:
        # Picked rows had no usable values — fall through to fallback
        return f"Neuro: no neuro assessment within {EXAM_WINDOW_HOURS}h of transfer."

    assert anchor_dt is not None
    tag = _exam_cite_tag("exam-neuro", anchor_dt, tz)

    if cite_registry is not None:
        registry_rows: list[dict[str, Any]] = []
        if gcs_total is not None:
            ts = _normalize_dttm(gcs_total_row.get("recorded_dttm"))
            label_str = (
                f"GCS (E{gcs_e} V{gcs_v} M{gcs_m})"
                if gcs_e is not None and gcs_v is not None and gcs_m is not None
                else "GCS"
            )
            registry_rows.append(
                {
                    "label": label_str,
                    "value": str(gcs_total),
                    "unit": "",
                    "time": ts.isoformat() if ts else None,
                }
            )
        if rass is not None:
            ts = _normalize_dttm(rass_row.get("recorded_dttm"))
            registry_rows.append(
                {
                    "label": "RASS",
                    "value": str(rass),
                    "unit": "",
                    "time": ts.isoformat() if ts else None,
                }
            )
        if cam_label is not None:
            ts = _normalize_dttm(cam_row.get("recorded_dttm"))
            registry_rows.append(
                {
                    "label": "CAM-ICU",
                    "value": cam_label,
                    "unit": "",
                    "time": ts.isoformat() if ts else None,
                }
            )
        cite_registry.setdefault(tag, []).extend(registry_rows)

    return f"Neuro: {', '.join(parts)} {tag}"


# Devices that count as "ventilator" vs "supplemental O2" — must agree with
# the QA/resident clinical definitions at qa.yaml + resident.yaml. Keep in
# sync with orchestrator._NON_VENT_DEVICES.
_VENT_DEVICE_TOKENS = ("vent", "imv", "bipap", "cpap")
_DEVICES_NEEDING_FLOW = (
    "nasal cannula", "nc", "hfnc", "high flow", "face mask", "venturi"
)


def _device_class(device_name: str) -> str:
    """Classify a device string as vent / hfnc / lowflow / unknown."""
    if not device_name:
        return "unknown"
    s = device_name.lower()
    if any(t in s for t in _VENT_DEVICE_TOKENS):
        return "vent"
    if "hfnc" in s or "high flow" in s:
        return "hfnc"
    if any(t in s for t in _DEVICES_NEEDING_FLOW):
        return "lowflow"
    if "room air" in s or s.strip() == "none":
        return "roomair"
    return "unknown"


def _fmt_fio2(v: Any) -> Optional[str]:
    """Render FiO2 as a percent, per Phase-0 verification of fraction storage."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # FiO2 stored as fraction (Phase 0: 95% of rows <1.0, max=1.0). Render
    # as percent.
    pct = f * 100 if f <= 1.0 else f
    pct_int = int(round(pct))
    return f"{pct_int}%"


def _build_exam_resp_subblock(
    respiratory_rows: list[dict[str, Any]],
    reference_dttm: datetime,
    cite_registry: Optional[dict[str, list[dict[str, Any]]]],
    tz: ZoneInfo,
) -> Optional[str]:
    """Build the deterministic Respiratory line.

    Format depends on device class:
      vent:    "Ventilated [mode], FiO2 X%, PEEP X, PS X"
      hfnc:    "HFNC X L/min, FiO2 X%"
      lowflow: "Nasal cannula X L/min" (or face mask, etc.)
      roomair: "Room air"
      unknown: "Respiratory support: <device_name>" + any populated params

    Only the most recent in-window row drives the rendering — earlier
    rows in the window may have been replaced (device escalation/de-esc).
    """
    if not respiratory_rows:
        return None

    win_lo, win_hi = _exam_window(reference_dttm)
    in_window = [
        row for row in respiratory_rows
        if isinstance(row, dict)
        and _normalize_dttm(row.get("recorded_dttm")) is not None
        and win_lo
        <= _normalize_dttm(row.get("recorded_dttm"))  # type: ignore[operator]
        <= win_hi
    ]

    if not in_window:
        latest_overall: Optional[datetime] = None
        for row in respiratory_rows or []:
            if not isinstance(row, dict):
                continue
            ts = _normalize_dttm(row.get("recorded_dttm"))
            if ts is not None and (latest_overall is None or ts > latest_overall):
                latest_overall = ts
        if latest_overall is None:
            return f"Respiratory: no respiratory data within {EXAM_WINDOW_HOURS}h of transfer."
        local = latest_overall.astimezone(tz)
        return (
            f"Respiratory: no respiratory data within {EXAM_WINDOW_HOURS}h of "
            f"transfer (last documented {local.month}-{local.day:02d} "
            f"{local.hour:02d}:{local.minute:02d})."
        )

    # Pick the latest in-window row.
    in_window.sort(
        key=lambda r: _normalize_dttm(r.get("recorded_dttm")),  # type: ignore[arg-type, return-value]
    )
    latest = in_window[-1]
    anchor_dt = _normalize_dttm(latest.get("recorded_dttm"))
    assert anchor_dt is not None  # filter above guarantees this

    device_name = (
        latest.get("device_category")
        or latest.get("device_name")
        or ""
    )
    cls = _device_class(str(device_name))

    fio2 = _fmt_fio2(latest.get("fio2_set"))
    peep = _fmt_exam_value(latest.get("peep_set"))
    ps = _fmt_exam_value(latest.get("pressure_support_set"))
    flow = _fmt_exam_value(latest.get("flow_rate_set") or latest.get("lpm_set"))
    mode = latest.get("mode_category") or latest.get("mode_name")

    parts: list[str] = []
    if cls == "roomair":
        parts.append("Room air")
    elif cls == "vent":
        m = f" {mode}" if mode else ""
        parts.append(f"Ventilated{m}")
        if fio2:
            parts.append(f"FiO2 {fio2}")
        if peep:
            parts.append(f"PEEP {peep}")
        if ps:
            parts.append(f"PS {ps}")
    elif cls == "hfnc":
        if flow:
            parts.append(f"HFNC {flow} L/min")
        else:
            parts.append("HFNC")
        if fio2:
            parts.append(f"FiO2 {fio2}")
    elif cls == "lowflow":
        label = str(device_name).strip() or "Nasal cannula"
        if flow:
            parts.append(f"{label} {flow} L/min")
        else:
            parts.append(label)
        if fio2:
            parts.append(f"FiO2 {fio2}")
    else:
        label = str(device_name).strip() or "Respiratory support"
        parts.append(label)
        if fio2:
            parts.append(f"FiO2 {fio2}")
        if peep:
            parts.append(f"PEEP {peep}")
        if ps:
            parts.append(f"PS {ps}")

    tag = _exam_cite_tag("exam-resp", anchor_dt, tz)

    if cite_registry is not None:
        registry_rows: list[dict[str, Any]] = [
            {
                "label": "device",
                "value": str(device_name) or "unknown",
                "unit": "",
                "time": anchor_dt.isoformat(),
            }
        ]
        if fio2:
            registry_rows.append(
                {"label": "FiO2", "value": fio2, "unit": "", "time": anchor_dt.isoformat()}
            )
        if peep:
            registry_rows.append(
                {"label": "PEEP", "value": peep, "unit": "cmH2O", "time": anchor_dt.isoformat()}
            )
        if ps:
            registry_rows.append(
                {"label": "PS", "value": ps, "unit": "cmH2O", "time": anchor_dt.isoformat()}
            )
        if flow:
            registry_rows.append(
                {"label": "flow", "value": flow, "unit": "L/min", "time": anchor_dt.isoformat()}
            )
        if mode:
            registry_rows.append(
                {"label": "mode", "value": str(mode), "unit": "", "time": anchor_dt.isoformat()}
            )
        cite_registry.setdefault(tag, []).extend(registry_rows)

    return f"Respiratory: {', '.join(parts)} {tag}"


def build_transfer_exam_block(
    vitals_rows: list[dict[str, Any]],
    assessment_rows: list[dict[str, Any]],
    respiratory_rows: list[dict[str, Any]],
    reference_dttm: Optional[datetime],
    adt_rows: list[dict[str, Any]],
    cite_registry: Optional[dict[str, list[dict[str, Any]]]] = None,
    tz: Optional[ZoneInfo] = None,
) -> str:
    """Assemble the full transfer-exam block.

    Order is **Neuro → Vitals → Respiratory** per clinical convention.
    Returns empty string when ``reference_dttm`` is None — the caller
    expects an empty string (not None) so the prompt-injection plumbing
    in intensivist.py stays simple.

    ``reference_dttm`` is trusted as supplied (externally from the MICU
    note IDs CSV); no ADT anchor cross-check is performed here. Note
    timestamps routinely precede ADT ``out_dttm`` by hours (transfer
    decision written before physical bed move) — that's normal workflow,
    not data corruption.
    """
    if reference_dttm is None:
        return ""

    tz = tz or ZoneInfo("America/Chicago")

    lines: list[str] = []
    neuro = _build_exam_neuro_subblock(
        assessment_rows, reference_dttm, cite_registry, tz
    )
    if neuro:
        lines.append(neuro)
    vitals = _build_exam_vitals_subblock(
        vitals_rows, reference_dttm, cite_registry, tz
    )
    if vitals:
        lines.append(vitals)
    resp = _build_exam_resp_subblock(
        respiratory_rows, reference_dttm, cite_registry, tz
    )
    if resp:
        lines.append(resp)

    if not lines:
        return ""

    header = (
        "TRANSFER EXAM (deterministic — DO NOT paraphrase or duplicate "
        "in narrative)"
    )
    return header + "\n" + "\n".join(lines)


def _resolve_renal_status_for_meds(
    ctx: "PatientContext",
    reference_dttm: Optional[datetime],
    lookback_hours: int,
) -> str:
    """Compute renal_status for the med-state classifier.

    Pulls the most recent creatinine in lookback window, age, sex, and
    CRRT presence; delegates the calculation to
    ``med_state.resolve_renal_status``.

    Returns "unknown" on any failure path so downstream classification
    falls back to the conservative (longest-interval) defaults.
    """
    from icu_pause.tools.med_state import resolve_renal_status

    if reference_dttm is None:
        return "unknown"

    # CRRT presence in window — short-circuits creatinine-based eGFR.
    on_rrt = False
    if ctx.crrt_therapy is not None and len(ctx.crrt_therapy) > 0:
        on_rrt = True
    # IHD/HD via procedures: any dialysis procedure timestamped within
    # window. Procedures parquet is small; iterating the python rows is fine.
    if not on_rrt and ctx.procedures is not None and len(ctx.procedures) > 0:
        proc_dicts = ctx.procedures.to_dicts()
        keywords = ("hemodialysis", "dialysis", "intermittent renal replacement")
        for row in proc_dicts:
            label = " ".join(
                str(row.get(k) or "")
                for k in ("procedure_name", "procedure_category")
            ).lower()
            if any(kw in label for kw in keywords):
                on_rrt = True
                break

    # Latest creatinine row (filtered to lab_category == "creatinine") —
    # caller-side filtering keeps med_state.resolve_renal_status pure.
    creatinine_rows: list[dict[str, Any]] = []
    if ctx.labs is not None and len(ctx.labs) > 0 and "lab_category" in ctx.labs.columns:
        cre_df = ctx.labs.filter(
            pl.col("lab_category").str.to_lowercase() == "creatinine"
        )
        if len(cre_df) > 0:
            creatinine_rows = cre_df.to_dicts()

    return resolve_renal_status(
        creatinine_rows=creatinine_rows,
        age_years=ctx.age_at_icu_admission if ctx.age_at_icu_admission is not None else ctx.age_at_admission,
        sex_category=ctx.sex_category,
        on_rrt=on_rrt,
    )


def serialize_to_json(
    ctx: PatientContext,
    lookback_hours: Optional[int] = None,
    agent_role: Optional[str] = None,
    cite_registry: Optional[dict[str, list[dict[str, Any]]]] = None,
    display_tz: Optional[ZoneInfo] = None,
) -> dict[str, Any]:
    """Serialize PatientContext to raw JSON-compatible dicts.

    Produces the same top-level keys that downstream agents expect via their
    ``required_context_keys`` properties, so no agent changes are needed for
    key names.

    When *cite_registry* is provided (not None), each data record that has a
    timestamp gets an injected ``"cite"`` field — a short parenthetical string
    like ``"(lab 1/17 08:00)"`` that agents can echo verbatim.  All emitted
    cite strings are accumulated in *cite_registry* (mutated in place) so the
    downstream QA step can verify provenance.

    Args:
        ctx: The patient data context.
        lookback_hours: The user-requested lookback window.
        agent_role: If provided, the ``notes`` key will contain only the
            note types routed to this agent (from ctx.agent_notes).
        cite_registry: If not None, mutable dict collecting all emitted cite
            strings mapped to their source rows.
        display_tz: Timezone for citation timestamps.  Defaults to
            America/Chicago — site-configurable for CLIF consortium deployment.

    Returns:
        Dict mapping context keys to JSON-serializable structures.
        ``demographics`` is a flat dict of scalars; DataFrame-backed keys are
        lists of row-dicts via Polars ``.to_dicts()``.  The ``meds`` key
        combines continuous and intermittent sub-keys.
    """
    icu_stay_hours = ContextSerializer._compute_icu_stay_hours(ctx)
    if lookback_hours is not None and icu_stay_hours is not None:
        effective_window_hours: Optional[float] = min(float(lookback_hours), icu_stay_hours)
    elif lookback_hours is not None:
        effective_window_hours = float(lookback_hours)
    else:
        effective_window_hours = icu_stay_hours

    demographics: dict[str, Any] = {
        # age_at_icu_admission is computed from birth_date + ICU admit time
        # (see DataRetriever.retrieve). Falls back to CLIF age_at_admission
        # only when birth_date is missing — CLIF's value is rounded to year
        # at *hospital* admit and can be off by 1 around the patient's
        # birthday or for long pre-ICU ward stays.
        "age_at_icu_admission": (
            ctx.age_at_icu_admission if ctx.age_at_icu_admission is not None
            else ctx.age_at_admission
        ),
        "sex_category": ctx.sex_category,
        "race_category": ctx.race_category,
        "admission_type_category": ctx.admission_type_category,
        "admission_dttm": ctx.admission_dttm,
        "icu_admission_dttm": ctx.icu_admission_dttm,
        "reference_dttm": str(ctx.reference_dttm) if ctx.reference_dttm is not None else None,
        "icu_stay_hours": round(icu_stay_hours, 1) if icu_stay_hours is not None else None,
        "effective_window_hours": (
            round(effective_window_hours, 1) if effective_window_hours is not None else None
        ),
    }

    # Resolve notes: per-agent routing if available, else legacy.
    # 1. Deduplicate near-identical notes (cosine similarity > 0.9)
    # 2. Apply physician-note context floor (floor-eligible agents only)
    # 3. Cap notes per type to control context window (Croxford et al. approach:
    #    control at document count level, not by truncating individual notes).
    from icu_pause.config import AGENT_MAX_NOTES_PER_TYPE
    from icu_pause.data.note_floor import (
        FLOOR_ELIGIBLE_AGENTS,
        FLOOR_PROTECTED_COL,
        ensure_physician_note_floor,
    )

    floor_metadata: Optional[dict[str, Any]] = None

    if agent_role and agent_role in ctx.agent_notes:
        notes_data: Any = {}

        # Step 1: deduplicate near-identical notes (cosine similarity > 0.9)
        # before capping. Done across all per-type frames first so the
        # floor check (Step 2) sees the post-dedup window.
        deduped_by_type: dict[str, pl.DataFrame] = {}
        for note_type_key, df in ctx.agent_notes[agent_role].items():
            if len(df) == 0:
                continue
            deduped_by_type[note_type_key] = _deduplicate_similar_notes(df)

        # Step 2: physician-note context floor. Runs only for agents that
        # consume progress / consults / hp notes; runs BEFORE the cap so
        # the floor row cannot be displaced. The full-stay frames carry
        # the same dedup discipline via the loader (note_id revision dedup).
        if (
            agent_role in FLOOR_ELIGIBLE_AGENTS
            and ctx.agent_notes_full_stay.get(agent_role)
        ):
            full_stay_by_type = ctx.agent_notes_full_stay[agent_role]
            deduped_by_type, floor_metadata = ensure_physician_note_floor(
                deduped_by_type,
                full_stay_by_type,
                reference_dttm=ctx.reference_dttm,
                lookback_hours=lookback_hours,
            )

        # Step 3: cap by note type (empirical p95 of post-dedup distribution
        # in MICU cohort; progress_note at p90 due to four-agent routing).
        # See AGENT_MAX_NOTES_PER_TYPE in config.py for rationale.  Floor
        # rows (carrying ``_floor_protected = True``) bypass the cap — the
        # cap drops the oldest non-protected row in that type to make
        # room.  Total post-cap count remains the cap value.
        for note_type_key, df in deduped_by_type.items():
            if len(df) == 0:
                continue
            cap = AGENT_MAX_NOTES_PER_TYPE.get(note_type_key, 3)
            df_trimmed = _cap_with_floor_protection(df, cap)
            # Drop the floor-protected sentinel column before serializing —
            # it is plumbing, not agent-visible content.
            if FLOOR_PROTECTED_COL in df_trimmed.columns:
                df_trimmed = df_trimmed.drop(FLOOR_PROTECTED_COL)
            notes_data[note_type_key] = _df_to_dicts(df_trimmed)
    elif ctx.agent_notes:
        # No specific agent_role — merge ALL agent notes into a combined dict
        # so that metadata.source_data includes every note for review/evaluation.
        notes_data = {}
        for _role, notes_by_type in ctx.agent_notes.items():
            for note_type_key, df in notes_by_type.items():
                if len(df) == 0:
                    continue
                df = _deduplicate_similar_notes(df)
                if note_type_key in notes_data:
                    # Same note type routed to multiple agents — merge and deduplicate
                    existing = notes_data[note_type_key]
                    new_records = _df_to_dicts(df)
                    # Deduplicate by note text to avoid showing the same note twice
                    seen_texts = {r.get("note_text", "") for r in existing}
                    for rec in new_records:
                        if rec.get("note_text", "") not in seen_texts:
                            existing.append(rec)
                            seen_texts.add(rec.get("note_text", ""))
                    notes_data[note_type_key] = existing
                else:
                    notes_data[note_type_key] = _df_to_dicts(df)
    else:
        notes_data = _df_to_dicts(ctx.clinical_notes)

    # Newest-first ordering for every notes branch. The per-agent path above
    # already sorts via Polars; this is a no-op there. The merge-all and
    # legacy paths rely on this pass. revision_dttm wins when present
    # (matches the per-agent path's preference); creation_dttm is the
    # fallback. Reviewer + agent share this dict, so a single sort here
    # keeps them in lockstep.
    if isinstance(notes_data, dict):
        for _v in notes_data.values():
            if isinstance(_v, list):
                _sort_rows_by_dttm_desc(_v, "revision_dttm", "creation_dttm")
    elif isinstance(notes_data, list):
        _sort_rows_by_dttm_desc(notes_data, "revision_dttm", "creation_dttm")

    # Build the base result dict
    vitals_data = aggregate_vitals_for_agent(ctx.vitals, agent_role=agent_role)
    # Class B audit item 5B: per-role vital_category allowlist applied AFTER
    # tiered compaction. Filters both bucketed_trends and recent_raw within
    # the shape this agent receives. Done before cite injection so dropped
    # rows don't accumulate orphaned cite tags. No-op for roles without an
    # allowlist entry (nurse/pharmacy/intensivist/case_manager/therapist).
    vitals_data = _filter_vitals_by_category(vitals_data, agent_role)
    labs_rows = _df_to_dicts(ctx.labs)
    # Class B audit item 5D: promote ABG components + high-volume clinical
    # labs out of the non-standardized "other" bucket. Runs BEFORE the
    # noise/residual filter — promoted rows have lab_category rewritten to
    # canonical values, so they survive the filter.
    labs_rows = _promote_lab_categories_from_other(labs_rows)
    # Class A audit (CBC differential noise) + Class B audit item 5C
    # (non-standardized "other" residual after promotion). Both before
    # cite injection so the registry doesn't accumulate orphaned cite tags.
    labs_rows = _filter_noisy_lab_categories(labs_rows)
    meds_continuous = _df_to_dicts(ctx.meds_continuous)
    meds_intermittent = _df_to_dicts(ctx.meds_intermittent)
    respiratory_rows = _df_to_dicts(ctx.respiratory_support)
    assessment_rows = _df_to_dicts(ctx.patient_assessments)
    code_status_rows = _df_to_dicts(ctx.code_status)
    procedure_rows = _df_to_dicts(ctx.procedures)
    micro_rows = _df_to_dicts(ctx.microbiology)
    # Class B audit item 5A: respiratory only sees pulmonary-relevant
    # specimens. Other agents (pharmacy in particular) get the full set.
    if agent_role == "respiratory":
        micro_rows = _filter_microbiology_for_respiratory(micro_rows)

    # Newest-first ordering for every time-series domain. Sorted BEFORE cite
    # injection so cite_<domain>_1 lines up with the topmost (newest) row the
    # agent sees. Reviewer + agent share these row dicts by reference, so a
    # single sort point covers both consumers. Domains with no timestamp
    # (diagnoses) or that are unpopulated (sofa_scores) are skipped.
    _sort_rows_by_dttm_desc(labs_rows, "lab_result_dttm")
    _sort_rows_by_dttm_desc(meds_continuous, "admin_dttm")
    _sort_rows_by_dttm_desc(meds_intermittent, "admin_dttm")
    _sort_rows_by_dttm_desc(respiratory_rows, "recorded_dttm")
    _sort_rows_by_dttm_desc(assessment_rows, "recorded_dttm")
    # CLIF code_status column varies by site (`code_status_dttm` canonical;
    # `start_dttm` / `recorded_dttm` seen in the wild) — accept all three so
    # ordering survives schema drift.
    _sort_rows_by_dttm_desc(
        code_status_rows, "code_status_dttm", "start_dttm", "recorded_dttm"
    )
    _sort_rows_by_dttm_desc(
        micro_rows, "result_dttm", "collect_dttm", "recorded_dttm"
    )
    _sort_rows_by_dttm_desc(procedure_rows, "procedure_dttm", "start_dttm")
    if isinstance(vitals_data, dict):
        for _sub_key, _sub_ts in (
            ("recent_raw", "recorded_dttm"),
            ("bucketed_trends", "bucket_end"),
        ):
            _sub_rows = vitals_data.get(_sub_key)
            if isinstance(_sub_rows, list):
                _sort_rows_by_dttm_desc(_sub_rows, _sub_ts)

    # Inject cite fields when citation is enabled
    if cite_registry is not None:
        tz = display_tz or ZoneInfo("America/Chicago")
        _add_cite_fields(labs_rows, "lab", "lab_result_dttm", tz, cite_registry)
        # Vitals: inject into recent_raw and bucketed_trends sub-lists
        if isinstance(vitals_data, dict):
            if "recent_raw" in vitals_data and isinstance(vitals_data["recent_raw"], list):
                _add_cite_fields(vitals_data["recent_raw"], "vital", "recorded_dttm", tz, cite_registry)
            if "bucketed_trends" in vitals_data and isinstance(vitals_data["bucketed_trends"], list):
                _add_cite_fields(vitals_data["bucketed_trends"], "vital", "bucket_end", tz, cite_registry)
        _add_cite_fields(meds_continuous, "med", "admin_dttm", tz, cite_registry)
        _add_cite_fields(meds_intermittent, "med", "admin_dttm", tz, cite_registry)
        _add_cite_fields(respiratory_rows, "resp", "recorded_dttm", tz, cite_registry)
        _add_cite_fields(assessment_rows, "assess", "recorded_dttm", tz, cite_registry)
        _add_cite_fields(code_status_rows, "code", "code_status_dttm", tz, cite_registry)
        _add_cite_fields(procedure_rows, "proc", "procedure_dttm", tz, cite_registry)

        # Notes: one cite tag per note row, keyed on note_type as the
        # source_type token. Tag format mirrors structured-data tags so
        # build_citation_index discovers note cites through the same
        # CITE_PATTERN scan path. Time anchor is revision_dttm (preferred)
        # with creation_dttm fallback — matches the newest-first sort
        # precedence above. Unknown note_type values are skipped (their
        # source_type would fail _add_cite_fields' assertion) — a logged
        # canary tells us if the routing config grows a new type we
        # haven't registered.
        if isinstance(notes_data, dict):
            for note_type_key, note_rows in notes_data.items():
                if not isinstance(note_rows, list) or not note_rows:
                    continue
                if note_type_key not in _CITE_SOURCE_TYPES:
                    logger.warning(
                        "note cite skipped: unknown note_type %r "
                        "(rows=%d). Register in _CITE_SOURCE_TYPES + "
                        "CITE_PATTERN to enable Focus-rationale citations.",
                        note_type_key, len(note_rows),
                    )
                    continue
                # Per-row time_col selection: revision_dttm preferred,
                # creation_dttm fallback when the note hasn't been
                # revised. Split the batch so each row gets tagged with
                # the right timestamp — batch-level selection would
                # mis-tag mixed-batch rows (some with revision_dttm,
                # some without).
                rows_with_revision = [
                    r for r in note_rows if r.get("revision_dttm")
                ]
                rows_without_revision = [
                    r for r in note_rows if not r.get("revision_dttm")
                ]
                if rows_with_revision:
                    _add_cite_fields(
                        rows_with_revision, note_type_key, "revision_dttm",
                        tz, cite_registry,
                    )
                if rows_without_revision:
                    _add_cite_fields(
                        rows_without_revision, note_type_key, "creation_dttm",
                        tz, cite_registry,
                    )
        elif isinstance(notes_data, list) and notes_data:
            # Legacy clinical_notes path: note_type lives on each row.
            # Group by type so each source_type call sees a homogeneous
            # row set.
            by_type: dict[str, list[dict[str, Any]]] = {}
            for r in notes_data:
                nt = r.get("note_type")
                if not isinstance(nt, str) or not nt:
                    continue
                by_type.setdefault(nt, []).append(r)
            for note_type_key, note_rows in by_type.items():
                if note_type_key not in _CITE_SOURCE_TYPES:
                    logger.warning(
                        "note cite skipped: unknown note_type %r "
                        "(rows=%d, legacy path). Register in "
                        "_CITE_SOURCE_TYPES + CITE_PATTERN to enable.",
                        note_type_key, len(note_rows),
                    )
                    continue
                rows_with_revision = [
                    r for r in note_rows if r.get("revision_dttm")
                ]
                rows_without_revision = [
                    r for r in note_rows if not r.get("revision_dttm")
                ]
                if rows_with_revision:
                    _add_cite_fields(
                        rows_with_revision, note_type_key, "revision_dttm",
                        tz, cite_registry,
                    )
                if rows_without_revision:
                    _add_cite_fields(
                        rows_without_revision, note_type_key, "creation_dttm",
                        tz, cite_registry,
                    )

    # Drop admin/coding columns AFTER cite injection so cite tags are built
    # from the original timestamps; the registry holds row-by-reference and
    # will see the trimmed shape, which is the intended end state.
    _drop_admin_columns(labs_rows, "lab")
    _drop_admin_columns(meds_continuous, "med")
    _drop_admin_columns(meds_intermittent, "med")
    _drop_admin_columns(respiratory_rows, "resp")
    _drop_admin_columns(assessment_rows, "assess")
    _drop_admin_columns(code_status_rows, "code")
    _drop_admin_columns(procedure_rows, "proc")
    _drop_admin_columns(micro_rows, "micro")
    if isinstance(vitals_data, dict):
        for sub in ("recent_raw", "bucketed_trends"):
            sub_rows = vitals_data.get(sub)
            if isinstance(sub_rows, list):
                _drop_admin_columns(sub_rows, "vital")

    # Class A audit: row-level redundancy drops. Run AFTER admin drops so the
    # cite registry holds the final row shape. These are conditional drops —
    # only fire when the canonical/numeric counterpart is present.
    _drop_lab_value_when_numeric_present(labs_rows)
    _drop_med_name_when_category_present(meds_continuous)
    _drop_med_name_when_category_present(meds_intermittent)
    _strip_null_assessment_value_columns(assessment_rows)

    adt_rows = _drop_admin_columns(_df_to_dicts(ctx.adt), "adt")
    _sort_rows_by_dttm_desc(
        adt_rows, "in_dttm", "out_dttm", "start_dttm", "recorded_dttm"
    )

    # Classify medication states at reference_dttm.  Done BEFORE
    # _drop_admin_columns so the classifier sees admin_dttm; the resulting
    # records carry their own copies of the timestamp string and won't be
    # affected by the column drop above.
    resolved_lookback = (
        int(lookback_hours) if lookback_hours is not None else 48
    )
    renal_status_for_meds = _resolve_renal_status_for_meds(
        ctx, ctx.reference_dttm, resolved_lookback
    )
    med_states_block = _build_med_states_block(
        meds_continuous,
        meds_intermittent,
        ctx.reference_dttm,
        renal_status=renal_status_for_meds,
        lookback_hours=resolved_lookback,
    )
    # active/recently_stopped/historical buckets are flat string lists for
    # narrative rendering — not chronological. Only records[] carries
    # per-drug timestamps and is sorted newest-first.
    if isinstance(med_states_block.get("records"), list):
        _sort_rows_by_dttm_desc(med_states_block["records"], "last_admin_dttm")
        # Reformat last_admin_dttm AFTER the sort so the LLM sees local
        # display strings (``M-DD HH:MM``) instead of raw ISO. The sort
        # above parses the ISO form via ``_parse_cite_timestamp``; doing
        # the reformat post-sort preserves the chronological order without
        # teaching the parser the local format. Companion to base.py's
        # ``_json_default_local_dttm`` (which catches raw datetime objects
        # but not pre-stringified ISO timestamps like this field).
        for record in med_states_block["records"]:
            iso = record.get("last_admin_dttm")
            if iso:
                record["last_admin_dttm"] = format_local_dttm(iso)

    crrt_rows = _df_to_dicts(ctx.crrt_therapy)
    ecmo_rows = _df_to_dicts(ctx.ecmo_mcs)
    position_rows = _df_to_dicts(ctx.position)
    _sort_rows_by_dttm_desc(crrt_rows, "recorded_dttm")
    _sort_rows_by_dttm_desc(ecmo_rows, "recorded_dttm")
    _sort_rows_by_dttm_desc(position_rows, "recorded_dttm")

    clinical_context_block = (
        ctx.clinical_context.to_dict()
        if ctx.clinical_context is not None
        else None
    )

    # Phase-3 deterministic transfer-exam block. Built from the raw ctx
    # DataFrames (NOT per-agent slices) so every agent sees the same
    # authoritative block — agent-role tiering doesn't change the
    # deterministic snapshot. Vitals: pulled directly from ``ctx.vitals``
    # to bypass per-agent allowlists / bucketed_trends collapsing. Assessment
    # and respiratory rows are agent-uniform already.
    vitals_for_exam = _df_to_dicts(ctx.vitals)
    transfer_exam_block = build_transfer_exam_block(
        vitals_for_exam,
        assessment_rows,
        respiratory_rows,
        ctx.reference_dttm,
        adt_rows,
        cite_registry=cite_registry,
        tz=display_tz or ZoneInfo("America/Chicago"),
    )

    result: dict[str, Any] = {
        "demographics": demographics,
        "adt": adt_rows,
        "vitals": vitals_data,
        "labs": labs_rows,
        "meds": {
            "continuous": meds_continuous,
            "intermittent": meds_intermittent,
            "states": med_states_block,
        },
        "respiratory": respiratory_rows,
        "assessments": assessment_rows,
        "code_status": code_status_rows,
        "diagnoses": _df_to_dicts(ctx.diagnoses),
        "microbiology": micro_rows,
        "procedures": procedure_rows,
        "sofa": _df_to_dicts(ctx.sofa_scores),
        "crrt": crrt_rows,
        "ecmo": ecmo_rows,
        "position": position_rows,
        "notes": notes_data,
        "clinical_context": clinical_context_block,
        "transfer_exam_block": transfer_exam_block,
    }
    # Internal plumbing key: underscore-prefixed so agents (which read via
    # required_context_keys) ignore it, and so workflow.py can lift it out
    # of the per-agent slice into GraphState before the slice reaches the
    # reviewer-facing source-data union.
    if floor_metadata is not None:
        result["_physician_floor"] = floor_metadata
    return result


# ---------------------------------------------------------------------------
# Critical flags — deterministic cross-domain summary for all agents
# ---------------------------------------------------------------------------

_VASOPRESSOR_CATEGORIES = {
    "norepinephrine", "epinephrine", "vasopressin", "phenylephrine",
    "dopamine", "dobutamine",
}


def build_critical_flags(patient_ctx: dict[str, Any]) -> str:
    """Build a deterministic ~150-token cross-domain summary from serialized patient data.

    Prepended to every agent's user message so agents can reference critical
    cross-domain signals without receiving full data tables. Always returns a
    string (empty-safe header when no flags fire).
    """
    flags: list[str] = []

    # --- Active infections ---
    micro = patient_ctx.get("microbiology") or []
    organisms = []
    for row in micro:
        org = row.get("organism") or row.get("organism_name") or ""
        if org and org.lower() not in ("pending", "no growth", "negative", ""):
            organisms.append(org)
    if organisms:
        flags.append(f"ACTIVE INFECTIONS: {', '.join(sorted(set(organisms)))}")

    # --- Hemodynamic concern (last MAP from vitals) ---
    # All time-series rows arrive newest-first (see serialize_to_json), so
    # iterating from index 0 yields the latest sample.
    vitals = patient_ctx.get("vitals") or {}
    bucketed = vitals.get("bucketed_trends") or []
    recent_raw = vitals.get("recent_raw") or []
    last_map = None
    # Try bucketed trends first (most agents see these). Use the bucket
    # mean — `last` was a noisy random sample at the bucket boundary.
    for row in bucketed:
        vname = str(row.get("vital_category", row.get("vital_name", ""))).lower()
        if vname == "map" and row.get("mean") is not None:
            last_map = row["mean"]
            break
    # Fall back to recent raw
    if last_map is None:
        for row in recent_raw:
            vname = str(row.get("vital_category", row.get("vital_name", ""))).lower()
            if vname == "map" and row.get("vital_value") is not None:
                last_map = row["vital_value"]
                break
    if last_map is not None:
        try:
            if float(last_map) < 65:
                flags.append(f"HEMODYNAMIC CONCERN: MAP {last_map} mmHg")
        except (ValueError, TypeError):
            pass

    # --- Respiratory support level ---
    resp = patient_ctx.get("respiratory") or []
    if resp:
        latest = resp[0] if isinstance(resp, list) else {}
        device = latest.get("device_category", "unknown")
        fio2 = latest.get("fio2_set")
        fio2_str = f" FiO2 {fio2}" if fio2 is not None else ""
        flags.append(f"RESPIRATORY SUPPORT: {device}{fio2_str}")

    # --- Vasopressor banner (state-gated) ---
    # Activity decision MUST come from the med_state classifier (built into
    # meds.states.records at retrieval time), NOT from the presence of admin
    # rows. The classifier compares the most recent admin against
    # reference_dttm with infusion-specific recency rules; the raw-rows path
    # used to call drugs "active" for hours after they were stopped, which
    # then leaked into every agent's CRITICAL FLAGS prefix and triggered
    # spurious cross-domain conflicts (iter-0: nurse vs pharmacy on
    # norepinephrine that was RECENTLY_STOPPED).
    #
    # Trend logic for ACTIVE drugs is preserved verbatim — it operates on
    # the dose history in meds.continuous (newest-first), which the
    # classifier doesn't touch. RECENTLY_STOPPED drugs surface in their own
    # framing so the prefix doesn't compute a meaningless trend on stopped
    # doses or claim active support that isn't running.
    meds = patient_ctx.get("meds") or {}
    continuous = meds.get("continuous") or []
    states_block = meds.get("states") or {}
    state_records = states_block.get("records") or []

    active_vasopressors: set[str] = set()
    recently_stopped_vasopressors: set[str] = set()
    for rec in state_records:
        drug = str(rec.get("drug_name", "")).lower()
        if drug not in _VASOPRESSOR_CATEGORIES:
            continue
        st = rec.get("state")
        if st == "ACTIVE":
            active_vasopressors.add(drug)
        elif st == "RECENTLY_STOPPED":
            recently_stopped_vasopressors.add(drug)

    if active_vasopressors:
        vaso_doses: dict[str, list[tuple]] = {}
        for row in continuous:
            med = str(row.get("med_category", row.get("medication_name", ""))).lower()
            if med in active_vasopressors:
                dose = row.get("med_dose") or row.get("dose")
                dttm = row.get("admin_dttm", "")
                vaso_doses.setdefault(med, []).append((dttm, dose))
        parts = []
        for med in sorted(active_vasopressors):
            entries = vaso_doses.get(med, [])
            # Trend from the two most recent entries. `continuous` arrives
            # newest-first so entries[0] is the latest dose.
            trend = ""
            if len(entries) >= 2:
                try:
                    d_prior = float(entries[1][1])
                    d_latest = float(entries[0][1])
                    if d_latest > d_prior * 1.1:
                        trend = ", dose increasing"
                    elif d_latest < d_prior * 0.9:
                        trend = ", dose decreasing"
                    else:
                        trend = ", dose stable"
                except (ValueError, TypeError):
                    pass
            parts.append(f"{med} active{trend}")
        flags.append(f"VASOPRESSORS: {'; '.join(parts)}")

    if recently_stopped_vasopressors:
        flags.append(
            "VASOPRESSORS RECENTLY DISCONTINUED: "
            + ", ".join(sorted(recently_stopped_vasopressors))
        )

    # --- Code status ---
    code_status = patient_ctx.get("code_status") or []
    if code_status:
        latest_cs = code_status[0] if isinstance(code_status, list) else {}
        status_val = latest_cs.get("code_status_category") or latest_cs.get("code_status")
        if status_val:
            flags.append(f"CODE STATUS: {status_val}")

    # --- Goals of care mismatch: CMO + aggressive interventions ---
    if code_status:
        latest_cs = code_status[0] if isinstance(code_status, list) else {}
        cs_val = (
            latest_cs.get("code_status_category")
            or latest_cs.get("code_status")
            or ""
        )
        if cs_val.upper() in ("CMO", "COMFORT MEASURES ONLY", "COMFORT CARE"):
            # Same state-gating rule as the vasopressor banner: only drugs
            # the classifier marked ACTIVE count as an active intervention.
            # A RECENTLY_STOPPED or HISTORICAL aggressive med is not a
            # goals-of-care conflict; flagging it would mislead the
            # intensivist into recommending unnecessary deescalation.
            _AGGRESSIVE_INTERMITTENT = {
                "vancomycin", "meropenem", "piperacillin", "cefepime",
                "ciprofloxacin", "levofloxacin", "tpn",
            }
            aggressive_meds: set[str] = set()
            for rec in state_records:
                if rec.get("state") != "ACTIVE":
                    continue
                drug = str(rec.get("drug_name", "")).lower()
                if drug in _VASOPRESSOR_CATEGORIES or drug in _AGGRESSIVE_INTERMITTENT:
                    aggressive_meds.add(drug)
            if aggressive_meds:
                flags.append(
                    f"GOALS OF CARE MISMATCH: Code status is {cs_val} but "
                    f"active aggressive interventions: "
                    f"{', '.join(sorted(aggressive_meds))}"
                )

    # --- Active CRRT/ECMO ---
    if patient_ctx.get("crrt"):
        flags.append("CRRT: Active")
    if patient_ctx.get("ecmo"):
        flags.append("ECMO/MCS: Active")

    # --- Patient clinical context (chronic conditions / baselines) ---
    # Agent-uniform: every domain agent sees the same chronic context the
    # downstream lab-warning reframer (PR 3) will use. Anticoag and steroid
    # contexts are deferred; see safety/clinical_context.py header note.
    cc_summary = _format_clinical_context_summary(patient_ctx.get("clinical_context"))
    if cc_summary:
        flags.append(f"PATIENT CONTEXT: {cc_summary}")

    if flags:
        return "CRITICAL FLAGS (reference when relevant to your domain):\n" + "\n".join(f"- {f}" for f in flags)
    return (
        "CRITICAL FLAGS: None active \u2014 patient hemodynamically stable, "
        "no active infections or vasoactive support."
    )


_CLINICAL_CONTEXT_LABELS: list[tuple[str, str]] = [
    ("has_esrd_dialysis", "ESRD/dialysis"),
    ("has_cirrhosis", "cirrhosis"),
    ("has_copd", "COPD"),
    ("has_chronic_afib", "chronic AF"),
    ("has_chronic_trach", "chronic tracheostomy"),
]


def _format_clinical_context_summary(cc: Any) -> str:
    """Render the chronic-condition flags as a short narrative phrase.

    ``cc`` is the ``clinical_context`` dict from ``serialize_to_json``
    (or None if inference didn't run). Empty when no flags are set, so
    the caller can skip emitting the line.

    Order is fixed (per ``_CLINICAL_CONTEXT_LABELS``) so the rendered
    text is stable across runs — agent-input determinism matters for
    cache-friendliness and reviewer comparisons.

    Therapeutic anticoagulation (added 2026-05-29) trails the chronic-
    condition labels when set, so agents reasoning about bleeding risk
    / INR see it on the same line as the chronic context.
    """
    if not isinstance(cc, dict):
        return ""
    labels = [label for key, label in _CLINICAL_CONTEXT_LABELS if cc.get(key)]
    anticoag = cc.get("on_therapeutic_anticoagulation")
    if anticoag:
        humanized = " + ".join(anticoag.split("|"))
        labels.append(f"on therapeutic anticoagulation ({humanized})")
    return "; ".join(labels)

"""Per-brief safety-drift metrics emitter (§8 of renal/electrolyte/VTE design).

Each completed brief emits a JSON line to ``<run_dir>/safety_drift.jsonl``
describing which safety WARNs and info-level signals fired plus the
structural denominators needed for rate computation by the sidecar
rollup (``scripts/safety_drift_rollup.py``).

The schema is the FULL union across PR1/PR2/PR3 — fields that aren't
wired yet emit as ``False``. This keeps the JSONL append-only across the
three-PR rollout: the rollup script never has to special-case missing
keys, and per-PR pre/post comparisons stay apples-to-apples.

Severity-tier discipline (v3.1):
- ``warns`` are safety-relevant; the rollup applies the 20% / 7-day
  threshold to these and exits non-zero on crossing.
- ``info_signals`` are population-trend signals (chronic-ESRD prevalence,
  elderly-low-weight prevalence) reported but never alerted on.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema constants — full PR1 + PR2 + PR3 union
# ---------------------------------------------------------------------------
#
# PR2 / PR3 keys ship as always-False in PR1 so the JSONL schema stays
# stable across the three-PR rollout. PR2 wires
# HYPONATREMIA_*, HYPERKALEMIA_*, HYPONATREMIA_RATE_EXCEEDS_CAP; PR3 wires
# VTE_MODALITY_DISAGREEMENT, VTE_DOSE_RENAL_MISMATCH,
# VTE_DOSE_CRCL_INDETERMINATE.

WARN_KEYS: tuple[str, ...] = (
    "AKI_PROBLEM_MISSING_BASELINE_REFERENCE",
    "ELECTROLYTE_PROBLEM_MERGED_WITHOUT_ATTRIBUTION",
    "HYPONATREMIA_CONTEXT_OMITTED_DESPITE_PIN",
    "HYPERKALEMIA_CONTEXT_OMITTED_DESPITE_PIN",
    "HYPONATREMIA_RATE_EXCEEDS_CAP",
    "KDIGO_STAGE_DISAGREEMENT",
    "VTE_MODALITY_DISAGREEMENT",
    "VTE_DOSE_RENAL_MISMATCH",
)

INFO_SIGNAL_KEYS: tuple[str, ...] = (
    "KDIGO_NOT_APPLICABLE_CHRONIC_BASELINE",
    "VTE_DOSE_CRCL_INDETERMINATE",
)

SAFETY_DRIFT_FILENAME = "safety_drift.jsonl"


# ---------------------------------------------------------------------------
# Problem-header detection on the S section
# ---------------------------------------------------------------------------

# Keyword set MUST stay aligned with
# ``orchestrator._RENAL_HEADER_KEYWORDS``. The metric's
# ``has_aki_problem`` denominator MUST agree with what the operating
# layer (nephrotoxin-dedup) considers a renal block — otherwise the
# drift metric reports rates over a population the dedup pass disagrees
# with, which is the exact silent-degradation pattern §8 was designed
# to detect. See design-doc §12 iter-1 finding "Denominator-alignment
# between metric and operating layer."
_AKI_HEADER_KEYWORDS: tuple[str, ...] = (
    "aki", "acute kidney", "ckd", "renal", "nephropathy", "kidney",
    "creatinine", "renal failure", "renal injury", "renal impairment",
    "chronic kidney",
)

_HYPONA_TOKENS: tuple[str, ...] = ("hyponatremia", "hypona")
_HYPERK_TOKENS: tuple[str, ...] = ("hyperkalemia", "hyperk")
_ELECTROLYTE_TOKENS: tuple[str, ...] = _HYPONA_TOKENS + _HYPERK_TOKENS


def _has_problem_header(s_text: str, tokens: tuple[str, ...]) -> bool:
    if not s_text:
        return False
    for line in s_text.splitlines():
        if not line.lstrip().startswith("#"):
            continue
        lower = line.lower()
        if any(t in lower for t in tokens):
            return True
    return False


def detect_has_aki_problem(s_text: str) -> bool:
    """True when S contains a `#Problem` header line that includes any
    of the renal keywords. Mirrors the orchestrator's loose-substring
    convention (``_is_renal_header``) so the metric denominator and the
    nephrotoxin-dedup pass agree on what a "renal block" is.
    """
    return _has_problem_header(s_text, _AKI_HEADER_KEYWORDS)


def detect_has_electrolyte_problem(s_text: str) -> bool:
    return _has_problem_header(s_text, _ELECTROLYTE_TOKENS)


def detect_has_hyponatremia_problem(s_text: str) -> bool:
    return _has_problem_header(s_text, _HYPONA_TOKENS)


def detect_has_hyperkalemia_problem(s_text: str) -> bool:
    return _has_problem_header(s_text, _HYPERK_TOKENS)


# ---------------------------------------------------------------------------
# §4.2.4 extraction-independent AKI signal (denominator for the
# scribe-emission rate; gates the §4.4.8 enoxaparin dose check in PR3)
# ---------------------------------------------------------------------------


def compute_structurally_indicated_aki(
    labs: list[dict[str, Any]] | None,
) -> bool:
    """Return True when structured labs indicate AKI without relying on
    any scribe extraction.

    Fires when EITHER condition holds:
      - sustained elevation: latest(Cr) > 1.5 AND min(Cr past 7d) > 1.2
      - acute rise:          latest(Cr) − min(Cr past 48h) ≥ 0.3

    Labs are expected newest-first per the retriever convention. Returns
    False when no creatinine rows are present or when no value parses.

    Why this is the right denominator (per §8.2): the v2 metric used
    ``has_aki_problem`` as the denominator, which is downstream of the
    scribe extraction the metric is supposed to monitor. The v3 fix
    breaks that circular dependency by using a labs-only signal.
    """
    if not labs or not isinstance(labs, list):
        return False

    creatinine_rows: list[tuple[Any, float]] = []
    for row in labs:
        if not isinstance(row, dict):
            continue
        cat = (
            str(row.get("lab_category") or "").strip().lower()
            or str(row.get("lab_name") or "").strip().lower()
        )
        if "creatinine" not in cat:
            continue
        val_raw = row.get("lab_value")
        if val_raw is None:
            continue
        try:
            val = float(str(val_raw).strip())
        except (TypeError, ValueError):
            continue
        ts = row.get("lab_collect_dttm") or row.get("collect_dttm")
        creatinine_rows.append((ts, val))

    if not creatinine_rows:
        return False

    from icu_pause.data.context import _parse_cite_timestamp

    latest_ts, latest_val = creatinine_rows[0]
    latest_dt = _parse_cite_timestamp(latest_ts) if latest_ts else None

    min_7d = latest_val
    min_48h = latest_val
    if latest_dt is not None:
        window_7d_start = latest_dt - timedelta(days=7)
        window_48h_start = latest_dt - timedelta(hours=48)
        for ts, val in creatinine_rows[1:]:
            dt = _parse_cite_timestamp(ts) if ts else None
            if dt is None:
                continue
            if dt >= window_7d_start:
                if val < min_7d:
                    min_7d = val
                if dt >= window_48h_start and val < min_48h:
                    min_48h = val
            else:
                # newest-first: once out of the 7d window, the rest are too
                break

    sustained = latest_val > 1.5 and min_7d > 1.2
    # round to 0.01 mg/dL (clinical Cr precision) to guard against
    # IEEE-754 traps at the 0.3 boundary — same precision discipline
    # as the KDIGO compute (§4.3.5, iter-1 finding #3).
    acute = round(latest_val - min_48h, 2) >= 0.3
    return bool(sustained or acute)


# ---------------------------------------------------------------------------
# Scribe-emission detectors. Each returns False until its companion
# scribe field lands (renal_context now; hyponatremia/hyperkalemia in
# PR2; vte_prophylaxis in PR3).
# ---------------------------------------------------------------------------


def detect_scribe_emitted_renal_context(state: dict[str, Any]) -> bool:
    ex = state.get("scribe_extraction") or {}
    if not ex.get("renal_context_validated"):
        return False
    rc = ex.get("renal_context") or {}
    # Counts as "emitted" when at least one of the substantive subfields
    # is populated. baseline_source_quote alone is the empty case (the
    # scribe couldn't find an anchor; the field is essentially null).
    informative_keys = (
        "baseline_creatinine",
        "baseline_creatinine_date",
        "kdigo_stage",
        "urine_output_pattern",
        "nephrology_status",
        "rrt_indications_documented",
    )
    return any(rc.get(k) for k in informative_keys)


def detect_scribe_emitted_hyponatremia_context(state: dict[str, Any]) -> bool:
    ex = state.get("scribe_extraction") or {}
    if not ex.get("hyponatremia_context_validated"):
        return False
    return bool(ex.get("hyponatremia_context"))


def detect_scribe_emitted_hyperkalemia_context(state: dict[str, Any]) -> bool:
    ex = state.get("scribe_extraction") or {}
    if not ex.get("hyperkalemia_context_validated"):
        return False
    return bool(ex.get("hyperkalemia_context"))


def detect_scribe_emitted_vte_prophylaxis(state: dict[str, Any]) -> bool:
    ex = state.get("scribe_extraction") or {}
    if not ex.get("vte_prophylaxis_validated"):
        return False
    return bool(ex.get("vte_prophylaxis"))


# ---------------------------------------------------------------------------
# Version pinning
# ---------------------------------------------------------------------------


def get_pipeline_version() -> str:
    """Return short git commit SHA of cwd, or "unknown" on failure."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return sha.decode().strip() or "unknown"
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return "unknown"


def get_scribe_version(state: dict[str, Any]) -> str:
    """Pull scribe prompt version off ``state['pipeline_metrics']``."""
    metrics = state.get("pipeline_metrics") or []
    for m in metrics:
        if isinstance(m, dict) and m.get("agent") == "scribe":
            v = m.get("prompt_version")
            if v:
                return str(v)
    return "unknown"


# ---------------------------------------------------------------------------
# Record builder + emitter
# ---------------------------------------------------------------------------


def _default_warns() -> dict[str, bool]:
    return {k: False for k in WARN_KEYS}


def _default_info_signals() -> dict[str, bool]:
    return {k: False for k in INFO_SIGNAL_KEYS}


def build_safety_drift_record(
    *,
    state: dict[str, Any],
    merged_sections: dict[str, str] | None,
    context: dict[str, Any] | None,
    hospitalization_id: str,
    reference_dttm: str | None = None,
    pipeline_version: str | None = None,
    scribe_version: str | None = None,
) -> dict[str, Any]:
    """Assemble the per-brief safety-drift record per §8.1.

    Always returns a record with the full WARN + info_signal key set so
    the sidecar rollup can rely on a stable schema even before PR2 / PR3
    wire their emissions.
    """
    s_text = (merged_sections or {}).get("S", "")
    labs = (context or {}).get("labs") or []

    emissions = state.get("safety_drift_emissions") or {}
    emitted_warns = emissions.get("warns") or {}
    emitted_info = emissions.get("info_signals") or {}

    warns = _default_warns()
    for key in WARN_KEYS:
        if emitted_warns.get(key):
            warns[key] = True

    info_signals = _default_info_signals()
    for key in INFO_SIGNAL_KEYS:
        if emitted_info.get(key):
            info_signals[key] = True

    return {
        "hospitalization_id": str(hospitalization_id),
        "reference_dttm": (
            str(reference_dttm) if reference_dttm is not None else None
        ),
        "pipeline_version": pipeline_version or get_pipeline_version(),
        "scribe_version": scribe_version or get_scribe_version(state),
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "has_aki_problem": detect_has_aki_problem(s_text),
        "has_electrolyte_problem": detect_has_electrolyte_problem(s_text),
        "has_hyponatremia_problem": detect_has_hyponatremia_problem(s_text),
        "has_hyperkalemia_problem": detect_has_hyperkalemia_problem(s_text),
        "structurally_indicated_aki": compute_structurally_indicated_aki(labs),
        # PR2 will wire; PR1 ships False.
        "structurally_indicated_hyperkalemia_trend": False,
        # PR3 will wire; PR1 ships False.
        "structurally_indicated_enoxaparin_dose_check": False,
        "scribe_emitted_renal_context": (
            detect_scribe_emitted_renal_context(state)
        ),
        "scribe_emitted_hyponatremia_context": (
            detect_scribe_emitted_hyponatremia_context(state)
        ),
        "scribe_emitted_hyperkalemia_context": (
            detect_scribe_emitted_hyperkalemia_context(state)
        ),
        "scribe_emitted_vte_prophylaxis": (
            detect_scribe_emitted_vte_prophylaxis(state)
        ),
        "warns": warns,
        "info_signals": info_signals,
    }


def emit_safety_drift_record(
    record: dict[str, Any],
    run_dir: str | Path | None = None,
) -> str | None:
    """Append ``record`` as one JSON line to
    ``<run_dir>/safety_drift.jsonl``.

    Resolves ``run_dir`` from the explicit argument, then
    ``ICUPAUSE_RUN_DIR`` env, then defaults to ``output/runs`` —
    matching the convention used by ``tracing.RunTrace``. Returns the
    file path on success, ``None`` on failure (logged but non-fatal:
    the drift metric must never break a pipeline run).
    """
    try:
        resolved = run_dir or os.environ.get(
            "ICUPAUSE_RUN_DIR", "output/runs"
        )
        out_dir = Path(resolved)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / SAFETY_DRIFT_FILENAME
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return str(path)
    except Exception:  # pragma: no cover — non-fatal guard
        # exc_info=True so disk / permission failures are diagnosable
        # without re-running the brief.
        logger.warning(
            "Failed to emit safety_drift record", exc_info=True
        )
        return None

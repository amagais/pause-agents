"""Section Merger: merges validated agent snippets into final ICU-PAUSE output."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from icu_pause.data.context import union_post_cap_contexts
from icu_pause.eval.safety_drift_metrics import (
    build_safety_drift_record,
    emit_safety_drift_record,
)
from icu_pause.schemas.icu_pause import (
    AgentSnippet,
    CompetingRisksEntry,
    ConflictSeverity,
    ICUPauseOutput,
    ICUPauseSection,
    ModifierConfirmation,
    OneLinerPMHEntry,
    Warning,
    WarningCategory,
    WarningSeverity,
)
from icu_pause.tools.text_normalize import (
    has_truncation_marker,
    normalize_for_pmh_match,
    normalize_for_validator,
)

# ---------------------------------------------------------------------------
# Deterministic vent-dependent status from respiratory support data
# ---------------------------------------------------------------------------

_VENT_DEPENDENT_DEVICES = {"imv", "nippv"}
# "trach" alone is intentionally NOT a vent keyword: CLIF device_category
# "Trach Collar" is supplemental O2 via the tracheostomy stoma, not positive-
# pressure ventilation. Patients on chronic invasive vent via trach surface as
# device_category "Vent"/"IMV" (caught above) or as a string containing
# "ventilat" (caught below).
_VENT_KEYWORDS = {"endotracheal", "ett", "ventilat"}
# Explicit non-vent device strings: these contain substrings that could be
# misread as ventilation but are oxygen-only modalities.
_NON_VENT_DEVICES = {"trach collar", "t-collar", "t collar", "tracheostomy collar"}


def _pick_latest_row(rows: Any, ts_key: str) -> Optional[dict]:
    """Return the dict in ``rows`` with the largest parseable ``ts_key``.

    Order-agnostic helper used by deterministic clinical-state extractors
    that may be called with either serialized (newest-first) or raw
    (insertion-order) inputs. Falls back to the first dict in the list
    when no row carries a parseable timestamp.
    """
    if not isinstance(rows, list) or not rows:
        return None
    from icu_pause.data.context import _parse_cite_timestamp

    best: Optional[dict] = None
    best_ts: Optional[datetime] = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_cite_timestamp(row.get(ts_key))
        if ts is None:
            continue
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        if best_ts is None or ts > best_ts:
            best, best_ts = row, ts
    if best is not None:
        return best
    for row in rows:
        if isinstance(row, dict):
            return row
    return None


def _determine_vent_status(respiratory: Any) -> str:
    """Determine vent-dependent status deterministically from structured data.

    Returns a clinical string suitable for the C section:
      "Y — [device details]"  or  "N"  or  "Unable to determine (no respiratory data)"

    Order-agnostic: callers pass either serialized rows (newest-first) or
    raw Parquet rows (insertion order). We pick the row with the latest
    ``recorded_dttm`` ourselves and fall back to the first dict if no
    timestamps are available.
    """
    if not respiratory or not isinstance(respiratory, list):
        return "Unable to determine (no respiratory data)"

    latest = _pick_latest_row(respiratory, "recorded_dttm")
    if not latest or not isinstance(latest, dict):
        return "Unable to determine (no respiratory data)"

    device = str(latest.get("device_category", "")).strip()
    mode = str(latest.get("mode_category", "")).strip()
    fio2 = latest.get("fio2_set")

    if not device:
        return "Unable to determine (no device recorded)"

    device_lower = device.lower()

    # Trach collar (supplemental O2 via tracheostomy stoma) is NOT mechanical
    # ventilation — short-circuit before the keyword check.
    if device_lower in _NON_VENT_DEVICES:
        return "N"

    # Check for invasive mechanical ventilation or NIPPV
    if device_lower in _VENT_DEPENDENT_DEVICES or any(
        kw in device_lower for kw in _VENT_KEYWORDS
    ):
        details = [device]
        if mode:
            details.append(mode)
        if fio2 is not None:
            details.append(f"FiO2 {fio2}")
        return f"Y — {', '.join(details)}"

    # Non-vent devices (nasal cannula, HFNC, room air, trach collar, etc.)
    return "N"


# ---------------------------------------------------------------------------
# S section helpers — deduplication, generic filtering, agent grouping
# ---------------------------------------------------------------------------

# Generic monitoring patterns that add no signal — apply to any ICU patient.
# Used both for S-section filtering and todo specificity tracking.
_GENERIC_PATTERNS = [
    "monitor vitals", "continue medications", "continue current",
    "follow up", "follow-up labs", "monitor respiratory",
    "monitor gcs", "continue dvt", "optimize patient",
    "continue management", "monitor status", "continue plan",
]


def _filter_generic_problems(content: str) -> str:
    """Remove generic monitoring problems from S-section content.

    A #Problem line is considered generic if it matches any pattern in
    _GENERIC_PATTERNS.  Its associated [] to-do lines are also removed.
    """
    lines = content.split("\n")
    result: list[str] = []
    skip_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("#") and not stripped.startswith("##"):
            # Problem header — check if generic
            if any(p in stripped.lower() for p in _GENERIC_PATTERNS):
                skip_block = True
                continue
            else:
                skip_block = False
                result.append(line)
        elif stripped.startswith("## "):
            # Domain header — never skip
            skip_block = False
            result.append(line)
        elif skip_block and (re.match(r'^\[[ x]?\]\s', stripped) or not stripped):
            # To-do or blank line belonging to a skipped generic problem
            continue
        else:
            skip_block = False
            result.append(line)

    return "\n".join(result)


# Action verbs that signal a to-do item (case-insensitive first word match)
_ACTION_VERBS = {
    "monitor", "continue", "assess", "follow", "coordinate",
    "confirm", "restart", "wean", "obtain", "check", "verify",
    "ensure", "notify", "schedule", "order", "titrate", "taper",
    "discontinue", "start", "stop", "hold", "resume", "request",
    "consult", "review", "update", "document", "finalize",
    "encourage", "maintain", "observe", "trend", "repeat",
    "initiate", "consider", "address", "taper", "transition",
}


def _format_ddi_review_block(
    meds_data: dict[str, Any],
    settings,
    reference_dttm: Optional[Any] = None,
) -> str:
    """Render a deterministic automated-screen block for U_unprescribing.

    Always shows what the screen examined (active med list + count) and
    whether any high-severity interactions were identified. Empty string
    if the DDI tool is disabled, the API is unavailable, or fewer than 2
    active meds.

    "Active" reflects med_state classification at ``reference_dttm`` —
    RECENTLY_STOPPED and HISTORICAL drugs are excluded.

    The header explicitly attributes the check to an automated screen,
    not a documented pharmacist review, to prevent clinicians from
    reading the block as a chart artifact (round-2 reviewer feedback,
    2026-05-26). Body wording ("Screened", "identified by screen")
    pairs with the header so a single line skimmed in isolation still
    carries the correct attribution.

    Format (matches the dotphrase checkbox style used elsewhere in U):

        Automated medication interaction screen:
        ☐ ✓ Screened: warfarin, amiodarone, fentanyl, heparin (4 active)
        ☐ No high-severity interactions identified by screen

    With high-severity hits:

        Automated medication interaction screen:
        ☐ ✓ Screened: warfarin, amiodarone, ... (12 active)
        ☐ ⚠ Screen identified 2 high-severity interactions — see QA Issues
    """
    if not getattr(settings, "drug_interaction_enabled", False):
        return ""
    if not meds_data:
        return ""

    from icu_pause.tools.drug_interactions import check_interactions

    result = check_interactions(
        meds_data,
        allow_network=getattr(settings, "drug_interaction_allow_network", False),
        timeout=getattr(settings, "drug_interaction_timeout_seconds", 5.0),
        reference_dttm=reference_dttm,
    )

    if not result.api_available:
        return ""
    drugs = result.checked_drug_names
    if len(drugs) < 2:
        return ""

    high_count = sum(1 for ix in result.interactions if ix.severity == "high")
    drug_list = ", ".join(drugs)
    n = len(drugs)

    lines = ["Automated medication interaction screen:"]
    lines.append(f"☐ ✓ Screened: {drug_list} ({n} active)")
    if high_count == 0:
        lines.append("☐ No high-severity interactions identified by screen")
    else:
        plural = "s" if high_count > 1 else ""
        lines.append(
            f"☐ ⚠ Screen identified {high_count} high-severity "
            f"interaction{plural} — see QA Issues"
        )
    return "\n".join(lines)


def _normalize_s_format(content: str) -> str:
    """Normalize S-section content into #Problem / [] to-do structure.

    Some models (especially GPT-4.1) output plain prose instead of the
    structured format. This deterministically prepends ``#`` to problem
    statements and ``[]`` to action items that are missing their prefix.

    Rules:
    - Lines already starting with ``#``, ``[]``, or ``## `` → unchanged
    - Lines whose first word is an action verb → prepend ``[]``
    - Lines containing a colon (problem: status pattern) → prepend ``#``
    - Other lines → left as-is (regular narrative)
    """
    lines = content.split("\n")
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue

        # Already formatted — leave alone
        if (stripped.startswith("#")
                or re.match(r'^\[[ x]?\]\s', stripped)
                or stripped.startswith("## ")):
            result.append(line)
            continue

        # Check first word for action verb
        first_word = stripped.split()[0].lower().rstrip(":")
        if first_word in _ACTION_VERBS:
            result.append(f"[] {stripped}")
            continue

        # Check for "Problem: status" pattern (colon within first ~60 chars)
        colon_pos = stripped.find(":")
        if 0 < colon_pos < 60:
            # Make sure it's not a time pattern like "08:00" or a URL
            before_colon = stripped[:colon_pos]
            if not re.match(r'^\d{1,2}$', before_colon.split()[-1] if before_colon.split() else ""):
                # Split the title from any inline body and emit them on
                # SEPARATE lines so the renderer treats the body as a body
                # line (citation-rendered), not as part of the bold header.
                title = stripped[:colon_pos].strip()
                body = stripped[colon_pos + 1:].strip()
                result.append(f"#{title}:")
                if body:
                    result.append(body)
                continue

        # Default: leave as-is
        result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Deterministic problem-list ordering by organ system
# ---------------------------------------------------------------------------

# Clinical priority order — matches how clinicians scan a problem list.
# This is a living config: extend keyword maps after reviewing cases.
PROBLEM_CATEGORY_ORDER = [
    "respiratory",
    "hemodynamic",
    "infection",
    "neurological",
    "renal",
    "hepatic",
    "hematologic",
    "metabolic",
    "nutritional",
    "gi",
    "musculoskeletal",
    "skin",
    "access",        # lines, drains, devices
    "disposition",
    "goals_of_care",
    "code_status",
]

# Categories that must never lead the S problem list — dispositions / reference
# info, not active problems for night-team watch. Enforced as a separate
# post-sort pass in _order_s_problems so the rule survives any future
# reordering of PROBLEM_CATEGORY_ORDER. See docs/problem_ordering_analysis.md
# (Ship 1, expert sign-off 2026-05-26).
BOTTOM_ANCHORED_CATEGORIES: set[str] = {
    "access",
    "disposition",
    "goals_of_care",
    "code_status",
}

# Keyword maps: category → list of keywords to match in #Problem header text.
# Case-insensitive substring match. Order within each list doesn't matter.
PROBLEM_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "respiratory": [
        "respiratory", "ventilat", "trach", "oxygen", "hypox", "pneumo",
        "airway", "extubat", "wean", "fio2", "hfnc", "nippv", "cpap",
        "bronch", "pleural", "chest tube", "capping", "spo2", "desaturat",
        "aspiration", "pulmonary", "ards", "copd", "asthma",
    ],
    "hemodynamic": [
        "hemodynamic", "hypoten", "hypertens", "vasopressor", "map ",
        "blood pressure", "tachycard", "bradycard", "arrhythm", "afib",
        "cardiac", "heart", "shock", "norepinephrine", "pressors",
        "fluid", "volume",
    ],
    "infection": [
        "infect", "sepsis", "antibiotic", "antimicrobial", "culture",
        "fever", "bacteremia", "fungal", "vre", "mrsa", "pseudomonas",
        "influenza", "covid", "pneumonia", "uti", "cellulitis",
        "osteomyelitis", "abscess", "stenotrop", "leukocyt", "wbc",
    ],
    "neurological": [
        "neuro", "mental status", "gcs", "delirium", "seizure", "stroke",
        "sedation", "agitat", "cam-icu", "rass", "encephalop", "spinal",
        "spasticity", "cognitive", "altered", "confusion",
    ],
    "renal": [
        "renal", "kidney", "creatinine", "dialysis", "crrt", "aki",
        "ckd", "urine", "oliguria", "anuria", "bun", "gfr",
    ],
    "hepatic": [
        "hepatic", "liver", "cirrhosis", "bilirubin", "ast", "alt",
        "coagulopathy", "hepatorenal", "encephalopath", "portal",
        "ascites", "varices",
    ],
    "hematologic": [
        "hematolog", "anemia", "thrombocytop", "coagul", "anticoagul",
        "dvt", "pe ", "pulmonary embolism", "bleeding", "transfus",
        "heparin", "warfarin", "apixaban", "enoxaparin", "vte",
        "hemoglobin", "platelet", "inr",
    ],
    "metabolic": [
        "electrolyte", "potassium", "sodium", "phosph", "magnesium",
        "calcium", "hyponatremia", "hyperkalemia", "hypophosphat",
        "hypokalemia", "acidosis", "alkalosis", "glucose", "insulin",
        "diabete", "a1c", "endocrine", "thyroid", "adrenal",
    ],
    "nutritional": [
        "nutrition", "malnutrition", "diet", "feeding", "enteral",
        "parenteral", "tpn", "albumin", "prealbumin", "npo",
        "swallow", "calori", "refeeding", "weight loss", "bmi",
    ],
    "gi": [
        "gastro", "gi ", "bowel", "ileus", "nausea", "vomit",
        "diarrhea", "constipat", "abdomin", "pancreat", "colitis",
        "obstruct", "gi bleed", "melena", "hematochezi",
    ],
    "musculoskeletal": [
        "mobility", "weakness", "physical therapy", "occupational",
        "ambulat", "deconditioning", "rehab", "fall", "fracture",
        "contracture", "quadripar", "parapar", "functional",
        "bed mobility", "transfer", "gait",
    ],
    "skin": [
        "wound", "skin", "pressure", "ulcer", "decubitus", "sacral",
        "heel", "braden", "incision", "surgical site", "drain",
    ],
    "access": [
        "central line", "picc", "arterial line", "foley", "catheter",
        "chest tube", "trach tube", "g-tube", "j-tube", "ngt",
        "peripheral iv", "port", "midline", "device", "line removal",
    ],
    "disposition": [
        "disposition", "discharge", "placement", "snf", "ltach",
        "rehab facility", "home health", "equipment", "insurance",
        "case manag", "social work", "family meeting", "transport",
        "dme", "prior auth",
    ],
    # code_status is intentionally listed BEFORE goals_of_care: first-match-wins
    # routes "DNR newly placed today" / "Comfort care discussion" to code_status,
    # while "Goals of care meeting scheduled" / "Hospice referral" fall through
    # to goals_of_care. Both are bottom-anchored, so the practical ordering is
    # identical — the split is for audit-trail clarity (expert sign-off
    # 2026-05-26, docs/problem_ordering_analysis.md Item 3).
    "code_status": [
        "code status", "dnr", "dni", "dnacpr", "comfort care",
        "allow natural death", "limited resuscitation", "molst", "polst",
    ],
    "goals_of_care": [
        "goals of care", "palliative", "hospice", "advance directive",
        "dpoa", "proxy", "family discuss", "prognos", "comfort measures",
        "comfort-focused",
    ],
}

# Precompute category index for fast lookup
_CATEGORY_INDEX = {cat: i for i, cat in enumerate(PROBLEM_CATEGORY_ORDER)}
_OTHER_INDEX = len(PROBLEM_CATEGORY_ORDER)  # "other" sorts last


def _classify_problem_category(header: str) -> str:
    """Classify a #Problem header into an organ-system category.

    Returns the category name (key in PROBLEM_CATEGORY_KEYWORDS) or "other".
    """
    header_lower = header.lower()
    for category, keywords in PROBLEM_CATEGORY_KEYWORDS.items():
        if any(kw in header_lower for kw in keywords):
            return category
    return "other"


def _order_s_problems(s_content: str) -> str:
    """Reorder #Problem blocks in S by clinical-priority organ system.

    Parses S into problem blocks (each starting with # and including
    subsequent non-# lines), classifies each by organ system, then
    reassembles in PROBLEM_CATEGORY_ORDER. Within a category, input
    order is preserved. Unmatched problems sort to the end.

    Lines before any # header (preamble) are preserved at the top.
    """
    lines = s_content.split("\n")
    preamble: list[str] = []
    blocks: list[tuple[str, list[str]]] = []  # (header_line, [body_lines])
    current_header = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            if current_header:
                blocks.append((current_header, current_lines))
            current_header = line
            current_lines = []
        elif current_header:
            current_lines.append(line)
        else:
            preamble.append(line)

    if current_header:
        blocks.append((current_header, current_lines))

    if len(blocks) <= 1:
        return s_content  # nothing to reorder

    # Classify and sort by (category priority, original insert order)
    categorized: list[tuple[int, int, str, str, list[str]]] = []
    for insert_order, (header, body) in enumerate(blocks):
        category = _classify_problem_category(header)
        if category == "other":
            logger.info(f"Problem category 'other': {header.strip()}")
        priority = _CATEGORY_INDEX.get(category, _OTHER_INDEX)
        categorized.append((priority, insert_order, category, header, body))

    categorized.sort(key=lambda x: (x[0], x[1]))

    # Bottom-anchor pass: move bottom-anchored categories to the tail,
    # preserving relative order within each partition. Survives any future
    # reordering of PROBLEM_CATEGORY_ORDER and prevents goals-of-care /
    # disposition / code-status / access from ever leading the list.
    non_anchored = [e for e in categorized if e[2] not in BOTTOM_ANCHORED_CATEGORIES]
    anchored = [e for e in categorized if e[2] in BOTTOM_ANCHORED_CATEGORIES]
    final = non_anchored + anchored

    # Reassemble
    result = list(preamble)
    for _, _, _, header, body in final:
        result.append(header)
        result.extend(body)

    return "\n".join(result).rstrip()


def _deduplicate_s_problems(s_content: str) -> str:
    """Remove semantically duplicate problem blocks within each domain group.

    The S section is organized by ``## Domain`` headers (e.g., ## Nursing,
    ## Respiratory). Deduplication runs *within* each group only — a nurse
    problem and a pharmacy problem that use similar keywords represent
    different clinical perspectives and must both be kept.

    Uses bag-of-words cosine similarity (threshold 0.7) to detect when the
    same clinical finding appears multiple times within one domain.
    """
    import math
    import re

    def _bow(text: str) -> dict[str, int]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return freq

    def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _dedup_blocks(blocks: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
        """Deduplicate a list of (header, [todo_lines]) blocks."""
        if len(blocks) <= 1:
            return blocks
        block_texts = [h + "\n" + "\n".join(tl) for h, tl in blocks]
        bows = [_bow(t) for t in block_texts]
        keep: list[int] = []
        for i, bow_i in enumerate(bows):
            is_dup = False
            for j in keep:
                if _cosine(bow_i, bows[j]) > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(i)
        return [blocks[i] for i in keep]

    # Parse into domain groups, each containing problem blocks
    lines = s_content.split("\n")
    output_parts: list[str] = []
    current_domain_header: str = ""
    current_blocks: list[tuple[str, list[str]]] = []
    current_problem_header: str = ""
    current_problem_lines: list[str] = []

    def _flush_domain():
        """Flush the accumulated domain group with deduplication."""
        nonlocal current_domain_header, current_blocks, current_problem_header, current_problem_lines
        if current_problem_header:
            current_blocks.append((current_problem_header, current_problem_lines))
        if current_domain_header or current_blocks:
            if current_domain_header:
                output_parts.append(current_domain_header)
            for header, todo_lines in _dedup_blocks(current_blocks):
                output_parts.append(header)
                output_parts.extend(todo_lines)
            output_parts.append("")  # blank line between groups
        current_blocks = []
        current_problem_header = ""
        current_problem_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            # New domain group — flush previous
            _flush_domain()
            current_domain_header = line
        elif stripped.startswith("#") and not stripped.startswith("##"):
            # New problem block within current domain
            if current_problem_header:
                current_blocks.append((current_problem_header, current_problem_lines))
            current_problem_header = line
            current_problem_lines = []
        elif current_problem_header:
            current_problem_lines.append(line)
        elif stripped:
            # Line before any # header — pass through
            output_parts.append(line)

    # Flush last domain
    _flush_domain()

    return "\n".join(output_parts).rstrip()


# ---------------------------------------------------------------------------
# Renal compute — Path A / Path B KDIGO with v3.1 gating fix
# ---------------------------------------------------------------------------
# See docs/renal_electrolyte_vte_extraction_design.md §4.3.5 + v3.1 §12.
# The v3.1 fix gates the `latest_cr ≥ 4.0` Stage 3 clause on FIRST meeting
# an AKI-definition criterion (ratio ≥ 1.5 OR — Path B only — absolute
# rise ≥ 0.3 mg/dL in 48h). Without this gating, every chronic ESRD
# patient with stable Cr ≥ 4.0 would falsely stage as KDIGO 3 AKI
# regardless of ratio. The defensive top-of-compute skip captures the
# even more extreme case (chart_baseline ≥ 4.0 AND no acute rise at
# all → KDIGO not applicable, INFO signal only).


_BASELINE_NUMERIC_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?"
)


def _parse_baseline_creatinine(s: Optional[str]) -> Optional[float]:
    """Parse the scribe's verbatim baseline_creatinine string into a
    single numeric mg/dL value.

    Handles:
    - Bare number: "1.4" → 1.4
    - Decorated number: "1.4 (per OSH records)" → 1.4
    - Range: "1.3-1.5" or "1.5-1.7 baseline range" → midpoint (1.4 / 1.6)

    Returns None when the string doesn't match a numeric pattern (e.g.,
    when the scribe emitted a sentinel that escaped validator filtering,
    or when the value is descriptive-only). Downstream code treats None
    as "Path A not evaluable" without WARN — silence is correct on
    indeterminate parsing.
    """
    if not s:
        return None
    m = _BASELINE_NUMERIC_RE.match(s)
    if not m:
        return None
    low = float(m.group(1))
    high = float(m.group(2)) if m.group(2) else None
    if high is not None:
        return round((low + high) / 2, 2)
    return low


_KDIGO_STAGE_RE = re.compile(r"stage\s*([1-3])", re.IGNORECASE)


def _parse_kdigo_stage_number(s: Optional[str]) -> Optional[int]:
    """Parse a chart-documented KDIGO stage string into an int 1-3.

    Handles: "KDIGO Stage 2", "AKI Stage 3 per nephrology",
    "Stage 1 AKI", "stage 2".
    Returns None when no stage number can be extracted.
    """
    if not s:
        return None
    m = _KDIGO_STAGE_RE.search(s)
    if not m:
        return None
    return int(m.group(1))


# Both ratio and absolute-change comparisons are rounded to clinical
# precision before bracket checks to defeat IEEE-754 representation
# traps. Examples that misclassify without rounding:
#   ratio: baseline 1.6 / current 2.4 → 1.4999...8 (true 1.5 exactly,
#          silently fails Stage 1 lower bound)
#   ratio: baseline 1.1 / current 3.3 → 2.9999...6 (true 3.0 exactly,
#          silently drops Stage 3 to Stage 2)
#   abs:   4.0 - 3.7 = 0.2999...8 (true 0.3 exactly, silently fails
#          Path B short-window absolute criterion)
# Cr is reported to 2 decimal places clinically; ratios to ~3
# significant figures. round(ratio, 3) covers all boundary cases at
# 1.5, 2.0, 3.0 with margin.
_RATIO_PRECISION = 3
_ABS_PRECISION = 2


def _classify_path_a_stage(
    latest_cr: float, chart_baseline: float,
) -> Optional[int]:
    """Path A KDIGO stage from long-window ratio.

    v3.1 gating: the `latest_cr ≥ 4.0` Stage 3 clause is gated on
    ratio ≥ 1.5 (the AKI definition). Without this gate, chronic ESRD
    patients with stable Cr 6.0 + chart-baseline 6.0 would stage as
    Stage 3 falsely.

    Ratio is rounded to 3 decimal places before bracket checks — see
    _RATIO_PRECISION rationale above.
    """
    ratio = round(latest_cr / chart_baseline, _RATIO_PRECISION)
    if ratio >= 3.0 or (latest_cr >= 4.0 and ratio >= 1.5):
        return 3
    if 2.0 <= ratio < 3.0:
        return 2
    if 1.5 <= ratio < 2.0:
        return 1
    return None


def _classify_path_b_stage(
    latest_cr: float, structured_48h_min: float,
) -> Optional[int]:
    """Path B KDIGO stage from 48h short-window.

    v3.1 gating: same as Path A, plus the absolute-change criterion
    (≥ 0.3 mg/dL in 48h) qualifies as an AKI-definition criterion for
    the `latest_cr ≥ 4.0` clause.

    Ratio rounded to 3 decimal places and absolute rise rounded to 2
    decimal places before bracket checks — see _RATIO_PRECISION /
    _ABS_PRECISION rationale above.
    """
    if structured_48h_min <= 0:
        return None
    ratio = round(latest_cr / structured_48h_min, _RATIO_PRECISION)
    abs_rise = round(latest_cr - structured_48h_min, _ABS_PRECISION)
    if ratio >= 3.0 or (
        latest_cr >= 4.0 and (ratio >= 1.5 or abs_rise >= 0.3)
    ):
        return 3
    if 2.0 <= ratio < 3.0:
        return 2
    if 1.5 <= ratio < 2.0 or abs_rise >= 0.3:
        return 1
    return None


_PATH_A_SKIP_TEXT = (
    "Path A not applicable: chart_baseline ≥ 4.0 mg/dL "
    "(chronic baseline); KDIGO staging requires acute-rise evidence "
    "not present (ratio < 1.5)."
)
_PATH_B_SKIP_TEXT = (
    "Path B not applicable: in-window min Cr ≥ 4.0 mg/dL "
    "(chronic baseline); KDIGO staging requires acute-rise evidence "
    "not present (ratio < 1.5 AND <0.3 mg/dL rise in 48h)."
)
_CHRONIC_BASELINE_QUALIFIER = (
    "Chronic baseline elevation; current Cr {latest_cr} at/near "
    "baseline. KDIGO staging not applicable without acute-rise "
    "evidence."
)


def compute_renal_delta_and_kdigo_path_a_b(
    scribe_renal_context: Optional[dict],
    latest_cr: Optional[float],
    structured_48h_min: Optional[float],
    chart_documented_kdigo_stage: Optional[str] = None,
) -> dict:
    """Compute delta + Path A + Path B KDIGO stages with v3.1 gating.

    Pure function — no state mutation, no side effects. Returns a
    dict with:

    - ``rendered_block`` (str): text block to feed back into the
      intensivist's input alongside the scribe RENAL CONTEXT pin.
      Contains the seven-component render described in design doc
      §4.3.6 (steps 3-4 of the ordered render — current Cr, delta,
      KDIGO paths, chronic-baseline qualifier when applicable).
    - ``delta_pct`` (Optional[int]), ``delta_abs_mg_dl`` (Optional[float])
    - ``path_a_stage``, ``path_b_stage`` (Optional[int])
    - ``computed_kdigo_stage`` (Optional[str]): the max-severity stage
      across Path A + Path B, rendered as "KDIGO Stage N".
    - ``info_signals`` (dict): includes ``KDIGO_NOT_APPLICABLE_CHRONIC_BASELINE``
      which fires when chart_baseline ≥ 4.0 AND ratio < 1.5 AND no
      chart-documented stage is provided (the chronic-baseline case).
    - ``warns`` (dict): includes ``KDIGO_STAGE_DISAGREEMENT`` which
      fires when ≥ 2 of {chart-documented, Path A, Path B} produced
      stages and they disagree on any pairwise comparison.

    Indeterminate cases (latest_cr None, scribe_renal_context None,
    baseline unparseable, etc.) return empty/None values silently —
    no WARN or INFO on indeterminate. The post-render check in
    ``_check_aki_problem_has_baseline`` is the canary that catches a
    rendered AKI #Problem missing baseline anchoring.
    """
    info_signals = {"KDIGO_NOT_APPLICABLE_CHRONIC_BASELINE": False}
    warns = {"KDIGO_STAGE_DISAGREEMENT": False}
    result: dict = {
        "rendered_block": "",
        "delta_pct": None,
        "delta_abs_mg_dl": None,
        "computed_kdigo_stage": None,
        "path_a_stage": None,
        "path_b_stage": None,
        "path_a_render": None,
        "path_b_render": None,
        "info_signals": info_signals,
        "warns": warns,
    }

    chart_baseline_str = (scribe_renal_context or {}).get("baseline_creatinine")
    chart_baseline = _parse_baseline_creatinine(chart_baseline_str)

    # Delta — computed when both numeric, regardless of whether Path A
    # or Path B produces a stage. Receiver value is independent.
    if chart_baseline is not None and latest_cr is not None:
        delta_abs = round(latest_cr - chart_baseline, 2)
        # Round-toward-positive guard: avoid -0.00 quirks on near-zero
        # deltas by snapping to 0.0 when within rounding epsilon.
        if abs(delta_abs) < 0.005:
            delta_abs = 0.0
        denom = chart_baseline if chart_baseline > 0 else None
        if denom:
            delta_pct = round(((latest_cr - chart_baseline) / denom) * 100)
        else:
            delta_pct = None
        result["delta_abs_mg_dl"] = delta_abs
        result["delta_pct"] = delta_pct

    # Path A — long-window ratio against chart-extracted baseline.
    # Both ratio and absolute comparisons use the rounded values per
    # _RATIO_PRECISION / _ABS_PRECISION constants — IEEE-754 traps at
    # the bracket boundaries (1.5, 2.0, 3.0, and the 0.3 abs threshold)
    # would otherwise silently mis-classify real clinical cases.
    path_a_skip = False
    if chart_baseline is not None and latest_cr is not None:
        path_a_ratio = (
            round(latest_cr / chart_baseline, _RATIO_PRECISION)
            if chart_baseline > 0 else None
        )
        if path_a_ratio is not None:
            if chart_baseline >= 4.0 and path_a_ratio < 1.5:
                result["path_a_render"] = _PATH_A_SKIP_TEXT
                path_a_skip = True
            else:
                result["path_a_stage"] = _classify_path_a_stage(
                    latest_cr, chart_baseline,
                )

    # Path B — 48h short-window against structured min.
    path_b_skip = False
    if structured_48h_min is not None and latest_cr is not None:
        path_b_ratio = (
            round(latest_cr / structured_48h_min, _RATIO_PRECISION)
            if structured_48h_min > 0 else None
        )
        path_b_abs = round(latest_cr - structured_48h_min, _ABS_PRECISION)
        if path_b_ratio is not None:
            if (
                structured_48h_min >= 4.0
                and path_b_ratio < 1.5
                and path_b_abs < 0.3
            ):
                result["path_b_render"] = _PATH_B_SKIP_TEXT
                path_b_skip = True
            else:
                result["path_b_stage"] = _classify_path_b_stage(
                    latest_cr, structured_48h_min,
                )

    # Final computed stage = max severity across paths.
    stages_produced = [
        s for s in (result["path_a_stage"], result["path_b_stage"])
        if s is not None
    ]
    if stages_produced:
        result["computed_kdigo_stage"] = f"KDIGO Stage {max(stages_produced)}"

    # INFO signal — chronic-baseline case. Fires when EVERY EVALUABLE
    # path skipped (chronic baseline) AND no chart-documented stage.
    chart_documented_number = _parse_kdigo_stage_number(
        chart_documented_kdigo_stage
    )
    chart_kdigo_none = chart_documented_number is None
    paths_evaluable = []
    paths_skipped = []
    if chart_baseline is not None and latest_cr is not None:
        paths_evaluable.append("A")
        if path_a_skip:
            paths_skipped.append("A")
    if structured_48h_min is not None and latest_cr is not None:
        paths_evaluable.append("B")
        if path_b_skip:
            paths_skipped.append("B")
    # Fire when at least one path was evaluable AND every evaluable
    # path skipped AND chart-documented is None. (A patient with NO
    # baseline data at all should not trigger this — that's a
    # "baseline missing" condition handled by a different post-render
    # check.)
    if (
        paths_evaluable
        and len(paths_skipped) == len(paths_evaluable)
        and chart_kdigo_none
    ):
        info_signals["KDIGO_NOT_APPLICABLE_CHRONIC_BASELINE"] = True

    # WARN — three-way disagreement (extends to all pairwise per v3.1
    # §4.3.5). Fires when ≥ 2 paths produced stages AND those stages
    # disagree.
    available_stages: dict[str, int] = {}
    if chart_documented_number is not None:
        available_stages["chart-documented"] = chart_documented_number
    if result["path_a_stage"] is not None:
        available_stages["Path A"] = result["path_a_stage"]
    if result["path_b_stage"] is not None:
        available_stages["Path B"] = result["path_b_stage"]
    if len(available_stages) >= 2:
        unique_vals = set(available_stages.values())
        if len(unique_vals) > 1:
            warns["KDIGO_STAGE_DISAGREEMENT"] = True

    # Render the block. The intensivist's RENAL CONTEXT pin prompt rule
    # (intensivist.yaml §S synthesis) expects this exact text in the
    # input — see design doc §4.3.6 ordered render steps 3-4.
    result["rendered_block"] = _render_kdigo_compute_block(
        scribe_renal_context=scribe_renal_context,
        latest_cr=latest_cr,
        structured_48h_min=structured_48h_min,
        chart_baseline=chart_baseline,
        chart_documented_kdigo_stage=chart_documented_kdigo_stage,
        available_stages=available_stages,
        result=result,
        info_signals=info_signals,
    )

    return result


def _render_kdigo_compute_block(
    *,
    scribe_renal_context: Optional[dict],
    latest_cr: Optional[float],
    structured_48h_min: Optional[float],
    chart_baseline: Optional[float],
    chart_documented_kdigo_stage: Optional[str],
    available_stages: dict[str, int],
    result: dict,
    info_signals: dict,
) -> str:
    """Render the compute output as a text block for the intensivist's
    input. Goes alongside the scribe RENAL CONTEXT pin and supplies
    steps 3-4 of the §4.3.6 ordered render."""
    parts = ["## RENAL COMPUTE (orchestrator — deterministic Path A / "
             "Path B KDIGO per v3.1 §4.3.5)\n"]
    parts.append("```")

    # Current Cr line
    if latest_cr is not None:
        parts.append(f"Current Cr: {latest_cr:.2f} mg/dL")
    else:
        parts.append("Current Cr: not available in structured data")

    # Delta line
    if result["delta_pct"] is not None and result["delta_abs_mg_dl"] is not None:
        delta_sign = "+" if result["delta_abs_mg_dl"] >= 0 else ""
        pct_sign = "+" if result["delta_pct"] >= 0 else ""
        parts.append(
            f"Delta: {delta_sign}{result['delta_abs_mg_dl']:.2f} mg/dL "
            f"({pct_sign}{result['delta_pct']}% vs chart baseline)"
        )
    elif chart_baseline is None:
        parts.append("Delta: not computable — no baseline anchor")
    else:
        parts.append("Delta: not computable — latest Cr unavailable")

    # KDIGO line(s)
    if info_signals["KDIGO_NOT_APPLICABLE_CHRONIC_BASELINE"]:
        # Chronic-baseline qualifier — render verbatim. Intensivist
        # yaml rule (RENAL CONTEXT PIN section) keys off this exact
        # phrasing to bypass the KDIGO render branch.
        if latest_cr is not None:
            parts.append(
                _CHRONIC_BASELINE_QUALIFIER.format(latest_cr=f"{latest_cr:.2f}")
            )
        else:
            parts.append(
                "Chronic baseline elevation; current Cr unavailable. "
                "KDIGO staging not applicable without acute-rise evidence."
            )
        # Surface the per-path skip text so the audit trail is in the
        # prompt input, not just internal state.
        if result["path_a_render"]:
            parts.append(result["path_a_render"])
        if result["path_b_render"]:
            parts.append(result["path_b_render"])
    elif len(available_stages) >= 2 and len(set(available_stages.values())) > 1:
        # Disagreement render — name every available source explicitly
        # with its stage. Append "(Cr-based; UOP not computed)" suffix
        # to flag that UOP-based KDIGO is deferred (R11 per §11.6).
        stage_renders = []
        for label, stage in available_stages.items():
            if label == "Path A":
                stage_renders.append(
                    f"Path A (long-window ratio) Stage {stage}"
                )
            elif label == "Path B":
                stage_renders.append(
                    f"Path B (48h short-window) Stage {stage}"
                )
            else:  # chart-documented
                stage_renders.append(f"chart-documented Stage {stage}")
        parts.append(
            f"KDIGO: {' / '.join(stage_renders)} (Cr-based; UOP not "
            "computed) — discrepancy flagged for chart review "
            "(KDIGO_STAGE_DISAGREEMENT)."
        )
    elif available_stages:
        # Single-source or fully-agreeing stages render
        stage_renders = []
        for label, stage in available_stages.items():
            if label == "Path A":
                stage_renders.append(
                    f"Path A (long-window ratio) Stage {stage}"
                )
            elif label == "Path B":
                stage_renders.append(
                    f"Path B (48h short-window) Stage {stage}"
                )
            else:
                stage_renders.append(f"chart-documented Stage {stage}")
        parts.append(
            f"KDIGO: {'; '.join(stage_renders)} (Cr-based; UOP not "
            "computed)."
        )
    else:
        # No path produced a stage AND not the chronic-baseline case
        not_evaluable_reasons = []
        if chart_baseline is None:
            not_evaluable_reasons.append(
                "Path A: no chart-extracted baseline (chart review "
                "required for KDIGO interpretation)"
            )
        elif result["path_a_stage"] is None and result["path_a_render"] is None:
            not_evaluable_reasons.append(
                "Path A: ratio < 1.5 (no AKI by ratio)"
            )
        if structured_48h_min is None:
            not_evaluable_reasons.append(
                "Path B: not evaluable — only 1 Cr in past 48h"
            )
        elif result["path_b_stage"] is None and result["path_b_render"] is None:
            not_evaluable_reasons.append(
                "Path B: ratio < 1.5 AND <0.3 mg/dL rise in 48h "
                "(no AKI by short-window criteria)"
            )
        if chart_documented_kdigo_stage:
            parts.append(
                f"KDIGO (chart-documented): {chart_documented_kdigo_stage}. "
                + ("Computed paths: " + "; ".join(not_evaluable_reasons)
                   if not_evaluable_reasons else "")
            )
        elif not_evaluable_reasons:
            parts.append("KDIGO: " + "; ".join(not_evaluable_reasons))
        else:
            parts.append(
                "KDIGO: not evaluable — insufficient data for "
                "computed staging (chart review required)"
            )

    parts.append("```")

    # Trailing instruction for the intensivist
    parts.append("")
    parts.append(
        "USE THE Current Cr + Delta + KDIGO lines AS-IS as components "
        "3-4 of the AKI/CKD #Problem render in Section S (per RENAL "
        "CONTEXT PIN rule in your system prompt). The scribe RENAL "
        "CONTEXT pin above supplies components 2, 5, 6, 7."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Nephrotoxin / renal med-review dedup (R6 §5.1) + electrolyte canary
# ---------------------------------------------------------------------------
# See docs/renal_electrolyte_vte_extraction_design.md §5.1 + §5.2.
# v3 replaced v2's BOW cosine ≥ 0.5 with a curated phrase list + cosine
# fallback ≥ 0.75 (reviewer §2a). The phrase list is the primary
# detector; cosine fallback catches near-paraphrases the list missed.
# The "no renal anchor present" branch (algorithm step 5) is the
# pre-flight item the reviewer called out — the to-do gets PROMOTED to
# the to-do checklist with a renal-context note rather than silently
# dropping. Easy to miss because the common case is "renal block exists,
# drop the duplicate."


_NEPHROTOXIN_REVIEW_PHRASES = frozenset({
    # Primary phrasings
    "review medications for renal impairment",
    "review medications for renal function",
    "review meds for renal",
    "review home meds for renal dosing",
    "review home medications for renal",
    "dose-adjust nephrotoxins",
    "dose adjust nephrotoxins",
    "adjust meds for renal function",
    "adjust meds for kidney function",
    "adjust medications for renal dose",
    "renal dose adjustment",
    "renal dose adjust",
    "review renally-cleared medications",
    "review renally cleared medications",
    "reconcile renally-cleared drugs",
    "renal medication reconciliation",
    "renal med reconciliation",
    # Consult / specialty phrasings
    "renal pharmacy consult",
    "pharmacy consult for renal dosing",
    # Holds / discontinuations
    "hold nephrotoxic agents",
    "hold nephrotoxins",
    "discontinue nephrotoxins",
    "stop nephrotoxic medications",
    "avoid nephrotoxic agents",
    "avoid nephrotoxins",
    # Specific-agent variants common in source data
    "review nsaids for renal",
    "hold nsaids",
    "review contrast for renal",
    "hold acei/arb",
    "hold acei",
    "hold arb",
})


# Renal #Problem header keywords — reuses the same set the §classifier
# uses for "renal" category problem matching. Headers containing any of
# these tokens identify the AKI/CKD/Renal block for dedup ownership.
_RENAL_HEADER_KEYWORDS = (
    "aki", "acute kidney", "ckd", "renal", "nephropathy", "kidney",
    "creatinine", "renal failure", "renal injury", "renal impairment",
)


_NEPHROTOXIN_COSINE_THRESHOLD = 0.75
# Renal-anchor tokens — the cosine fallback path requires at least one
# of these to be present in the candidate line. Without this guard, the
# cosine fallback over-fires on generic "review home medications" lines
# that share token-overlap with curated phrases like "review home meds
# for renal dosing" but have NO renal semantic content. The token guard
# matches the clinical intent — a nephrotoxin / renal-dose-review action
# MUST anchor on a renal token.
_RENAL_ANCHOR_TOKENS = frozenset({
    "renal", "renally", "nephrotoxic", "nephrotoxins", "nephrotoxin",
    "kidney", "kidneys", "gfr", "crcl",
})
_PROMOTED_TODO_NO_RENAL_ANCHOR = (
    "[]Review medications against current renal function "
    "(nephrotoxin avoidance + dose-adjusted dosing) — no AKI/CKD "
    "problem in S; receiver should evaluate renal status on arrival"
)


def _bow_for_phrase(text: str) -> dict[str, int]:
    """Bag-of-words vector for cosine fallback. Lowercase, word tokens."""
    freq: dict[str, int] = {}
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        freq[w] = freq.get(w, 0) + 1
    return freq


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    """Cosine similarity between two bag-of-words vectors."""
    import math
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# Pre-compute bow vectors for phrase list — used by cosine fallback.
_NEPHROTOXIN_PHRASE_BOWS = tuple(
    _bow_for_phrase(p) for p in _NEPHROTOXIN_REVIEW_PHRASES
)


def _is_nephrotoxin_review_line(line: str) -> bool:
    """Return True if the given to-do line matches any curated
    nephrotoxin-review phrase OR cosine ≥ 0.75 against the phrase
    list (paraphrase fallback)."""
    # Normalize: strip the `[]` to-do prefix and surrounding whitespace,
    # lowercase. Drop punctuation EXCEPT inside multi-char phrases like
    # "acei/arb" — the phrase list itself contains "/", so we keep it
    # to match correctly.
    stripped = re.sub(r"^\s*\[[ x]?\]\s*", "", line).strip().lower()
    if not stripped:
        return False

    # Primary substring match against curated phrases
    for phrase in _NEPHROTOXIN_REVIEW_PHRASES:
        if phrase in stripped:
            return True

    # Cosine fallback — only when primary match failed AND the line
    # contains at least one renal-anchor token. The token guard prevents
    # generic-med-review lines from falsely matching (the v1 iter-1
    # finding: 0.75 alone over-fires on "review home medications" which
    # shares 3/5 tokens with "review home medications for renal").
    line_bow = _bow_for_phrase(stripped)
    if not line_bow:
        return False
    if not (set(line_bow) & _RENAL_ANCHOR_TOKENS):
        return False
    for phrase_bow in _NEPHROTOXIN_PHRASE_BOWS:
        if _cosine(line_bow, phrase_bow) >= _NEPHROTOXIN_COSINE_THRESHOLD:
            return True
    return False


def _is_renal_header(header: str) -> bool:
    """Return True if a #Problem header identifies an AKI/CKD/Renal
    problem (the etiologic owner of the nephrotoxin med-review action).
    """
    h = header.lower()
    return any(kw in h for kw in _RENAL_HEADER_KEYWORDS)


def _dedup_nephrotoxin_med_review_across_problems(
    s_content: str,
) -> tuple[str, list[str], list[str]]:
    """Dedup nephrotoxin / renal-dose-review to-dos across S problems.

    Algorithm (per design doc §5.1):
    1. Parse S into (header, [body_lines]) problem blocks.
    2. Identify renal blocks via _is_renal_header.
    3. For each non-renal block: scan to-do lines against the curated
       phrase list + cosine ≥ 0.75 fallback.
    4. If a renal block exists in S: drop nephrotoxin-review lines from
       non-renal blocks (the renal block is the etiologic owner). If
       multiple nephrotoxin-review lines exist within the renal block
       itself, keep the first and drop subsequent duplicates.
    5. If NO renal block exists in S (the "no renal anchor present"
       branch — reviewer's pre-flight #2): drop the lines from
       non-renal blocks AND promote a canonical renal-review action
       to a returned ``promoted_todos`` list. Caller is responsible
       for adding this to the to-do checklist.

    Returns ``(deduped_s_content, dedup_actions_log, promoted_todos)``.
    """
    # Parse S into preamble + per-problem blocks (preserve original
    # header lines so we can rebuild without re-rendering them).
    lines = s_content.split("\n")
    preamble: list[str] = []
    blocks: list[tuple[str, list[str]]] = []  # (header, body)
    current_header: Optional[str] = None
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            if current_header is not None:
                blocks.append((current_header, current_body))
            current_header = line
            current_body = []
        elif current_header is not None:
            current_body.append(line)
        else:
            preamble.append(line)
    if current_header is not None:
        blocks.append((current_header, current_body))

    # Identify renal block(s). If multiple, the first is canonical
    # owner; later renal blocks dedup against the first.
    renal_indices = [
        i for i, (h, _) in enumerate(blocks) if _is_renal_header(h)
    ]
    has_renal_block = bool(renal_indices)

    dedup_actions: list[str] = []
    promoted_todos: list[str] = []

    # Track whether we've kept the FIRST nephrotoxin-review line in
    # the canonical renal block (so within-block dedup leaves exactly
    # one). When there's no renal block, this flag stays False and
    # any non-renal hit triggers the promoted-todo branch (once,
    # since multiple promotion entries would just clutter the
    # checklist).
    renal_owner_index = renal_indices[0] if has_renal_block else None
    renal_owner_has_one = False
    promoted_already = False

    for i, (header, body) in enumerate(blocks):
        kept_body: list[str] = []
        for body_line in body:
            if not _is_nephrotoxin_review_line(body_line):
                kept_body.append(body_line)
                continue
            # This line IS a nephrotoxin-review to-do.
            if i == renal_owner_index:
                # Inside the canonical renal block — keep the first,
                # drop subsequent.
                if renal_owner_has_one:
                    dedup_actions.append(
                        f"DEDUP: nephrotoxin med-review dropped from "
                        f"{header.strip()} (within-block duplicate)"
                    )
                else:
                    kept_body.append(body_line)
                    renal_owner_has_one = True
                continue
            # This is a non-renal block with a nephrotoxin-review hit.
            if has_renal_block:
                dedup_actions.append(
                    f"DEDUP: nephrotoxin med-review dropped from "
                    f"{header.strip()} — already present under "
                    f"{blocks[renal_owner_index][0].strip()}"
                )
            else:
                # No renal anchor in S — promote to the to-do checklist
                # instead of silently dropping. Promote at most once;
                # multiple identical promotions would just clutter.
                dedup_actions.append(
                    f"DEDUP: nephrotoxin med-review dropped from "
                    f"{header.strip()} — no renal anchor present; "
                    f"promoting renal-review action to to-do checklist"
                )
                if not promoted_already:
                    promoted_todos.append(_PROMOTED_TODO_NO_RENAL_ANCHOR)
                    promoted_already = True
            # Either way: drop this line from the block.
        blocks[i] = (header, kept_body)

    # Rebuild S content
    out_parts: list[str] = []
    if preamble:
        out_parts.extend(preamble)
    for header, body in blocks:
        out_parts.append(header)
        out_parts.extend(body)

    return "\n".join(out_parts), dedup_actions, promoted_todos


# Electrolyte atomicity canary (R6 §5.2) — fires WARN when a single
# #Problem header contains BOTH hyponatremia AND hyperkalemia tokens
# without an AKI/renal-mediated attribution. The canary does NOT
# auto-split — that's prompt territory; this is the regression
# detector.

_HYPONA_TOKENS = ("hyponatremia", "hypona", "hyponatrem")
_HYPERK_TOKENS = ("hyperkalemia", "hyperk", "hyperkalem")
_ATTRIBUTION_TOKENS = (
    "aki", "renal-mediated", "renal mediated", "renal", "ckd", "kidney",
)


def _warn_on_merged_electrolyte_problems(
    s_content: str,
) -> list[str]:
    """Scan S problem headers for unattributed hyponatremia +
    hyperkalemia merges. Returns a list of WARN strings (one per
    offending header); the caller routes each to the qa_process log
    as ELECTROLYTE_PROBLEM_MERGED_WITHOUT_ATTRIBUTION.
    """
    warns: list[str] = []
    for line in s_content.split("\n"):
        stripped = line.strip()
        if not (
            stripped.startswith("#") and not stripped.startswith("##")
        ):
            continue
        header_lower = stripped.lower()
        has_hypona = any(t in header_lower for t in _HYPONA_TOKENS)
        has_hyperk = any(t in header_lower for t in _HYPERK_TOKENS)
        if not (has_hypona and has_hyperk):
            continue
        has_attribution = any(
            t in header_lower for t in _ATTRIBUTION_TOKENS
        )
        if not has_attribution:
            warns.append(
                f"ELECTROLYTE_PROBLEM_MERGED_WITHOUT_ATTRIBUTION: "
                f"{stripped} — merged electrolyte problems require "
                f"explicit AKI/renal etiology attribution per §5.2"
            )
    return warns


# AKI baseline reference post-render check (§4.3.7)
# Fires WARN when any AKI/CKD/Renal #Problem is rendered AND the body
# is missing baseline-Cr reference text. The check accepts either the
# pinned baseline_creatinine value verbatim OR the explicit fallback
# sentinel ("no baseline anchor in source"). Without either, the
# receiver cannot interpret the current Cr against the patient's
# baseline.

_BASELINE_FALLBACK_SENTINELS = (
    "no baseline anchor in source",
    "no baseline anchor extracted",
    "baseline not documented",
    "chart review required for interpretation",
)


def _check_aki_problem_has_baseline(
    s_content: str,
    pinned_baseline_value: Optional[str] = None,
) -> list[str]:
    """Scan S for any AKI/CKD/Renal #Problem and verify each renders
    baseline-Cr context.

    ``pinned_baseline_value`` is the chart-extracted baseline string
    from scribe (e.g., "1.4" or "1.3-1.5"). When provided, the check
    accepts that exact value appearing in the problem body OR any
    sentinel from _BASELINE_FALLBACK_SENTINELS.

    Returns a list of WARN strings (one per offending header). Caller
    routes each to qa_process as
    AKI_PROBLEM_MISSING_BASELINE_REFERENCE.
    """
    warns: list[str] = []
    lines = s_content.split("\n")
    current_header: Optional[str] = None
    # current_block_text accumulates the FULL block (header + body) for
    # the baseline-presence check. Inline header references like
    # "#AKI on CKD: Cr 2.4 (baseline 1.4)" are valid anchoring per
    # §4.3.6 (the receiver still sees the baseline value in the
    # rendered text), so the check looks at the full block, not
    # body-only.
    current_block_text = ""

    def _emit_for_current():
        if not current_header:
            return
        if not _is_renal_header(current_header):
            return
        block_lower = current_block_text.lower()
        # Pass if either: the pinned baseline value appears anywhere in
        # the block (header or body), OR any explicit fallback sentinel
        # appears.
        baseline_present = False
        if pinned_baseline_value:
            if pinned_baseline_value.lower() in block_lower:
                baseline_present = True
        if not baseline_present:
            for sentinel in _BASELINE_FALLBACK_SENTINELS:
                if sentinel in block_lower:
                    baseline_present = True
                    break
        if not baseline_present:
            warns.append(
                f"AKI_PROBLEM_MISSING_BASELINE_REFERENCE: "
                f"{current_header.strip()} — block missing baseline-Cr "
                f"reference (neither pinned value nor explicit "
                f"fallback sentinel present)"
            )

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("##"):
            # Flush previous block
            _emit_for_current()
            current_header = stripped
            current_block_text = stripped  # header IS part of the block
        else:
            current_block_text += " " + stripped
    # Flush final block
    _emit_for_current()
    return warns


# ---------------------------------------------------------------------------
# S section temporal bucketing — stratify to-dos by temporal ownership
# ---------------------------------------------------------------------------

_PRE_TRANSFER_SIGNALS = [
    "prior to transfer", "prior to discharge", "before transfer",
    "confirm", "verify", "update", "align", "finalize",
    "assess readiness", "check before", "document", "reconcil",
]
_WARD_ONGOING_SIGNALS = [
    "each shift", "daily", "as needed", "monitor for", "continue",
    "tid", "prn", "q4h", "q8h", "q12h", "ongoing", "reassess",
    "trend", "wean as", "titrate", "maintain", "observe",
]
_DISCHARGE_SIGNALS = [
    "outpatient", "home health", "post-discharge", "community",
    "follow-up appointment", "discharge plan", "snf", "equipment",
    "transportation", "caregiver training", "home care", "dme",
]


def _classify_todo_bucket(todo_text: str) -> str:
    """Classify a to-do item into a temporal bucket.

    Returns one of: "pre_transfer", "ward_ongoing", "discharge".
    Default: "ward_ongoing" if no signals match.
    """
    lower = todo_text.lower()

    # --- Hard overrides (checked first, before any signal scanning) ---
    # Explicit transfer-related phrases always mean pre_transfer, even if
    # "discharge" appears as a substring (e.g. "Confirm discharge summary sent").
    _HARD_PRE_TRANSFER = ["before transfer", "prior to transfer", "prior to discharge"]
    for phrase in _HARD_PRE_TRANSFER:
        if phrase in lower:
            return "pre_transfer"
    # Tasks the RECEIVING/ward team performs once the patient has arrived are
    # ward_ongoing, even when phrased with a pre-transfer verb ("Verify ... on
    # arrival", "Inspect ... after transfer"). This timing signal must win over
    # the leading-verb and pre_transfer-signal heuristics below. Keyed on
    # arrival/after-transfer *timing* only — not the bare words "receiving team",
    # which also appear in pre-transfer handoff prep ("Confirm X for the
    # receiving team").
    _AFTER_TRANSFER_SIGNALS = [
        "on arrival", "upon arrival", "after arrival",
        "after transfer", "post-transfer", "once on the ward",
        "once on the floor", "after the patient arrives",
        "when the patient arrives", "on the receiving unit",
    ]
    for phrase in _AFTER_TRANSFER_SIGNALS:
        if phrase in lower:
            return "ward_ongoing"
    # Emergency equipment / airway tasks are pre-transfer safety checks
    if "emergency" in lower and any(
        kw in lower for kw in ("equipment", "airway", "trach")
    ):
        return "pre_transfer"

    # --- Start-of-item verb matching ---
    # Logistical coordination verbs at the start indicate ICU-team pre-transfer tasks,
    # regardless of the clinical system the action relates to.
    _PRE_TRANSFER_VERBS = {
        "confirm", "coordinate", "arrange", "ensure", "verify",
        "finalize", "secure", "obtain", "reconcile", "prepare",
    }
    first_word = lower.split()[0].rstrip(":") if lower.split() else ""
    if first_word in _PRE_TRANSFER_VERBS:
        return "pre_transfer"

    # --- Signal scanning (pre_transfer and ward first, discharge last) ---
    for signal in _PRE_TRANSFER_SIGNALS:
        if signal in lower:
            return "pre_transfer"
    for signal in _WARD_ONGOING_SIGNALS:
        if signal in lower:
            return "ward_ongoing"
    for signal in _DISCHARGE_SIGNALS:
        if signal in lower:
            return "discharge"
    return "ward_ongoing"  # default bucket


# Inline temporal markers for to-do lines
_BUCKET_MARKERS = {
    "pre_transfer": "\U0001f6cf PRE-TRANSFER",  # 🛏️ PRE-TRANSFER (patient in bed)
    "ward_ongoing": "\U0001f3e5 ON WARD",       # 🏥 ON WARD (hospital)
    "discharge": "\U0001f3e0 DISCHARGE",         # 🏠 DISCHARGE (home)
}

# Plain-text markers for EMR (no emoji)
_BUCKET_MARKERS_EMR = {
    "pre_transfer": "[PRE-TRANSFER]",
    "ward_ongoing": "[ON WARD]",
    "discharge": "[DISCHARGE]",
}


def _bucket_marker(todo_text: str) -> str:
    """Return the emoji temporal marker for a to-do line."""
    bucket = _classify_todo_bucket(todo_text)
    return _BUCKET_MARKERS.get(bucket, "")


def _mark_todo_buckets(s_content: str) -> str:
    """Add temporal bucket markers to [] to-do lines in S content."""
    lines = s_content.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\[[ x]?\]\s', stripped):
            # Extract the to-do text after the checkbox
            text = re.sub(r'^\[[ x]?\]\s*', '', stripped)
            # Don't double-mark if already has a marker
            if not any(m in text for m in _BUCKET_MARKERS.values()):
                marker = _bucket_marker(text)
                result.append(f"[] {marker} {text}")
            else:
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


_TEMPORAL_BUCKET_HEADERS = {
    "before transfer (icu team):", "before transfer:",
    "on the ward (receiving team):", "on the ward:",
    "at discharge (case manager/team):", "at discharge:",
}

_TODO_LINE_PREFIXES = ("[]", "☐", "-  ")

_TEMPLATE_HEADERS_EXTRACT = [
    "to-do list", "to-do's", "todos:", "to-do:",
    "prior to transfer:", "action items:",
    "to do list", "follow up pending cultures: unknown",
]


def _extract_todos_from_s(s_content: str) -> tuple[str, list[str]]:
    """Separate to-do items from narrative content in the S section.

    Scans line-by-line and pulls out:
      - Lines starting with [], ☐, or '-  ' (indented bullet from bucketing)
      - Temporal bucket headers (BEFORE TRANSFER:, ON THE WARD:, AT DISCHARGE:)

    Returns:
        (narrative_text, extracted_todo_texts)
        where narrative_text has no checkbox or bucket-header lines,
        and extracted_todo_texts are the cleaned item strings.
    """
    narrative_lines: list[str] = []
    todos: list[str] = []

    for line in s_content.split("\n"):
        stripped = line.strip()
        if not stripped:
            narrative_lines.append(line)
            continue

        lower = stripped.lower()

        # Strip temporal bucket headers — they'll be regenerated
        if lower.rstrip(":").rstrip() + ":" in _TEMPORAL_BUCKET_HEADERS or lower in _TEMPORAL_BUCKET_HEADERS:
            continue

        # Strip orphaned section-title text (e.g., "To-do list prior to transfer:")
        if any(h in lower for h in _TEMPLATE_HEADERS_EXTRACT):
            continue

        # Detect to-do lines
        is_todo = False
        for prefix in _TODO_LINE_PREFIXES:
            if stripped.startswith(prefix):
                is_todo = True
                break

        if is_todo:
            clean = stripped.lstrip("[]- ☐").strip()
            if not clean:
                continue
            # Skip template artifact headers
            if any(h in clean.lower() for h in _TEMPLATE_HEADERS_EXTRACT):
                continue
            todos.append(clean)
        else:
            narrative_lines.append(line)

    return "\n".join(narrative_lines).rstrip(), todos


def _smart_truncate_problem(text: str, max_words: int = 50) -> str:
    """Truncate a problem description while preserving numeric values.

    If truncation would drop numeric patterns (lab values, vitals, dates),
    extends the cutoff to include them.
    """
    import re

    words = text.split()
    if len(words) <= max_words:
        return text

    # Find positions of all numeric/date patterns worth preserving
    numeric_pattern = re.compile(
        r'\d+\.?\d*'  # numbers like 4.3, 100, 0.3
        r'|'
        r'\d{1,2}/\d{1,2}(?:/\d{2,4})?'  # dates like 1/25 or 1/25/24
    )
    preserve_positions: set[int] = set()
    for i, word in enumerate(words):
        if numeric_pattern.search(word):
            preserve_positions.add(i)

    # Find the furthest numeric position beyond max_words
    max_needed = max_words
    for pos in preserve_positions:
        if pos >= max_words and pos < max_words + 15:  # look up to 15 words ahead
            max_needed = pos + 1

    truncated = " ".join(words[:max_needed])
    if max_needed < len(words):
        truncated += "..."
    return truncated


# ---------------------------------------------------------------------------
# Structured rules engine for deterministic To-Do checklist items (C.4)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agent-grouped S section — organize problems by contributing domain
# ---------------------------------------------------------------------------

# Display labels for agent names (clinical role → readable header)
_AGENT_DISPLAY: dict[str, str] = {
    "nurse": "Nursing",
    "respiratory": "Respiratory",
    "pharmacy": "Medications",
    "dietitian": "Nutrition",
    "case_manager": "Disposition & Goals of Care",
    "therapist": "Mobility & Rehab",
    "intensivist": "Clinical Synthesis",
}

# Order agents should appear in the grouped S section
_AGENT_S_ORDER = [
    "nurse", "respiratory", "pharmacy", "dietitian",
    "case_manager", "therapist", "intensivist",
]


def _build_grouped_s_section(
    effective_snippets: list,
    intensivist_output: Any,
) -> str:
    """Build the S section grouped by contributing agent/domain.

    Instead of merging all S contributions into a single flat list,
    this preserves per-agent grouping with domain headers so the
    physician can scan by clinical area.

    Each agent's S content is rendered under a ## Domain header.
    The Intensivist's S content (if any) appears last as "Clinical
    Synthesis" — it typically contains gap-fills and cross-domain items.
    """
    # Collect S contributions per agent
    agent_s: dict[str, list[str]] = {}
    for snippet in effective_snippets:
        for sec in snippet.sections:
            sec_key = sec.section
            if sec_key.lower() == "s" and sec.content and sec.content != NOT_ENOUGH_INFO:
                agent_s.setdefault(snippet.agent_name, []).append(sec.content)

    # Add Intensivist's S if present
    if intensivist_output and intensivist_output.sections:
        for sec in intensivist_output.sections:
            if sec.section.lower() == "s" and sec.content and sec.content != NOT_ENOUGH_INFO:
                agent_s.setdefault("intensivist", []).append(sec.content)

    if not agent_s:
        return NOT_ENOUGH_INFO

    # Assemble with domain headers, filtering generic items
    parts: list[str] = []
    for agent_name in _AGENT_S_ORDER:
        contents = agent_s.get(agent_name)
        if not contents:
            continue
        display = _AGENT_DISPLAY.get(agent_name, agent_name.replace("_", " ").title())
        # Filter generic monitoring items and collect non-empty lines
        domain_lines: list[str] = []
        for content in contents:
            filtered = _filter_generic_problems(_normalize_s_format(content))
            for line in filtered.split("\n"):
                stripped = line.strip()
                if stripped and not stripped.startswith("## "):
                    domain_lines.append(stripped)
        # Only add domain header if there's content after filtering
        if domain_lines:
            parts.append(f"## {display}")
            parts.extend(domain_lines)
            parts.append("")  # blank line between groups

    # Catch any agents not in the predefined order
    for agent_name, contents in agent_s.items():
        if agent_name not in _AGENT_S_ORDER:
            display = _AGENT_DISPLAY.get(agent_name, agent_name.replace("_", " ").title())
            domain_lines = []
            for content in contents:
                filtered = _filter_generic_problems(_normalize_s_format(content))
                for line in filtered.split("\n"):
                    stripped = line.strip()
                    if stripped and not stripped.startswith("## "):
                        domain_lines.append(stripped)
            if domain_lines:
                parts.append(f"## {display}")
                parts.extend(domain_lines)
                parts.append("")

    return "\n".join(parts).rstrip()


def generate_structured_todos(patient_data: dict[str, Any]) -> list[str]:
    """Generate deterministic to-do items from structured CLIF data.

    These rules fire based on the presence of specific signals in the
    patient data (devices, medications, pending results, etc.) and produce
    actionable checklist items that are independent of LLM text generation.

    This is the "rules engine" described in the proposal (C.4): it ensures
    critical handoff items are never missed even if agents don't mention them.
    """
    todos: list[str] = []

    # --- Lines / Devices / Airways ---
    respiratory = patient_data.get("respiratory", [])
    if respiratory:
        # Check for active invasive devices at latest timepoint
        # (respiratory rows are newest-first from serialize_to_json).
        latest_resp = respiratory[0] if isinstance(respiratory, list) else None
        if latest_resp and isinstance(latest_resp, dict):
            device = latest_resp.get("device_category", "")
            if device and "trach" in str(device).lower():
                todos.append("Assess tracheostomy: weaning plan and decannulation readiness")
            if device and any(k in str(device).lower() for k in ["ett", "endotracheal"]):
                todos.append("Assess extubation readiness")

    # --- Medication safety ---
    meds = patient_data.get("meds", {})
    continuous = meds.get("continuous", []) if isinstance(meds, dict) else []
    intermittent = meds.get("intermittent", []) if isinstance(meds, dict) else []

    # Active vasopressors / sedatives → ensure taper plan
    high_risk_categories = {"vasopressor", "sedative", "inotrope", "paralytic"}
    for med in continuous:
        if isinstance(med, dict):
            cat = str(med.get("med_category", "")).lower()
            if any(hr in cat for hr in high_risk_categories):
                todos.append(f"Review taper/transition plan for continuous {cat}")
                break  # One reminder is enough

    # Anticoagulants → DVT prophylaxis transition
    all_meds = continuous + intermittent
    has_anticoag = any(
        "anticoag" in str(m.get("med_category", "")).lower()
        or "heparin" in str(m.get("med_category", "")).lower()
        for m in all_meds if isinstance(m, dict)
    )
    if has_anticoag:
        todos.append("Confirm anticoagulation plan post-transfer (dose adjustment or DVT prophylaxis)")

    # --- Pending cultures ---
    # Only fire if there are genuinely pending (no organism result yet) cultures.
    # Exclude empty/null/unknown organism values — those are data quality issues,
    # not actionable pending results.
    micro = patient_data.get("microbiology", [])
    _PENDING_ORGANISM_VALUES = {"pending", "in progress", "preliminary"}
    pending_cultures = [
        m for m in micro if isinstance(m, dict)
        and str(m.get("organism", "")).strip().lower() in _PENDING_ORGANISM_VALUES
    ]
    if pending_cultures:
        specimens = set()
        for m in pending_cultures:
            spec = str(m.get("specimen_type", m.get("specimen", ""))).strip()
            if spec and spec.lower() not in ("", "none", "unknown", "pending"):
                specimens.add(spec)
        if specimens:
            todos.append(f"Follow up pending cultures: {', '.join(sorted(specimens))}")

    # --- Code status ---
    code_status = patient_data.get("code_status", [])
    if not code_status:
        todos.append("Verify code status documentation (not found in structured data)")

    # --- Active CRRT / ECMO ---
    crrt = patient_data.get("crrt", [])
    if crrt:
        todos.append("CRRT active: confirm renal replacement therapy transition plan")

    ecmo = patient_data.get("ecmo", [])
    if ecmo:
        todos.append("ECMO/MCS active: confirm decannulation or continuation plan")

    # --- Foley / urinary catheter (from assessments) ---
    assessments = patient_data.get("assessments", [])
    for a in assessments if isinstance(assessments, list) else []:
        if isinstance(a, dict):
            cat = str(a.get("assessment_category", "")).lower()
            if "foley" in cat or "urinary catheter" in cat:
                todos.append("Reassess Foley catheter necessity (daily assessment per CAUTI bundle)")
                break

    # --- Device dwell time (critical flags auto-promote to checklist) ---
    procedures = patient_data.get("procedures", [])
    demographics = patient_data.get("demographics", {})
    ref_dttm = demographics.get("reference_dttm") or demographics.get("icu_admission_dttm")
    if procedures and ref_dttm:
        from icu_pause.tools.device_dwell import check_device_dwell

        dwell_result = check_device_dwell(procedures, ref_dttm)
        for flag in dwell_result.flags:
            if flag.severity == "critical":
                device_name = flag.device_type.replace("_", " ").title()
                todos.append(
                    f"URGENT: {device_name} in place {flag.dwell_days}d "
                    f"— {flag.recommended_action} (infection risk)"
                )

    return todos

logger = logging.getLogger(__name__)

# Ordered section metadata for rendering
SECTION_ORDER = [
    (ICUPauseSection.I, "ICU Admission Reason & Brief ICU Course"),
    (ICUPauseSection.C, "Code Status / DPOA / Goals of Care / ACP Note"),
    (ICUPauseSection.U_UNPRESCRIBING, "Unprescribing & Pertinent High-Risk Medications"),
    (ICUPauseSection.P, "Pending Tests at Time of Transfer"),
    (ICUPauseSection.A, "Active Consultants (including Rehab: PT, OT, SLP, Wound Care)"),
    (ICUPauseSection.U_UNCERTAINTY, "Uncertainty Measure / Diagnostic Pause"),
    (ICUPauseSection.S, "Summary of Major Problems and To-Do's"),
    (ICUPauseSection.E, "Exam at Transfer, Lines/Drains/Airways & Data Review"),
]

NOT_ENOUGH_INFO = "Not enough information from structured data."


_CAUSAL_PENDING_RE = re.compile(
    r"\b(less|more)\s+likely\s+[^,;.]{0,80}"
    r"\b(given|based\s+on|due\s+to|because|since|owing\s+to|with)\s+"
    r"[^,;.]*\bpending\b",
    re.IGNORECASE,
)

_ORGANISM_CLASSES = (
    "bacterial", "viral", "fungal", "atypical",
    "mycobacterial", "parasitic",
)
_ORGANISM_CLASS_RE = re.compile(
    r"\b(" + "|".join(_ORGANISM_CLASSES) + r")\b",
    re.IGNORECASE,
)


def _detect_cross_domain_basis(diagnosis: str, basis: str) -> bool:
    """Return True iff diagnosis and basis name DIFFERENT organism classes.

    Heuristic only; fires when BOTH sides name an organism class token.
    Non-organism failure modes rely on HITL review.
    """
    dx_classes = {m.group(1).lower() for m in _ORGANISM_CLASS_RE.finditer(diagnosis)}
    basis_classes = {m.group(1).lower() for m in _ORGANISM_CLASS_RE.finditer(basis)}
    if not dx_classes or not basis_classes:
        return False
    return dx_classes.isdisjoint(basis_classes)


def _drop_to_clause_boundary(text: str, match: re.Match) -> str:
    """Drop from match.start() to the next clause delimiter (, ; .) or line end.

    Used outside the Less likely section where the causal-pending construction
    appears in narrative prose rather than as a per-line entry.
    """
    start = match.start()
    rest = text[match.end():]
    boundary = re.search(r"[,;.\n]", rest)
    end = match.end() + boundary.end() if boundary else len(text)
    return text[:start] + text[end:]


_U_SECTION_HEADER_RE = re.compile(
    r"^\s*(less\s+likely|pending\s+data)\b",
    re.IGNORECASE,
)
_U_NONE_BULLET_RE = re.compile(
    r"^\s*-\s*none(\s+with\s+documented\s+reasoning)?\s*\.?\s*$",
    re.IGNORECASE,
)
_U_BULLET_RE = re.compile(r"^\s*-\s")


def _strip_stray_none_bullets(text: str) -> str:
    """Drop '- none' bullets that coexist with real items under Less likely
    or Pending data. The model sometimes emits a stale-template 'none'
    placeholder alongside populated items; the sentinel is meaningful only
    when the section is otherwise empty.
    """
    lines = text.split("\n")
    drop: set[int] = set()
    i = 0
    while i < len(lines):
        if not _U_SECTION_HEADER_RE.match(lines[i]):
            i += 1
            continue
        # Walk subsequent bullet/blank lines under this header.
        bullets: list[tuple[int, str]] = []
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped:
                j += 1
                continue
            if _U_BULLET_RE.match(lines[j]):
                bullets.append((j, lines[j]))
                j += 1
                continue
            break
        has_real = any(not _U_NONE_BULLET_RE.match(b) for _, b in bullets)
        has_none = any(_U_NONE_BULLET_RE.match(b) for _, b in bullets)
        if has_real and has_none:
            for idx, b in bullets:
                if _U_NONE_BULLET_RE.match(b):
                    drop.add(idx)
                    logger.warning(
                        "U_uncertainty: stripped stray 'none' bullet coexisting with real items"
                    )
        i = j
    if not drop:
        return text
    return "\n".join(l for k, l in enumerate(lines) if k not in drop)


def _enforce_u_uncertainty(text: str) -> str:
    """Re-emit U_uncertainty in canonical templated form.

    Structure: Working diagnosis / Differential / Less likely (per-diagnosis) /
    Pending data (to confirm/exclude) / Diagnostic certainty prose line.

    Validator behavior:
    - Causal-pending detector: within Less likely, drop the whole entry line;
      outside Less likely, drop from the matched span to the next clause
      delimiter. Drop-with-warning is the default.
    - Organism-class mismatch flagged in Less likely entries; cross-domain
      bases dropped. Heuristic only — non-organism shapes rely on HITL.
    - Empty slots render as "none with documented reasoning" / "none"
      rather than being suppressed.
    """
    has_working_dx = bool(re.search(r"(?i)working\s+diagnos", text))
    has_ddx = bool(re.search(r"(?i)(differential|ddx\s+includes)", text))

    if not has_working_dx or not has_ddx:
        logger.warning(
            "U_uncertainty missing required format "
            "(Working diagnosis / Differential) — replacing with template stub"
        )
        return (
            "Working diagnosis at the time of transfer: "
            "[see clinical narrative in section I]\n"
            "Differential includes: [not specified]\n"
            "Less likely:\n"
            "  - none with documented reasoning\n"
            "Pending data (to confirm/exclude):\n"
            "  - none"
        )

    # Preserve the LLM's prose; strip out any incoming certainty rendering
    # (the Diagnostic certainty line is dropped from the dotphrase output —
    # the post-processor canonicalizes legacy shapes by removing them).
    header = text
    header = re.split(r"(?i)select\s+from\s+the\s+following\s*:?", header, maxsplit=1)[0]
    header = re.sub(r"(?im)^\s*diagnostic\s+certainty\s*:.*$", "", header)
    header = re.sub(r"(?im)^\s*certainty\s+level\s*[:=].*$", "", header)
    header = re.sub(r"(?im)^\s*[123]\s*:\s*(High|Some|Marked)[^\n]*$", "", header)
    header = re.sub(r"(?m)^\s*[☐☒]\s*(High|Some|Marked)[^\n]*$", "", header)
    header = re.sub(
        r"(?m)^\s*-\s*\[\s*[xX ]?\s*\]\s*(High|Some|Marked)[^\n]*$", "", header,
    )

    # Per-line walk: drop causal-pending and cross-domain Less likely entries;
    # apply clause-boundary causal-pending drop outside Less likely.
    cleaned_lines: list[str] = []
    in_less_likely = False
    for line in header.split("\n"):
        stripped = line.strip()
        if re.match(r"(?i)less\s+likely\s*:?\s*$", stripped):
            in_less_likely = True
            cleaned_lines.append(line)
            continue
        if in_less_likely and (
            not stripped or re.match(r"(?i)pending\s+data", stripped)
        ):
            in_less_likely = False
            cleaned_lines.append(line)
            continue
        if in_less_likely and stripped.startswith("-"):
            if _CAUSAL_PENDING_RE.search(line):
                logger.warning(
                    "U_uncertainty causal-pending Less likely entry dropped: %r",
                    stripped,
                )
                continue
            m = re.match(r"-\s*([^:]+):\s*based\s+on\s+(.+)", stripped, re.IGNORECASE)
            if m and _detect_cross_domain_basis(m.group(1), m.group(2)):
                logger.warning(
                    "U_uncertainty cross-domain basis dropped: dx=%r basis=%r",
                    m.group(1).strip(), m.group(2).strip(),
                )
                continue
        else:
            # Outside Less likely: causal-pending in prose gets clause-boundary drop.
            while True:
                match = _CAUSAL_PENDING_RE.search(line)
                if not match:
                    break
                logger.warning(
                    "U_uncertainty causal-pending clause dropped: %r",
                    match.group(0),
                )
                line = _drop_to_clause_boundary(line, match)
        cleaned_lines.append(line)
    header = "\n".join(cleaned_lines)

    # Empty-slot rendering: render placeholder rather than suppressing the header.
    # The empty-slot lookahead `(?=\s*$|\s*\n\s*$)` was historically too
    # permissive (matches at end-of-line trivially), so this can over-inject
    # when bullets follow. The strip pass below corrects both that and the
    # model-side bug where the LLM emits a stale-template "- none" alongside
    # real items.
    header = re.sub(
        r"(?im)(^\s*less\s+likely\s*:\s*$)(?=\s*\n\s*(?:pending\s+data|$))",
        r"\1\n  - none with documented reasoning",
        header,
    )
    header = re.sub(
        r"(?im)(^\s*pending\s+data\s*\(to\s+confirm/exclude\)\s*:\s*$)(?=\s*$|\s*\n\s*$)",
        r"\1\n  - none",
        header,
    )

    # Strip stray "- none" bullets coexisting with real items (post-injection
    # cleanup; the sentinel is meaningful only in a truly-empty section).
    header = _strip_stray_none_bullets(header)

    return header.rstrip()


# ---------------------------------------------------------------------------
# Competing-risks indication grounding validator (v1.8)
# See docs/competing_risks_indication_grounding_v1.8_design.md
# ---------------------------------------------------------------------------

_QUANT_PRED_RE = re.compile(
    r"\b("
    r"within\s+\d+\s*(?:hours?|hrs?|h|days?|d|weeks?|wks?|months?)|"
    r"\d+\s*(?:units?|mL|mg|g/dL|mEq/L|mmol/L)|"
    r"INR\s*[<>]\s*\d|"
    r"fall(?:ing)?\s+by\s+\d|"
    r"transfusion\s+of\s+\d|"
    r"drop\s+(?:of|by)\s+\d|"
    r"\d+\s*%|"
    r"\d+\s*in\s*\d+"
    r")\b",
    re.IGNORECASE,
)

_INDICATION_FRAMING_RE = re.compile(
    # bare "for <Capitalized phrase>" or "for <hedge-qualifier> <phrase>"
    # — catches "for acute on chronic segmental PE", "for presumed PE",
    # "for AFib", "for atrial fibrillation".
    r"\bfor\s+(?:acute|chronic|presumed|likely|suspected|known)?\s*"
    r"([A-Z][^.,;]+|atrial\s+fibrillation|pulmonary\s+embol[a-z]*|"
    r"deep\s+vein\s+thromb[a-z]*)",
    re.IGNORECASE,
)


# Event types that can carry routed-note bodies. Different versions of the
# trace have used different event-type names; scan all of them so the
# validator doesn't silently force-hedge when the schema shifts.
_NOTE_BODY_EVENT_TYPES = frozenset({
    "note_load", "data_retrieval", "agent_input", "note_routing",
})

# Field names a note row might use for the body text. The HPC trace as of
# 2026-05-28 uses "text_preview" (truncated). Older shapes used "note_text"
# or "text". Newer shapes may use "body" or "content". Scan all of them.
_NOTE_BODY_FIELD_NAMES = ("note_text", "text", "body", "content", "text_preview")


def _routed_note_index_from_trace(
    trace_events: list[dict[str, Any]],
) -> dict[str, str]:
    """Build a mapping of routed note_id -> note body from the trace event
    stream. Reads the post-routing body (what the LLM actually saw) — for
    synthetic perturbation runs this is the redacted body, NOT the
    pre-redaction source. See design doc §6 Substep 2.

    Defensive scan: walks every event whose type is in
    _NOTE_BODY_EVENT_TYPES, recursively descends into nested dicts/lists,
    and accepts a body under any field name in _NOTE_BODY_FIELD_NAMES.
    Tolerates trace-schema drift — the v1.8.0 implementation only looked
    for note_text/text under note_load events, silently returning an
    empty mapping when the actual schema used text_preview, which caused
    every model citation to be rejected as "not in routed-note set."
    """
    routed: dict[str, str] = {}

    def _maybe_record(row: Any) -> None:
        if not isinstance(row, dict):
            return
        nid = row.get("note_id")
        if not nid:
            return
        for fname in _NOTE_BODY_FIELD_NAMES:
            txt = row.get(fname)
            if isinstance(txt, str) and txt:
                # First non-empty body wins; don't overwrite a fuller body
                # (e.g., note_text) with a truncated preview seen later.
                existing = routed.get(str(nid), "")
                if len(txt) > len(existing):
                    routed[str(nid)] = txt
                break

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            _maybe_record(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for ev in trace_events or []:
        if ev.get("type") not in _NOTE_BODY_EVENT_TYPES:
            continue
        _walk(ev.get("data") or {})
    return routed


def _notes_by_id_from_agent_context(
    agent_context_text: dict[str, Any],
) -> dict[str, str]:
    """Build {note_id -> full note_text} from per-agent post-cap context.

    Authoritative source of full note bodies for the v1.8 indication-grounding
    validator. The trace-derived path (_routed_note_index_from_trace) reads
    text_preview which on HPC truncates to ~150 chars — any quote past that
    cutoff would be falsely rejected. Fixes 2026-05-29 hosp <hospitalization_id>
    finding where the validator over-fired on a legit citation because the
    quote lived past char 153 of a 10,669-char progress note.

    The actual production shape of agent_context_text is
    ``dict[agent_role, {"notes": dict[ntype, list[note_dict]], ...other_keys}]``
    — notes live one level deeper than the original implementation assumed.
    Rather than hard-code that shape (and re-break the next time it shifts),
    walks the structure recursively picking up any dict with both note_id
    and a body field, mirroring _routed_note_index_from_trace's discipline.
    """
    out: dict[str, str] = {}

    def _maybe_record(row: Any) -> None:
        if not isinstance(row, dict):
            return
        nid = row.get("note_id")
        if nid is None:
            return
        txt = row.get("note_text") or row.get("text") or ""
        if not isinstance(txt, str) or not txt:
            return
        key = str(nid)
        if len(txt) > len(out.get(key, "")):
            out[key] = txt

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            _maybe_record(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(agent_context_text or {})
    return out


def _scrub_warning_text_for_drug(
    warnings: list[Warning],
    drug: str,
    *,
    conflict_condition: Optional[str],
) -> tuple[list[Warning], int]:
    """Rewrite warning(s) naming ``drug`` to use the dedicated-clause
    hedge form. Returns (updated_warnings, count_rewritten).

    Heuristic (per design doc §6.2):
      1. Find warnings whose message contains the drug name (case-insensitive
         substring).
      2. In each matched warning, regex-replace `for <indication phrase>`
         with the dedicated-clause hedge form. The replacement is
         positioned mid-sentence as a clause rather than a parenthetical.
      3. If the indication-framing regex doesn't match the warning text
         (warning drifted from the entry's indication), drop the warning
         entirely — safer than leaving a possibly-confabulated warning.
    """
    out: list[Warning] = []
    rewritten = 0
    cond_clause = (
        f"in the setting of {conflict_condition}"
        if conflict_condition
        else "in the setting of the documented competing condition"
    )
    # Idempotency sentinel — the literal hedge phrase the rewriter inserts.
    # If a warning's message already contains this fragment, the rewriter
    # has run before and we MUST skip; re-running produces garbled output
    # because the inserted "indication for {drug} not documented" itself
    # contains a "for {drug}" substring that _INDICATION_FRAMING_RE re-matches
    # on Substep 4's pass, triggering another rewrite. The v1.8.0 doc-cell
    # run surfaced this with duplicated clauses ("...in the setting of X
    # — indication in the setting of X — indication for {drug}...").
    hedge_sentinel = (
        f"indication for {drug.lower()} not documented in available notes"
    )
    for w in warnings:
        msg = w.message
        if drug.lower() not in msg.lower():
            out.append(w)
            continue
        if hedge_sentinel in msg.lower():
            # Already rewritten on a prior substep — preserve verbatim.
            out.append(w)
            continue
        if not _INDICATION_FRAMING_RE.search(msg):
            # warning text doesn't have an indication-framing phrase to
            # replace — drop the warning entirely, safer than leaving it
            rewritten += 1
            continue
        new_msg = _INDICATION_FRAMING_RE.sub(
            f"{cond_clause} — indication for {drug} not documented in "
            f"available notes; competing risks framing therefore preliminary",
            msg,
            count=1,
        )
        out.append(Warning(
            category=w.category,
            severity=w.severity,
            message=new_msg,
            source_agent=w.source_agent,
            source_section=w.source_section,
            cite=w.cite,
        ))
        rewritten += 1
    return out, rewritten


def validate_competing_risks_grounding(
    intensivist_output: Any,
    trace_events: list[dict[str, Any]],
    *,
    notes_by_id: dict[str, str] | None = None,
) -> list[Warning]:
    """Post-pass validator for the Phase-1 competing_risks_check entries.

    For every entry whose indication is not the hedge phrase, verifies that
    (a) source_note_id is in the case's routed-note set, and (b) source_quote
    substring-matches source_note_id's body (with normalization). On miss
    or after substring failure, rewrites the entry's indication to the
    hedge phrase, scrubs the corresponding warning text via
    ``_scrub_warning_text_for_drug``, and returns a qa_process WARN.

    Additionally checks (warning↔entry consistency, §6.2.1 Substep 4) that
    for every entry whose final indication is the hedge phrase, no warning
    naming the entry's drug carries an indication-framing phrase — catches
    the parallel-emission failure where the structured entry hedges but the
    warning text independently confabulates.

    Finally, runs a qualitative-only canary on each arm string — emits a
    qa_process WARN if a quantitative prediction is detected (no rewrite).

    Mutates ``intensivist_output.reasoning_log.competing_risks_check`` and
    ``intensivist_output.warnings`` in place. Returns the list of newly
    emitted WARN objects to be folded into the warning stream.
    """
    out_warnings: list[Warning] = []
    if intensivist_output is None:
        return out_warnings
    rl = getattr(intensivist_output, "reasoning_log", None)
    if rl is None:
        return out_warnings
    entries: list[CompetingRisksEntry] = list(
        getattr(rl, "competing_risks_check", []) or []
    )
    if not entries:
        return out_warnings

    routed = _routed_note_index_from_trace(trace_events)
    # State-supplied full bodies override the (truncated) trace-derived
    # previews. The trace path stays as a fallback for note_ids that the
    # state lookup doesn't carry (e.g., legacy state shapes or perturbation
    # runs that only register through the trace).
    if notes_by_id:
        for nid, full in notes_by_id.items():
            if len(full) > len(routed.get(nid, "")):
                routed[nid] = full

    HEDGE = CompetingRisksEntry.HEDGE_PHRASE

    new_entries: list[CompetingRisksEntry] = []
    for entry in entries:
        # Pass-through for already-hedged entries (Pydantic already enforced
        # source fields are None); fall through to Substep 4 below.
        if entry.indication == HEDGE:
            new_entries.append(entry)
        else:
            # Substep 1a: truncation check (distinct WARN identifier from
            # the substring-miss case so we can track fill-pressure
            # truncation separately in audit logs)
            if entry.source_quote and has_truncation_marker(entry.source_quote):
                out_warnings.append(Warning(
                    category=WarningCategory.QA_PROCESS,
                    severity=WarningSeverity.INFO,
                    message=(
                        f"INDICATION_QUOTE_TRUNCATED: drug={entry.drug!r} "
                        f"source_quote contains ellipsis "
                        f"({entry.source_quote!r}); indication rewritten "
                        f"to hedge"
                    ),
                    source_agent="orchestrator",
                ))
                entry = entry.model_copy(update={
                    "indication": HEDGE,
                    "source_note_id": None,
                    "source_quote": None,
                })
                new_warnings, _ = _scrub_warning_text_for_drug(
                    list(intensivist_output.warnings),
                    entry.drug,
                    conflict_condition=entry.conflict_condition,
                )
                intensivist_output.warnings = new_warnings
                new_entries.append(entry)
                continue
            # Substep 1: source-set check
            if entry.source_note_id not in routed:
                out_warnings.append(Warning(
                    category=WarningCategory.QA_PROCESS,
                    severity=WarningSeverity.INFO,
                    message=(
                        f"INDICATION_NOT_GROUNDED_IN_SOURCE: drug="
                        f"{entry.drug!r} cited source_note_id="
                        f"{entry.source_note_id!r} not in routed-note "
                        f"set ({len(routed)} routed notes); indication "
                        f"rewritten to hedge"
                    ),
                    source_agent="orchestrator",
                ))
                entry = entry.model_copy(update={
                    "indication": HEDGE,
                    "source_note_id": None,
                    "source_quote": None,
                })
                new_warnings, _ = _scrub_warning_text_for_drug(
                    list(intensivist_output.warnings),
                    entry.drug,
                    conflict_condition=entry.conflict_condition,
                )
                intensivist_output.warnings = new_warnings
                new_entries.append(entry)
                continue
            # Substep 2: substring check (post-normalization)
            note_body = routed.get(entry.source_note_id, "")
            quote_norm = normalize_for_validator(entry.source_quote or "")
            body_norm = normalize_for_validator(note_body)
            if not quote_norm or quote_norm not in body_norm:
                out_warnings.append(Warning(
                    category=WarningCategory.QA_PROCESS,
                    severity=WarningSeverity.INFO,
                    message=(
                        f"INDICATION_NOT_GROUNDED_IN_SOURCE: drug="
                        f"{entry.drug!r} source_quote not found in "
                        f"body of source_note_id={entry.source_note_id!r}; "
                        f"indication rewritten to hedge"
                    ),
                    source_agent="orchestrator",
                ))
                entry = entry.model_copy(update={
                    "indication": HEDGE,
                    "source_note_id": None,
                    "source_quote": None,
                })
                new_warnings, _ = _scrub_warning_text_for_drug(
                    list(intensivist_output.warnings),
                    entry.drug,
                    conflict_condition=entry.conflict_condition,
                )
                intensivist_output.warnings = new_warnings
                new_entries.append(entry)
                continue
            # Citation resolved cleanly
            new_entries.append(entry)

        # Substep 3 (qualitative-only canary, applies to all entries):
        for arm_name in ("risk_of_continuing", "risk_of_holding_or_reducing"):
            arm = getattr(entry, arm_name)
            if arm and _QUANT_PRED_RE.search(arm):
                out_warnings.append(Warning(
                    category=WarningCategory.QA_PROCESS,
                    severity=WarningSeverity.INFO,
                    message=(
                        f"ARM_HAS_UNSUPPORTED_QUANTIFIER: drug="
                        f"{entry.drug!r} {arm_name} contains a "
                        f"quantitative prediction without source backing: "
                        f"{arm!r}"
                    ),
                    source_agent="orchestrator",
                ))

    # Substep 4 (warning↔entry consistency, §6.2.1):
    # For every entry whose FINAL indication is the hedge phrase, check
    # that no warning naming the drug carries an indication-framing
    # phrase. Catches the parallel-emission failure where the structured
    # entry hedges but the warning independently confabulates.
    for entry in new_entries:
        if entry.indication != HEDGE:
            continue
        offending = [
            w for w in intensivist_output.warnings
            if entry.drug.lower() in w.message.lower()
            and _INDICATION_FRAMING_RE.search(w.message)
        ]
        if not offending:
            continue
        out_warnings.append(Warning(
            category=WarningCategory.QA_PROCESS,
            severity=WarningSeverity.INFO,
            message=(
                f"INDICATION_CONFABULATED_IN_WARNING: drug="
                f"{entry.drug!r} entry hedges indication but warning "
                f"text contains indication-framing phrase ('for <X>'); "
                f"warning rewritten to dedicated-clause hedge"
            ),
            source_agent="orchestrator",
        ))
        new_warnings, _ = _scrub_warning_text_for_drug(
            list(intensivist_output.warnings),
            entry.drug,
            conflict_condition=entry.conflict_condition,
        )
        intensivist_output.warnings = new_warnings

    # Write back the mutated entry list.
    rl.competing_risks_check = new_entries
    return out_warnings


def _extract_section_i_lead_block(section_i_content: str) -> str:
    """Return the lead-sentence paragraph of a rendered Section I.

    Section I per the v2.0 FORMAT renders as three paragraph blocks
    separated by blank lines:
      1. one-liner lead sentence (age / sex / one-liner PMH lead /
         admission framing)
      2. "Full PMH per chart: <verbatim scribe.pmh>"
      3. "ICU course c/b ... Currently ..."

    The alignment lint operates on block 1. We split on blank lines
    and return the first non-empty block. If the LLM emitted a
    monolithic single-paragraph Section I (model didn't follow the
    new FORMAT exactly), we fall back to "the prose preceding the
    'Full PMH per chart:' marker" so the lint still has a target.
    """
    if not section_i_content:
        return ""
    # Primary path: blank-line-separated paragraphs.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_i_content) if p.strip()]
    if paragraphs:
        first = paragraphs[0]
        # Reject the first paragraph if it IS the PMH paragraph (some
        # models may flip the order). Detect by the literal label.
        if "Full PMH per chart" not in first:
            return first
    # Fallback: split on the literal PMH-paragraph marker.
    marker_match = re.search(r"Full PMH per chart\s*:", section_i_content, re.I)
    if marker_match:
        return section_i_content[: marker_match.start()].strip()
    # Last-resort fallback: the whole Section I content. The lint will
    # be over-permissive (every condition substring-matches somewhere)
    # but won't false-fail.
    return section_i_content.strip()


def _extract_section_i_pmh_paragraph(section_i_content: str) -> str:
    """Return the verbatim PMH paragraph block, or empty string if absent.

    Looks for the literal ``Full PMH per chart:`` label and returns the
    content following it, stopping at the next blank-line paragraph
    boundary. Used by the alignment lint to verify each entry's
    ``source_clause_anchor`` traces to the rendered paragraph slot, not
    just the original scribe.pmh string (defends against the model
    rendering a paraphrased / reordered PMH paragraph despite the
    AS-IS instruction).
    """
    if not section_i_content:
        return ""
    m = re.search(
        r"Full PMH per chart\s*:\s*(.+?)(?:\n\s*\n|\Z)",
        section_i_content,
        re.I | re.S,
    )
    return m.group(1).strip() if m else ""


def validate_one_liner_pmh_selection(
    intensivist_output: Any,
    state: dict[str, Any],
    trace_events: list[dict[str, Any]],
    *,
    notes_by_id: dict[str, str] | None = None,
) -> list[Warning]:
    """Post-render validator for the Section I one-liner PMH selection.

    Runs four checks per one_liner_pmh_selection entry:

    1. **Source-clause anchor check** — ``source_clause_anchor`` must
       substring-match ``scribe_extraction.pmh`` (or the rendered
       ``Full PMH per chart:`` paragraph slot when that's available
       and longer). Failure → ``ONE_LINER_PMH_ANCHOR_UNRESOLVED`` WARN;
       the entry survives in the structured output for audit, but the
       reviewer panel flags it.

    2. **In-prose alignment check** — each entry's ``display`` must
       have a normalized-form match in the rendered Section I lead
       sentence (block 1 of the v2.0 FORMAT). Normalization is
       ``normalize_for_pmh_match`` so routine abbreviation pairs
       (BrCa ↔ breast cancer, s/p ↔ status post, c/b ↔ complicated by,
       mets ↔ metastases) don't false-fail. Failure →
       ``ONE_LINER_PMH_ALIGNMENT_MISMATCH`` WARN.

    3. **Modifier-confirmation completeness check** — if ``display``
       contains a time-sensitive modifier (`on [drug]`,
       `requiring [intervention]`, `with active [condition]`,
       `recurrent`, `naive`, `chronic [organ-failure]-dependent`),
       a matching ``modifier_confirmation`` entry MUST be present.
       Failure → ``ONE_LINER_PMH_MODIFIER_UNCONFIRMED`` WARN.

    4. **Modifier-confirmation grounding check** — for each
       ModifierConfirmation, ``confirmed_in_note_id`` must resolve
       to a routed note, and ``confirmation_quote`` must
       substring-match that note's body (with
       ``normalize_for_validator``). Truncation markers (``...``,
       ``…``) in the quote fail validation. Failure →
       ``ONE_LINER_PMH_MODIFIER_QUOTE_UNRESOLVED`` WARN.

    Lint-only — does NOT mutate the rendered Section I text or the
    structured pin. The WARNs are the audit handle; reviewer-panel
    surfacing of failures motivates intensivist-prompt iteration in
    pilot rounds. Future PR can layer a "drop unconfirmed modifier
    from rendered display" rewrite on top of this lint without
    touching the lint contract.

    Mutates ``intensivist_output.warnings`` in place (appends) and
    returns the list of newly emitted WARN objects.
    """
    out_warnings: list[Warning] = []
    if intensivist_output is None:
        return out_warnings

    entries: list[OneLinerPMHEntry] = list(
        getattr(intensivist_output, "one_liner_pmh_selection", []) or []
    )
    if not entries:
        return out_warnings

    # Build the routed-note body lookup (state-supplied bodies win over
    # trace-derived previews, mirroring the competing_risks pattern).
    routed = _routed_note_index_from_trace(trace_events)
    if notes_by_id:
        for nid, full in notes_by_id.items():
            if len(full) > len(routed.get(nid, "")):
                routed[nid] = full

    scribe_extraction = state.get("scribe_extraction") or {}
    scribe_pmh = (scribe_extraction.get("pmh") or "").strip()

    # Find Section I content among the contributions.
    section_i_content = ""
    for contrib in getattr(intensivist_output, "sections", []) or []:
        sect = getattr(contrib, "section", None) or (
            contrib.get("section") if isinstance(contrib, dict) else None
        )
        if str(sect) in ("I", ICUPauseSection.I.value):
            section_i_content = (
                getattr(contrib, "content", None)
                or (contrib.get("content") if isinstance(contrib, dict) else "")
                or ""
            )
            break

    lead_block = _extract_section_i_lead_block(section_i_content)
    pmh_paragraph = _extract_section_i_pmh_paragraph(section_i_content)

    lead_normalized = normalize_for_pmh_match(lead_block)
    scribe_pmh_normalized = normalize_for_validator(scribe_pmh)
    pmh_paragraph_normalized = normalize_for_validator(pmh_paragraph)

    def _emit(message: str) -> None:
        w = Warning(
            category=WarningCategory.QA_PROCESS,
            severity=WarningSeverity.INFO,
            message=message,
            source_agent="orchestrator",
        )
        out_warnings.append(w)
        intensivist_output.warnings = list(intensivist_output.warnings) + [w]

    for entry in entries:
        display = (entry.display or "").strip()
        anchor = (entry.source_clause_anchor or "").strip()
        rank = entry.rank

        # 1. Anchor → scribe.pmh (or rendered paragraph slot, whichever
        # contains it). The paragraph slot is checked separately because
        # a rendered paraphrase is itself a violation worth surfacing as
        # a different signal — but for anchor resolution we accept either.
        anchor_norm = normalize_for_validator(anchor)
        anchor_resolved = (
            (scribe_pmh_normalized and anchor_norm in scribe_pmh_normalized)
            or (pmh_paragraph_normalized and anchor_norm in pmh_paragraph_normalized)
        )
        if anchor and not anchor_resolved:
            _emit(
                "ONE_LINER_PMH_ANCHOR_UNRESOLVED: "
                f"rank={rank} display={display!r} "
                f"source_clause_anchor={anchor!r} not substring-present "
                "in scribe.pmh or rendered PMH paragraph slot."
            )

        # 2. display ↔ Section I lead sentence
        display_norm = normalize_for_pmh_match(display)
        if display and lead_normalized and display_norm not in lead_normalized:
            _emit(
                "ONE_LINER_PMH_ALIGNMENT_MISMATCH: "
                f"rank={rank} display={display!r} "
                "not found (normalized) in Section I lead sentence."
            )

        # 3. Modifier-confirmation completeness
        time_sensitive_patterns = [
            r"\bon\s+\w",
            r"\brequiring\b",
            r"\bwith\s+active\b",
            r"\brecurrent\b",
            r"\bnaive\b",
            r"\bchronic\s+\w+[-\s]dependent\b",
        ]
        time_sensitive_modifiers = [
            re.search(p, display, re.I).group(0)
            for p in time_sensitive_patterns
            if re.search(p, display, re.I)
        ]
        confirmations: list[ModifierConfirmation] = list(
            entry.modifier_confirmation or []
        )
        confirmed_texts_norm = {
            normalize_for_validator(c.modifier_text or "")
            for c in confirmations
        }
        for mod_text in time_sensitive_modifiers:
            if normalize_for_validator(mod_text) not in confirmed_texts_norm:
                # Be tolerant: substring on the confirmation text counts.
                # Catches cases where modifier_text is the broader phrase
                # (e.g., "chronic vent-dependent") and the regex match is
                # the narrower trigger (e.g., "chronic vent-dependent").
                if not any(
                    normalize_for_validator(mod_text) in c
                    for c in confirmed_texts_norm
                ):
                    _emit(
                        "ONE_LINER_PMH_MODIFIER_UNCONFIRMED: "
                        f"rank={rank} display={display!r} "
                        f"contains time-sensitive modifier {mod_text!r} "
                        "without a matching modifier_confirmation entry."
                    )

        # 4. Modifier-confirmation grounding
        for conf in confirmations:
            mod_text = (conf.modifier_text or "").strip()
            nid = (conf.confirmed_in_note_id or "").strip()
            quote = (conf.confirmation_quote or "").strip()

            if has_truncation_marker(quote):
                _emit(
                    "ONE_LINER_PMH_MODIFIER_QUOTE_UNRESOLVED: "
                    f"rank={rank} modifier={mod_text!r} "
                    "confirmation_quote contains truncation marker."
                )
                continue

            note_body = routed.get(nid, "")
            if not note_body:
                _emit(
                    "ONE_LINER_PMH_MODIFIER_QUOTE_UNRESOLVED: "
                    f"rank={rank} modifier={mod_text!r} "
                    f"confirmed_in_note_id={nid!r} not in routed-notes set."
                )
                continue

            quote_norm = normalize_for_validator(quote)
            body_norm = normalize_for_validator(note_body)
            if quote_norm and quote_norm not in body_norm:
                _emit(
                    "ONE_LINER_PMH_MODIFIER_QUOTE_UNRESOLVED: "
                    f"rank={rank} modifier={mod_text!r} "
                    f"confirmation_quote not substring-present in "
                    f"note_id={nid!r} body."
                )

    return out_warnings


# Section I "scribe pin empty" fallback string (must match intensivist prompt).
_PMH_FALLBACK_STR = "PMH not extracted from available notes — chart review required"


def apply_pmh_fallback_render(
    merged_sections: dict[str, str], pmh_fallback: Any
) -> Optional[dict[str, Any]]:
    """Render hand-off + #12 provenance for the intensivist PMH fallback.

    When the intensivist composed PMH from the H&P/progress-note opener
    (``pmh_fallback`` populated with non-empty ``text``), GUARANTEE the rendered
    Section I one-liner carries that text and not the "chart review required"
    string — defends against the model populating the structured field but
    leaving the fallback string in the prose. Mutates ``merged_sections`` in
    place. Returns a ``pmh_fallback_fired`` trace event (sans timestamp) for the
    #12 manifest fire-rate, or None when there's nothing to apply.
    """
    text = (
        (getattr(pmh_fallback, "text", "") or "").strip()
        if pmh_fallback is not None else ""
    )
    if not text:
        return None
    sec_i = merged_sections.get("I", "") or ""
    if _PMH_FALLBACK_STR in sec_i:
        merged_sections["I"] = sec_i.replace(_PMH_FALLBACK_STR, text)
    return {
        "type": "pmh_fallback_fired",
        "node": "orchestrator",
        "level": "info",
        "message": (
            f"intensivist PMH fallback fired: "
            f"source={getattr(pmh_fallback, 'pmh_source', '?')} "
            f"note_types={getattr(pmh_fallback, 'note_types', [])} "
            f"chars={len(text)}"
        ),
        "data": {
            "pmh_source": getattr(pmh_fallback, "pmh_source", None),
            "note_ids": getattr(pmh_fallback, "note_ids", []),
            "note_types": getattr(pmh_fallback, "note_types", []),
            "text": text,
            "pmh_fallback_chars": len(text),
        },
    }


class SectionMerger:
    """Step 5: Merge all validated agent snippets into the final ICU-PAUSE brief."""

    def __init__(self, settings: Any):
        self.settings = settings

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Merge agent outputs into the final ICU-PAUSE output.

        When the Intensivist agent has produced output, its harmonized sections
        are used as the PRIMARY source for each section. Domain agent outputs
        serve as fallback for any sections the Intensivist did not cover.
        """
        snippets: list[AgentSnippet] = state.get("agent_snippets", [])
        revised: list[AgentSnippet] = state.get("revised_snippets", [])
        intensivist_output: AgentSnippet | None = state.get("intensivist_output")
        delib_log: list[dict] = state.get("deliberation_log", [])
        qa_issues: list[str] = state.get("qa_issues", [])

        # If deliberation produced revised snippets, prefer them over originals
        revised_agents = {s.agent_name for s in revised}
        if revised_agents:
            effective_snippets = [
                s for s in snippets if s.agent_name not in revised_agents
            ] + list(revised)
            logger.info(
                f"SectionMerger: using revised snippets from {sorted(revised_agents)}"
            )
        else:
            effective_snippets = list(snippets)

        # Build Intensivist section lookup (primary source when available)
        # Normalize section keys: Intensivist may output "U_UNPRESCRIBING" but
        # the canonical keys are "U_unprescribing". Build a case-insensitive map.
        canonical_keys = {s.value.lower(): s.value for s, _ in SECTION_ORDER}
        intensivist_sections: dict[str, Any] = {}
        if intensivist_output and intensivist_output.sections:
            for sec in intensivist_output.sections:
                if sec.content and sec.content != NOT_ENOUGH_INFO:
                    # Normalize key: try exact, then lowercase match
                    key = sec.section
                    if key not in canonical_keys.values():
                        normalized = canonical_keys.get(key.lower(), key)
                        if normalized != key:
                            logger.info(f"SectionMerger: normalized key '{key}' -> '{normalized}'")
                            key = normalized
                    intensivist_sections[key] = sec
            logger.info(
                f"SectionMerger: Intensivist provided {len(intensivist_sections)} sections"
            )

        # Merge: for each section, choose best source between Intensivist and domain agents
        merged_sections: dict[str, str] = {}
        section_confidences: dict[str, float] = {}
        for section_enum, _label in SECTION_ORDER:
            section_key = section_enum.value

            # Collect domain agent contributions for this section
            contributions = []
            for snippet in effective_snippets:
                for sec in snippet.sections:
                    # Case-insensitive section key matching for domain agents too
                    sec_key = sec.section
                    if sec_key not in canonical_keys.values():
                        sec_key = canonical_keys.get(sec_key.lower(), sec_key)
                    if sec_key == section_key and sec.content and sec.content != NOT_ENOUGH_INFO:
                        contributions.append(sec)

            # Best domain agent contribution (by field coverage)
            best_agent = None
            if contributions:
                contributions.sort(
                    key=lambda c: (self._field_coverage_score(c), len(c.content)),
                    reverse=True,
                )
                best_agent = contributions[0]

            # Choose source: Intensivist vs domain agent
            intensivist_sec = intensivist_sections.get(section_key)

            if intensivist_sec and best_agent:
                # Both exist: use Intensivist UNLESS its confidence is very low
                # and domain agent has substantially better content
                if intensivist_sec.confidence <= 0.2 and best_agent.confidence >= 0.4:
                    logger.info(
                        f"SectionMerger: {section_key} — using domain agent "
                        f"(conf {best_agent.confidence}) over Intensivist (conf {intensivist_sec.confidence})"
                    )
                    merged_sections[section_key] = best_agent.content
                    section_confidences[section_key] = best_agent.confidence
                else:
                    merged_sections[section_key] = intensivist_sec.content
                    section_confidences[section_key] = intensivist_sec.confidence
            elif intensivist_sec:
                merged_sections[section_key] = intensivist_sec.content
                section_confidences[section_key] = intensivist_sec.confidence
            elif best_agent:
                merged_sections[section_key] = best_agent.content
                section_confidences[section_key] = best_agent.confidence
            else:
                merged_sections[section_key] = NOT_ENOUGH_INFO
                section_confidences[section_key] = 0.0

        # Coerce any list-typed section values to strings (LLMs sometimes
        # return arrays instead of strings for section content)
        for key, val in merged_sections.items():
            if isinstance(val, list):
                merged_sections[key] = "\n".join(str(item) for item in val)

        # Get patient data for deterministic post-processing.  This stays
        # uncapped because the safety net (template / safety-tools) is a floor
        # that should always run on the most complete data, independent of
        # what any LLM agent happened to receive after caps.
        patient_data = state.get("patient_context_text", {})

        # Per-agent post-cap, post-routing slices — used as the source of
        # truth for what reviewers should see (see metadata wiring below).
        agent_context_text = state.get("agent_context_text", {}) or {}

        # --- Deterministic post-processing: enforce dotphrase template ---
        self._enforce_dotphrase_template(merged_sections, patient_data)

        # --- Append deterministic automated-screen block to U_unprescribing ---
        # Mirrors the safety-checkbox pattern in _enforce_dotphrase_template:
        # always state what the tool checked, so silence is never ambiguous
        # between "no interactions" and "tool didn't run."
        ddi_block = _format_ddi_review_block(
            patient_data.get("meds", {}),
            self.settings,
            reference_dttm=(patient_data.get("demographics") or {}).get(
                "reference_dttm"
            ),
        )
        if ddi_block:
            existing = merged_sections.get("U_unprescribing", "").rstrip()
            sep = "\n\n" if existing else ""
            merged_sections["U_unprescribing"] = existing + sep + ddi_block

        # --- Citation deduplication (before provenance check) ---
        from icu_pause.tools.citation_check import (
            check_citation_preservation,
            check_citation_provenance,
            deduplicate_citations,
            expand_concatenated_citations,
        )

        for key in merged_sections:
            # Expand "(a; b; c)" → "(a) (b) (c)" before dedup so the renderer
            # and judge both see one source tag per paren.
            merged_sections[key] = expand_concatenated_citations(merged_sections[key])
            merged_sections[key] = deduplicate_citations(merged_sections[key])

        # --- Consolidate Section E into the fixed 5-bucket layout (pre-render) ---
        # Re-buckets E (lead / lines-drains-airways / skin / isolation /
        # positioning) so the STORED brief.json is canonical, not just the
        # reviewer-app display. Content-preserving (placement only); the
        # reviewer-app router re-parses the result idempotently.
        if merged_sections.get("E"):
            from icu_pause.rendering.e_section import consolidate_e_section
            merged_sections["E"] = consolidate_e_section(merged_sections["E"])

        # --- Normalize S format (fix models that output plain prose) ---
        if merged_sections.get("S"):
            merged_sections["S"] = _normalize_s_format(merged_sections["S"])

        # --- Build todo_checklist from structured rules + S-section extraction ---
        # Data layer keeps S narrative and todo_checklist as separate fields.
        # Render layer combines them visually into one S section.

        # (1) Generate deterministic todos from the rules engine
        structured_todos = generate_structured_todos(patient_data)

        # (2) Extract checkbox / bullet todos that the Intensivist wrote into S
        s_raw = merged_sections.get("S", "")
        narrative_s, extracted_todos = _extract_todos_from_s(s_raw)

        # (3) Set S to narrative-only content
        merged_sections["S"] = narrative_s

        # (3b) R6 §5.1 — Nephrotoxin med-review cross-block dedup. Drops
        # nephrotoxin / renal-dose-review to-dos from non-renal problems
        # so the AKI/CKD block is the etiologic owner. The "no renal
        # anchor present" branch promotes the action to the to-do
        # checklist instead of silently dropping. Reasoning-log
        # entries surface each action for audit.
        nephrotoxin_promoted_todos: list[str] = []
        if merged_sections["S"]:
            (
                merged_sections["S"],
                nephrotoxin_actions,
                nephrotoxin_promoted_todos,
            ) = _dedup_nephrotoxin_med_review_across_problems(
                merged_sections["S"]
            )
            for action in nephrotoxin_actions:
                logger.info(action)

        # (4) Deduplicate S problems
        if merged_sections["S"]:
            merged_sections["S"] = _deduplicate_s_problems(merged_sections["S"])

        # (4b) Order S problems by clinical priority (organ system)
        if merged_sections["S"]:
            merged_sections["S"] = _order_s_problems(merged_sections["S"])

        # (4c) R6 §5.2 — Electrolyte atomicity canary. Fires WARNs on
        # merged hyponatremia + hyperkalemia problems without explicit
        # AKI/renal attribution. Does NOT modify S — auto-splitting is
        # prompt territory; this is the regression detector.
        electrolyte_canary_warns: list[str] = []
        if merged_sections["S"]:
            electrolyte_canary_warns = (
                _warn_on_merged_electrolyte_problems(merged_sections["S"])
            )

        # (4d) §4.3.7 — AKI baseline reference post-render check. Reads
        # the pinned baseline value from scribe_extraction (when
        # validated) and asserts each AKI/CKD/Renal #Problem renders
        # either that value verbatim or an explicit "no baseline anchor
        # in source" fallback sentinel.
        aki_baseline_warns: list[str] = []
        if merged_sections["S"]:
            scribe_ex = state.get("scribe_extraction") or {}
            rc = (
                scribe_ex.get("renal_context") or {}
                if scribe_ex.get("renal_context_validated") else {}
            )
            pinned_baseline = rc.get("baseline_creatinine") if rc else None
            aki_baseline_warns = _check_aki_problem_has_baseline(
                merged_sections["S"],
                pinned_baseline_value=pinned_baseline,
            )

        # Accumulate canary WARNs on state for the drift-metric module
        # (next chunk) to ingest. Logged as warnings here for immediate
        # visibility in trace events.
        if electrolyte_canary_warns or aki_baseline_warns:
            emissions = state.setdefault("safety_drift_emissions", {})
            warns_emit = emissions.setdefault("warns", {})
            for w in electrolyte_canary_warns:
                logger.warning(w)
                warns_emit["ELECTROLYTE_PROBLEM_MERGED_WITHOUT_ATTRIBUTION"] = True
            for w in aki_baseline_warns:
                logger.warning(w)
                warns_emit["AKI_PROBLEM_MISSING_BASELINE_REFERENCE"] = True

        # (5) Combine extracted + structured + nephrotoxin-promoted todos
        combined_todos = (
            extracted_todos + structured_todos + nephrotoxin_promoted_todos
        )

        # (6) Exact dedup
        seen: set[str] = set()
        deduped_todos: list[str] = []
        for item in combined_todos:
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                deduped_todos.append(item)

        # (7) Classify into temporal buckets → todo_checklist
        todo_checklist: list[dict[str, str]] = []
        for item in deduped_todos:
            bucket = _classify_todo_bucket(item)
            todo_checklist.append({"bucket": bucket, "text": item})

        # Competing-risks indication grounding validator (v1.8) — runs
        # BEFORE warning collection so any rewrites / new qa_process WARNs
        # flow through the dedup pipeline downstream. Mutates
        # intensivist_output.reasoning_log.competing_risks_check and
        # intensivist_output.warnings in place when a citation fails to
        # resolve. See docs/competing_risks_indication_grounding_v1.8_design.md.
        cr_validation_warnings: list[Warning] = []
        if intensivist_output is not None:
            trace_events = state.get("trace_events", []) or []
            # Capture the model's raw competing_risks_check entries BEFORE
            # the validator mutates them in place. This audit trail lets us
            # distinguish "model self-hedged" from "validator rewrote
            # model's citation to hedge" — different failure modes with
            # different fixes. Without the raw snapshot, the post-validator
            # state is the only thing that survives, and the model's
            # original behavior is lost.
            raw_entries: list[dict[str, Any]] = []
            if (intensivist_output.reasoning_log
                    and intensivist_output.reasoning_log.competing_risks_check):
                raw_entries = [
                    e.model_dump() if hasattr(e, "model_dump") else dict(e)
                    for e in intensivist_output.reasoning_log.competing_risks_check
                ]
            cr_validation_warnings = validate_competing_risks_grounding(
                intensivist_output, trace_events,
                notes_by_id=_notes_by_id_from_agent_context(
                    state.get("agent_context_text", {}) or {}
                ),
            )
            final_entries: list[dict[str, Any]] = []
            if (intensivist_output.reasoning_log
                    and intensivist_output.reasoning_log.competing_risks_check):
                final_entries = [
                    e.model_dump() if hasattr(e, "model_dump") else dict(e)
                    for e in intensivist_output.reasoning_log.competing_risks_check
                ]
            if raw_entries or final_entries:
                trace_events.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "competing_risks_validated",
                    "node": "orchestrator",
                    "level": "info",
                    "message": (
                        f"competing_risks_check: {len(raw_entries)} raw -> "
                        f"{len(final_entries)} final; "
                        f"{len(cr_validation_warnings)} qa_process WARN(s)"
                    ),
                    "data": {
                        "competing_risks_check_raw": raw_entries,
                        "competing_risks_check": final_entries,
                        "qa_warns_emitted": [
                            w.message for w in cr_validation_warnings
                        ],
                    },
                })

        # One-liner PMH selection validator (v2.0) — runs the four
        # alignment checks against the rendered Section I lead sentence
        # and the scribe-pinned PMH paragraph:
        #   1. source_clause_anchor in scribe.pmh
        #   2. display ↔ Section I lead sentence normalized-form match
        #   3. modifier_confirmation completeness for time-sensitive
        #      modifiers in display
        #   4. modifier_confirmation grounding (note resolution +
        #      substring quote match)
        # Lint-only — does NOT rewrite Section I or drop pins. Appends
        # qa_process WARNs to intensivist_output.warnings for reviewer-
        # panel surfacing. See feedback_field_conflation_render.md +
        # project_icu_pause_oneliner_pmh_criterion.md.
        oneliner_pmh_warnings: list[Warning] = []
        if intensivist_output is not None:
            raw_oneliner_pmh = [
                e.model_dump() if hasattr(e, "model_dump") else dict(e)
                for e in (
                    getattr(intensivist_output, "one_liner_pmh_selection", [])
                    or []
                )
            ]
            oneliner_pmh_warnings = validate_one_liner_pmh_selection(
                intensivist_output,
                state,
                trace_events,
                notes_by_id=_notes_by_id_from_agent_context(
                    state.get("agent_context_text", {}) or {}
                ),
            )
            if raw_oneliner_pmh or oneliner_pmh_warnings:
                trace_events.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": "one_liner_pmh_validated",
                    "node": "orchestrator",
                    "level": "warn" if oneliner_pmh_warnings else "info",
                    "message": (
                        f"one_liner_pmh_selection: "
                        f"{len(raw_oneliner_pmh)} entries; "
                        f"{len(oneliner_pmh_warnings)} qa_process WARN(s)"
                    ),
                    "data": {
                        "one_liner_pmh_selection": raw_oneliner_pmh,
                        "qa_warns_emitted": [
                            w.message for w in oneliner_pmh_warnings
                        ],
                    },
                })

        # ── Intensivist PMH fallback (scribe pin empty) — render hand-off + #12.
        pmh_fb = getattr(intensivist_output, "pmh_fallback", None) if intensivist_output else None
        _fb_event = apply_pmh_fallback_render(merged_sections, pmh_fb)
        if _fb_event is not None:
            _fb_event["timestamp"] = datetime.now(timezone.utc).isoformat()
            trace_events.append(_fb_event)

        # Collect all warnings (domain agents + intensivist + resident conflicts +
        # citation checks) as structured Warning objects. Categories are assigned
        # at the producer boundary (see base.wrap_llm_snippet, intensivist
        # wrap_llm_intensivist) so the routing decision is centralized: clinician
        # panel filters by category at render time, while the audit log retains
        # everything. Dedup is by (category, message) — collapses identical
        # entries without the substring blocklist that previously hid genuine
        # safety signals.
        all_warnings: list[Warning] = []
        seen: set[tuple[str, str]] = set()

        def _add(w: Warning) -> None:
            key = (w.category.value, w.message.strip().lower())
            if key in seen:
                return
            seen.add(key)
            all_warnings.append(w)

        for snippet in effective_snippets:
            for w in snippet.warnings:
                _add(w)
        if intensivist_output and intensivist_output.warnings:
            for w in intensivist_output.warnings:
                _add(w)
        for w in cr_validation_warnings:
            _add(w)

        # Fold Resident cross-domain conflicts into the warning stream as
        # CROSS_DOMAIN_CONFLICT. The Resident already classified severity as
        # safety_critical | clinical | logistical; map directly.
        resident_brief = state.get("resident_pre_brief")
        _CONFLICT_SEVERITY_MAP = {
            ConflictSeverity.SAFETY_CRITICAL.value: WarningSeverity.SAFETY_CRITICAL,
            ConflictSeverity.CLINICAL.value: WarningSeverity.CLINICAL,
            ConflictSeverity.LOGISTICAL.value: WarningSeverity.LOGISTICAL,
        }
        if resident_brief and isinstance(resident_brief, dict):
            for conflict in resident_brief.get("cross_domain_conflicts", []) or []:
                if not isinstance(conflict, dict):
                    continue
                sev_str = conflict.get("severity", "clinical")
                severity = _CONFLICT_SEVERITY_MAP.get(
                    sev_str, WarningSeverity.CLINICAL,
                )
                domain_a = conflict.get("domain_a", "?")
                domain_b = conflict.get("domain_b", "?")
                desc = conflict.get("conflict_description", "")
                relevant = conflict.get("relevant_sections") or []
                _add(Warning(
                    category=WarningCategory.CROSS_DOMAIN_CONFLICT,
                    severity=severity,
                    message=f"{domain_a} vs {domain_b}: {desc}",
                    source_agent="resident",
                    source_section=(relevant[0] if relevant else None),
                ))

        # --- Citation verification ---
        # Provenance failures (cite tag in output that doesn't exist in the
        # registry) are evidence of fabrication — route to SAFETY_FLAG so the
        # clinician sees them. Preservation drops (Intensivist stripped a tag
        # the domain agent emitted) are bookkeeping integrity checks — route
        # to QA_PROCESS so they only show in dev mode.
        cite_registry = state.get("cite_registry", {})
        if cite_registry:
            provenance_issues = check_citation_provenance(merged_sections, cite_registry)
            for issue in provenance_issues:
                _add(Warning(
                    category=WarningCategory.SAFETY_FLAG,
                    severity=WarningSeverity.CLINICAL,
                    message=issue,
                    source_agent="orchestrator",
                ))
            preservation_issues = check_citation_preservation(
                effective_snippets, merged_sections,
            )
            for issue in preservation_issues:
                _add(Warning(
                    category=WarningCategory.QA_PROCESS,
                    severity=WarningSeverity.INFO,
                    message=issue,
                    source_agent="orchestrator",
                ))
        else:
            provenance_issues = []
            preservation_issues = []
        citation_metadata = {
            "provenance_issues": len(provenance_issues),
            "preservation_dropped": len(preservation_issues),
            "dropped": preservation_issues[:20],
        }

        # Per-category counts for diagnostics + paper-relevant metrics.
        warnings_by_category: dict[str, int] = {}
        for w in all_warnings:
            warnings_by_category[w.category.value] = (
                warnings_by_category.get(w.category.value, 0) + 1
            )
        clinician_facing_count = sum(
            1 for w in all_warnings
            if w.category in (
                WarningCategory.SAFETY_FLAG,
                WarningCategory.CROSS_DOMAIN_CONFLICT,
                WarningCategory.DATA_GAP,
                WarningCategory.DETERMINISTIC_OVERRIDE,
            )
        )
        logger.info(
            f"Warnings: {len(all_warnings)} total, "
            f"{clinician_facing_count} clinician-facing, "
            f"by category: {warnings_by_category}"
        )

        # --- Citation index for renderer tooltips ---
        # Order is intentional: filter registry to tags actually referenced in
        # the final text first, then anything in text but missing from the
        # filtered index is flagged ``unverified`` by build_citation_index.
        from icu_pause.tools.citation_index import build_citation_index
        citation_index = build_citation_index(merged_sections, cite_registry)

        # Compute metadata
        sections_filled = sum(
            1 for v in merged_sections.values() if v != NOT_ENOUGH_INFO
        )

        # --- Output length per section (word count) ---
        section_word_counts = {
            key: len(content.split()) if isinstance(content, str) and content != NOT_ENOUGH_INFO else 0
            for key, content in merged_sections.items()
            if not key.startswith("_")  # skip internal metadata keys
        }
        total_word_count = sum(section_word_counts.values())

        # --- Section-level content hash: detect intensivist rewrites ---
        import hashlib
        intensivist_rewrite_rate = {}
        if intensivist_output:
            for sec in intensivist_output.sections:
                int_hash = hashlib.md5(sec.content.encode()).hexdigest()[:8]
                # Find matching domain agent content for this section
                domain_content = ""
                for snippet in effective_snippets:
                    for agent_sec in snippet.sections:
                        if agent_sec.section == sec.section:
                            domain_content = agent_sec.content
                            break
                    if domain_content:
                        break
                if domain_content:
                    dom_hash = hashlib.md5(domain_content.encode()).hexdigest()[:8]
                    intensivist_rewrite_rate[sec.section] = "rewritten" if int_hash != dom_hash else "verbatim"
                else:
                    intensivist_rewrite_rate[sec.section] = "new"  # Intensivist-owned (I, P)

        # --- To-do specificity: flag generic to-dos ---
        generic_todos = []
        specific_todos = []
        for todo in todo_checklist:
            todo_text = todo["text"]
            if any(p in todo_text.lower() for p in _GENERIC_PATTERNS):
                generic_todos.append(todo_text)
            else:
                specific_todos.append(todo_text)
        todo_specificity_rate = (
            len(specific_todos) / len(todo_checklist) if todo_checklist else 1.0
        )

        # --- Prompt truncation tracking ---
        truncated_agents = [
            m.get("agent", "?") for m in state.get("pipeline_metrics", [])
            if m.get("input_tokens", 0) >= 30000  # Near 32k limit
        ]

        # Summarize pipeline metrics
        all_metrics = state.get("pipeline_metrics", [])
        total_input = sum(m.get("input_tokens", 0) for m in all_metrics)
        total_output = sum(m.get("output_tokens", 0) for m in all_metrics)
        total_latency = sum(m.get("latency_ms", 0) for m in all_metrics)

        output = ICUPauseOutput(
            hospitalization_id=state.get("hospitalization_id", "unknown"),
            generated_at=datetime.now(timezone.utc).isoformat(),
            sections=merged_sections,
            todo_checklist=todo_checklist,
            warnings=all_warnings,
            qa_issues=qa_issues,
            section_confidences=section_confidences,
            metadata={
                # Reviewer-facing source data: union of the per-agent slices
                # AFTER per-agent caps (AGENT_MAX_NOTES_PER_TYPE) and routing
                # (_AGENT_DATA_KEYS) have been applied.  Every record in this
                # union was seen by at least one agent — the correct denominator
                # for document-level omission/hallucination judgments.  Falls
                # back to the uncapped patient_data only if agent_context_text
                # is empty (e.g. legacy state shapes), so reviewers never get
                # zero source data.
                "source_data": (
                    union_post_cap_contexts(agent_context_text)
                    if agent_context_text else patient_data
                ),
                # Per-agent slices preserved for audit / future per-claim
                # provenance work.  Mirrors the in-flight workflow state.
                "agent_source_data": agent_context_text,
                "agent_count": len(effective_snippets) + (1 if intensivist_output else 0),
                "intensivist_sections": len(intensivist_sections),
                "sections_filled": sections_filled,
                "sections_total": len(SECTION_ORDER),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_latency_ms": round(total_latency, 1),
                "per_agent_metrics": all_metrics,
                "deliberation": {
                    "enabled": bool(revised_agents),
                    "agents_revised": sorted(revised_agents),
                    "issues_addressed": len(delib_log),
                    "log": delib_log,
                },
                "warnings_total": len(all_warnings),
                "warnings_clinician_facing": clinician_facing_count,
                "warnings_by_category": warnings_by_category,
                "section_word_counts": section_word_counts,
                "total_word_count": total_word_count,
                "intensivist_rewrite_rate": intensivist_rewrite_rate,
                "todo_specificity": {
                    "total": len(todo_checklist),
                    "specific": len(specific_todos),
                    "generic": len(generic_todos),
                    "specificity_rate": round(todo_specificity_rate, 2),
                    "generic_items": generic_todos,
                },
                "truncated_agents": truncated_agents,
                "citation_preservation": citation_metadata,
                "citation_index": citation_index,
                # Per-agent physician-note floor metadata (intensivist,
                # respiratory, pharmacy, dietitian). Each entry carries
                # floor_applied, reason, primary_team, floor_note_id,
                # floor_note_type, floor_specialty, floor_age_hours so the
                # validation cohort can audit floor activation rates and
                # the ratio of named-team vs. broad-physician fallbacks.
                "physician_floor": state.get("physician_floor", {}),
            },
        )

        logger.info(
            f"SectionMerger: {sections_filled}/{len(SECTION_ORDER)} sections filled, "
            f"{len(todo_checklist)} to-do items, {len(qa_issues)} QA issues"
        )

        # Per-brief safety-drift metrics emission (§8 of renal/
        # electrolyte/VTE design). Non-fatal: emission must never break
        # a pipeline run. Logged with traceback (exc_info=True) so a
        # structural bug — every brief silently failing → denominator
        # collapse in the sidecar rollup — is diagnosable from a single
        # brief's log without re-running.
        try:
            drift_record = build_safety_drift_record(
                state=state,
                merged_sections=merged_sections,
                context=patient_data,
                hospitalization_id=output.hospitalization_id,
                reference_dttm=(
                    (patient_data.get("demographics") or {}).get(
                        "reference_dttm"
                    )
                ),
            )
            emit_safety_drift_record(drift_record)
        except Exception:  # pragma: no cover — guard only
            logger.warning(
                "safety_drift emission failed", exc_info=True
            )

        return {"icu_pause_output": output.model_dump()}

    @staticmethod
    def _enforce_dotphrase_template(
        sections: dict[str, str], patient_data: dict[str, Any]
    ) -> None:
        """Post-process merged sections to enforce dotphrase template structure.

        Adds deterministic safety fields that must always be present regardless
        of LLM output quality. Modifies sections dict in place.
        """
        # --- E section: ensure safety checkboxes have an answer ---
        # Two distinct failure modes to handle:
        #   1. Line is entirely absent from E → append with deterministic default
        #   2. Line is present but unanswered (e.g. "Difficult airway? [Y/N]" or
        #      bare "Difficult airway?") → REPLACE in place. Otherwise the
        #      claim extractor pulls the unanswered template into Step 1 and
        #      asks the reviewer to verify a question instead of a claim
        #      (observed 2026-05-08 in a round-3 brief).
        e_content = sections.get("E", "")
        safety_lines = []

        # Difficult airway? — deterministic from respiratory data
        respiratory = patient_data.get("respiratory", {})
        has_trach = False
        if isinstance(respiratory, dict):
            for key, values in respiratory.items():
                if isinstance(values, list):
                    for v in values:
                        if isinstance(v, dict):
                            trach_val = v.get("tracheostomy", "")
                            device = v.get("device_name", "") or v.get("device_category", "")
                            if str(trach_val).lower() in ("true", "1", "yes") or "trach" in str(device).lower():
                                has_trach = True
                                break

        difficult_airway_replacement = (
            "Difficult airway? Yes — tracheostomy in place"
            if has_trach
            else "Difficult airway? ☐"
        )
        # Match "Difficult airway?" optionally followed by an unfilled
        # placeholder ([Y/N], [ ], [_], etc.) and nothing else on the line.
        # Lines that already carry a real answer (Y/N/Yes/No/☐/☑/etc.) don't
        # match and are left alone.
        unanswered_airway_re = re.compile(
            r"(?im)^[ \t]*Difficult\s+airway\?\s*(?:\[[^\]]*\])?\s*$"
        )
        e_content_new, n_airway = unanswered_airway_re.subn(
            difficult_airway_replacement, e_content
        )
        if n_airway:
            e_content = e_content_new
        elif "ifficult airway" not in e_content.lower():
            safety_lines.append(difficult_airway_replacement)

        unanswered_lines_re = re.compile(
            r"(?im)^[ \t]*Lines?/drains?\s+assessed\s+for\s+removal\?\s*(?:\[[^\]]*\])?\s*$"
        )
        e_content_new, n_lines = unanswered_lines_re.subn(
            "Lines/drains assessed for removal? ☐", e_content
        )
        if n_lines:
            e_content = e_content_new
        elif "ines/drains assessed" not in e_content.lower():
            safety_lines.append("Lines/drains assessed for removal? ☐")

        if safety_lines or n_airway or n_lines:
            sections["E"] = (
                e_content.rstrip()
                + (("\n" + "\n".join(safety_lines)) if safety_lines else "")
            )

        # --- U_unprescribing: normalize checkbox format ---
        u_content = sections.get("U_unprescribing", "")
        if u_content:
            u_content = u_content.replace("[x]", "☑").replace("[X]", "☑")
            u_content = u_content.replace("[ ]", "☐")
            sections["U_unprescribing"] = u_content

        # --- C section: deterministic vent-dependent status ---
        c_content = sections.get("C", "")
        respiratory = patient_data.get("respiratory", [])
        vent_status = _determine_vent_status(respiratory)
        # Replace any LLM-generated vent-dependent line with deterministic value
        import re as _re
        c_lines = []
        for line in c_content.split("\n"):
            if _re.search(r"(?i)vent.?dependent|currently vent", line):
                continue  # drop LLM-generated vent line
            c_lines.append(line)
        c_content = "\n".join(c_lines).rstrip()
        c_content += f"\nCurrently vent-dependent: {vent_status}"

        # Append Goals of Care / ACP field if missing
        if (
            "ACP" not in c_content
            and "advance care" not in c_content.lower()
            and "goals of care" not in c_content.lower()
        ):
            c_content += "\nGoals of Care / ACP: Not documented"
        sections["C"] = c_content

        # --- S section: normalize format + content filtering ---
        s_content = sections.get("S", "")
        # Guard: coerce list to string (LLM sometimes returns S as a list)
        if isinstance(s_content, list):
            s_content = "\n".join(str(item) for item in s_content)
            sections["S"] = s_content
        if s_content:
            # `re` is already imported at module scope (orchestrator.py:6).
            # Do NOT re-import inside this function — that creates a function-
            # local binding which shadows module-level re for the WHOLE
            # function and breaks earlier `re.compile(...)` calls in this
            # method (UnboundLocalError, observed 2026-05-08).
            # Split inline "[]" that follows "#Problem" on same line
            s_content = re.sub(r'(\#[^#\[\]]+)\s*\[\]', r'\1\n[]', s_content)
            # Fix "/" separators: "#Respiratory/[]Monitor" → "#Respiratory\n[] Monitor"
            s_content = re.sub(r'(\#[^#/\[\]]+)/\[\]', r'\1\n[]', s_content)
            # Strip trailing "/" from header names (e.g., "Respiratory/" → "Respiratory").
            # Anchored to end-of-line: without $ the greedy match strips internal slashes,
            # e.g. "#Delirium/sedation recovery" → "#Deliriumsedation recovery" and units
            # like "mL/min" collapse. Non-greedy + MULTILINE $ keeps the fix scoped to
            # actual trailing slashes only.
            s_content = re.sub(r'(#[^\n]*?)/+\s*$', r'\1', s_content, flags=re.MULTILINE)
            # Ensure [] has a space after it
            s_content = re.sub(r'\[\]([^\s])', r'[] \1', s_content)

            # Filter out generic monitoring and medication items
            filtered_lines = []
            generic_patterns = [
                r"monitor\s+(respiratory|GCS|RASS|MAP|vital|hemodynamic)",
                r"continue\s+monitoring",
                r"adjust\s+(ventilator|vasoactive|oxygen)",
            ]
            med_names = [
                "propofol", "fentanyl", "venlafaxine", "midazolam",
                "dexmedetomidine", "ketamine", "rocuronium", "cisatracurium",
            ]
            for line in s_content.split("\n"):
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                # Skip generic monitoring lines
                if any(re.search(p, lower) for p in generic_patterns):
                    continue
                # Skip medication-named items (belong in U_unprescribing)
                if stripped.startswith("#") and any(med in lower for med in med_names):
                    continue
                filtered_lines.append(line)

            sections["S"] = "\n".join(filtered_lines)

        # Note: S dedup and todo extraction now happen in the main merge()
        # method after _enforce_dotphrase_template() returns. This avoids
        # the dual-write bug where both this path and the injection block
        # independently wrote checkbox items into S.

        # --- U_uncertainty: enforce diagnostic pause format ---
        u_unc = sections.get("U_uncertainty", "")
        if u_unc:
            sections["U_uncertainty"] = _enforce_u_uncertainty(u_unc)

        # --- P section: ensure not empty ---
        p_content = sections.get("P", "")
        if not p_content or not p_content.strip():
            sections["P"] = "None"

        # --- A section: ensure rehab checkboxes present ---
        a_content = sections.get("A", "")
        if a_content:
            rehab_services = {
                "PT": "☐ PT",
                "OT": "☐ OT",
                "SLP": "☐ SLP",
                "Wound Care": "☐ Wound Care",
            }
            for svc, checkbox in rehab_services.items():
                # Check if service is mentioned (either checked or unchecked)
                if svc not in a_content:
                    a_content = a_content.rstrip() + f"\n{checkbox}"
            sections["A"] = a_content

        # --- C section: ensure DPOA field present ---
        c_final = sections.get("C", "")
        if c_final and "DPOA" not in c_final and "power of attorney" not in c_final.lower():
            c_final = c_final.rstrip() + "\nDPOA: Not documented"
            sections["C"] = c_final

    @staticmethod
    def _field_coverage_score(contribution: Any) -> float:
        """Compute a deterministic coverage score for a section contribution.

        Coverage = number of data sources the agent cited as used.  This is
        deterministic (based on which structured data keys the agent consumed)
        and does not rely on LLM self-reported confidence.

        A higher score means the agent had more data available to inform its
        contribution, making it the more authoritative fallback choice.
        """
        sources = getattr(contribution, "data_sources_used", [])
        return float(len(sources))

    @staticmethod
    def _extract_todos(summary_text: str) -> list[str]:
        """Extract actionable to-do items from the Summary section."""
        if not summary_text or summary_text == NOT_ENOUGH_INFO:
            return []

        # Headers/labels that contain "to-do" but are NOT actionable items
        _TODO_HEADER_PATTERNS = [
            "to-do list", "to-do's", "todos:", "to-do:", "to do list",
            "prior to transfer", "action items:",
        ]

        todos = []
        for line in summary_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()

            # Skip section headers that mention "to-do" (e.g., "To-do list prior to transfer:")
            if any(p in lower for p in _TODO_HEADER_PATTERNS):
                continue
            # Skip lines that are just a header marker (start with # or end with : only)
            if stripped.startswith("#") or (stripped.endswith(":") and len(stripped.split()) <= 6):
                continue
            # Skip temporal bucket headers (from _bucket_todos_temporally)
            if lower.startswith("☐ ") and lower.endswith(":"):
                continue

            # Match bullet points, checkbox markers, numbered items, or "To-do" markers
            if (
                stripped.startswith(("-", "*", "•", "☐"))
                or stripped.startswith("[]")
                or stripped.startswith("  - ")  # indented bullet (from temporal bucketing)
                or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
                or "to-do" in lower
                or "todo" in lower
                or "action:" in lower
            ):
                cleaned = stripped.lstrip("-*•☐0123456789.) ").lstrip("[] ").strip()
                if cleaned and len(cleaned) > 5:
                    todos.append(cleaned)
        return todos

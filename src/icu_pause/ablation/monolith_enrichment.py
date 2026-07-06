"""Deterministic clinical enrichment for the single-agent (monolith) arm.

Runs the SAME deterministic (no-LLM) tools the multiagent pipeline uses — DDI,
device dwell, lab reference ranges — on the shared bundle and formats the results
as a text block injected into the monolith's single prompt. This holds the
tool/data layer constant across arms, so the single-vs-multi contrast isolates
GENERATION architecture (one call vs. routed agents), not tool access. These are
deterministic lookups, not agents, so injecting their output keeps the arm a true
single agent (one LLM call).

Each tool is best-effort: a failure is logged and skipped — enrichment never
blocks generation. Same call shapes the QA agent uses (qa.py).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_enrichment_block(
    patient_data: dict[str, Any],
    reference_dttm: Any = None,
    icu_admission_dttm: Any = None,
    allow_network: bool = False,
    timeout: float = 5.0,
) -> str:
    """Return a formatted '=== DETERMINISTIC CLINICAL ENRICHMENT ===' block, or ''.

    patient_data is the shared serialized bundle (same dict the pipeline agents
    and the monolith receive). Mirrors qa.py's tool calls.
    """
    if not isinstance(patient_data, dict):
        return ""
    parts: list[str] = []

    # --- DDI (drug_interactions.check_interactions) ---
    try:
        meds = patient_data.get("meds", {})
        if meds:
            from icu_pause.tools.drug_interactions import check_interactions
            res = check_interactions(meds, allow_network=allow_network,
                                     timeout=timeout, reference_dttm=reference_dttm)
            ix = getattr(res, "interactions", None) or []
            if ix:
                lines = [f"  - {i.drug_a} + {i.drug_b}: {i.description} "
                         f"(severity: {i.severity})" for i in ix]
                parts.append("DRUG-DRUG INTERACTIONS DETECTED:\n" + "\n".join(lines))
    except Exception as e:  # noqa: BLE001
        logger.warning("monolith enrichment: DDI skipped (%s)", e)

    # --- device dwell (device_dwell.check_device_dwell) ---
    try:
        procs = patient_data.get("procedures", [])
        if procs and reference_dttm:
            from icu_pause.tools.device_dwell import check_device_dwell
            res = check_device_dwell(procs, reference_dttm, icu_admission_dttm)
            flags = getattr(res, "flags", None) or []
            if flags:
                lines = [f"  - {f.device_type.replace('_', ' ').title()}: "
                         f"{f.dwell_days}d in place (threshold {f.threshold_days}d, "
                         f"{f.severity}). {f.recommended_action}" for f in flags]
                parts.append("DEVICE DWELL FLAGS:\n" + "\n".join(lines))
    except Exception as e:  # noqa: BLE001
        logger.warning("monolith enrichment: device_dwell skipped (%s)", e)

    # --- lab reference ranges (lab_ranges.check_lab_ranges) ---
    try:
        labs = patient_data.get("labs", [])
        if labs:
            from icu_pause.tools.lab_ranges import check_lab_ranges
            cc = patient_data.get("clinical_context")
            if isinstance(cc, dict):
                from icu_pause.safety.clinical_context import PatientClinicalContext
                cc = PatientClinicalContext.from_dict(cc)
            res = check_lab_ranges(labs, "", clinical_context=cc)
            flags = getattr(res, "flags", None) or []
            if flags:
                lines = []
                for f in flags:
                    name = getattr(f, "lab_name", None) or getattr(f, "name", "?")
                    val = getattr(f, "value", "?")
                    status = getattr(f, "status", "?")
                    lines.append(f"  - {name}: {val} ({status})")
                parts.append("ABNORMAL / CRITICAL LABS:\n" + "\n".join(lines))
    except Exception as e:  # noqa: BLE001
        logger.warning("monolith enrichment: lab_ranges skipped (%s)", e)

    if not parts:
        return ""
    return ("=== DETERMINISTIC CLINICAL ENRICHMENT (computed by the same pipeline "
            "tools; weave the relevant items into the appropriate sections, with "
            "citations) ===\n" + "\n\n".join(parts))

"""Top-3 prioritization concordance — secondary metric.

Question: does each arm's brief surface the SAME active acute problems the
clinician prioritized in the contemporaneous human transfer note? This is where
section-owned decomposition should pay off (vs. raw numeric fidelity, which a
monolith matches). NOT whole-list Jaccard — it's recall of the clinician's top-3.

Ground truth = the human ``progress_note`` whose creation_dttm IS ``reference_dttm``
— the ICU→ward note picked to define transfer time. This site has no
``transfer_note`` type; the reference is the progress note at the transfer
timestamp. The pipeline's strict-< leakage guard excludes the note AT
reference_dttm, so it's a valid held-out reference (the brief never saw it). We
recover it by loading progress_note WITHOUT the leakage filter and matching on
the reference timestamp.

Pipeline (LLM judge; model is pluggable — local Gemma for screening, GPT-o3-mini
for the publication number):
  1. Extract the clinician's top-3 active acute problems from the human note (once/case).
  2. Per arm: judge how many of those 3 the brief clearly addresses.
  3. Concordance = matched / 3, averaged over cases.

Parsing is line-based (PROBLEM: / 1: YES|NO), not JSON — robust on local models.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from icu_pause.llm.provider import create_llm

logger = logging.getLogger(__name__)

# No "transfer_note" type at this site — the reference is the progress_note at
# reference_dttm (the ICU→ward note that defined transfer time).
_REF_NOTE_TYPE = "progress_note"


# ---------------------------------------------------------------------------
# Reference-note retrieval
# ---------------------------------------------------------------------------
def retrieve_reference_note(retriever, hid: str, reference_dttm) -> str | None:
    """Return the human transfer note's text at/near ``reference_dttm``.

    Loads the transfer_note WITHOUT the leakage guard (reference_dttm=None) so the
    note *at* transfer time — normally excluded — is recovered, then picks the row
    nearest reference_dttm.
    """
    try:
        df = retriever._load_notes_for_hospitalization(_REF_NOTE_TYPE, hid, None, None)
    except Exception as e:  # noqa: BLE001
        logger.warning("reference-note load failed for %s: %s", hid, e)
        return None
    if df is None or len(df) == 0 or "note_text" not in df.columns:
        return None

    rows = df.to_dicts()
    ref_utc = None
    try:
        ref_utc = retriever._to_utc(reference_dttm) if reference_dttm else None
    except Exception:  # noqa: BLE001
        ref_utc = None

    def _dist(r):
        cd = r.get("creation_dttm")
        if ref_utc is None or cd is None:
            return 0.0
        try:
            from datetime import datetime
            t = cd if hasattr(cd, "year") else datetime.fromisoformat(str(cd).replace("Z", "+00:00"))
            return abs((t - ref_utc).total_seconds())
        except Exception:  # noqa: BLE001
            return 0.0

    best = min(rows, key=_dist)
    logger.info("reference note %s: matched creation_dttm=%s of %d %s candidates",
                hid, best.get("creation_dttm"), len(rows), _REF_NOTE_TYPE)
    txt = best.get("note_text")
    return str(txt) if txt else None


# ---------------------------------------------------------------------------
# LLM-judge steps
# ---------------------------------------------------------------------------
_EXTRACT_SYS = (
    "You are a critical-care physician reviewing an ICU→ward transfer note. "
    "Identify the active acute problems the ICU team is prioritizing at transfer."
)
_EXTRACT_USER = (
    "From the transfer note below, list the TOP 3 active acute medical problems "
    "the clinician is prioritizing for the ward team — the issues most central to "
    "ongoing management. Be specific (e.g. 'septic shock from pneumonia', not "
    "'infection'). Output exactly three lines, each starting 'PROBLEM: ', nothing "
    "else.\n\n=== TRANSFER NOTE ===\n{note}\n"
)

_COVER_SYS = (
    "You are auditing whether an AI-generated ICU transfer brief surfaces the "
    "clinician's priority problems. Judge only whether each problem is clearly "
    "addressed; do not reward vague mentions."
)
_COVER_USER = (
    "Clinician's top-3 priority problems:\n{problems}\n\n"
    "Does the AI brief below clearly address each one? For each numbered problem "
    "answer on its own line as '<n>: YES' or '<n>: NO' (YES only if the brief "
    "substantively addresses that specific problem). Output exactly three lines.\n\n"
    "=== AI BRIEF ===\n{brief}\n"
)

_PROBLEM_RE = re.compile(r"^\s*PROBLEM:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_YESNO_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(YES|NO)\b", re.IGNORECASE | re.MULTILINE)


def extract_priorities(llm, note_text: str, k: int = 3) -> list[str]:
    if not note_text:
        return []
    resp = llm.invoke(_EXTRACT_SYS, _EXTRACT_USER.format(note=note_text[:12000]),
                      response_format=None)
    text = resp if isinstance(resp, str) else str(resp)
    probs = [m.strip() for m in _PROBLEM_RE.findall(text)]
    if not probs:  # fallback: non-empty lines
        probs = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    return probs[:k]


def judge_coverage(llm, priorities: list[str], brief_text: str) -> list[bool]:
    if not priorities:
        return []
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(priorities, 1))
    resp = llm.invoke(_COVER_SYS,
                      _COVER_USER.format(problems=numbered, brief=brief_text[:14000]),
                      response_format=None)
    text = resp if isinstance(resp, str) else str(resp)
    verdict = {int(n): (yn.upper() == "YES") for n, yn in _YESNO_RE.findall(text)}
    return [verdict.get(i, False) for i in range(1, len(priorities) + 1)]


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------
def score_case_prioritization(judge_llm, reference_note: str,
                              brief_text: str, priorities: list[str] | None = None
                              ) -> dict[str, Any]:
    """Concordance for one (brief, human-note) pair.

    ``priorities`` can be passed in to reuse the once-per-case extraction across
    arms (the human note is identical for every arm of a case).
    """
    if priorities is None:
        priorities = extract_priorities(judge_llm, reference_note)
    if not priorities:
        return {"priorities": [], "covered": [], "matched": 0, "of": 0, "concordance": None}
    covered = judge_coverage(judge_llm, priorities, brief_text)
    matched = sum(1 for c in covered if c)
    return {
        "priorities": priorities,
        "covered": covered,
        "matched": matched,
        "of": len(priorities),
        "concordance": matched / len(priorities),
    }

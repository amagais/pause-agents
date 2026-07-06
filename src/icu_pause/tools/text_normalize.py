"""Shared text-normalization helpers for chart-quote substring validators.

Used by validators that substring-match LLM-emitted quotes against chart
note bodies — the competing-risks indication grounding validator
(orchestrator), and (anticipated, per admission_antibiotics_design.md) the
admission_antibiotics scribe validator.

Chart exports routinely contain NBSP (U+00A0), curly quotes (U+2018/2019/
201C/201D), en-dash (U+2013), em-dash (U+2014), and unicode horizontal
ellipsis (U+2026). Without normalization, an LLM-emitted quote like
"cefepime (3/3 - 3/5)" (with hyphen) won't substring-match a body that
renders "cefepime (3/3 – 3/5)" (with en-dash), and validators will
false-reject and drive the model toward over-hedging.
"""

from __future__ import annotations

import re
import unicodedata


_PUNCT_NORMALIZATION = str.maketrans({
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "–": "-",  # en-dash
    "—": "-",  # em-dash
    " ": " ",  # NBSP
})


def normalize_for_validator(s: str) -> str:
    """NFKC + whitespace collapse + curly-quote/dash normalization + lowercase.

    The validator substring check runs on the output of this helper for
    both sides (quote and body) so the comparison is symmetric.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_PUNCT_NORMALIZATION)
    s = " ".join(s.split())
    return s.lower()


def has_truncation_marker(s: str) -> bool:
    """True if the string contains ASCII ellipsis ('...') or unicode
    horizontal ellipsis ('…'). Used by competing-risks indication
    grounding validator to reject quotes the LLM truncated under
    schema fill pressure.
    """
    return "..." in s or "…" in s


# ---------------------------------------------------------------------------
# PMH-specific normalization for the Section I one-liner ↔ structured-pin
# alignment lint. Used by the orchestrator to match
# OneLinerPMHEntry.display against the rendered lead-sentence prose.
#
# Without abbreviation expansion, raw-string matching false-fails on routine
# medical abbreviation pairs that the intensivist may use interchangeably
# between the lead sentence and the structured display field (e.g., model
# emits "metastatic BrCa" in prose and "metastatic breast cancer" in
# display). The dictionary is bidirectional — normalization expands the
# short form to the canonical long form on BOTH sides of the match.
#
# Dictionary scope: high-frequency PMH/oncology/cardiology/respiratory
# abbreviations seen in the icu_pause pilot corpus. Add as needed; the
# guiding principle is "abbreviation pairs that a clinician would read as
# the same concept." Drug names go in lowercase form because case
# normalization runs before expansion.
# ---------------------------------------------------------------------------


_PMH_ABBREVIATIONS: dict[str, str] = {
    # general clinical
    "s/p": "status post",
    "c/b": "complicated by",
    "h/o": "history of",
    "w/": "with",
    "w/o": "without",
    "pmh": "past medical history",
    "pmhx": "past medical history",
    "hx": "history",
    # oncology
    "brca": "breast cancer",
    "ca": "cancer",
    "mets": "metastases",
    "scc": "squamous cell carcinoma",
    "crc": "colorectal cancer",
    "idc": "invasive ductal carcinoma",
    "ilc": "invasive lobular carcinoma",
    "nsclc": "non small cell lung cancer",
    # cardiology
    "htn": "hypertension",
    "hld": "hyperlipidemia",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "hfpef": "heart failure preserved ejection fraction",
    "hfref": "heart failure reduced ejection fraction",
    "afib": "atrial fibrillation",
    "as": "aortic stenosis",
    "tia": "transient ischemic attack",
    # respiratory
    "copd": "chronic obstructive pulmonary disease",
    "ahrf": "acute hypoxemic respiratory failure",
    "pe": "pulmonary embolism",
    "dvt": "deep vein thrombosis",
    "ards": "acute respiratory distress syndrome",
    "trach": "tracheostomy",
    # renal / endocrine / metabolic
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "esrd": "end stage renal disease",
    "dm": "diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "dka": "diabetic ketoacidosis",
    # gi / neuro / misc
    "gib": "gastrointestinal bleed",
    "uti": "urinary tract infection",
    "mca": "middle cerebral artery",
    "pca": "posterior cerebral artery",
    "cva": "cerebrovascular accident",
    "goc": "goals of care",
}


_ABBREV_PATTERN = re.compile(
    r"(?<![A-Za-z0-9/])(?:"
    + "|".join(re.escape(k) for k in sorted(_PMH_ABBREVIATIONS, key=len, reverse=True))
    + r")(?![A-Za-z0-9/])"
)


def expand_pmh_abbreviations(s: str) -> str:
    """Expand medical abbreviations in ``s`` using the in-module
    ``_PMH_ABBREVIATIONS`` dictionary. Matches at non-word boundaries
    that also exclude '/' (so 's/p' is treated as one token even though
    '/' is not a word character in regex's default sense). Longest-key-
    first ordering prevents 'ca' from clobbering longer 'cad', 'crc', etc.

    Assumes input is already lowercased + punctuation-normalized (i.e.,
    the output of ``normalize_for_validator``-style preprocessing).
    Returns the string with each known abbreviation replaced by its
    canonical long form; unknown tokens pass through untouched.
    """
    return _ABBREV_PATTERN.sub(lambda m: _PMH_ABBREVIATIONS[m.group(0)], s)


def normalize_for_pmh_match(s: str) -> str:
    """NFKC + punctuation normalize + lowercase + abbreviation expansion.

    Used by the orchestrator's Section I one-liner ↔ structured
    OneLinerPMHEntry.display alignment lint to tolerate routine
    medical abbreviation pairs (BrCa ↔ breast cancer, s/p ↔ status
    post, c/b ↔ complicated by, mets ↔ metastases) that would
    otherwise false-fail a raw-string match.

    Run on BOTH sides of the comparison so the match is symmetric.
    """
    return expand_pmh_abbreviations(normalize_for_validator(s))

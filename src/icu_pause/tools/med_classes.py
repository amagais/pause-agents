"""Drug-class lookup for sedation / analgesia / paralytic tense rendering.

Single source of truth for the Section I sedation-tense fix:
- ``intensivist._format_med_state_block`` reads ``SEDATION_ANALGESIA_PARALYTIC_DRUGS``
  to decide which meds.states.records entries to pin into the Section I context.
- ``intensivist.yaml`` Section I rule references this allowlist by NAME (not
  enumerated) so the YAML and the code can't drift.
- ``qa._check_sedation_tense_conflict`` reads the same dict to build the
  active-class list for the awake+sedation contradiction check, and reads
  ``QA_FIRE_SEVERITY`` for the class-aware severity matrix.

Why name-keyed and not mCIDE-keyed: CLIF v2.1 mCIDE provides a ``med_group``
taxonomy with clean ``sedation`` / ``analgesia`` / ``paralytics`` buckets
(see docs/clif_data_gaps_investigation.md:312-315), but the QA layer needs
to distinguish dexmedetomidine (arousal-preserving, awake patients are
clinically normal) from propofol/midazolam (true sedatives, awake patients
should be flagged). mCIDE groups all three under ``sedation`` — the
discriminating signal does not exist at the schema level, so a drug-name
override is required regardless. Keying the whole map by name (rather
than maintaining a parallel mCIDE plumbing path AND a drug-name override
map) keeps the source of truth in one place.

Asymmetry note for future PRs: vasopressors DO map cleanly to mCIDE
(``vasopressor`` / ``vasoactives``). When/if vasopressor tense rules
ship, they should key off ``med_group`` (after plumbing it through
``MedStateRecord`` and ``meds.states.records``) — not by extending this
file. The two lookup mechanisms exist because the clinical
discrimination requirements differ, not by oversight.
"""

from __future__ import annotations

from typing import Literal, Optional

SedationDrugClass = Literal[
    "true_sedative",
    "arousal_preserving",
    "analgesic",
    "dissociative",
    "paralytic",
]


SEDATION_ANALGESIA_PARALYTIC_DRUGS: dict[str, SedationDrugClass] = {
    # Awake patients on dex are clinically routine; do NOT flag in QA.
    "dexmedetomidine": "arousal_preserving",
    # True sedatives at infusion rates suppress arousal; awake+active = flag.
    "propofol": "true_sedative",
    "midazolam": "true_sedative",
    "lorazepam": "true_sedative",
    # Ketamine is ambiguous (low-dose analgesia preserves arousal,
    # sedation-dose does not). Mark dissociative; QA fires at low severity
    # to surface for review without crying wolf.
    "ketamine": "dissociative",
    # Opioid infusions are analgesia, not sedation; awake patients on them
    # are clinically normal. Do NOT flag in QA.
    "fentanyl": "analgesic",
    "hydromorphone": "analgesic",
    "morphine": "analgesic",
    "remifentanil": "analgesic",
    # NMBAs paralyze; an awake-charted patient on an active paralytic is a
    # documentation or clinical emergency. Fire at high severity.
    "cisatracurium": "paralytic",
    "rocuronium": "paralytic",
    "vecuronium": "paralytic",
}


# QA fire matrix per the class-aware tense-conflict rule.
# Severity is the QA issue severity when:
#   (a) a drug in this class is ACTIVE at transfer (state ∈ {ACTIVE,
#       ACTIVE_SCHEDULED, ACTIVE_PRN}), AND
#   (b) Section I or Section E text contains an awake/interactive token.
# "no_flag" classes are clinically normal in awake patients (analgesia,
# arousal-preserving sedation) and must NOT fire — alert-fatigue defeats
# the backstop.
QA_FIRE_SEVERITY: dict[SedationDrugClass, Literal["high", "medium", "low", "no_flag"]] = {
    "paralytic": "high",
    "true_sedative": "medium",
    "dissociative": "low",
    "arousal_preserving": "no_flag",
    "analgesic": "no_flag",
}


def classify_drug(drug_name: Optional[str]) -> Optional[SedationDrugClass]:
    """Return the sedation/analgesia/paralytic class for a drug, or None.

    Case-insensitive substring-anchored lookup against the canonical name
    set. ``meds.states.records[*].drug_name`` is populated from
    ``med_category`` or ``medication_name`` (see ``tools/med_state.py:437``),
    and CLIF med_category values are typically canonical lowercase strings
    (e.g., "dexmedetomidine", "propofol"). Brand names and combination
    products are not handled here — those should be normalized upstream in
    ``tools/drug_canonicalization.py`` if they become a real failure mode.
    """
    if not drug_name:
        return None
    key = drug_name.strip().lower()
    if not key:
        return None
    direct = SEDATION_ANALGESIA_PARALYTIC_DRUGS.get(key)
    if direct is not None:
        return direct
    # Tolerate CLIF med_category variants that pad with route/form
    # qualifiers (e.g., "propofol infusion", "fentanyl gtt"). Match on
    # whole-word containment of any allowlist key.
    tokens = set(key.replace("/", " ").replace("-", " ").split())
    for name, cls in SEDATION_ANALGESIA_PARALYTIC_DRUGS.items():
        if name in tokens:
            return cls
    return None


def is_pin_eligible(drug_name: Optional[str]) -> bool:
    """True when the drug should be considered for the Section I pin block.

    Membership-only; does not consult med-state. The pin block itself
    applies the state filter (ACTIVE / RECENTLY_STOPPED / trending_to_zero).
    """
    return classify_drug(drug_name) is not None


def drug_pattern_for_qa(active_classes: set[SedationDrugClass]) -> str:
    """Return a regex alternation of drug names whose class is in
    ``active_classes`` AND has a fire severity that isn't ``no_flag``.

    Used by the QA tense-conflict check to build the bare-active rendering
    detector. The caller composes the surrounding context (state qualifier
    negative-lookahead, past-tense lookbehind).

    Returns "" when no drug in the requested classes is fire-eligible.
    """
    names = [
        name
        for name, cls in SEDATION_ANALGESIA_PARALYTIC_DRUGS.items()
        if cls in active_classes and QA_FIRE_SEVERITY[cls] != "no_flag"
    ]
    if not names:
        return ""
    # Sort longest-first so the regex engine prefers "hydromorphone" over
    # "morphine" when both could match a prefix.
    names.sort(key=len, reverse=True)
    return "|".join(names)

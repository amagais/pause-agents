"""Antibiotic drug-name canonicalization for the admission_antibiotics
pathway (scribe validator + QA contract check + schema model_validator).

NOT a single source of truth. Two parallel canonicalization maps already
exist:

  - med_state.py:_DRUG_ALIASES — underscore canonical form, keys into
    DOSING_INTERVALS_HOURS (load-bearing for the drug-aware med-state
    classifier).
  - drug_interactions.py:CLIF_TO_RXNORM — hyphen canonical form, keys
    into STATIC_ICU_INTERACTIONS (load-bearing for the safety-critical
    DDI table).

Consolidation into a single helper-with-form-parameter is staged as
separate tracked work (see pending_work.md "Drug canonicalization
consolidation"). Rekeying the downstream lookup tables is NOT the
migration path — the helper returns the form each consumer needs,
leaving DOSING_INTERVALS_HOURS and STATIC_ICU_INTERACTIONS untouched.

This module uses HYPHEN canonical form because it's the display form
that ships in clinician-facing pin blocks.

Initial alias set is the UNION of antimicrobials from the two existing
maps PLUS cefazolin (absent from both, but the canonical worked-example
case in docs/admission_antibiotics_design.md §1 — the spec's own
schema model_validator would reject the worked example without it).

Provenance of each alias is noted by inline comment so the next
consolidation pass can audit without re-doing the diff.
"""

from __future__ import annotations

import re
from typing import Optional


# Hyphen canonical form. Provenance:
#   (m)  = med_state.py:_DRUG_ALIASES
#   (di) = drug_interactions.py:CLIF_TO_RXNORM
#   (new) = added in this PR (admission_antibiotics PR)
DRUG_ALIASES: dict[str, str] = {
    # --- Cephalosporins ---
    "cefepime": "cefepime",                                  # (m, di)
    "maxipime": "cefepime",                                  # (m)
    "ceftriaxone": "ceftriaxone",                            # (m)
    "rocephin": "ceftriaxone",                               # (m)
    "cefazolin": "cefazolin",                                # (new)
    "ancef": "cefazolin",                                    # (new)
    "kefzol": "cefazolin",                                   # (new)

    # --- Carbapenems ---
    "meropenem": "meropenem",                                # (m, di)
    "merrem": "meropenem",                                   # (m)
    "ertapenem": "ertapenem",                                # (m)
    "invanz": "ertapenem",                                   # (m)

    # --- Monobactams ---
    "aztreonam": "aztreonam",                                # (m)
    "azactam": "aztreonam",                                  # (m)

    # --- Penicillins + beta-lactamase inhibitors ---
    # Canonical = hyphenated. All slash/space/underscore variants alias to
    # the hyphen form. The underscore variant is present so charts that
    # paste from the med_state module's underscore form still resolve.
    "piperacillin-tazobactam": "piperacillin-tazobactam",    # (m canonical lhs, di canonical rhs)
    "piperacillin/tazobactam": "piperacillin-tazobactam",    # (m)
    "piperacillin / tazobactam": "piperacillin-tazobactam",  # (di)
    "piperacillin tazobactam": "piperacillin-tazobactam",    # (m)
    "piperacillin_tazobactam": "piperacillin-tazobactam",    # (m — interop w/ underscore form)
    "pip-tazo": "piperacillin-tazobactam",                   # (m, di)
    "pip/tazo": "piperacillin-tazobactam",                   # (m)
    "zosyn": "piperacillin-tazobactam",                      # (m, di)
    "tazocin": "piperacillin-tazobactam",                    # (m)
    "ampicillin-sulbactam": "ampicillin-sulbactam",          # (di canonical rhs)
    "ampicillin/sulbactam": "ampicillin-sulbactam",          # (new, hyphen-canonical interop)
    "ampicillin / sulbactam": "ampicillin-sulbactam",        # (di)
    "unasyn": "ampicillin-sulbactam",                        # (di)

    # --- Glycopeptides ---
    "vancomycin": "vancomycin",                              # (m)
    "vanco": "vancomycin",                                   # (di)
    "vancocin": "vancomycin",                                # (m)

    # --- Lipopeptides ---
    "daptomycin": "daptomycin",                              # (m)
    "cubicin": "daptomycin",                                 # (m)

    # --- Fluoroquinolones ---
    "levofloxacin": "levofloxacin",                          # (m)
    "levaquin": "levofloxacin",                              # (m)
    "levo": "levofloxacin",                                  # (di)
    "ciprofloxacin": "ciprofloxacin",                        # (new — di only had "cipro" alias, no canonical self-map)
    "cipro": "ciprofloxacin",                                # (di)

    # --- Nitroimidazoles ---
    "metronidazole": "metronidazole",                        # (m)
    "flagyl": "metronidazole",                               # (m)
}


def canonicalize_drug(name: str) -> Optional[str]:
    """Return canonical (hyphen-form) key for an arbitrary drug name, or
    None if the name isn't in DRUG_ALIASES. Case-insensitive, whitespace-
    stripped."""
    if not name:
        return None
    key = name.lower().strip()
    return DRUG_ALIASES.get(key)


def drug_aliases_for(canonical_or_alias: str) -> set[str]:
    """All alias spellings (including the canonical form itself) for the
    canonical key the input maps to. When the input isn't in DRUG_ALIASES,
    returns a singleton set containing the lowercased+stripped input — so
    callers can still do a word-boundary check against the literal name
    for drugs outside the antimicrobial alias map."""
    canonical = canonicalize_drug(canonical_or_alias)
    if canonical is None:
        return {canonical_or_alias.lower().strip()}
    return {a for a, c in DRUG_ALIASES.items() if c == canonical}


def drug_appears_in_text(drug: str, text: str) -> bool:
    """Word-boundary check: does drug (or any of its canonical aliases)
    appear in text? Used by:

      - _validate_admission_antibiotics (scribe runtime validator — sole
        enforcement point since the schema model_validator was removed 2026-06-08)
      - _check_antibiotic_pin_contract (QA)

    Word-boundary semantics intentional: 'cefepime' must NOT match
    'cefepimexyz', 'pip-tazo' must NOT match 'pip-tazol'."""
    aliases = drug_aliases_for(drug)
    text_lower = text.lower()
    for a in aliases:
        if re.search(rf"\b{re.escape(a)}\b", text_lower):
            return True
    return False

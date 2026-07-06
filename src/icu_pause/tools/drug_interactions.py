"""Deterministic drug-drug interaction checker (hybrid static + openFDA).

Replaces the deprecated RxNav interaction API (decommissioned by NLM in Jan 2024).

Architecture
------------
Two-tier lookup:

1. Static ICU-critical table (authoritative for severity='high').
   Clinician-reviewable, deterministic, always runs. Severity is curated
   — never assigned by an LLM or external service. The QA agent's
   severity='high' gate (qa.py) traces entirely to entries here.

2. openFDA drug-label API (secondary broadener).
   Free, official, no API key. Returns 'moderate' severity only — never
   escalates a pair to 'high'. Skipped when allow_network=False (eval
   runs, reproducibility tests).

Public entry point: ``check_interactions(meds_data, allow_network=True)``
returns a backward-compatible ``InteractionCheckResult`` for qa.py.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

Severity = Literal["high", "moderate", "low"]


# ---------------------------------------------------------------------------
# Public models (stable contract with qa.py)
# ---------------------------------------------------------------------------


class DrugInteraction(BaseModel):
    """A single drug-drug interaction hit."""

    drug_a: str
    drug_b: str
    severity: Severity
    description: str
    source: Literal["static", "openfda"]


class InteractionCheckResult(BaseModel):
    """Aggregated result of a drug interaction check for one patient."""

    interactions: list[DrugInteraction]
    unresolved_drugs: list[str]  # drugs whose openFDA lookup failed
    checked_drug_count: int
    checked_drug_names: list[str] = []  # canonical names of drugs actually checked
    api_available: bool
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# CLIF med_category → canonical drug name normalization
# ---------------------------------------------------------------------------
#
# Anticoagulant / antiplatelet scope:
# Registered classes match the CLIF v2.1 mCIDE continuous-meds scope —
# heparin, bivalirudin, argatroban (anticoagulants); cangrelor,
# eptifibatide (GPIIb/IIIa antiplatelets). Direct oral anticoagulants
# (apixaban, rivaroxaban, dabigatran, edoxaban) and subcutaneous LMWH
# (enoxaparin used prophylactically) are intentionally NOT registered
# for routine cohort runs: they are not categorized in CLIF v2.1's
# intermittent-meds layer, and the project's continuous-meds extract is
# IV+inhaled-route-only by scope. Surface them only via clinical-note
# extraction.
#
# Do not "fix" this by adding apixaban/rivaroxaban/etc as dead code —
# without rows in the source data, registry entries become misleading.
# See feedback_icu_pause_first_transfer_anchor.md +
# project_icu_pause_anticoag_gap.md for the full data-layer reasoning.

CLIF_TO_RXNORM: dict[str, str] = {
    "norepi": "norepinephrine",
    "norepinephrine bitartrate": "norepinephrine",
    "epi": "epinephrine",
    "epinephrine hcl": "epinephrine",
    "pip-tazo": "piperacillin-tazobactam",
    "piperacillin / tazobactam": "piperacillin-tazobactam",
    "zosyn": "piperacillin-tazobactam",
    "unasyn": "ampicillin-sulbactam",
    "ampicillin / sulbactam": "ampicillin-sulbactam",
    "vanco": "vancomycin",
    "levo": "levofloxacin",
    "cipro": "ciprofloxacin",
    "meropenem": "meropenem",
    "cefepime": "cefepime",
    # Anticoagulants (direct thrombin inhibitors, indirect FXa)
    "heparin sodium": "heparin",
    "heparin drip": "heparin",
    "enoxaparin sodium": "enoxaparin",
    "lovenox": "enoxaparin",
    "angiomax": "bivalirudin",
    "argatroban hydrochloride": "argatroban",
    # Antiplatelets — GPIIb/IIIa antagonists. CLIF mCIDE files them under
    # the anticoag bucket, but pharmacologically these are antiplatelets.
    # Keep them OUT of the prompt's "Anticoagulation" field so the brief
    # doesn't conflate antithrombotic mechanisms.
    "kengreal": "cangrelor",
    "integrilin": "eptifibatide",
    # Thrombolytics — listed for interaction-pair lookup only.
    "tpa": "alteplase",
    "activase": "alteplase",
    "insulin regular": "insulin regular",
    "insulin lispro": "insulin lispro",
    "phenylephrine hcl": "phenylephrine",
    "vasopressin": "vasopressin",
    "milrinone lactate": "milrinone",
    "dobutamine hcl": "dobutamine",
    "fentanyl citrate": "fentanyl",
    "propofol": "propofol",
    "midazolam hcl": "midazolam",
    "dexmedetomidine": "dexmedetomidine",
    "precedex": "dexmedetomidine",
    "tpn": "parenteral nutrition",
    "amiodarone hcl": "amiodarone",
    "metoprolol tartrate": "metoprolol",
    "hydralazine hcl": "hydralazine",
    "nicardipine hcl": "nicardipine",
    "furosemide": "furosemide",
    "lasix": "furosemide",
    "pantoprazole sodium": "pantoprazole",
    "famotidine": "famotidine",
    "haldol": "haloperidol",
    "morphine sulfate": "morphine",
}

# Drugs in the antiplatelet bucket — surfaced separately from
# anticoagulants in the brief so the Pharmacy/Intensivist agents don't
# call cangrelor or eptifibatide "anticoagulation."
ANTIPLATELET_CANONICAL: frozenset[str] = frozenset({
    "cangrelor",
    "eptifibatide",
    "aspirin",
    "clopidogrel",
})

# Drugs in the anticoagulant bucket. Limited to what's available in
# CLIF v2.1 mCIDE continuous meds (no DOACs, no LMWH prophylaxis).
ANTICOAGULANT_CANONICAL: frozenset[str] = frozenset({
    "heparin",
    "bivalirudin",
    "argatroban",
    "warfarin",
    "enoxaparin",  # therapeutic dosing only; prophylactic subQ unavailable
})


def _normalize(name: str) -> str:
    """Lowercase, strip, and apply CLIF→canonical mapping."""
    key = name.strip().lower()
    return CLIF_TO_RXNORM.get(key, key)


# ---------------------------------------------------------------------------
# Static ICU-critical interaction table
# ---------------------------------------------------------------------------
#
# Curation notes:
#   - Keys are frozensets so lookup is order-invariant.
#   - Severity is clinician-reviewable; sources noted per-pair.
#   - Canonical drug names match the output of _normalize() above.
#   - This table is a SAFETY-CRITICAL ARTIFACT. Changes require review.
#
# Sources referenced (by short tag in each entry's description):
#   ONCHigh   — ONC High-Priority Drug-Drug Interaction List (2016 update)
#   Lexicomp  — Lexicomp Drug Interactions (commercial, cited from published lists)
#   UTD       — UpToDate drug-drug interactions
#   PubMed    — primary literature

STATIC_ICU_INTERACTIONS: dict[frozenset[str], tuple[Severity, str]] = {
    # --- Sedation + respiratory depression --------------------------------
    frozenset({"fentanyl", "midazolam"}): (
        "high",
        "Synergistic respiratory depression and additive sedation; monitor RR, "
        "SpO2, and sedation depth (ONCHigh; UTD).",
    ),
    frozenset({"fentanyl", "propofol"}): (
        "high",
        "Additive hypotension and respiratory depression; titrate with "
        "hemodynamic monitoring (UTD).",
    ),
    frozenset({"hydromorphone", "midazolam"}): (
        "high",
        "Synergistic respiratory depression; dose reduction required in "
        "non-intubated patients (ONCHigh).",
    ),
    frozenset({"morphine", "midazolam"}): (
        "high",
        "Additive respiratory depression and sedation (ONCHigh).",
    ),
    frozenset({"dexmedetomidine", "propofol"}): (
        "high",
        "Additive hypotension and bradycardia; titrate one before adding the "
        "other (UTD).",
    ),
    # --- Serotonergic / MAO-inhibitor -------------------------------------
    frozenset({"linezolid", "fentanyl"}): (
        "high",
        "Linezolid is a weak reversible MAOI; risk of serotonin syndrome with "
        "serotonergic opioids (ONCHigh; FDA boxed warning).",
    ),
    frozenset({"linezolid", "methadone"}): (
        "high",
        "Serotonergic interaction and additive QT prolongation; avoid or "
        "monitor closely (ONCHigh).",
    ),
    frozenset({"linezolid", "tramadol"}): (
        "high",
        "Risk of serotonin syndrome; avoid combination (ONCHigh).",
    ),
    # --- QT prolongation / torsades ---------------------------------------
    frozenset({"amiodarone", "ondansetron"}): (
        "high",
        "Additive QT prolongation; torsades risk, especially with "
        "hypomagnesemia or hypokalemia (ONCHigh; FDA).",
    ),
    frozenset({"amiodarone", "haloperidol"}): (
        "high",
        "Additive QT prolongation; consider ECG monitoring and electrolyte "
        "correction (Lexicomp).",
    ),
    frozenset({"amiodarone", "methadone"}): (
        "high",
        "Additive QT prolongation plus CYP inhibition elevating methadone "
        "levels (ONCHigh).",
    ),
    frozenset({"methadone", "ondansetron"}): (
        "high",
        "Additive QT prolongation; torsades risk (Lexicomp).",
    ),
    frozenset({"haloperidol", "ondansetron"}): (
        "high",
        "Additive QT prolongation (Lexicomp).",
    ),
    # --- Nephrotoxicity ---------------------------------------------------
    frozenset({"vancomycin", "piperacillin-tazobactam"}): (
        "high",
        "Increased risk of acute kidney injury vs vancomycin alone; monitor "
        "SCr and trough (PubMed: multiple cohort studies).",
    ),
    frozenset({"vancomycin", "tacrolimus"}): (
        "high",
        "Additive nephrotoxicity; monitor SCr and tacrolimus trough (UTD).",
    ),
    frozenset({"vancomycin", "gentamicin"}): (
        "high",
        "Additive nephrotoxicity and ototoxicity (ONCHigh).",
    ),
    frozenset({"vancomycin", "tobramycin"}): (
        "high",
        "Additive nephrotoxicity and ototoxicity (ONCHigh).",
    ),
    # --- Hemodynamic / sedation crosstalk ---------------------------------
    frozenset({"propofol", "norepinephrine"}): (
        "high",
        "Propofol-induced vasodilation may mask or worsen hypotension "
        "requiring escalating pressor support (clinical).",
    ),
    # --- Bleeding ---------------------------------------------------------
    frozenset({"warfarin", "aspirin"}): (
        "high",
        "Additive bleeding risk; indication-specific assessment required "
        "(ONCHigh).",
    ),
    frozenset({"heparin", "aspirin"}): (
        "high",
        "Additive bleeding risk in ICU patients on anticoagulation (UTD).",
    ),
    frozenset({"enoxaparin", "aspirin"}): (
        "high",
        "Additive bleeding risk; re-evaluate indication in ICU (UTD).",
    ),
    # Argatroban + warfarin: highest-yield missed interaction during DTI→VKA
    # transition. Argatroban inflates INR by ~0.5-2.0 above what warfarin
    # alone produces, so INR during overlap doesn't reflect VKA effect alone.
    frozenset({"argatroban", "warfarin"}): (
        "high",
        "Argatroban inflates INR independent of warfarin effect during "
        "DTI→VKA transition. Overlap target is institution-specific "
        "(commonly INR > 4.0 on combined therapy, or chromogenic factor X "
        "assay where available); standard practice is to hold argatroban "
        "4-6h then recheck INR before discontinuing the DTI (Lexicomp; UTD).",
    ),
    # Anticoag/antiplatelet × thrombolytic — additive bleeding to a degree
    # that often exceeds clinical tolerance; require explicit reassessment.
    frozenset({"heparin", "alteplase"}): (
        "high",
        "Additive bleeding risk with thrombolytic; reassess heparin dosing "
        "and bleeding parameters per stroke/PE protocol (UTD).",
    ),
    frozenset({"bivalirudin", "alteplase"}): (
        "high",
        "Additive bleeding risk with thrombolytic; reassess DTI dosing per "
        "PCI/protocol (UTD).",
    ),
    frozenset({"argatroban", "alteplase"}): (
        "high",
        "Additive bleeding risk with thrombolytic (UTD).",
    ),
    frozenset({"cangrelor", "alteplase"}): (
        "high",
        "Additive bleeding risk; antiplatelet plus thrombolytic (UTD).",
    ),
    frozenset({"eptifibatide", "alteplase"}): (
        "high",
        "Additive bleeding risk; GPIIb/IIIa plus thrombolytic (UTD).",
    ),
    # Bivalirudin + antiplatelets — mechanism-based severity. PCI/ACS
    # protocol context (where these are intentional co-administrations)
    # belongs in the suppression layer (TODO: clinical_context_suppression
    # rule), NOT in the severity field. Severity here reflects bleeding-
    # risk pharmacology so the table is consumable by code that lacks
    # clinical context.
    #
    # TODO(citation-audit): refresh GPIIb/IIIa pair citations when doing
    # the broader citation refresh. ONCHigh's 2016 list predates the
    # MATRIX/ISAR-REACT 5 era; ACC/AHA/SCAI 2021 PCI guideline or current
    # SCAI consensus is more current for the bleeding-risk framing here.
    frozenset({"bivalirudin", "cangrelor"}): (
        "moderate",
        "Bleeding risk in PCI co-administration; review ACT, sheath site, "
        "and post-procedure timing (Lexicomp).",
    ),
    frozenset({"bivalirudin", "eptifibatide"}): (
        "moderate",
        "Bleeding risk; GPIIb/IIIa with DTI in PCI/ACS settings (Lexicomp).",
    ),
    frozenset({"bivalirudin", "aspirin"}): (
        "high",
        "Additive bleeding risk; antiplatelet plus direct thrombin "
        "inhibitor. Mechanism-equivalent to heparin+aspirin (also high). "
        "PCI/ACS protocol contexts where this is expected co-administration "
        "are handled by suppression rules at the surfacing layer, not by "
        "severity downgrade (Lexicomp).",
    ),
    frozenset({"bivalirudin", "clopidogrel"}): (
        "moderate",
        "Bleeding risk; antiplatelet plus DTI in ACS/PCI (Lexicomp).",
    ),
    # Argatroban + antiplatelets — same rationale as bivalirudin; argatroban
    # is the preferred DTI in HIT, often co-administered with antiplatelets.
    frozenset({"argatroban", "cangrelor"}): (
        "moderate",
        "Bleeding risk in HIT/PCI co-administration (Lexicomp).",
    ),
    frozenset({"argatroban", "eptifibatide"}): (
        "moderate",
        "Bleeding risk; GPIIb/IIIa with DTI (Lexicomp).",
    ),
    frozenset({"argatroban", "aspirin"}): (
        "moderate",
        "Bleeding risk; antiplatelet plus DTI (Lexicomp).",
    ),
    frozenset({"argatroban", "clopidogrel"}): (
        "moderate",
        "Bleeding risk; antiplatelet plus DTI (Lexicomp).",
    ),
    # Heparin + GPIIb/IIIa antiplatelets — often co-administered in PCI;
    # flag as moderate to surface monitoring need without over-escalating.
    frozenset({"heparin", "cangrelor"}): (
        "moderate",
        "Bleeding risk in PCI co-administration (Lexicomp).",
    ),
    frozenset({"heparin", "eptifibatide"}): (
        "moderate",
        "Bleeding risk; GPIIb/IIIa plus heparin in PCI/ACS (Lexicomp).",
    ),
    # --- CYP-mediated level changes ---------------------------------------
    frozenset({"amiodarone", "warfarin"}): (
        "high",
        "Amiodarone inhibits CYP2C9/3A4/P-gp; warfarin levels rise — reduce "
        "warfarin dose and monitor INR (ONCHigh).",
    ),
    frozenset({"fluconazole", "tacrolimus"}): (
        "high",
        "Fluconazole inhibits CYP3A4; tacrolimus levels rise (ONCHigh).",
    ),
    # --- Electrolyte / digoxin --------------------------------------------
    frozenset({"digoxin", "furosemide"}): (
        "high",
        "Diuretic-induced hypokalemia potentiates digoxin toxicity (ONCHigh).",
    ),
    frozenset({"digoxin", "amiodarone"}): (
        "high",
        "Amiodarone raises digoxin levels via P-gp inhibition — halve digoxin "
        "dose and monitor levels (ONCHigh).",
    ),
    # --- Neuromuscular blockade -------------------------------------------
    frozenset({"rocuronium", "gentamicin"}): (
        "high",
        "Aminoglycosides potentiate non-depolarizing NMB; risk of prolonged "
        "paralysis (UTD).",
    ),
    frozenset({"rocuronium", "tobramycin"}): (
        "high",
        "Aminoglycosides potentiate non-depolarizing NMB; risk of prolonged "
        "paralysis (UTD).",
    ),
}


# ---------------------------------------------------------------------------
# Tier 1: static table lookup
# ---------------------------------------------------------------------------


def check_static(drug_a: str, drug_b: str) -> Optional[DrugInteraction]:
    """Look up an interaction in the clinician-reviewed static table."""
    a = _normalize(drug_a)
    b = _normalize(drug_b)
    if a == b:
        return None
    hit = STATIC_ICU_INTERACTIONS.get(frozenset({a, b}))
    if hit is None:
        return None
    severity, description = hit
    return DrugInteraction(
        drug_a=a,
        drug_b=b,
        severity=severity,
        description=description,
        source="static",
    )


# ---------------------------------------------------------------------------
# Tier 2: openFDA drug-label lookup
# ---------------------------------------------------------------------------


OPENFDA_URL = "https://api.fda.gov/drug/label.json"


@lru_cache(maxsize=512)
def _fetch_openfda_label(drug: str, timeout: float = 5.0) -> Optional[str]:
    """Return the concatenated drug_interactions text for a drug, or None.

    Cached per-process. openFDA anonymous rate limit is ~240/min.
    Returns None (not raise) on any error so the caller degrades gracefully.
    """
    try:
        r = httpx.get(
            OPENFDA_URL,
            params={
                "search": f'openfda.generic_name:"{drug}"',
                "limit": 1,
            },
            timeout=timeout,
        )
        if r.status_code == 404:
            # openFDA returns 404 when no records match — normal, not an error.
            logger.debug("openFDA: no label found for %r", drug)
            return ""
        if r.status_code != 200:
            logger.debug(
                "openFDA returned status %d for %r", r.status_code, drug
            )
            return None
        results = r.json().get("results", [])
        if not results:
            return ""
        label = results[0]
        # drug_interactions is a list of strings per FDA label structure.
        return " ".join(label.get("drug_interactions", [])).lower()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("openFDA lookup failed for %r: %s", drug, exc)
        return None


def check_openfda(
    drug_a: str, drug_b: str, timeout: float = 5.0
) -> Optional[DrugInteraction]:
    """Secondary broadener. Never returns severity='high'.

    Heuristic: fetch drug_a's FDA label, check if drug_b's normalized name
    appears in the drug_interactions section text. Returns 'moderate'
    severity for any hit — escalation to 'high' requires a static-table entry.
    """
    a = _normalize(drug_a)
    b = _normalize(drug_b)
    if a == b:
        return None

    interaction_text = _fetch_openfda_label(a, timeout=timeout)
    if interaction_text is None:
        # Fetch error — caller can track this drug as unresolved.
        return None
    if not interaction_text:
        return None

    # openFDA labels use a variety of name forms; match on the first token
    # of a multi-word canonical name to tolerate combination drug phrasing
    # (e.g. "piperacillin-tazobactam" → match on "piperacillin").
    needle = b.split("-")[0].split()[0]
    if needle not in interaction_text:
        return None

    return DrugInteraction(
        drug_a=a,
        drug_b=b,
        severity="moderate",
        description=(
            f"openFDA label for {a} references interaction with {b}; review "
            "FDA prescribing information for clinical detail."
        ),
        source="openfda",
    )


# ---------------------------------------------------------------------------
# Hybrid per-pair check
# ---------------------------------------------------------------------------


def check_pair(
    drug_a: str,
    drug_b: str,
    *,
    allow_network: bool = True,
    timeout: float = 5.0,
) -> Optional[DrugInteraction]:
    """Hybrid check for one drug pair.

    Static table first (deterministic, authoritative for severity='high').
    openFDA second (broader but severity='moderate' only).
    """
    hit = check_static(drug_a, drug_b)
    if hit is not None:
        return hit
    if not allow_network:
        return None
    return check_openfda(drug_a, drug_b, timeout=timeout)


# ---------------------------------------------------------------------------
# Cohort entry point (called by qa.py)
# ---------------------------------------------------------------------------


def extract_drug_names(
    meds_data: dict,
    *,
    reference_dttm: Optional[Any] = None,
    active_only: bool = False,
) -> list[str]:
    """Extract unique drug names from CLIF-formatted medication data.

    When ``active_only`` is True and ``reference_dttm`` is provided, only
    drugs classified ACTIVE by ``med_state.classify_med_states`` are
    returned. Drugs RECENTLY_STOPPED or HISTORICAL are excluded — this is
    the gate that prevents false-positive DDI alerts on weaned-off
    infusions.
    """
    if active_only and reference_dttm is not None:
        from icu_pause.tools.med_state import (
            active_drug_names,
            classify_med_states,
        )

        records = classify_med_states(meds_data, reference_dttm)
        return sorted(active_drug_names(records))

    names: set[str] = set()
    for section_key in ("continuous", "intermittent"):
        rows = meds_data.get(section_key) or []
        for row in rows:
            name = str(
                row.get("med_category", row.get("medication_name", ""))
            ).strip()
            if name:
                names.add(name)
    return sorted(names)


def check_interactions(
    meds_data: dict,
    *,
    allow_network: bool = True,
    timeout: float = 5.0,
    reference_dttm: Optional[Any] = None,
    # Legacy kwargs retained for backward compatibility with qa.py wiring.
    base_url: Optional[str] = None,  # unused; kept for call-site compat
) -> InteractionCheckResult:
    """Run a hybrid drug interaction check on CLIF medication data.

    1. Extract unique ACTIVE drugs (per med_state classifier when
       ``reference_dttm`` is given) and normalize names.
    2. For each unordered pair, call the static table; fall back to openFDA
       if ``allow_network`` and no static hit.
    3. Return structured result (never raises).

    When ``reference_dttm`` is provided, RECENTLY_STOPPED and HISTORICAL
    drugs are excluded so DDI alerts only fire on co-exposures the patient
    is actually receiving. When ``reference_dttm`` is None, falls back to
    legacy behavior (any drug seen in the window counts).
    """
    del base_url  # accepted for call-site compat; unused in hybrid design

    drug_names = extract_drug_names(
        meds_data,
        reference_dttm=reference_dttm,
        active_only=reference_dttm is not None,
    )
    if len(drug_names) < 2:
        return InteractionCheckResult(
            interactions=[],
            unresolved_drugs=[],
            checked_drug_count=len(drug_names),
            checked_drug_names=sorted({_normalize(n) for n in drug_names}),
            api_available=True,
        )

    normalized = [_normalize(n) for n in drug_names]
    # Preserve pair uniqueness while tolerating duplicate normalized names.
    unique_drugs = sorted(set(normalized))

    interactions: list[DrugInteraction] = []
    unresolved: set[str] = set()
    seen: set[frozenset[str]] = set()

    for i, a in enumerate(unique_drugs):
        for b in unique_drugs[i + 1 :]:
            key = frozenset({a, b})
            if key in seen:
                continue
            seen.add(key)

            static_hit = check_static(a, b)
            if static_hit is not None:
                interactions.append(static_hit)
                continue

            if not allow_network:
                continue

            # openFDA tier. Track per-drug lookup failures in unresolved.
            label_a = _fetch_openfda_label(a, timeout=timeout)
            if label_a is None:
                unresolved.add(a)
                continue
            label_b = _fetch_openfda_label(b, timeout=timeout)
            if label_b is None:
                unresolved.add(b)
                continue

            # Try both directions — A's label referencing B, or B's label
            # referencing A — since FDA labels are asymmetric.
            fda_hit = check_openfda(a, b, timeout=timeout) or check_openfda(
                b, a, timeout=timeout
            )
            if fda_hit is not None:
                interactions.append(fda_hit)

    logger.info(
        "Drug interaction check: %d drugs, %d interactions found "
        "(%d static, %d openFDA), allow_network=%s",
        len(unique_drugs),
        len(interactions),
        sum(1 for ix in interactions if ix.source == "static"),
        sum(1 for ix in interactions if ix.source == "openfda"),
        allow_network,
    )

    return InteractionCheckResult(
        interactions=interactions,
        unresolved_drugs=sorted(unresolved),
        checked_drug_count=len(unique_drugs),
        checked_drug_names=unique_drugs,
        api_available=True,
    )

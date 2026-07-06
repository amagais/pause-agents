"""Deterministic patient clinical-context inference.

Builds a ``PatientClinicalContext`` from CLIF tables. Detection is
deterministic and conservative — every detected condition records the
row/code that triggered it, so reframing decisions downstream are auditable.

# ---------------------------------------------------------------------------
# Anticoagulation detection (added 2026-05-29)
# ---------------------------------------------------------------------------
#
# ``on_therapeutic_anticoagulation`` is now detected from the CLIF parquet
# (refresh shipped 2026-05-04 — see project_icu_pause_anticoag_gap.md). The
# anticoag taxonomy is shared with ``tools.drug_interactions``:
#
#   ANTICOAGULANT_CANONICAL = {heparin, bivalirudin, argatroban, warfarin,
#                              enoxaparin}
#
# Detection rule (conservative): if ANY of these canonical agents is present
# in the patient's continuous or intermittent meds, the field is set to the
# pipe-joined list of canonical names that fired ("heparin|warfarin").
# Cangrelor and eptifibatide are filed under the CLIF mCIDE anticoag bucket
# but are GPIIb/IIIa antiplatelets — they are NOT counted as therapeutic
# anticoagulation here (kept in sync with drug_interactions.py).
#
# Known scope limits (carried over from the prior deferred-PR note):
#   - SubQ prophylactic LMWH is NOT distinguishable from therapeutic LMWH
#     at the data layer in CLIF v2.1. Any enoxaparin presence trips the
#     flag conservatively. Acceptable because INR/PTT reframes treat
#     "on anticoag" as "don't claim cirrhosis-only explanation"; a false
#     positive de-escalates a confident chronic claim into a softer one.
#   - Oral DOACs (apixaban, rivaroxaban, dabigatran, edoxaban) remain
#     structurally unavailable in CLIF v2.1 mCIDE. A DOAC-only patient on
#     a cirrhotic INR↑ will still see the chronic-cirrhosis reframe
#     (without "not anticoagulated"). Logged as a Limitations paper item.
#   - ``baseline_inr`` remains absent — anticoag detection alone doesn't
#     justify computing one; cirrhosis context handles the chronic-INR
#     story without a numeric anchor.
#
# ---------------------------------------------------------------------------
# Deferred — active systemic corticosteroids
# ---------------------------------------------------------------------------
#
# Steroid-induced leukocytosis reframing remains deferred. Not deferred on
# clinical grounds — the WBC demargination reframe is high-value precisely
# because it tells clinicians not to chase a phantom sepsis signal — but
# pending verification that ``dose_amount`` / ``dose_unit`` / ``route``
# columns reliably distinguish systemic prednisone from inhaled fluticasone
# or topical hydrocortisone.
#
# A permissive fallback ("active steroid med_category, dose-deferred") would
# produce false-positive WBC reframes that *mask* sepsis — strictly worse
# than no detector for this case.
#
# When activating, the gate is unambiguous: confirm dose/route data supports
# the systemic-vs-inhaled-vs-topical distinction on a sample of real cases
# before turning detection on. Then implement the dose thresholds in the
# spec (Prednisone ≥20mg/day equiv, etc.) and re-add the WBC reframe rule.
#
# ---------------------------------------------------------------------------
# Baseline semantics
# ---------------------------------------------------------------------------
#
# ``baseline_creatinine`` and ``baseline_hgb`` are computed from the EARLIEST
# resulted lab in the lookback window passed to the retriever (typically 48h
# pre-reference). They are NOT pre-admission baselines.
#
# Implications worth being explicit about:
#
#  - For a chronic ESRD patient, the in-window "baseline" is whatever Cr
#    happened to be drawn at the start of the 48h window — which itself
#    reflects the patient's position in the inter-dialytic cycle (Cr peaks
#    immediately before HD, troughs after). The "baseline" computed here is
#    not the patient's true HD-cycle steady state.
#
#  - Reframing in PR 3 has two modes. The qualitative mode
#    ("Cr X — chronic elevation from ESRD on HD") doesn't use this value at
#    all and is the high-value reframe. The quantitative mode ("Cr X — above
#    patient's HD baseline of Y, consider workup") DOES use it; consumers
#    should treat ``baseline_creatinine`` as "earliest in-window Cr" rather
#    than "the patient's chronic baseline."
#
#  - The threshold ``MIN_BASELINE_MEASUREMENTS`` requires at least 3 resulted
#    values in window. With daily Cr (typical ICU cadence), 48h gives 2
#    measurements — below threshold → baseline = None for many patients.
#    That's by design (avoids noisy single-point baselines), but means most
#    patients won't get a numeric baseline. A real fix is Tier 2: pull
#    pre-admission labs from outside the lookback window, or relax to ≥2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import polars as pl

if TYPE_CHECKING:
    from icu_pause.data.context import PatientContext


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class PatientClinicalContext:
    """Deterministic patient-level chronic conditions and baselines.

    Each True flag is paired with a list of evidence strings under the same
    key in ``evidence`` (e.g., ``evidence['has_esrd_dialysis'] =
    ['icd:N18.6', 'crrt_active']``) so downstream reframing is auditable.
    """

    # Chronic conditions (PR 1 scope: 5 contexts).
    has_esrd_dialysis: bool = False
    has_cirrhosis: bool = False
    has_copd: bool = False
    has_chronic_afib: bool = False
    has_chronic_trach: bool = False

    # Baselines computed from in-window labs. Earliest value when at least
    # ``MIN_BASELINE_MEASUREMENTS`` measurements are available; otherwise None.
    # ``baseline_inr`` deferred — see module header.
    baseline_creatinine: Optional[float] = None
    baseline_hgb: Optional[float] = None

    # Therapeutic anticoagulation status. ``None`` when no anticoag agent
    # is detected; pipe-joined canonical names (e.g. "heparin|warfarin")
    # when one or more are present. See module header for the agent set
    # and known scope limits (subQ prophylaxis indistinguishable from
    # therapeutic LMWH; oral DOACs structurally unavailable).
    on_therapeutic_anticoagulation: Optional[str] = None

    # Audit trail. Key matches the flag attribute name; value is a list of
    # short evidence strings (icd code, crrt_active, med:sevelamer, etc.).
    evidence: dict[str, list[str]] = field(default_factory=dict)

    def any_flag_set(self) -> bool:
        """True if any chronic-condition flag is set. Drives gating in
        downstream reframing — when all flags are False, reframing is a no-op
        and the warning stream is byte-identical to the pre-feature output."""
        return (
            self.has_esrd_dialysis
            or self.has_cirrhosis
            or self.has_copd
            or self.has_chronic_afib
            or self.has_chronic_trach
        )

    def to_dict(self) -> dict:
        """Serialize for ``serialize_to_json`` / reviewer-app metadata."""
        return {
            "has_esrd_dialysis": self.has_esrd_dialysis,
            "has_cirrhosis": self.has_cirrhosis,
            "has_copd": self.has_copd,
            "has_chronic_afib": self.has_chronic_afib,
            "has_chronic_trach": self.has_chronic_trach,
            "baseline_creatinine": self.baseline_creatinine,
            "baseline_hgb": self.baseline_hgb,
            "on_therapeutic_anticoagulation": self.on_therapeutic_anticoagulation,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PatientClinicalContext":
        """Inverse of ``to_dict``. Used by qa.py to rehydrate the context
        after it round-trips through ``serialize_to_json``'s dict form.
        """
        return cls(
            has_esrd_dialysis=bool(data.get("has_esrd_dialysis", False)),
            has_cirrhosis=bool(data.get("has_cirrhosis", False)),
            has_copd=bool(data.get("has_copd", False)),
            has_chronic_afib=bool(data.get("has_chronic_afib", False)),
            has_chronic_trach=bool(data.get("has_chronic_trach", False)),
            baseline_creatinine=data.get("baseline_creatinine"),
            baseline_hgb=data.get("baseline_hgb"),
            on_therapeutic_anticoagulation=data.get("on_therapeutic_anticoagulation"),
            evidence=dict(data.get("evidence") or {}),
        )


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_BASELINE_MEASUREMENTS = 3

# ICD code prefixes for each chronic condition. Matched as *startswith*
# against the diagnosis_code / icd_code column.
#
# Both ICD-10-CM and ICD-9-CM prefixes are included. At Northwestern's
# CLIF parquet (verified 2026-05-04) ~22% of diagnoses are ICD-9-CM,
# ~78% ICD-10-CM. Without ICD-9 prefixes, chronic conditions encoded
# under the older system silently miss detection. ICD-9 prefixes don't
# collide with ICD-10 (digit-prefix vs letter-prefix), so a single
# combined startswith match is safe.
_ICD_PREFIXES_ESRD = (
    # ICD-10
    "N18.5", "N185",     # CKD stage 5
    "N18.6", "N186",     # ESRD
    "Z99.2", "Z992",     # Dialysis dependence
    "Z91.15", "Z9115",   # Noncompliance with renal dialysis
    "T82.4", "T824",     # Mechanical complication of vascular dialysis catheter
    # ICD-9
    "585.5", "5855",     # CKD stage V
    "585.6", "5856",     # End-stage renal disease
    "V45.11", "V4511",   # Renal dialysis status
    "V56",               # Encounter for dialysis (V56.0/.1/.2/.3/.31/.32/.8)
    "996.73", "99673",   # Complications due to renal dialysis (catheter, etc.)
)
_ICD_PREFIXES_CIRRHOSIS = (
    # ICD-10
    "K70.3",  "K703",    # Alcoholic cirrhosis
    "K74.0",  "K740",    # Hepatic fibrosis
    "K74.1",  "K741",
    "K74.2",  "K742",
    "K74.3",  "K743",
    "K74.4",  "K744",
    "K74.5",  "K745",
    "K74.6",  "K746",    # Other / unspecified cirrhosis
    "K76.6",  "K766",    # Portal hypertension
    "I85",               # Esophageal varices (with/without bleeding)
    # ICD-9
    "571.2",  "5712",    # Alcoholic cirrhosis of liver
    "571.5",  "5715",    # Cirrhosis of liver without mention of alcohol
    "571.6",  "5716",    # Biliary cirrhosis
    "572.3",  "5723",    # Portal hypertension
    "456.0",  "4560",    # Esophageal varices with bleeding
    "456.1",  "4561",    # Esophageal varices without bleeding
    "456.2",  "4562",    # Esophageal varices in diseases classified elsewhere
)
_ICD_PREFIXES_COPD = (
    # ICD-10
    "J41",  # Simple/mucopurulent chronic bronchitis
    "J42",  # Unspecified chronic bronchitis
    "J43",  # Emphysema
    "J44",  # COPD
    # ICD-9 (parent prefixes catch all sub-codes — 491.x, 492.x, 496)
    "491",  # Chronic bronchitis (.0/.1/.2/.20/.21/.22/.8/.9)
    "492",  # Emphysema (.0/.8)
    "496",  # Chronic airway obstruction NEC (= COPD)
)
_ICD_PREFIXES_AFIB = (
    # ICD-10 (subtype-specific: paroxysmal vs persistent vs chronic vs unspecified)
    "I48.0",  "I480",   # Paroxysmal AF
    "I48.1",  "I481",   # Persistent AF
    "I48.2",  "I482",   # Chronic / longstanding persistent AF
    "I48.91", "I4891",  # Unspecified AF
    # ICD-9
    # KNOWN LIMITATION: ICD-9 has a single 427.31 code for ALL atrial
    # fibrillation subtypes. Detection from ICD-9 alone cannot
    # distinguish paroxysmal from chronic AF; we accept this as a known
    # site-data limitation rather than trying to disambiguate by other
    # signals (rate-control meds aren't reliably present in CLIF — see
    # project_icu_pause_chronic_meds_gap.md).
    "427.31", "42731",
)
_ICD_PREFIXES_CHRONIC_TRACH = (
    # ICD-10
    "Z93.0",  "Z930",
    # ICD-9
    "V44.0",  "V440",   # Tracheostomy status
)

# Medication signal sets. Match against ``med_category`` then ``med_name``
# (case-insensitive substring). Categorical match is preferred; the
# substring fallback handles sites whose med_category mapping is sparse.
_CKD_MED_SIGNALS = {
    "sevelamer", "calcium acetate", "cinacalcet",
    "epoetin", "epoetin alfa", "darbepoetin",
}
_HE_REGIMEN_LACTULOSE = {"lactulose"}
_HE_REGIMEN_RIFAXIMIN = {"rifaximin"}
_COPD_MED_SIGNALS = {
    "tiotropium", "umeclidinium",
    # ICS+LABA combinations (chronic-use brand names + generics)
    "fluticasone-salmeterol", "advair",
    "budesonide-formoterol", "symbicort",
    "fluticasone-vilanterol", "breo",
    "umeclidinium-vilanterol", "anoro",
    "fluticasone-umeclidinium-vilanterol", "trelegy",
}

# Procedure name signals. Substring match against ``procedure_name`` or
# ``procedure_category`` (case-insensitive).
_DIALYSIS_PROCEDURE_KEYWORDS = (
    "hemodialysis", "peritoneal dialysis",
    "av fistula", "av graft", "arteriovenous fistula", "arteriovenous graft",
)
_TRACH_PROCEDURE_KEYWORDS = ("tracheostomy",)
_CIRRHOSIS_PROCEDURE_KEYWORDS = ("tips", "transjugular intrahepatic")
_PARACENTESIS_KEYWORDS = ("paracentesis",)

# Respiratory device categories that indicate chronic trach when consistent.
_CHRONIC_TRACH_DEVICE_CATEGORIES = {
    "trach collar", "trach_collar", "tracheostomy",
    "trach mask", "trach_mask",
}

# Threshold for "tracheostomy >30 days before admission" being chronic.
_CHRONIC_TRACH_DAYS_BEFORE_ADMIT = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_icd_codes(diagnoses: Optional[pl.DataFrame]) -> list[str]:
    """Pull ICD codes from the diagnoses DataFrame.

    CLIF schema drift: column is ``diagnosis_code`` at some sites,
    ``icd_code`` at others. Returns an upper-cased, dot-or-no-dot tolerant
    list — callers match by ``startswith`` against the prefix sets above.
    """
    if diagnoses is None or len(diagnoses) == 0:
        return []
    col = next(
        (c for c in ("diagnosis_code", "icd_code") if c in diagnoses.columns),
        None,
    )
    if col is None:
        return []
    return [
        str(c).strip().upper()
        for c in diagnoses[col].drop_nulls().to_list()
        if c
    ]


def _icd_matches(codes: list[str], prefixes: tuple[str, ...]) -> list[str]:
    """Return the codes that match any of the given prefixes (uppercased)."""
    upper_prefixes = tuple(p.upper() for p in prefixes)
    return [c for c in codes if c.startswith(upper_prefixes)]


def _med_rows(
    df: Optional[pl.DataFrame],
) -> list[dict]:
    """Convert a meds DataFrame to row-dicts; tolerate None / empty."""
    if df is None or len(df) == 0:
        return []
    return df.to_dicts()


def _med_label(row: dict) -> str:
    """Lowercase med-name proxy for substring matching.

    ``med_category`` is the canonical CLIF field; ``med_name`` is the raw
    string. Both are queried because category mapping is sparse at some
    sites.
    """
    cat = str(row.get("med_category") or "").strip().lower()
    name = str(row.get("med_name") or row.get("medication_name") or "").strip().lower()
    if cat and name:
        return f"{cat} {name}"
    return cat or name


def _has_med_signal(rows: list[dict], signals: set[str]) -> list[str]:
    """Return the matched signals (deduped, sorted) found in any row."""
    hits: set[str] = set()
    for row in rows:
        label = _med_label(row)
        if not label:
            continue
        for sig in signals:
            if sig in label:
                hits.add(sig)
    return sorted(hits)


def _proc_rows(df: Optional[pl.DataFrame]) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    return df.to_dicts()


def _proc_label(row: dict) -> str:
    name = str(row.get("procedure_name") or "").strip().lower()
    cat = str(row.get("procedure_category") or "").strip().lower()
    if name and cat:
        return f"{name} {cat}"
    return name or cat


def _proc_keyword_hits(
    rows: list[dict], keywords: tuple[str, ...]
) -> list[dict]:
    """Return procedure rows whose name/category contains any keyword."""
    hits = []
    for row in rows:
        label = _proc_label(row)
        if not label:
            continue
        if any(kw in label for kw in keywords):
            hits.append(row)
    return hits


def _earliest_lab_value(
    labs: Optional[pl.DataFrame], lab_category: str
) -> Optional[float]:
    """Earliest in-window numeric value for a lab category, gated by
    ``MIN_BASELINE_MEASUREMENTS`` to avoid noisy single-point baselines.

    Uses ``lab_collect_dttm`` (when present) or ``lab_result_dttm`` for
    ordering. Pending rows (masked numeric) are excluded by the
    drop_nulls on ``lab_value_numeric``.
    """
    if labs is None or len(labs) == 0:
        return None
    if "lab_category" not in labs.columns or "lab_value_numeric" not in labs.columns:
        return None

    df = labs.filter(
        (pl.col("lab_category").str.to_lowercase() == lab_category.lower())
        & pl.col("lab_value_numeric").is_not_null()
    )
    if len(df) < MIN_BASELINE_MEASUREMENTS:
        return None

    sort_col = next(
        (c for c in ("lab_collect_dttm", "lab_result_dttm") if c in df.columns),
        None,
    )
    if sort_col is None:
        # No timestamp to order by — can't pick "earliest" deterministically.
        return None

    df = df.sort(sort_col, nulls_last=True)
    earliest = df["lab_value_numeric"].drop_nulls()
    if len(earliest) == 0:
        return None
    return float(earliest[0])


def _parse_dttm(value) -> Optional[datetime]:
    """Best-effort parse of a CLIF timestamp string/datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _add_evidence(
    ctx: PatientClinicalContext, flag: str, item: str
) -> None:
    """Append an evidence string to the audit trail under ``flag``."""
    ctx.evidence.setdefault(flag, []).append(item)


# ---------------------------------------------------------------------------
# Detection rules
# ---------------------------------------------------------------------------


def _detect_esrd_dialysis(
    ctx: PatientClinicalContext,
    icd_codes: list[str],
    meds_continuous: list[dict],
    meds_intermittent: list[dict],
    crrt: Optional[pl.DataFrame],
    procedures: list[dict],
    reference_dttm: Optional[datetime],
) -> None:
    """Rule 1 — ESRD / chronic dialysis.

    Trigger on any of: matching ICD code, dialysis-procedure name match,
    ≥2 CKD-specific meds. CRRT presence is supportive but does NOT alone
    set the flag — AKI requiring temporary CRRT vs chronic ESRD-on-HD is a
    different clinical scenario; conservative-False with explicit evidence
    keeps the receiving team aware that renal replacement is acute.
    """
    icd_hits = _icd_matches(icd_codes, _ICD_PREFIXES_ESRD)
    proc_hits = _proc_keyword_hits(procedures, _DIALYSIS_PROCEDURE_KEYWORDS)
    med_hits = _has_med_signal(meds_continuous + meds_intermittent, _CKD_MED_SIGNALS)
    crrt_present = crrt is not None and len(crrt) > 0

    chronicity_signal = bool(icd_hits) or bool(proc_hits) or len(med_hits) >= 2

    if chronicity_signal:
        ctx.has_esrd_dialysis = True
        for c in icd_hits:
            _add_evidence(ctx, "has_esrd_dialysis", f"icd:{c}")
        for p in proc_hits:
            label = _proc_label(p)
            _add_evidence(ctx, "has_esrd_dialysis", f"procedure:{label}")
        for m in med_hits:
            _add_evidence(ctx, "has_esrd_dialysis", f"med:{m}")
        if crrt_present:
            _add_evidence(ctx, "has_esrd_dialysis", "crrt_active")
    elif crrt_present:
        # CRRT present without chronicity signal — mark explicitly so
        # downstream reframing knows this is acute, not chronic.
        _add_evidence(
            ctx,
            "has_esrd_dialysis",
            "crrt_active_but_chronicity_unconfirmed",
        )


def _detect_cirrhosis(
    ctx: PatientClinicalContext,
    icd_codes: list[str],
    meds_continuous: list[dict],
    meds_intermittent: list[dict],
    procedures: list[dict],
) -> None:
    """Rule 2 — Cirrhosis / chronic liver disease.

    Trigger on any of: cirrhosis ICD codes, TIPS or recurrent paracentesis
    (≥2 in window), or active lactulose AND rifaximin (HE regimen).
    """
    icd_hits = _icd_matches(icd_codes, _ICD_PREFIXES_CIRRHOSIS)

    proc_hits_tips = _proc_keyword_hits(procedures, _CIRRHOSIS_PROCEDURE_KEYWORDS)
    paracenteses = _proc_keyword_hits(procedures, _PARACENTESIS_KEYWORDS)
    recurrent_para = len(paracenteses) >= 2

    all_meds = meds_continuous + meds_intermittent
    has_lactulose = bool(_has_med_signal(all_meds, _HE_REGIMEN_LACTULOSE))
    has_rifaximin = bool(_has_med_signal(all_meds, _HE_REGIMEN_RIFAXIMIN))
    he_regimen = has_lactulose and has_rifaximin

    if icd_hits or proc_hits_tips or recurrent_para or he_regimen:
        ctx.has_cirrhosis = True
        for c in icd_hits:
            _add_evidence(ctx, "has_cirrhosis", f"icd:{c}")
        for p in proc_hits_tips:
            _add_evidence(
                ctx, "has_cirrhosis", f"procedure:{_proc_label(p)}"
            )
        if recurrent_para:
            _add_evidence(
                ctx,
                "has_cirrhosis",
                f"procedure:paracentesis_x{len(paracenteses)}",
            )
        if he_regimen:
            _add_evidence(ctx, "has_cirrhosis", "med:lactulose+rifaximin")


def _detect_copd(
    ctx: PatientClinicalContext,
    icd_codes: list[str],
    meds_continuous: list[dict],
    meds_intermittent: list[dict],
) -> None:
    """Rule 3 — COPD / chronic hypoxemia.

    Trigger on COPD ICD codes OR active long-acting bronchodilator
    (tiotropium/umeclidinium) OR active ICS+LABA combination.
    """
    icd_hits = _icd_matches(icd_codes, _ICD_PREFIXES_COPD)
    med_hits = _has_med_signal(
        meds_continuous + meds_intermittent, _COPD_MED_SIGNALS
    )

    if icd_hits or med_hits:
        ctx.has_copd = True
        for c in icd_hits:
            _add_evidence(ctx, "has_copd", f"icd:{c}")
        for m in med_hits:
            _add_evidence(ctx, "has_copd", f"med:{m}")


def _detect_chronic_afib(
    ctx: PatientClinicalContext, icd_codes: list[str]
) -> None:
    """Rule 5 — Chronic atrial fibrillation (ICD-only after PR 1 amendment).

    Original spec also accepted "rhythm-control medication AND active
    anticoagulation" as a signal. Anticoag detection is deferred (see
    module header), so the medication-coupled path is dropped — chronic
    AF detection here is ICD-only. Rhythm-strip / vitals-derived inference
    is intentionally excluded (not deterministic enough).
    """
    icd_hits = _icd_matches(icd_codes, _ICD_PREFIXES_AFIB)
    if icd_hits:
        ctx.has_chronic_afib = True
        for c in icd_hits:
            _add_evidence(ctx, "has_chronic_afib", f"icd:{c}")


def _detect_chronic_trach(
    ctx: PatientClinicalContext,
    icd_codes: list[str],
    procedures: list[dict],
    respiratory: Optional[pl.DataFrame],
    admission_dttm: Optional[datetime],
) -> None:
    """Rule 7 — Chronic tracheostomy / vent dependence.

    Trigger on any of:
      - Tracheostomy procedure dated >30 days before admission (chronic
        timing). A trach within the current admission is recorded as
        ``acute_trach`` evidence but does NOT set the chronic flag.
      - ``device_category`` consistently in the chronic-trach set across
        all respiratory rows in the lookback (not transient).
      - ICD code Z93.0 (tracheostomy status).

    Cross-check note: ``orchestrator._determine_vent_status`` reads the
    same ``device_category`` field directly, so detection here and the
    deterministic vent narrative use the same underlying signal.
    """
    icd_hits = _icd_matches(icd_codes, _ICD_PREFIXES_CHRONIC_TRACH)

    trach_procs = _proc_keyword_hits(procedures, _TRACH_PROCEDURE_KEYWORDS)
    chronic_proc_evidence = []
    acute_proc_evidence = []
    if trach_procs and admission_dttm is not None:
        for p in trach_procs:
            proc_dttm = _parse_dttm(
                p.get("procedure_dttm")
                or p.get("start_dttm")
                or p.get("recorded_dttm")
            )
            if proc_dttm is None:
                # Undated procedure — ambiguous; skip rather than guess.
                continue
            delta_days = (admission_dttm - proc_dttm).days
            if delta_days > _CHRONIC_TRACH_DAYS_BEFORE_ADMIT:
                chronic_proc_evidence.append(_proc_label(p))
            else:
                acute_proc_evidence.append(_proc_label(p))

    # Device-category consistency: every non-null device row in the
    # lookback must be in the chronic-trach set. Transient trach mention
    # (e.g., post-op recovery) doesn't qualify.
    consistent_trach_device = False
    if (
        respiratory is not None
        and len(respiratory) > 0
        and "device_category" in respiratory.columns
    ):
        devices = (
            respiratory["device_category"]
            .drop_nulls()
            .to_list()
        )
        if devices:
            normalized = [str(d).strip().lower() for d in devices]
            if all(d in _CHRONIC_TRACH_DEVICE_CATEGORIES for d in normalized):
                consistent_trach_device = True

    if icd_hits or chronic_proc_evidence or consistent_trach_device:
        ctx.has_chronic_trach = True
        for c in icd_hits:
            _add_evidence(ctx, "has_chronic_trach", f"icd:{c}")
        for label in chronic_proc_evidence:
            _add_evidence(
                ctx, "has_chronic_trach", f"procedure:{label}_pre_admit_30d+"
            )
        if consistent_trach_device:
            _add_evidence(
                ctx, "has_chronic_trach", "device_category:trach_consistent"
            )

    # Acute trach is recorded as evidence but does not set the chronic flag.
    for label in acute_proc_evidence:
        _add_evidence(
            ctx, "has_chronic_trach", f"acute_trach:{label}"
        )


def _compute_baselines(
    ctx: PatientClinicalContext, labs: Optional[pl.DataFrame]
) -> None:
    """Set ``baseline_creatinine`` and ``baseline_hgb`` when ≥3 in-window
    measurements exist for that lab. ``baseline_inr`` remains deferred —
    cirrhosis context handles the chronic-INR story without a numeric
    anchor."""
    ctx.baseline_creatinine = _earliest_lab_value(labs, "creatinine")
    ctx.baseline_hgb = _earliest_lab_value(labs, "hemoglobin")


# Therapeutic anticoagulation signals. Kept in sync with
# ``tools.drug_interactions.ANTICOAGULANT_CANONICAL`` — same agent set,
# same intent (treat presence as therapeutic-dose). Cangrelor and
# eptifibatide are filed under the CLIF mCIDE anticoag bucket but are
# GPIIb/IIIa antiplatelets — they are intentionally NOT counted here.
_THERAPEUTIC_ANTICOAG_SIGNALS = {
    "heparin",
    "bivalirudin",
    "argatroban",
    "warfarin",
    "enoxaparin",
    # CLIF brand-name / synonym fall-throughs that the lowercased
    # _med_label() substring matcher needs to catch directly when
    # med_category is absent.
    "angiomax",      # bivalirudin
    "lovenox",       # enoxaparin
    "coumadin",      # warfarin
    "jantoven",      # warfarin
}


def _detect_therapeutic_anticoagulation(
    ctx: PatientClinicalContext,
    meds_continuous: list[dict],
    meds_intermittent: list[dict],
) -> None:
    """Detect therapeutic-dose anticoagulation from the patient's med list.

    Sets ``ctx.on_therapeutic_anticoagulation`` to a pipe-joined list of
    canonical agent names (e.g. "heparin|warfarin") when one or more
    therapeutic anticoag agents are present in either meds layer.
    Otherwise leaves the field ``None``.

    Scope (see module header for full reasoning):
      - Includes IV continuous (heparin, bivalirudin, argatroban) and
        oral/SC intermittent (warfarin, enoxaparin).
      - Excludes cangrelor / eptifibatide (GPIIb/IIIa antiplatelets,
        not anticoag — kept in sync with drug_interactions.py).
      - Does NOT distinguish enoxaparin therapeutic from prophylactic
        dose (CLIF v2.1 lacks reliable dose-to-purpose mapping);
        prophylaxis trips the flag conservatively.
      - Oral DOACs (apixaban, rivaroxaban, dabigatran, edoxaban) are
        not pipeline-visible in CLIF v2.1 mCIDE and will silently miss.
    """
    hits = _has_med_signal(
        meds_continuous + meds_intermittent, _THERAPEUTIC_ANTICOAG_SIGNALS
    )
    if not hits:
        return

    # Canonicalize: collapse brand-name hits into their generic so the
    # output field reads cleanly. Maps each detected substring to a
    # canonical agent name from ANTICOAGULANT_CANONICAL.
    _BRAND_TO_CANONICAL = {
        "angiomax": "bivalirudin",
        "lovenox": "enoxaparin",
        "coumadin": "warfarin",
        "jantoven": "warfarin",
    }
    canonical = {_BRAND_TO_CANONICAL.get(h, h) for h in hits}
    ctx.on_therapeutic_anticoagulation = "|".join(sorted(canonical))
    for h in hits:
        _add_evidence(ctx, "on_therapeutic_anticoagulation", f"med:{h}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def infer_clinical_context(patient: "PatientContext") -> PatientClinicalContext:
    """Build a ``PatientClinicalContext`` from a populated ``PatientContext``.

    Detection runs against the same DataFrames the retriever attaches to
    ``PatientContext`` after data loading, so this function should be
    invoked at the end of ``DataRetriever.retrieve()``.

    Reads (all optional — missing tables degrade detection, never crash):
      - ``patient.diagnoses`` (ICD codes)
      - ``patient.meds_continuous`` / ``patient.meds_intermittent``
      - ``patient.respiratory_support`` (device_category for chronic trach)
      - ``patient.crrt_therapy`` (supportive ESRD signal)
      - ``patient.procedures`` (dialysis/TIPS/paracentesis/trach)
      - ``patient.labs`` (creatinine/hgb baselines)
      - ``patient.admission_dttm`` (chronic-trach timing)
      - ``patient.reference_dttm`` (currently unused; kept for future
        time-windowed detection)
    """
    ctx = PatientClinicalContext()

    icd_codes = _get_icd_codes(patient.diagnoses)
    meds_cont = _med_rows(patient.meds_continuous)
    meds_int = _med_rows(patient.meds_intermittent)
    procedures = _proc_rows(patient.procedures)
    admission_dttm = _parse_dttm(patient.admission_dttm)

    _detect_esrd_dialysis(
        ctx,
        icd_codes=icd_codes,
        meds_continuous=meds_cont,
        meds_intermittent=meds_int,
        crrt=patient.crrt_therapy,
        procedures=procedures,
        reference_dttm=patient.reference_dttm,
    )
    _detect_cirrhosis(
        ctx,
        icd_codes=icd_codes,
        meds_continuous=meds_cont,
        meds_intermittent=meds_int,
        procedures=procedures,
    )
    _detect_copd(
        ctx,
        icd_codes=icd_codes,
        meds_continuous=meds_cont,
        meds_intermittent=meds_int,
    )
    _detect_chronic_afib(ctx, icd_codes=icd_codes)
    _detect_chronic_trach(
        ctx,
        icd_codes=icd_codes,
        procedures=procedures,
        respiratory=patient.respiratory_support,
        admission_dttm=admission_dttm,
    )
    _detect_therapeutic_anticoagulation(
        ctx,
        meds_continuous=meds_cont,
        meds_intermittent=meds_int,
    )

    _compute_baselines(ctx, patient.labs)

    return ctx

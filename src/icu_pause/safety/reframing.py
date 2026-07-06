"""Patient-context-aware reframing of lab range warnings.

Given a lab flag (lab_name, value, unit, status) plus a deterministic
``PatientClinicalContext`` (chronic conditions), returns a context phrase
that downstream stringification appends to the human-readable warning —
e.g. ``chronic elevation from ESRD on HD`` next to a Cr 4.2 high flag,
instead of a naked "creatinine high" warning that would prompt
unnecessary workup.

Causation framing, not baseline-prediction framing
--------------------------------------------------

Non-critical reframes use *causation* language ("chronic elevation from
ESRD on HD", "chronic depression from cirrhosis (synthetic dysfunction)")
rather than *prediction* language ("at baseline for ESRD/HD", "expected
for chronic ESRD"). Rationale: the system only has the patient's
problem list. It does NOT have the patient's actual prior values, so
"at baseline" overclaims patient-specific knowledge.

CHRONIC vs REVIEW tier
----------------------

Each non-critical reframe declares a tier on the ``Reframing`` it
returns:

  * **CHRONIC** — the system has either (a) a patient-priors anchor for
    this lab (e.g. ``baseline_creatinine``, ``baseline_hgb``) or (b) a
    self-verifying mechanism+action phrase that already asks the
    clinician to verify (e.g. K+ "confirm session schedule"). Rendered
    by qa.py as ``[LAB_RANGE/CHRONIC] ... — {text} (vs. general
    reference X)``.

  * **REVIEW** — chronic context applies but the system has no patient-
    priors mechanism for this lab. Rendered as ``[LAB_RANGE/REVIEW] ...
    — {text}; confirm consistent with prior (vs. general reference X)``.
    The "confirm consistent with prior" prompt is appended by qa.py so
    each reframe doesn't have to remember it.

Current tier assignment:

  * Cr high w/ baseline → CHRONIC; w/o baseline → REVIEW.
  * K+ high → CHRONIC (mechanism+action, self-verifying).
  * Hgb low w/ baseline_hgb → CHRONIC; w/o → REVIEW.
  * Cirrhosis bili/albumin/plt → REVIEW (no per-lab priors mechanism).
  * Cirrhosis INR w/ anticoag detected → CHRONIC (anticoag-aware wording).
  * Cirrhosis INR w/o anticoag → REVIEW.

Anticoag-aware INR
------------------

The cirrhosis INR↑ reframe now branches on
``ctx.on_therapeutic_anticoagulation`` (set by ``clinical_context``
detection — see that file's header). When anticoag is detected the
reframe text becomes "elevated INR on therapeutic {agent}; verify
within target range" — the meds explain the elevation more naturally
than cirrhosis. Without anticoag, the cirrhosis context applies under
REVIEW tier with explicit "not anticoagulated" qualifier.

Severity-aware design (clinical-safety constraint)
--------------------------------------------------

Reframe text is split into two modes based on flag ``status``:

  * **Non-critical (high / low)** — causation phrasings ("chronic
    elevation from ESRD on HD", "chronic depression from cirrhosis").
    Chronic disease explains the abnormality; not a prediction that
    the value is at the patient's baseline.

  * **Critical (critical_high / critical_low)** — must be
    *action-guiding* (e.g. "common in ESRD between HD sessions;
    confirm session schedule and trend") or *suppressed entirely*
    (return None). Soft / chronic-causation wording on a critical line
    undermines severity preservation for a human reader. A Hgb 6.5 in
    an ESRD patient is below CKD-anemia targets and the action is
    "transfuse" regardless of chronicity.

Returned text is a *context phrase only* — it does NOT repeat the lab
name, value, or unit. The caller (``check_lab_ranges`` →
``qa.py`` stringification) owns the lab/value/reference-range prefix.
This avoids double-stating the value in rendered QA lines like
"potassium = 6.7 mEq/L is critically high — context: K 6.7 mEq/L —
common in ESRD".

Severity floor at the data layer
--------------------------------

The flag's ``status`` field is the canonical severity and is NEVER
modified here. Reframing only affects the textual context layer.

Multi-context precedence
------------------------

Each lab maps to one context (Cr/K+/Hgb → ESRD; INR/bili/alb/plt →
cirrhosis), so there's no conflict in the PR 3 scope. When anticoag
detection ships, INR will additionally route to anticoag context for
warfarin-on patients (anticoag wins over cirrhosis for INR — see
``clinical_context.py`` header note).

Out of scope (lab_ranges path only)
-----------------------------------

Vitals-based reframes (SpO2 in COPD target range, irregular rhythm in
chronic AF, "not on ventilator" for chronic trach) and pCO2 (not in
``LAB_REFERENCE_RANGES``) need different code paths and ship separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from icu_pause.safety.clinical_context import PatientClinicalContext


# How far above ``baseline_creatinine`` is "above baseline, evaluate"
# rather than "at baseline." 20% per spec.
_CR_ABOVE_BASELINE_THRESHOLD = 0.20


@dataclass
class Reframing:
    """A reframing decision for a single lab flag.

    ``text`` is a context phrase only — never repeats lab name, value,
    or unit (the caller owns the prefix). Examples:

      * ``"chronic elevation from ESRD on HD"`` (tier=CHRONIC)
      * ``"common between HD sessions; confirm session schedule"`` (tier=CHRONIC)
      * ``"above patient's HD baseline of 4.0, consider workup"`` (tier=CHRONIC)
      * ``"chronic ESRD/HD context"`` (tier=REVIEW — qa.py appends "confirm consistent with prior")
      * ``"chronic ESRD/HD; assess for AKI-on-ESRD vs chronic baseline elevation"`` (critical)

    ``tier`` controls the rendered tag and trailing action prompt in
    qa.py:
      * ``"CHRONIC"`` — system has either patient-priors backing the
        chronic claim (e.g. baseline_creatinine) or a self-verifying
        mechanism+action phrase (e.g. K+ "confirm session schedule").
        Renders as ``[LAB_RANGE/CHRONIC] ... — {text} (vs. general
        reference X)``.
      * ``"REVIEW"`` — chronic context applies but the system has no
        patient-priors mechanism for this lab. Renders as
        ``[LAB_RANGE/REVIEW] ... — {text}; confirm consistent with
        prior (vs. general reference X)``.
    Defaults to CHRONIC.

    ``context_applied`` lists the ``PatientClinicalContext`` flag(s)
    that produced this reframing (audit trail).
    """

    text: str
    context_applied: list[str]
    tier: str = "CHRONIC"


def reframe_lab_warning(
    lab_name: str,
    value: float,
    unit: str,
    status: str,
    ctx: "PatientClinicalContext",
) -> Optional[Reframing]:
    """Return a ``Reframing`` if the patient's chronic context applies
    to this lab+status, else ``None``.

    For *critical* statuses, returns either an action-guiding context
    phrase (the chronic context changes how the receiving team should
    act) or ``None`` (chronic context doesn't change action — let the
    bare critical alarm render unmodified).
    """
    lab_lower = lab_name.lower()

    # ----- ESRD / chronic dialysis -------------------------------------
    if ctx.has_esrd_dialysis:
        if lab_lower == "creatinine":
            return _reframe_esrd_creatinine(value, status, ctx)
        if lab_lower == "potassium":
            if status == "high":
                # Non-critical hyperkalemia in ESRD is the typical
                # inter-dialytic upward drift. CHRONIC tier — the
                # mechanism+action text ("confirm session schedule")
                # is self-verifying, so no REVIEW prompt needed.
                return Reframing(
                    text="common between HD sessions; confirm session schedule",
                    context_applied=["has_esrd_dialysis"],
                    tier="CHRONIC",
                )
            if status == "critical_high":
                # Critical K+ in ESRD is still an emergency — the action
                # is "treat hyperkalemia + arrange urgent dialysis." The
                # context here is action-guiding, not soothing.
                return Reframing(
                    text=(
                        "common in ESRD between HD sessions; confirm "
                        "session schedule and trend"
                    ),
                    context_applied=["has_esrd_dialysis"],
                )
        if lab_lower == "hemoglobin":
            if status == "low":
                # Non-critical anemia of CKD. REVIEW tier when no
                # baseline_hgb is available — clinician should verify
                # this Hgb is consistent with the patient's chronic
                # anemia rather than acute bleeding. CHRONIC tier when
                # baseline_hgb is known (we have priors backing the
                # claim).
                if ctx.baseline_hgb is None:
                    return Reframing(
                        text="chronic CKD anemia context",
                        context_applied=["has_esrd_dialysis"],
                        tier="REVIEW",
                    )
                return Reframing(
                    text="chronic depression from anemia of CKD",
                    context_applied=["has_esrd_dialysis"],
                    tier="CHRONIC",
                )
            if status == "critical_low":
                # Hgb in critical-low range (<7) is below CKD-anemia
                # targets — transfuse regardless of chronicity. No
                # reframe text added; the critical alarm stands alone.
                return None

    # ----- Cirrhosis ---------------------------------------------------
    if ctx.has_cirrhosis:
        if lab_lower == "inr":
            if status == "high":
                # Anticoag-aware: if the patient is on therapeutic
                # anticoagulation, the meds explain the INR more
                # naturally than cirrhosis. Route to anticoag wording.
                # If not anticoagulated, fall back to chronic-cirrhosis
                # context under REVIEW (no INR-priors in clinical_ctx).
                anticoag = ctx.on_therapeutic_anticoagulation
                if anticoag:
                    # Humanize the canonical pipe-joined field
                    # ("heparin|warfarin") to read naturally
                    # ("heparin + warfarin"). The raw field stays
                    # pipe-joined for machine-readable parsing
                    # downstream.
                    agents = " + ".join(anticoag.split("|"))
                    return Reframing(
                        text=(
                            f"elevated INR on therapeutic {agents}; "
                            "verify within target range"
                        ),
                        context_applied=[
                            "has_cirrhosis",
                            "on_therapeutic_anticoagulation",
                        ],
                        tier="CHRONIC",
                    )
                return Reframing(
                    text="chronic cirrhosis context, not anticoagulated",
                    context_applied=["has_cirrhosis"],
                    tier="REVIEW",
                )
            if status == "critical_high":
                # Critical INR (>4.0) means bleeding risk regardless of
                # cirrhosis. Action is FFP / vitamin K / PCC. Suppress.
                return None
        if lab_lower == "bilirubin_total":
            if status == "high":
                # No bilirubin-priors mechanism in clinical_context →
                # REVIEW. Clinician verifies this is chronic, not new
                # decompensation / acute biliary obstruction.
                return Reframing(
                    text="chronic cirrhosis context",
                    context_applied=["has_cirrhosis"],
                    tier="REVIEW",
                )
            if status == "critical_high":
                # Critical bili (>10) is decompensated-liver-disease
                # / transplant territory; chronic context doesn't
                # change the action (call hepatology). Suppress.
                return None
        if lab_lower == "albumin":
            if status == "low":
                # No albumin-priors → REVIEW. Clinician verifies the
                # depression is chronic synthetic dysfunction rather
                # than new third-spacing / acute illness.
                return Reframing(
                    text="chronic cirrhosis context (synthetic dysfunction)",
                    context_applied=["has_cirrhosis"],
                    tier="REVIEW",
                )
            if status == "critical_low":
                # Critical albumin (<1.5) is volume/transfusion concern
                # regardless of cirrhosis. Suppress.
                return None
        if lab_lower == "platelets":
            if status == "low":
                # No platelet-priors → REVIEW. Splenic-sequestration
                # mechanism kept in the body text (clinically useful);
                # REVIEW tag tells the clinician to verify this is
                # chronic rather than acute consumption / DIC.
                return Reframing(
                    text="chronic cirrhosis context (splenic sequestration)",
                    context_applied=["has_cirrhosis"],
                    tier="REVIEW",
                )
            if status == "critical_low":
                # Critical plt (<50) is bleeding-risk territory. Even
                # cirrhotic plt <50 is a transfusion call when bleeding;
                # chronic context doesn't change the action. Suppress.
                return None

    return None


def _reframe_esrd_creatinine(
    value: float, status: str, ctx: "PatientClinicalContext"
) -> Optional[Reframing]:
    """Cr-specific reframing — split by status and by whether the system
    has a patient-specific anchor (``baseline_creatinine``).

    Non-critical Cr high in ESRD:
      * Baseline known + value >20% above → CHRONIC,
        "above patient's HD baseline of {baseline}, consider workup"
        (patient-specific anchor backs the deviation claim).
      * Baseline known + value within 20% → CHRONIC,
        "chronic elevation from ESRD on HD" (priors anchor the chronic
        claim, even if the earliest-in-window value is imperfect).
      * Baseline unknown → REVIEW,
        "chronic ESRD/HD context" (no patient priors; clinician should
        verify this is consistent with the patient's prior values).

    Critical Cr (>4.0) in ESRD:
      * Always action-guiding. Receiving team needs to assess AKI-on-
        ESRD vs chronic baseline elevation.
    """
    baseline = ctx.baseline_creatinine

    if status == "high":
        if baseline is not None and baseline > 0:
            ratio = (value - baseline) / baseline
            if ratio > _CR_ABOVE_BASELINE_THRESHOLD:
                return Reframing(
                    text=f"above patient's HD baseline of {baseline}, consider workup",
                    context_applied=["has_esrd_dialysis"],
                    tier="CHRONIC",
                )
            return Reframing(
                text="chronic elevation from ESRD on HD",
                context_applied=["has_esrd_dialysis"],
                tier="CHRONIC",
            )
        # No patient-specific baseline → REVIEW tier.
        return Reframing(
            text="chronic ESRD/HD context",
            context_applied=["has_esrd_dialysis"],
            tier="REVIEW",
        )

    if status == "critical_high":
        # Frame the differential ("rule out AKI-on-ESRD") without claiming
        # the system computed an HD-cycle baseline. The receiving team
        # needs to think about superimposed acute injury vs. the patient's
        # chronic elevation; we name the clinical entity, not a number.
        return Reframing(
            text=(
                "chronic ESRD/HD; assess for AKI-on-ESRD "
                "vs chronic baseline elevation"
            ),
            context_applied=["has_esrd_dialysis"],
        )

    return None

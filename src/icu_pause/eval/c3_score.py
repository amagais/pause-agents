"""C3 two-sided fidelity scorer — data model + the DETERMINISTIC recall core.

Consumes the I-PASS/JC gold registry (c3_definitions) and scores a brief against
the FULL CHART two ways (plan C3):
  - RECALL    : gold element present in chart -> present + correct in brief? (miss = omission)
  - PRECISION : brief asserts a value for the element -> matches chart? (mismatch = fabrication)

This module implements the DETERMINISTIC half now (the structured-numeric elements
backed by numeric_fidelity: vitals, pressor dose, vent FiO2/PEEP, antibiotic days,
CRRT/dialysis). Prose-dependent elements (code status, DPOA, contingencies, ...) are
marked PENDING_LLM with a clean hook for the hybrid LLM extractor/verifier that
follows. Coverage (how much of the harm-weight is deterministically resolved) is
reported honestly so nothing is silently scored as present-and-correct.

The harm-weighted composite is later VALIDATED as a proxy against clinician PDSQI-9
`accurate`/`cited` (the demote-to-proxy step) — this module does not assert it is the
ground-truth harm metric, only computes it.

Pure-python except for the numeric_fidelity import; no LLM, no network here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from statistics import harmonic_mean

from icu_pause.eval import c3_definitions as C3
from icu_pause.eval import numeric_fidelity as NF

# numeric_fidelity GTValue.data_type + label -> gold element id
_VALUE_TO_ELEMENT = {
    NF.VITALS: lambda lbl: "vitals_trends",
    NF.PRESSOR: lambda lbl: "pressor_dose",
    NF.ABX: lambda lbl: "antibiotics_day",
    NF.DIALYSIS: lambda lbl: "crrt_ecmo",
    NF.VENT: lambda lbl: "vent_peep" if "peep" in lbl.lower() else "vent_fio2",
}

# status codes
CORRECT = "correct"            # salient in chart, brief reproduces it correctly
OMISSION = "omission"          # salient in chart, brief missing it (recall miss)
VALUE_MISMATCH = "value_mismatch"  # brief asserts element but value wrong (precision fail)
FABRICATION = "fabrication"    # brief asserts element value absent/contradicted in chart
NA_ABSENT = "na_absent"        # not salient in chart, brief silent -> not scored
PENDING_LLM = "pending_llm"    # routed to the hybrid LLM pass (not yet run)
PENDING_DET = "pending_det"    # deterministic-capable but matcher not yet built (labs, vent_dependence)
LLM_PARSE_FAIL = "llm_parse_fail"  # LLM returned no usable verdict for this element

# statuses that represent a REAL resolution (everything else = unresolved/failed,
# and must NOT count toward coverage_resolved — the honesty guard).
RESOLVED_STATUSES = frozenset({CORRECT, OMISSION, VALUE_MISMATCH, FABRICATION, NA_ABSENT})


@dataclass
class ElementVerdict:
    element_id: str
    harm: int
    salient: bool                 # present/applicable in the chart
    status: str
    recall_ok: bool | None = None  # brief contains the chart fact, correct (None if N/A)
    precision_ok: bool | None = None  # brief's assertion matches chart (None if no assertion)
    detail: str = ""


@dataclass
class C3Result:
    hosp_id: str
    arm: str
    generator: str
    verdicts: list = field(default_factory=list)

    # ---- harm-weighted two-sided aggregates over RESOLVED salient elements ----
    def _w(self, predicate):
        return sum(v.harm for v in self.verdicts if predicate(v))

    def recall(self):
        denom = self._w(lambda v: v.salient and v.recall_ok is not None)
        num = self._w(lambda v: v.salient and v.recall_ok is True)
        return (num / denom) if denom else None

    def precision(self):
        denom = self._w(lambda v: v.precision_ok is not None)
        num = self._w(lambda v: v.precision_ok is True)
        return (num / denom) if denom else None

    def f_two_sided(self):
        r, p = self.recall(), self.precision()
        if r is None or p is None or r == 0 or p == 0:
            return None
        return harmonic_mean([r, p])

    def omissions(self):
        return [v.element_id for v in self.verdicts if v.status == OMISSION]

    def fabrications(self):
        return [v.element_id for v in self.verdicts if v.status == FABRICATION]

    def coverage(self):
        """Fraction of total harm-weight that is REALLY resolved — honesty guard.
        Anything not in RESOLVED_STATUSES (pending OR llm_parse_fail) counts as
        unresolved, so a failed LLM pass shows as low coverage, never 1.0."""
        tot = sum(g.harm for g in C3.GOLD_REGISTRY)
        unresolved = self._w(lambda v: v.status not in RESOLVED_STATUSES)
        return (tot - unresolved) / tot if tot else None

    def to_dict(self):
        return {"hosp_id": self.hosp_id, "arm": self.arm, "generator": self.generator,
                "recall": self.recall(), "precision": self.precision(),
                "f_two_sided": self.f_two_sided(), "coverage_resolved": self.coverage(),
                "omissions": self.omissions(), "fabrications": self.fabrications(),
                "verdicts": [asdict(v) for v in self.verdicts]}


def score_deterministic(brief_sections: dict, serialized_context: dict,
                        hosp_id: str = "", arm: str = "", generator: str = "") -> C3Result:
    """Deterministic two-sided pass over the numeric-backed gold elements.

    brief_sections: {section_key: text}
    serialized_context: the patient_context_text dict numeric_fidelity expects
                        (from retrieve_bundle / serialize_to_json).
    """
    gt = NF.extract_ground_truth(serialized_context)
    fr = NF.score_brief(brief_sections, gt)

    # group numeric_fidelity per-value results by gold element
    per_elem: dict[str, list[bool]] = {}
    for gtv, retained in fr.value_results:
        mapper = _VALUE_TO_ELEMENT.get(gtv.data_type)
        if not mapper:
            continue
        eid = mapper(gtv.label)
        per_elem.setdefault(eid, []).append(bool(retained))

    by_id = {g.id: g for g in C3.GOLD_REGISTRY}
    verdicts: list[ElementVerdict] = []
    resolved_ids = set()

    for eid, hits in per_elem.items():
        g = by_id.get(eid)
        if not g:
            continue
        resolved_ids.add(eid)
        salient = len(hits) > 0
        n_hit = sum(hits)
        # recall: per element, treat "all required values reproduced" via the
        # element's tolerance.recall threshold (default 1.0 = all).
        thr = (g.tolerance or {}).get("recall", 1.0)
        recall_ok = (n_hit / len(hits)) >= thr if hits else None
        status = CORRECT if recall_ok else OMISSION
        verdicts.append(ElementVerdict(
            eid, g.harm, salient, status, recall_ok=recall_ok,
            detail=f"{n_hit}/{len(hits)} chart values reproduced (thr={thr})"))

    # everything else in the registry -> pending (LLM, or deterministic-not-yet-built)
    DET_TODO = {"critical_labs", "vent_dependence"}  # structured but need a bespoke matcher
    for g in C3.GOLD_REGISTRY:
        if g.id in resolved_ids:
            continue
        st = PENDING_DET if g.id in DET_TODO else PENDING_LLM
        verdicts.append(ElementVerdict(g.id, g.harm, salient=False, status=st,
                                       detail="hybrid LLM pass / bespoke matcher TODO"))

    return C3Result(hosp_id, arm, generator, verdicts)


if __name__ == "__main__":
    # tiny smoke: a chart with pressor + vent + a vital; a brief that keeps the
    # pressor dose and FiO2 but DROPS the PEEP (an omission).
    ctx = {
        "medications": {"continuous": [
            {"medication_name": "norepinephrine", "dose": 0.10, "dose_unit": "mcg/kg/min"}]},
        "respiratory": [{"fio2_set": 0.40, "peep_set": 8, "recorded_dttm": "2026-01-01 00:00"}],
        "vitals": {"recent_raw": [{"vital_category": "heart_rate", "vital_value": 88}]},
    }
    brief = {"E": "On norepinephrine 0.1 mcg/kg/min. FiO2 40%. HR 88.",  # PEEP omitted
             "S": "Septic shock, improving."}
    res = score_deterministic(brief, ctx, hosp_id="TEST", arm="full", generator="demo")
    d = res.to_dict()
    print(f"recall={d['recall']} precision={d['precision']} f={d['f_two_sided']} "
          f"coverage_resolved={d['coverage_resolved']:.2f}")
    print("omissions:", d["omissions"])
    for v in res.verdicts:
        if v.status not in (PENDING_LLM, PENDING_DET):
            print(f"  {v.element_id:16s} harm={v.harm} salient={v.salient} "
                  f"{v.status:14s} {v.detail}")

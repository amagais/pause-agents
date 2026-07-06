"""C3 hybrid LLM pass — the prose extractor/verifier for the gold elements that
deterministic matching can't resolve (code status, DPOA, contingencies, allergies,
lines, consultants, narrative course, ...) plus the prose precision side.

Default model = LOCAL DeepSeek-R1 (free on the HPC GPU, and — critically — NOT one
of the four generators under test, so it cannot self-prefer any arm's briefs;
gemma-4 / medgemma / qwen-3.6 are disqualified as C3 judges for exactly that
reason). Provider/model are configurable so a small o4-mini calibration backstop is
a one-flag switch.

For each requested gold element the model returns, grounded in the FULL CHART:
  salient    : is there ground truth for this element in the chart?
  brief_states: does the brief assert anything about it?
  correct    : if both, does the brief's assertion match the chart?
which maps to the two-sided verdict (recall = omission side, precision = fabrication
side) used by c3_score.C3Result.

Reasoning-model safe: DeepSeek emits <think> blocks; we strip them and extract the
JSON array defensively (same failure mode that caused the o4-mini all-zeros).
"""

from __future__ import annotations

import json
import os
import re

from icu_pause.eval import c3_definitions as C3
from icu_pause.eval.c3_score import (
    ElementVerdict, CORRECT, OMISSION, VALUE_MISMATCH, FABRICATION, NA_ABSENT,
    LLM_PARSE_FAIL,
)

_SYSTEM = """You are a clinical hand-off FIDELITY auditor. You compare an ICU-to-ward \
transfer brief against the FULL SOURCE CHART and judge, for each listed hand-off \
element, whether the brief faithfully represents the chart.

For EACH element you are given, decide three booleans, judging ONLY from the chart \
and the brief (never outside knowledge):
  - "salient": does the CHART contain ground truth for this element (is it present / \
applicable for this patient)? If the chart has no information bearing on it, salient=false.
  - "brief_states": does the BRIEF assert anything specific about this element?
  - "correct": if salient AND brief_states, does the brief's assertion MATCH the chart \
(right value/category, not contradicted, not stale)? If not both, set correct=null.

Rules:
  - A correct OMISSION of a non-salient element is fine (salient=false, brief_states=false).
  - Asserting something the chart does not support is a FABRICATION (salient=false, \
brief_states=true).
  - Missing something the chart documents is an OMISSION (salient=true, brief_states=false).
  - Quote the minimal chart evidence and brief evidence you used.

Return ONLY a JSON array, one object per element, no prose outside the JSON:
[{"element_id": "...", "salient": true/false, "brief_states": true/false, \
"correct": true/false/null, "chart_evidence": "...", "brief_evidence": "..."}]"""


def _checklist(elements) -> str:
    lines = []
    for g in elements:
        lines.append(
            f'- element_id="{g.id}" | {g.label} | I-PASS {g.ipass} | '
            f'salient when: {g.salient_when}'
            + (f' | note: {g.notes}' if g.notes else ''))
    return "\n".join(lines)


def build_messages(brief_text: str, chart_json: str, elements):
    user = (
        f"## FULL SOURCE CHART (ground truth)\n{chart_json}\n\n"
        f"## TRANSFER BRIEF TO AUDIT\n{brief_text}\n\n"
        f"## HAND-OFF ELEMENTS TO JUDGE\n{_checklist(elements)}\n\n"
        f"Return the JSON array (one object per element_id above)."
    )
    return _SYSTEM, user


def _extract_json_array(raw: str):
    """Strip <think> and pull the first balanced JSON array. Returns [] on failure."""
    if not isinstance(raw, str):
        return []
    txt = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # also drop an unterminated leading <think> (truncated reasoning)
    if "<think>" in txt and "</think>" not in txt:
        txt = txt.split("<think>")[0]
    start = txt.find("[")
    if start < 0:
        return []
    depth = 0
    for i in range(start, len(txt)):
        c = txt[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(txt[start:i + 1])
                except Exception:  # noqa: BLE001
                    return []
    return []


def _verdict_from(obj, harm) -> ElementVerdict:
    eid = str(obj.get("element_id", ""))
    sal = bool(obj.get("salient"))
    states = bool(obj.get("brief_states"))
    corr = obj.get("correct")
    detail = (str(obj.get("chart_evidence", ""))[:160]
              + " || " + str(obj.get("brief_evidence", ""))[:160])
    if sal and states:
        if corr is True:
            return ElementVerdict(eid, harm, True, CORRECT, recall_ok=True,
                                  precision_ok=True, detail=detail)
        return ElementVerdict(eid, harm, True, VALUE_MISMATCH, recall_ok=False,
                              precision_ok=False, detail=detail)
    if sal and not states:
        return ElementVerdict(eid, harm, True, OMISSION, recall_ok=False, detail=detail)
    if (not sal) and states:
        return ElementVerdict(eid, harm, False, FABRICATION, precision_ok=False, detail=detail)
    return ElementVerdict(eid, harm, False, NA_ABSENT, detail=detail)


def evaluate_elements(llm, brief_text: str, chart_json: str, elements,
                      batch_size: int = 6, retries: int = 1) -> list:
    """Evaluate `elements` in small BATCHES (default 6). One huge call over all ~22
    prose elements makes a reasoning model (DeepSeek-R1) spend its whole output
    budget thinking and never emit the JSON array (the observed all-`llm_parse_fail`
    failure). Batching bounds the per-call reasoning so the JSON reliably lands; a
    retry recovers a transient empty parse. Elements still missing are flagged
    LLM_PARSE_FAIL (never silently scored — counts as unresolved in coverage)."""
    verdicts = []
    for i in range(0, len(elements), max(1, batch_size)):
        chunk = elements[i:i + max(1, batch_size)]
        by_id = {g.id: g for g in chunk}
        system, user = build_messages(brief_text, chart_json, chunk)
        got = {}
        for _ in range(retries + 1):
            raw = str(llm.invoke(system=system, user=user))
            dbg = os.environ.get("ICUPAUSE_C3_DEBUG_RAW")
            if dbg:
                with open(dbg, "a") as fh:
                    fh.write(f"\n\n===== batch {[g.id for g in chunk]} (len={len(raw)}) =====\n{raw}\n")
            arr = _extract_json_array(raw)
            for obj in arr:
                if isinstance(obj, dict) and obj.get("element_id") in by_id and obj["element_id"] not in got:
                    got[obj["element_id"]] = _verdict_from(obj, by_id[obj["element_id"]].harm)
            if len(got) == len(chunk):
                break
        for g in chunk:
            verdicts.append(got.get(g.id) or ElementVerdict(
                g.id, g.harm, salient=False, status=LLM_PARSE_FAIL,
                detail="element absent from model JSON"))
    return verdicts


def build_llm(provider: str = "local", model: str = "deepseek-r1"):
    """LLM at temperature 0.0 via the project's eval factory (same plumbing the
    PDSQI judge uses). DeepSeek/local served on the HPC vLLM = free."""
    from copy import copy
    from icu_pause.config import Settings
    from icu_pause.eval import create_eval_llm
    settings = copy(Settings())
    object.__setattr__(settings, "llm_temperature", 0.0)
    return create_eval_llm(settings, provider, model)

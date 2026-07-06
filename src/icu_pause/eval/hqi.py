"""ICU-PAUSE Handoff Quality Instrument (HQI) evaluator.

Trimmed to 3 attributes that are genuinely unique from PDSQI-9 and the
per-agent grounding check:

1. Schema Completeness (deterministic — checks section presence)
2. To-Do Actionability (LLM-as-judge — checks patient-specificity)
3. Data Currency (deterministic — binary pass/fail)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICU-PAUSE section keys that should be populated
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = ["I", "C", "U_unprescribing", "P", "A", "U_uncertainty", "S", "E"]
PLACEHOLDER_TEXT = "Not enough information from structured data."


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class HQIScore(BaseModel):
    """Scores for all 3 ICU-PAUSE-HQI attributes."""

    schema_completeness: int = 0  # 1-5 (deterministic)
    todo_actionability: int = 0  # 1-5 (LLM-as-judge)
    data_currency: bool = True  # Pass/Fail (deterministic)


class HQIEvaluation(BaseModel):
    """Complete ICU-PAUSE-HQI evaluation result."""

    scores: HQIScore
    reasoning: dict[str, str]  # attribute -> reasoning text
    total_score: float  # todo_actionability score (only Likert attribute)
    evaluator_model: str
    missing_sections: list[str]  # sections that were absent or placeholder


# ---------------------------------------------------------------------------
# Deterministic: Schema Completeness
# ---------------------------------------------------------------------------


def score_schema_completeness(
    sections: dict[str, str],
    todo_checklist: list[dict[str, str] | str] | None = None,
) -> tuple[int, str, list[str]]:
    """Deterministically score schema completeness.

    Returns (score 1-5, reasoning string, list of missing section keys).
    """
    missing = []
    placeholder = []

    for key in REQUIRED_SECTIONS:
        content = sections.get(key, "")
        if not content or not content.strip():
            missing.append(key)
        elif content.strip() == PLACEHOLDER_TEXT:
            placeholder.append(key)

    # Check to-do checklist
    if not todo_checklist or len(todo_checklist) == 0:
        # Only count as missing if S section doesn't already contain to-dos
        s_content = sections.get("S", "")
        if "[]" not in s_content and "to-do" not in s_content.lower():
            missing.append("todo_checklist")

    absent_count = len(missing)
    placeholder_count = len(placeholder)

    if absent_count >= 3:
        score = 1
    elif absent_count == 2:
        score = 2
    elif absent_count == 1 or placeholder_count > 0:
        score = 3
    elif placeholder_count == 0 and absent_count == 0:
        # All present — check if substantive (more than a short sentence)
        short_sections = [
            k for k in REQUIRED_SECTIONS
            if len(sections.get(k, "").strip()) < 20
            and sections.get(k, "").strip() != ""
            and sections.get(k, "").strip() != PLACEHOLDER_TEXT
        ]
        if short_sections:
            score = 4
        else:
            score = 5
    else:
        score = 4

    parts = []
    if missing:
        parts.append(f"Missing/empty: {', '.join(missing)}")
    if placeholder:
        parts.append(f"Placeholder only: {', '.join(placeholder)}")
    if not parts:
        parts.append("All sections present and substantively populated")
    reasoning = ". ".join(parts)

    return score, reasoning, missing


# ---------------------------------------------------------------------------
# LLM prompt for To-Do Actionability
# ---------------------------------------------------------------------------

_TODO_SYSTEM_PROMPT = """\
You are a hospitalist (ward attending physician) evaluating the To-Do list \
from an AI-generated ICU-to-ward transition brief. Score ONLY the quality \
of the pre-transfer to-do items.

## TO-DO LIST ACTIONABILITY

Evaluate whether the to-do items are:
1. PATIENT-SPECIFIC — refers to this patient's actual clinical situation, \
not generic tasks that would apply to any ICU patient
2. ACTIONABLE — can be completed before transfer, with clear next steps
3. ROLE-APPROPRIATE — clear who should do it (nurse, physician, pharmacy, etc.)
4. TIME-ANCHORED — when it should be done (before transfer, by specific time)

RED FLAGS for poor to-do quality (score low):
- Generic tasks: "continue medications", "monitor vitals", "follow up labs", \
"optimize patient", "continue current management"
- Tasks that apply to EVERY ICU patient: "monitor respiratory status", \
"monitor GCS", "continue DVT prophylaxis" (unless there is a specific \
reason for this patient)
- Post-transfer tasks framed as pre-transfer (e.g., "arrange outpatient \
follow-up" when patient hasn't transferred yet)
- Vague language: "consider", "as needed", "if appropriate"

GOOD to-do examples (score high):
- "Wean FiO2 from 50% to 40% and reassess SpO2 before transfer"
- "Obtain vancomycin trough at 14:00 — dose adjustment pending"
- "Reassess Foley catheter day 3 per CAUTI bundle — document indication"
- "Pharmacy to reconcile home metoprolol 50mg BID with current IV esmolol"
- "PT/OT to clear for ward-level mobility before transfer"

## SCORING

1 = No To-Do list present, or all items are non-specific (e.g., 'optimize \
patient', 'continue current plan')
2 = Tasks present but predominantly generic — could apply to any ICU patient. \
Few if any are specific to this patient's situation.
3 = Some tasks are patient-specific and actionable; others are generic or \
vague. Mix of quality.
4 = Most tasks are specific to this patient, pre-transfer achievable, and \
role-appropriate. Minor gaps — one or two could be more specific.
5 = All tasks are patient-specific, time-anchored, role-assigned, and \
clearly achievable before ICU discharge. No generic filler tasks.

## OUTPUT FORMAT
Return a JSON object:
{
  "score": <1-5>,
  "reasoning": "<justification noting specific good/bad to-do items>"
}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class HQIEvaluator:
    """Evaluate a transition brief using the trimmed ICU-PAUSE-HQI instrument.

    - Schema Completeness: deterministic (no LLM)
    - To-Do Actionability: LLM-as-judge (1 call)
    - Data Currency: deterministic (no LLM)
    """

    def __init__(self, settings: Settings):
        from icu_pause.eval import create_eval_llm

        self.llm: BaseLLM = create_eval_llm(
            settings, settings.hqi_llm_provider, settings.hqi_llm_model,
        )
        self._load_prompt(settings)

    def _load_prompt(self, settings: Settings) -> None:
        path = Path(settings.prompts_dir) / "hqi_todo_evaluator.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.todo_prompt = data.get("system_prompt", _TODO_SYSTEM_PROMPT)
        else:
            self.todo_prompt = _TODO_SYSTEM_PROMPT

    def _evaluate_todo_actionability(
        self, summary: str, source_notes: str,
    ) -> tuple[int, str]:
        """Score to-do actionability using LLM-as-judge."""
        user_message = (
            f"## SOURCE CLINICAL DATA\n{source_notes}\n\n"
            f"## ICU-TO-WARD TRANSITION BRIEF\n{summary}\n\n"
            f"Evaluate ONLY the To-Do list items in this brief. "
            f"Score their actionability and patient-specificity. Return the JSON."
        )

        try:
            response = self.llm.invoke(
                system=self.todo_prompt,
                user=user_message,
            )
            from icu_pause.llm.provider import _clean_llm_output

            cleaned = _clean_llm_output(response or "")
            parsed = json.loads(cleaned, strict=False)
            score = max(1, min(5, parsed.get("score", 0)))
            reasoning = parsed.get("reasoning", "")
            return score, reasoning
        except Exception as e:
            logger.warning(f"To-Do actionability evaluation failed: {e}")
            return 0, f"Evaluation failed: {e}"

    def _evaluate_data_currency(
        self, summary: str, source_notes: str,
    ) -> tuple[bool, str]:
        """Evaluate data currency.

        For now, this returns True (pass) as a placeholder. A full
        implementation would compare numeric values in the summary against
        the most recent values in the source data timestamps.
        """
        # TODO: Implement deterministic comparison of values in summary
        # against most recent source data timestamps
        return True, "Data currency check: placeholder (pass)"

    def evaluate(
        self,
        source_notes: str,
        summary: str,
        sections: dict[str, str] | None = None,
        todo_checklist: list[dict[str, str] | str] | None = None,
    ) -> HQIEvaluation:
        """Score a transition brief using the trimmed HQI instrument.

        Args:
            source_notes: Original clinical data.
            summary: The transition brief text.
            sections: Dict of section_key -> content (for schema completeness).
            todo_checklist: List of to-do items (for schema completeness).

        Returns:
            HQIEvaluation with 3 attribute scores.
        """
        # 1. Schema Completeness (deterministic)
        if sections:
            schema_score, schema_reasoning, missing = score_schema_completeness(
                sections, todo_checklist,
            )
        else:
            schema_score, schema_reasoning, missing = 0, "No sections provided", []

        # 2. To-Do Actionability (LLM)
        todo_score, todo_reasoning = self._evaluate_todo_actionability(
            summary, source_notes,
        )

        # 3. Data Currency (deterministic)
        currency_pass, currency_reasoning = self._evaluate_data_currency(
            summary, source_notes,
        )

        scores = HQIScore(
            schema_completeness=schema_score,
            todo_actionability=todo_score,
            data_currency=currency_pass,
        )

        reasoning = {
            "schema_completeness": schema_reasoning,
            "todo_actionability": todo_reasoning,
            "data_currency": currency_reasoning,
        }

        # Total score = to-do actionability (only Likert attribute)
        # Schema completeness is deterministic, data currency is binary
        total = float(todo_score) if 1 <= todo_score <= 5 else 0.0

        return HQIEvaluation(
            scores=scores,
            reasoning=reasoning,
            total_score=round(total, 2),
            evaluator_model=self.llm.last_usage.model,
            missing_sections=missing,
        )

"""ICU-PAUSE-specific quality rubric evaluator (LLM-as-a-Judge).

.. deprecated::
    This custom 6-attribute rubric is superseded by the validated PDSQI-9
    instrument in ``eval/pdsqi9.py`` (Croxford et al., npj Digital Medicine
    2025).  PDSQI-9 is a strict superset (9 attributes) with published,
    clinician-validated scoring definitions.  New evaluation workflows
    should use ``PDSQI9Evaluator`` instead.  This module is kept for
    backwards compatibility with existing evaluation results.

Original description:
Draft rubric — pending clinician review. Attributes are adapted from the
validated PDSQI-9 instrument, tailored to ICU handoff note generation
from structured data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RubricScore(BaseModel):
    """Score for a single rubric attribute."""

    attribute: str
    score: int  # 1-5 Likert
    reasoning: str


class RubricEvaluation(BaseModel):
    """Complete rubric evaluation result."""

    scores: list[RubricScore]
    overall_score: float  # mean of individual scores
    evaluator_model: str


# ---------------------------------------------------------------------------
# Default prompt (used if config/prompts/rubric_evaluator.yaml is absent)
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a clinical quality reviewer evaluating an AI-generated ICU handoff \
note against the source structured patient data. Score the note on each \
attribute below using a 1-5 Likert scale.

## RUBRIC ATTRIBUTES

<accurate>
ACCURATE — Do the clinical facts in the note match the source data?
1 = Multiple fabricated or incorrect values
2 = Several inaccuracies or unsupported claims
3 = Mostly accurate with minor discrepancies
4 = Accurate with at most one trivial error
5 = All facts verifiable from source data
</accurate>

<thorough>
THOROUGH — Does the note address all clinically relevant findings?
1 = Major clinical findings omitted
2 = Several relevant findings missing
3 = Covers key findings, some gaps
4 = Comprehensive with minor omissions
5 = All relevant clinical data addressed
</thorough>

<organized>
ORGANIZED — Does the note follow ICU-PAUSE structure logically?
1 = Information scattered, hard to follow
2 = Some structure but significant disorganization
3 = Follows ICU-PAUSE sections, minor flow issues
4 = Well-organized, clear section delineation
5 = Excellent structure, logical flow within and across sections
</organized>

<succinct>
SUCCINCT — Is the note concise without unnecessary repetition?
1 = Excessively verbose or heavily redundant across sections
2 = Significant repetition or unnecessary detail
3 = Mostly concise with some redundancy
4 = Concise with minimal repetition
5 = Optimally concise, no wasted content
</succinct>

<actionable>
ACTIONABLE — Does the note provide clear handoff priorities and to-do items?
1 = No actionable items or priorities identified
2 = Vague or incomplete action items
3 = Some action items but lacking specificity
4 = Clear action items with good specificity
5 = Comprehensive, prioritized, specific action items
</actionable>

<cited>
CITED — Does the note reference specific data values from the source?
1 = No specific values cited
2 = Few values cited, mostly narrative
3 = Some values cited for key findings
4 = Good use of specific values throughout
5 = Consistently cites specific values (e.g., HR 110→85, Cr 2.1→1.2)
</cited>

## OUTPUT FORMAT
Return a JSON object:
{
  "scores": [
    {"attribute": "accurate", "score": <1-5>, "reasoning": "<brief justification>"},
    {"attribute": "thorough", "score": <1-5>, "reasoning": "..."},
    {"attribute": "organized", "score": <1-5>, "reasoning": "..."},
    {"attribute": "succinct", "score": <1-5>, "reasoning": "..."},
    {"attribute": "actionable", "score": <1-5>, "reasoning": "..."},
    {"attribute": "cited", "score": <1-5>, "reasoning": "..."}
  ]
}
"""

RUBRIC_ATTRIBUTES = ["accurate", "thorough", "organized", "succinct", "actionable", "cited"]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ICUPauseRubricEvaluator:
    """Evaluate an ICU-PAUSE note against source data using a custom rubric."""

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = self._create_eval_llm(settings)
        self.system_prompt = self._load_prompt(settings)

    @staticmethod
    def _create_eval_llm(settings: Settings) -> BaseLLM:
        """Use eval-specific LLM settings if configured, else pipeline LLM."""
        if settings.eval_llm_provider and settings.eval_llm_model:
            from copy import copy

            eval_settings = copy(settings)
            object.__setattr__(eval_settings, "llm_provider", settings.eval_llm_provider)
            object.__setattr__(eval_settings, "llm_model", settings.eval_llm_model)
            return create_llm(eval_settings)
        return create_llm(settings)

    @staticmethod
    def _load_prompt(settings: Settings) -> str:
        path = Path(settings.prompts_dir) / "rubric_evaluator.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        return _DEFAULT_SYSTEM_PROMPT

    def evaluate(
        self,
        patient_data: dict[str, Any],
        generated_note: dict[str, Any],
    ) -> RubricEvaluation:
        """Score a generated ICU-PAUSE note against source patient data.

        Args:
            patient_data: The patient_context_text dict (source data).
            generated_note: The ICUPauseOutput dict (sections, etc.).

        Returns:
            RubricEvaluation with per-attribute scores and overall score.
        """
        sections_text = "\n\n".join(
            f"### {key}\n{value}"
            for key, value in generated_note.get("sections", {}).items()
        )

        user_message = (
            f"## SOURCE PATIENT DATA\n```json\n"
            f"{json.dumps(patient_data, indent=2, default=str)}\n```\n\n"
            f"## GENERATED ICU-PAUSE NOTE\n{sections_text}\n\n"
            f"Evaluate the note against the source data using the rubric. "
            f"Return the JSON scores."
        )

        try:
            response = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
            )
            from icu_pause.llm.provider import _clean_llm_output

            cleaned = _clean_llm_output(response or "")
            parsed = json.loads(cleaned, strict=False)

            scores = [
                RubricScore(**s) for s in parsed.get("scores", [])
            ]
        except Exception as e:
            logger.warning(f"Rubric evaluation failed: {e}")
            scores = [
                RubricScore(attribute=attr, score=0, reasoning=f"Evaluation failed: {e}")
                for attr in RUBRIC_ATTRIBUTES
            ]

        valid_scores = [s.score for s in scores if 1 <= s.score <= 5]
        overall = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return RubricEvaluation(
            scores=scores,
            overall_score=round(overall, 2),
            evaluator_model=self.llm.last_usage.model,
        )

"""Golden Case Battery evaluator (LLM-as-a-Judge).

Compares system-generated ICU-PAUSE briefs against expert-written reference
briefs using a multi-dimensional rubric. Designed for regression testing:
run the same golden cases after every prompt change to ensure quality.

Golden cases are stored as JSON files in a directory with the structure:
    golden_cases/
        <hospitalization_id>.json
            {
                "hospitalization_id": "...",
                "reference_brief": { "I": "...", "C": "...", ... },
                "critical_checks": [
                    "Pharmacy Agent must flag renal dosing for vancomycin",
                    "Respiratory Agent must note weaning from ventilator"
                ]
            }

The critical_checks field allows defining specific clinical patterns that
MUST appear in the output — these serve as deterministic unit tests for
clinical reasoning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, _clean_llm_output, _strip_code_fences, create_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GoldenCase(BaseModel):
    """A single golden case with expert-written reference."""

    hospitalization_id: str
    reference_brief: dict[str, str]  # section_key -> expert text
    critical_checks: list[str] = Field(default_factory=list)


class GoldenCaseMatchResult(BaseModel):
    """LLM-as-judge comparison of generated vs reference brief."""

    hospitalization_id: str
    clinical_accuracy: int = Field(ge=0, le=5, default=0)
    completeness: int = Field(ge=0, le=5, default=0)
    hallucination: int = Field(ge=0, le=5, default=0)  # 5 = no hallucination
    actionability: int = Field(ge=0, le=5, default=0)
    overall_score: float = 0.0
    reasoning: dict[str, str] = Field(default_factory=dict)
    critical_checks_passed: list[str] = Field(default_factory=list)
    critical_checks_failed: list[str] = Field(default_factory=list)


class GoldenCaseBatchResult(BaseModel):
    """Results across all golden cases."""

    results: list[GoldenCaseMatchResult]
    mean_clinical_accuracy: float
    mean_completeness: float
    mean_hallucination: float
    mean_actionability: float
    mean_overall: float
    critical_check_pass_rate: float
    evaluator_model: str


# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a hospitalist (ward attending physician) comparing a system-generated \
ICU-to-ward transition brief against an expert-written reference note. As the \
receiving physician who would accept this patient from the ICU, evaluate how \
well the generated brief matches the reference on each dimension.

## SCORING DIMENSIONS

<clinical_accuracy>
CLINICAL ACCURACY — Are the key clinical facts correct and matching the reference?
1 = Major clinical errors, critical facts wrong
2 = Several inaccuracies compared to reference
3 = Mostly accurate, some discrepancies
4 = Accurate with at most one trivial difference
5 = All clinical facts match the reference
</clinical_accuracy>

<completeness>
COMPLETENESS — Are all critical items from the reference present?
1 = Major clinical elements missing
2 = Several important items from reference omitted
3 = Most key items present, some gaps
4 = Nearly complete, minor omissions
5 = All critical items from reference are present
</completeness>

<hallucination>
HALLUCINATION — Does the generated note add incorrect information?
1 = Multiple fabricated facts not in reference or source data
2 = Several unsupported claims
3 = Minor unsupported additions
4 = At most one trivial unsupported detail
5 = No hallucinated content
</hallucination>

<actionability>
ACTIONABILITY — Are the to-do items and handoff priorities appropriate?
1 = No actionable items or completely wrong priorities
2 = Vague or mostly incorrect action items
3 = Some appropriate action items, missing key ones
4 = Good action items, minor gaps
5 = All appropriate action items present and well-prioritized
</actionability>

## OUTPUT FORMAT
Return a JSON object:
{
  "clinical_accuracy": <1-5>,
  "completeness": <1-5>,
  "hallucination": <1-5>,
  "actionability": <1-5>,
  "reasoning": {
    "clinical_accuracy": "<justification>",
    "completeness": "<justification>",
    "hallucination": "<justification>",
    "actionability": "<justification>"
  }
}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class GoldenCaseEvaluator:
    """Compare generated briefs against expert-written golden cases."""

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = self._create_eval_llm(settings)
        self.system_prompt = self._load_prompt(settings)

    @staticmethod
    def _create_eval_llm(settings: Settings) -> BaseLLM:
        if settings.eval_llm_provider and settings.eval_llm_model:
            from copy import copy

            eval_settings = copy(settings)
            object.__setattr__(eval_settings, "llm_provider", settings.eval_llm_provider)
            object.__setattr__(eval_settings, "llm_model", settings.eval_llm_model)
            return create_llm(eval_settings)
        return create_llm(settings)

    @staticmethod
    def _load_prompt(settings: Settings) -> str:
        path = Path(settings.prompts_dir) / "golden_case_evaluator.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        return _DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def load_golden_cases(golden_dir: str) -> list[GoldenCase]:
        """Load all golden case files from a directory.

        Args:
            golden_dir: Path to directory containing golden case JSON files.

        Returns:
            List of GoldenCase objects.
        """
        cases: list[GoldenCase] = []
        golden_path = Path(golden_dir)
        if not golden_path.exists():
            logger.warning(f"Golden cases directory not found: {golden_dir}")
            return cases

        for json_file in sorted(golden_path.glob("*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                cases.append(GoldenCase(**data))
            except Exception as e:
                logger.warning(f"Failed to load golden case {json_file}: {e}")

        logger.info(f"Loaded {len(cases)} golden cases from {golden_dir}")
        return cases

    def _check_critical(
        self,
        generated_text: str,
        critical_checks: list[str],
    ) -> tuple[list[str], list[str]]:
        """Run deterministic critical checks against generated output.

        Each critical check is a natural-language description of something
        that MUST appear in the output. We use simple substring/keyword
        matching for speed; the LLM rubric catches semantic misses.

        Returns:
            (passed, failed) lists of check descriptions.
        """
        passed: list[str] = []
        failed: list[str] = []
        generated_lower = generated_text.lower()

        for check in critical_checks:
            # Extract key terms from the check description for matching
            # This is deliberately simple — the LLM rubric is the real judge
            check_lower = check.lower()

            # Look for key clinical terms in the check
            key_terms = []
            for word in check_lower.split():
                # Keep meaningful clinical terms (skip common words)
                if len(word) > 4 and word not in {
                    "agent", "should", "about", "their", "which",
                    "these", "those", "would", "could", "being",
                    "noted", "include", "including",
                }:
                    key_terms.append(word)

            # Check passes if at least half of key terms appear in output
            if key_terms:
                matches = sum(1 for t in key_terms if t in generated_lower)
                if matches >= len(key_terms) / 2:
                    passed.append(check)
                else:
                    failed.append(check)
            else:
                passed.append(check)  # No key terms to check

        return passed, failed

    def evaluate_case(
        self,
        golden_case: GoldenCase,
        generated_sections: dict[str, str],
    ) -> GoldenCaseMatchResult:
        """Compare a single generated brief against its golden reference.

        Args:
            golden_case: The expert-written reference case.
            generated_sections: Dict of section_key -> generated text.

        Returns:
            GoldenCaseMatchResult with scores and critical check results.
        """
        # Format reference and generated briefs
        ref_text = "\n\n".join(
            f"### {k}\n{v}" for k, v in golden_case.reference_brief.items()
        )
        gen_text = "\n\n".join(
            f"### {k}\n{v}" for k, v in generated_sections.items()
            if v and v != "Not enough information from structured data."
        )

        # LLM-as-judge evaluation
        user_message = (
            f"## REFERENCE ICU-PAUSE brief (expert-written):\n{ref_text}\n\n"
            f"## SYSTEM-GENERATED brief:\n{gen_text}\n\n"
            f"Evaluate the generated note against the reference. Return the JSON."
        )

        scores = {"clinical_accuracy": 0, "completeness": 0, "hallucination": 0, "actionability": 0}
        reasoning: dict[str, str] = {}

        try:
            response = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
            )
            cleaned = _clean_llm_output(response or "")
            parsed = json.loads(cleaned, strict=False)

            for dim in scores:
                scores[dim] = max(1, min(5, parsed.get(dim, 0)))
            reasoning = parsed.get("reasoning", {})
        except Exception as e:
            logger.warning(
                f"Golden case evaluation failed for {golden_case.hospitalization_id}: {e}"
            )
            reasoning = {dim: f"Evaluation failed: {e}" for dim in scores}

        # Critical checks (deterministic)
        passed, failed = self._check_critical(gen_text, golden_case.critical_checks)

        overall = sum(scores.values()) / len(scores) if any(scores.values()) else 0.0

        return GoldenCaseMatchResult(
            hospitalization_id=golden_case.hospitalization_id,
            clinical_accuracy=scores["clinical_accuracy"],
            completeness=scores["completeness"],
            hallucination=scores["hallucination"],
            actionability=scores["actionability"],
            overall_score=round(overall, 2),
            reasoning=reasoning,
            critical_checks_passed=passed,
            critical_checks_failed=failed,
        )

    def evaluate_batch(
        self,
        golden_cases: list[GoldenCase],
        generated_outputs: dict[str, dict[str, str]],
    ) -> GoldenCaseBatchResult:
        """Evaluate all golden cases against generated outputs.

        Args:
            golden_cases: List of golden cases to evaluate.
            generated_outputs: Dict of hospitalization_id -> {section_key -> text}.

        Returns:
            GoldenCaseBatchResult with per-case results and aggregated scores.
        """
        results: list[GoldenCaseMatchResult] = []

        for case in golden_cases:
            generated = generated_outputs.get(case.hospitalization_id)
            if generated is None:
                logger.warning(
                    f"No generated output for golden case {case.hospitalization_id}"
                )
                continue

            result = self.evaluate_case(case, generated)
            results.append(result)

        if not results:
            return GoldenCaseBatchResult(
                results=[],
                mean_clinical_accuracy=0.0,
                mean_completeness=0.0,
                mean_hallucination=0.0,
                mean_actionability=0.0,
                mean_overall=0.0,
                critical_check_pass_rate=0.0,
                evaluator_model=self.llm.last_usage.model,
            )

        def _mean(attr: str) -> float:
            vals = [getattr(r, attr) for r in results if getattr(r, attr) > 0]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        total_checks = sum(
            len(r.critical_checks_passed) + len(r.critical_checks_failed)
            for r in results
        )
        total_passed = sum(len(r.critical_checks_passed) for r in results)
        check_rate = total_passed / total_checks if total_checks > 0 else 1.0

        return GoldenCaseBatchResult(
            results=results,
            mean_clinical_accuracy=_mean("clinical_accuracy"),
            mean_completeness=_mean("completeness"),
            mean_hallucination=_mean("hallucination"),
            mean_actionability=_mean("actionability"),
            mean_overall=_mean("overall_score"),
            critical_check_pass_rate=round(check_rate, 4),
            evaluator_model=self.llm.last_usage.model,
        )

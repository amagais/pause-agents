"""Data Utilization Quality evaluator (LLM-as-a-Judge).

Evaluates whether each domain agent actually used the clinically significant
structured data it was given. This distinguishes the system from simple
summarization — agents should be doing targeted reasoning over specific data,
not just paraphrasing notes.

Analog to Google's "Tool Use Quality" metric adapted for clinical agents
that consume structured data rather than call tools.
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


class DataUtilizationResult(BaseModel):
    """Data utilization evaluation for a single agent."""

    agent_name: str
    score: int = Field(ge=0, le=5, default=0)  # 1-5 Likert, 0 = eval failed
    critical_fields_used: list[str] = Field(default_factory=list)
    critical_fields_ignored: list[str] = Field(default_factory=list)
    reasoning: str = ""


class DataUtilizationEvaluation(BaseModel):
    """Data utilization evaluation across all agents for a single case."""

    results: list[DataUtilizationResult]
    mean_score: float
    evaluator_model: str


# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are evaluating how well a clinical agent utilized the structured patient \
data it was provided. The agent's job is not just to summarize — it must \
reason over specific clinical values to produce actionable ICU handoff content.

Evaluate the agent's output against the source data on these criteria:
- Did the agent reference the most clinically significant data values?
- Did the agent reason about trends (improving/worsening) when time-series \
data was available?
- Were critical abnormal values (e.g., creatinine 3.2, SpO2 trending down, \
high-risk medications) acknowledged and addressed?
- Did the agent miss any data that would change clinical management?

## SCORING (1-5)
1 = Agent ignored most of the important structured data
2 = Agent used some data but missed critical values
3 = Agent used key data but missed some clinically significant fields
4 = Agent used most important data with good clinical reasoning
5 = Agent comprehensively utilized all relevant data with appropriate reasoning

## OUTPUT FORMAT
Return a JSON object:
{
  "score": <1-5>,
  "critical_fields_used": ["<field that was appropriately referenced>", "..."],
  "critical_fields_ignored": ["<clinically important field that was missed>", "..."],
  "reasoning": "<brief assessment of data utilization quality>"
}
"""


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class DataUtilizationEvaluator:
    """Evaluate how well each agent used its input data."""

    def __init__(self, settings: Settings):
        from icu_pause.eval import create_eval_llm

        # Data utilization uses the general eval LLM (no dedicated override)
        self.llm: BaseLLM = create_eval_llm(settings)
        self.system_prompt = self._load_prompt(settings)

    @staticmethod
    def _load_prompt(settings: Settings) -> str:
        path = Path(settings.prompts_dir) / "data_utilization_evaluator.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        return _DEFAULT_SYSTEM_PROMPT

    def evaluate_agent(
        self,
        agent_name: str,
        agent_input: dict[str, Any],
        agent_output: str,
        domain_keys: list[str] | None = None,
    ) -> DataUtilizationResult:
        """Evaluate a single agent's data utilization.

        Args:
            agent_name: Name of the agent (e.g., "nurse", "pharmacy").
            agent_input: The context dict the agent received as input.
            agent_output: The agent's section text output (concatenated).
            domain_keys: The agent's assigned data domain keys (e.g.,
                ["vitals", "assessments", "notes"]). Used to scope the
                evaluation to within-domain utilization only.

        Returns:
            DataUtilizationResult with score and field-level details.
        """
        if not agent_output or not agent_output.strip():
            return DataUtilizationResult(
                agent_name=agent_name,
                reasoning="No output to evaluate",
            )

        domain_clause = ""
        if domain_keys:
            domain_clause = (
                f"\n\nThis agent's assigned data domain: {', '.join(domain_keys)}. "
                f"Only evaluate utilization of data within this domain and the "
                f"CRITICAL FLAGS prefix. Do NOT penalize for missing cross-domain data."
            )

        user_message = (
            f"## STRUCTURED DATA provided to the {agent_name} agent:\n"
            f"```json\n{json.dumps(agent_input, indent=2, default=str)}\n```\n\n"
            f"## AGENT OUTPUT:\n{agent_output}\n\n"
            f"Rate how well the {agent_name} agent utilized the most clinically "
            f"significant data fields within its assigned domain.{domain_clause} "
            f"Return the JSON."
        )

        try:
            response = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
            )
            cleaned = _clean_llm_output(response)
            parsed = json.loads(cleaned, strict=False)

            return DataUtilizationResult(
                agent_name=agent_name,
                score=max(1, min(5, parsed.get("score", 0))),
                critical_fields_used=parsed.get("critical_fields_used", []),
                critical_fields_ignored=parsed.get("critical_fields_ignored", []),
                reasoning=parsed.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Data utilization evaluation failed for {agent_name}: {e}")
            return DataUtilizationResult(
                agent_name=agent_name,
                reasoning=f"Evaluation failed: {e}",
            )

    def evaluate_all_agents(
        self,
        agent_snippets: list[Any],
        agent_contexts: dict[str, dict[str, Any]],
        patient_context: dict[str, Any],
    ) -> DataUtilizationEvaluation:
        """Evaluate data utilization for all domain agents in a pipeline run.

        Args:
            agent_snippets: List of AgentSnippet objects from the pipeline.
            agent_contexts: Per-agent context dicts (agent_name -> data dict).
            patient_context: Shared patient context (fallback).

        Returns:
            DataUtilizationEvaluation with per-agent results and mean score.
        """
        results: list[DataUtilizationResult] = []

        for snippet in agent_snippets:
            context = agent_contexts.get(snippet.agent_name, patient_context)

            agent_text = "\n\n".join(
                f"[{sec.section}]: {sec.content}"
                for sec in snippet.sections
                if sec.content and sec.content != "Not enough information from structured data."
            )

            # Derive domain keys from the per-agent context (excludes notes
            # and other non-structured keys that aren't domain-specific).
            domain_keys = list(context.keys()) if context else None

            result = self.evaluate_agent(
                agent_name=snippet.agent_name,
                agent_input=context,
                agent_output=agent_text,
                domain_keys=domain_keys,
            )
            results.append(result)

        valid_scores = [r.score for r in results if r.score > 0]
        mean = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        return DataUtilizationEvaluation(
            results=results,
            mean_score=round(mean, 2),
            evaluator_model=self.llm.last_usage.model,
        )

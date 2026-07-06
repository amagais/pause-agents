"""Evaluation tools: rubrics, PDSQI-9, grounding, data utilization, golden cases, and prompt versioning."""

from __future__ import annotations

from copy import copy

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm


def create_eval_llm(
    settings: Settings,
    specific_provider: str = "",
    specific_model: str = "",
) -> BaseLLM:
    """Create an LLM for evaluation with 3-tier fallback.

    Priority: evaluator-specific → eval_llm → pipeline LLM.

    Args:
        settings: Application settings.
        specific_provider: Evaluator-specific provider (e.g., grounding_llm_provider).
        specific_model: Evaluator-specific model (e.g., grounding_llm_model).
    """
    provider = specific_provider or settings.eval_llm_provider
    model = specific_model or settings.eval_llm_model

    if provider and model:
        eval_settings = copy(settings)
        object.__setattr__(eval_settings, "llm_provider", provider)
        object.__setattr__(eval_settings, "llm_model", model)
        return create_llm(eval_settings)
    return create_llm(settings)

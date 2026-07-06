"""Resident Agent: pre-synthesis for the Intensivist.

Mirrors actual clinical workflow: a senior ICU resident reviews all domain
agent outputs, flags cross-domain conflicts and gaps, and drafts a structured
pre-synthesis brief (~2-3k tokens) that the Intensivist refines.

The Resident does NOT have access to raw structured data or clinical notes —
only domain agent outputs. This keeps its token footprint small and makes its
function distinct from the Intensivist's.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.schemas.icu_pause import AgentSnippet, ResidentPreBrief

logger = logging.getLogger(__name__)


class ResidentAgent:
    """Senior ICU resident agent that produces a pre-synthesis brief.

    Receives post-QA domain agent outputs, identifies cross-domain conflicts
    and gaps, and drafts a structured brief for the Intensivist.
    """

    agent_name = "resident"

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.settings = settings
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = Path(self.settings.prompts_dir) / "resident.yaml"
        if not path.exists():
            logger.warning(f"Prompt file not found: {path}. Using default prompt.")
            return self._default_prompt()
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("system_prompt", self._default_prompt())

    @staticmethod
    def _default_prompt() -> str:
        return (
            "You are a senior ICU resident preparing a pre-synthesis brief for the "
            "attending intensivist.\n\n"
            "You will receive the validated outputs of six domain agents (Nurse, "
            "Respiratory Therapist, Pharmacist, Dietitian, Case Manager, "
            "Physical/Occupational Therapist) after QA review.\n\n"
            "Your job is NOT to rewrite their findings — it is to synthesize across "
            "domains and identify what the intensivist needs to know.\n\n"
            "STRICT SCOPE RULES:\n"
            "- You receive domain agent outputs ONLY. You do not have access to raw "
            "CLIF data or clinical notes. Do not hallucinate values not present in "
            "agent outputs.\n"
            "- If a domain agent flagged 'data unavailable', preserve that — do not infer.\n"
            "- Do not generate full ICU-PAUSE sections. That is the Intensivist's job.\n\n"
            "YOUR THREE TASKS:\n"
            "1. FLAG CROSS-DOMAIN CONFLICTS: Identify contradictions between agents "
            "(e.g., Pharmacy recommends fluid restriction; Dietitian recommends "
            "high-volume enteral feeds). List each conflict with severity "
            "(safety_critical, clinical, logistical). Do not resolve — flag only.\n"
            "2. IDENTIFY CRITICAL GAPS: Note ICU-PAUSE fields where no agent provided "
            "content. Distinguish 'explicitly_unavailable' (agent flagged it) from "
            "'silently_absent' (no agent addressed it).\n"
            "3. DRAFT PRE-BRIEF NARRATIVE: Write a concise cross-domain summary "
            "(≤250 words) with: (a) dominant clinical theme of this transfer, "
            "(b) 2-3 key inter-domain dependencies the intensivist must harmonize, "
            "(c) highest-priority to-do items surfaced across agents.\n\n"
            "Return valid JSON matching the ResidentPreBrief schema."
        )

    def _format_agent_outputs(self, snippets: list[AgentSnippet]) -> str:
        """Format domain agent outputs for the Resident's review."""
        parts = ["## DOMAIN AGENT OUTPUTS (POST-QA)\n"]
        for snippet in snippets:
            parts.append(f"### {snippet.agent_name.upper()} AGENT")
            if not snippet.sections:
                parts.append("(No sections produced — agent execution may have failed)")
            for sec in snippet.sections:
                content = sec.content or "(empty)"
                parts.append(
                    f"**[{sec.section}]** (confidence {sec.confidence}):\n"
                    f"{content}"
                )
            if snippet.warnings:
                parts.append(
                    f"Warnings: {'; '.join(w.message for w in snippet.warnings)}"
                )
            parts.append("")
        return "\n".join(parts)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the Resident agent.

        Reads domain agent snippets (post-QA, post-deliberation if enabled)
        and produces a ResidentPreBrief for the Intensivist.
        """
        t0 = datetime.now(timezone.utc)
        trace_events: list[dict] = []

        snippets: list[AgentSnippet] = state.get("agent_snippets", [])
        revised: list[AgentSnippet] = state.get("revised_snippets", [])

        # Use revised snippets where available
        revised_agents = {s.agent_name for s in revised}
        effective_snippets = [
            s for s in snippets if s.agent_name not in revised_agents
        ] + list(revised)

        agent_outputs_text = self._format_agent_outputs(effective_snippets)
        qa_issues = state.get("qa_issues", [])
        qa_text = ""
        if qa_issues:
            qa_text = (
                "\n## QA ISSUES IDENTIFIED\n"
                + "\n".join(f"- {issue}" for issue in qa_issues)
                + "\n"
            )

        user_message = (
            f"{agent_outputs_text}\n"
            f"{qa_text}\n"
            f"Review the domain agent outputs above. Produce a ResidentPreBrief "
            f"with cross-domain conflicts, critical gaps, and a pre-brief narrative.\n\n"
            f"Return valid JSON only."
        )

        trace_events.append({
            "timestamp": t0.isoformat(),
            "type": "agent_input",
            "node": self.agent_name,
            "level": "info",
            "message": f"Reviewing {len(effective_snippets)} domain agent outputs",
            "data": {
                "agents_reviewed": [s.agent_name for s in effective_snippets],
                "qa_issues_count": len(qa_issues),
            },
        })

        try:
            pre_brief = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=ResidentPreBrief,
            )
            logger.info(
                f"Resident: {len(pre_brief.cross_domain_conflicts)} conflicts, "
                f"{len(pre_brief.critical_gaps)} gaps, "
                f"confidence={pre_brief.resident_confidence}"
            )
        except Exception as e:
            logger.error(f"Resident agent failed: {e}")
            pre_brief = ResidentPreBrief(
                pre_brief_narrative={
                    "dominant_clinical_theme": "Resident synthesis unavailable",
                    "inter_domain_dependencies": [],
                    "priority_todo_items": [],
                },
                self_critique_passed=False,
                self_critique_flags=[f"Execution failed: {e}"],
                resident_confidence="low",
            )

        usage = self.llm.last_usage
        elapsed_ms = (datetime.now(timezone.utc) - t0).total_seconds() * 1000
        metrics = {
            "agent": self.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "model": usage.model,
        }

        trace_events.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "agent_output",
            "node": self.agent_name,
            "level": "info",
            "message": (
                f"Pre-brief: {len(pre_brief.cross_domain_conflicts)} conflicts, "
                f"{len(pre_brief.critical_gaps)} gaps"
            ),
            "data": {
                "conflicts": len(pre_brief.cross_domain_conflicts),
                "gaps": len(pre_brief.critical_gaps),
                "confidence": pre_brief.resident_confidence,
                "self_critique_passed": pre_brief.self_critique_passed,
                "metrics": metrics,
            },
        })

        return {
            "resident_pre_brief": pre_brief.model_dump(),
            "pipeline_metrics": [metrics],
            "trace_events": trace_events,
        }

"""Base class for LLM-powered domain agents."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from icu_pause.config import Settings
from icu_pause.data.context import format_local_dttm
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.schemas.icu_pause import (
    AgentSnippet,
    AgentSnippetLLM,
    Warning,
    WarningCategory,
    WarningSeverity,
    wrap_llm_snippet,
)

logger = logging.getLogger(__name__)


def _json_default_local_dttm(obj: Any) -> str:
    """``json.dumps(..., default=...)`` hook that routes ``datetime`` values
    through ``format_local_dttm`` so every ``_dttm`` field the LLM sees
    renders as ``M-DD HH:MM`` in the display timezone instead of
    ``str(datetime)`` (``"2024-05-08 13:02:00.598988+00:00"``).

    Closes the timestamp leak documented on branch
    ``fix/timezone-leaks-in-prompts``: the per-block formatters
    (pending-tests, demographics, active-consults) were already converted
    in commit 50e673d, but the raw row dicts the LLM also sees still flowed
    through ``default=str`` and let UTC ISO strings into the brief
    (Section P, demographics line, ad-hoc model paraphrases). Non-datetime
    values fall back to ``str(obj)`` exactly as before.
    """
    if isinstance(obj, datetime):
        return format_local_dttm(obj)
    return str(obj)


def format_data_sections_block(relevant_data: dict[str, Any]) -> str:
    """Format a relevant_data dict as the ``## KEY\\n{json}`` block used in the
    early_fusion user message.

    Pure utility, extracted from ``_build_user_message`` and ``revise()`` so
    the hybrid_v1 per-domain graceful-degradation fallback (pre-reg §1.3
    Stage F) can produce the identical block when an extractor fails for one
    domain. Byte-identical to the inline code at both existing call sites by
    construction.

    Locked formatting:
    - Uppercased key as a level-2 markdown header.
    - JSON serialization with compact separators ``(",", ":")``.
    - ``_json_default_local_dttm`` for datetime fields (timezone-leak guard).
    - Blocks joined with double newline.
    """
    return "\n\n".join(
        f"## {key.upper()}\n{json.dumps(value, separators=(',', ':'), default=_json_default_local_dttm)}"
        for key, value in relevant_data.items()
    )


class BaseDomainAgent(ABC):
    """Abstract base for all ICU-PAUSE domain agents.

    Subclasses define:
    - agent_name: identifier used for logging and prompt file lookup
    - required_context_keys: which keys from patient_context_text to include
    - target_sections: which ICU-PAUSE sections this agent contributes to
    """

    def __init__(self, settings: Settings):
        # Use per-agent max_tokens if configured, else fall back to global
        agent_max = settings.agent_max_tokens.get(
            self._agent_name_for_init(), settings.llm_max_tokens
        )
        self.llm: BaseLLM = create_llm(settings, max_tokens_override=agent_max)
        self._agent_max_tokens = agent_max
        self.settings = settings
        self.system_prompt, self.prompt_version = self._load_prompt()

    def _agent_name_for_init(self) -> str:
        """Return agent_name for use during __init__ (before property is accessible)."""
        return self.agent_name

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Unique agent identifier, e.g. 'nurse', 'respiratory'."""
        ...

    @property
    @abstractmethod
    def required_context_keys(self) -> list[str]:
        """Keys from patient_context_text dict this agent needs."""
        ...

    @property
    @abstractmethod
    def target_sections(self) -> list[str]:
        """ICU-PAUSE section keys this agent contributes to."""
        ...

    def _load_prompt(self) -> tuple[str, str]:
        """Load the system prompt and version from the YAML config file.

        Returns:
            Tuple of (system_prompt, version). Version defaults to "unknown"
            if not specified in the YAML.
        """
        path = Path(self.settings.prompts_dir) / f"{self.agent_name}.yaml"
        if not path.exists():
            logger.warning(f"Prompt file not found: {path}. Using default prompt.")
            return self._default_prompt(), "unknown"
        with open(path) as f:
            data = yaml.safe_load(f)
        prompt = data.get("system_prompt", self._default_prompt())
        version = data.get("version", "unknown")
        return prompt, version

    def _default_prompt(self) -> str:
        sections_str = ", ".join(self.target_sections)
        return (
            f"You are a {self.agent_name} agent generating ICU-PAUSE handoff content. "
            f"Produce content for sections: {sections_str}. "
            f"Use only the data provided. If data is insufficient for a section, "
            f'set content to "Not enough information from structured data." '
            f"Output valid JSON matching the AgentSnippet schema."
        )

    def _citation_rule(self) -> str:
        """Return the CITATION RULE prompt block, or empty string if citations are off."""
        mode = getattr(self.settings, "citation_mode", "decision_critical")
        if mode == "off":
            return ""
        scope = (
            "Cite decision-critical values: vent settings, doses, lab "
            "values, code status, line placement dates.\n"
            "  Skip citations for implicit-source items like mobility "
            "descriptions or subjective assessments.\n"
            if mode == "decision_critical"
            else "Cite every referenced value.\n"
        )
        return (
            "- CITATION RULE: Each data record includes a \"cite\" field "
            "(e.g. \"(lab 1/17 08:00)\").\n"
            "  When you reference a value, append its cite field verbatim.\n"
            "  Do NOT generate timestamps yourself — only echo cite fields "
            "from the data.\n"
            "  When multiple values share the same cite tag, group them:\n"
            "  GOOD: 'BP 118/72, HR 84, RR 18 (vital 1/17 08:00)'\n"
            "  BAD: 'BP 118/72 (vital 1/17 08:00), HR 84 (vital 1/17 "
            "08:00)'\n"
            f"  {scope}\n"
        )

    def _format_scribe_pins(self, state: dict[str, Any]) -> str:
        """Render scribe-extracted pin blocks (admission_antibiotics and
        future siblings) for prepending to this agent's user_message.

        Default is no-op (empty string). Agents that consume scribe pins
        override this to render the block they're authorized to receive —
        see :class:`icu_pause.agents.pharmacy.PharmacyAgent` for the
        admission_antibiotics path.
        """
        return ""

    def _build_user_message(self, relevant_data: dict[str, Any]) -> str:
        """Build the user message from the relevant patient data (s0,n0 path).

        Kept byte-identical to the historical early_fusion message so the
        clinician-validated (S0,N0) and hybrid_v1 cells do not drift.
        """
        data_sections = format_data_sections_block(relevant_data)
        return f"Patient data:\n\n{data_sections}\n\n" + self._task_instructions_block()

    def _build_user_message_composed(
        self, structured_block: str, notes_block: str
    ) -> str:
        """Build the user message from independently-rendered S and N blocks.

        Used for every cell except (s0,n0). Reuses the IDENTICAL task-instructions
        tail as the s0/n0 path so the only thing that varies across cells is the
        data representation (single-factor isolation, compression sub-study).
        """
        data_sections = "\n\n".join(
            b for b in (structured_block, notes_block) if b
        )
        return f"Patient data:\n\n{data_sections}\n\n" + self._task_instructions_block()

    def _structured_block(
        self, axis: str, context: dict[str, Any], state: dict[str, Any]
    ) -> str:
        """Render the structured-axis block (S0 raw tables | S1 summary | S2 view)."""
        if not getattr(self.settings, "structured_data_enabled", True):
            return ""
        if axis == "s1":
            summary = state.get("structured_summaries", {}).get(self.agent_name, "")
            return (
                "## STRUCTURED DATA SUMMARY\n"
                + (summary or "No structured data summary available.")
            )
        if axis == "s2":
            view = state.get("structured_views", {}).get(self.agent_name, "")
            return (
                "## STRUCTURED DATA (salience-selected)\n"
                + (view or "Not documented.")
            )
        # s0 — raw per-agent tiered tables, excluding notes (the N axis owns notes).
        structured_keys = [k for k in self.required_context_keys if k != "notes"]
        data = {k: context.get(k, None) for k in structured_keys}
        return format_data_sections_block(data)

    def _notes_block(
        self, axis: str, context: dict[str, Any], state: dict[str, Any]
    ) -> str:
        """Render the notes-axis block (N0 raw | N1 summary | N2 extracted anchors).

        N2 is SUBSTITUTIVE: the extracted anchors replace raw notes in the prompt.
        """
        if not getattr(self.settings, "notes_enabled", True):
            return ""
        if axis == "n1":
            summary = state.get("note_summaries", {}).get(self.agent_name, "")
            return (
                "## CLINICAL NOTES SUMMARY\n"
                + (summary or "No clinical notes summary available.")
            )
        if axis == "n2":
            extraction = state.get("per_domain_extractions", {}).get(self.agent_name)
            return self._render_extracted_notes_block(extraction or {})
        # n0 — raw routed notes (only for agents that route notes).
        if "notes" not in self.required_context_keys:
            return ""
        return format_data_sections_block({"notes": context.get("notes", None)})

    def _render_extracted_notes_block(self, extraction: dict[str, Any]) -> str:
        """Render the N2 substitutive notes block from per-domain extracted facts."""
        if not extraction:
            return (
                "## CLINICAL NOTES (extracted facts)\n"
                "Not documented in this domain's routed clinical notes."
            )
        field_lines = "\n".join(
            f"  {field}: {value}" for field, value in extraction.items()
        )
        return (
            "## CLINICAL NOTES (extracted facts)\n"
            "Discrete facts extracted from this domain's routed clinical notes "
            '(substitutive representation; "Not documented" = absent in notes).\n'
            f"{field_lines}"
        )

    def _resolve_axes(self, state: dict[str, Any]) -> tuple[str, str]:
        """Resolve (structured_axis, notes_axis) for this run.

        data_retrieval normally writes both into state. For hand-built states
        that carry only fusion_mode (e.g. unit tests, the LangGraph server), fall
        back to the legacy fusion_mode→cell mapping so behavior is unchanged.
        """
        structured_axis = state.get("structured_axis")
        notes_axis = state.get("notes_axis")
        if structured_axis and notes_axis:
            return structured_axis, notes_axis
        from icu_pause.config import axes_from_fusion_mode
        s_fb, n_fb = axes_from_fusion_mode(state.get("fusion_mode", "early_fusion"))
        return (structured_axis or s_fb), (notes_axis or n_fb)

    def _task_instructions_block(self) -> str:
        """The post-data task/schema instructions, shared across all cells.

        Extracted from _build_user_message verbatim so (s0,n0) stays byte-identical.
        """
        sections_list = ', '.join(f'"{s}"' for s in self.target_sections)
        return (
            # INSTRUCTIONS AFTER DATA — in the model's recent attention window
            f"## YOUR TASK\n\n"
            f"Generate ICU-PAUSE section contributions for EXACTLY these sections: "
            f"{', '.join(self.target_sections)}.\n\n"
            f"CRITICAL REMINDERS (read these carefully before generating):\n"
            f"- Each section's \"section\" field MUST be one of: {sections_list}\n"
            f"- Do NOT invent your own section names\n"
            f"- Do NOT write patient history, PMH, or clinical narratives in ANY section\n"
            f"  unless your section specifically requires it\n"
            f"- Do NOT write sentences starting with the patient's name or age\n"
            f"- Section A = checklist only (consultant names, not narratives)\n"
            f"- Section S = #problem/[]to-do format only. Each #problem must name a\n"
            f"  SPECIFIC clinical problem for THIS patient (not generic organ system\n"
            f"  monitoring like 'Monitor respiratory status' or 'Monitor GCS').\n"
            f"  Ask: would this to-do appear for EVERY ICU patient? If yes, exclude it.\n"
            f"  Do NOT include medication names in S — those belong in U_unprescribing.\n"
            f"- Section P = pending items only (not current treatments)\n"
            f"- Section E = exam findings only (use MOST RECENT data)\n"
            f"{self._citation_rule()}"
            f"Respond with a JSON object matching this schema:\n"
            f'{{\n'
            f'  "agent_name": "{self.agent_name}",\n'
            f'  "sections": [\n'
            f'    {{\n'
            f'      "section": "{self.target_sections[0]}",\n'
            f'      "content": "<clinical text for {self.target_sections[0]}>",\n'
            f'      "confidence": 0.0-1.0,\n'
            f'      "data_sources_used": ["<key1>", "<key2>"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "warnings": ["<any data quality warnings>"]\n'
            f'}}'
        )

    def revise(self, state: dict[str, Any], qa_issue: str, conflicting_output: str) -> dict[str, Any]:
        """Re-evaluate output based on a specific QA conflict.

        The agent sees its own original output, the conflicting agent's output,
        the specific QA issue, and the original patient data. Returns a revised
        snippet with deliberation metrics and log entry.
        """
        agent_contexts = state.get("agent_context_text", {})
        context = agent_contexts.get(self.agent_name, state.get("patient_context_text", {}))
        structured_axis, notes_axis = self._resolve_axes(state)

        # Find this agent's original snippet
        original_snippet = next(
            (s for s in state.get("agent_snippets", []) if s.agent_name == self.agent_name),
            None,
        )
        original_text = ""
        if original_snippet:
            original_text = "\n".join(
                f"[{s.section}]: {s.content}" for s in original_snippet.sections
            )

        # Build the data block from the SAME compression axes the first pass used,
        # so deliberation does not silently revert to raw early_fusion data.
        if structured_axis == "s0" and notes_axis == "n0":
            relevant_data = {k: context.get(k, None) for k in self.required_context_keys}
            data_sections = format_data_sections_block(relevant_data)
        else:
            structured_block = self._structured_block(structured_axis, context, state)
            notes_block = self._notes_block(notes_axis, context, state)
            data_sections = "\n\n".join(
                b for b in (structured_block, notes_block) if b
            )

        user_message = (
            f"## QA CONFLICT RESOLUTION\n\n"
            f"Your previous output for this patient was flagged by QA for a "
            f"contradiction with another agent.\n\n"
            f"### QA Issue\n{qa_issue}\n\n"
            f"### Your Original Output\n{original_text}\n\n"
            f"### Conflicting Agent Output\n{conflicting_output}\n\n"
            f"### Patient Data\n{data_sections}\n\n"
            f"Re-evaluate your output for sections: {', '.join(self.target_sections)}.\n"
            f"Review the source data carefully to resolve the contradiction. "
            f"If the data supports your original claim, reinforce it with specific "
            f"citations. If the data supports the other agent's claim, revise your "
            f"output accordingly. Only change the sections relevant to the conflict.\n\n"
            f"Respond with a JSON object matching this schema:\n"
            f'{{\n'
            f'  "agent_name": "{self.agent_name}",\n'
            f'  "sections": [\n'
            f'    {{\n'
            f'      "section": "<section_key>",\n'
            f'      "content": "<clinical text>",\n'
            f'      "confidence": 0.0-1.0,\n'
            f'      "data_sources_used": ["<key1>", "<key2>"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "warnings": ["<any data quality warnings>"]\n'
            f'}}'
        )

        original_sections = {}
        if original_snippet:
            original_sections = {s.section: s.content for s in original_snippet.sections}

        try:
            snippet = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=AgentSnippet,
            )
            logger.info(
                f"{self.agent_name} deliberation produced {len(snippet.sections)} sections"
            )
        except Exception as e:
            logger.warning(f"{self.agent_name} deliberation failed: {e}")
            usage = self.llm.last_usage
            metrics = {
                "agent": f"{self.agent_name}_deliberation",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": round(usage.latency_ms, 1),
                "model": usage.model,
            }
            return {
                "revised_snippets": [],
                "pipeline_metrics": [metrics],
                "deliberation_log": [
                    {
                        "agent": self.agent_name,
                        "round": 1,
                        "issue": qa_issue,
                        "original_sections": original_sections,
                        "revised_sections": {},
                        "error": str(e),
                    }
                ],
            }

        usage = self.llm.last_usage
        metrics = {
            "agent": f"{self.agent_name}_deliberation",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "model": usage.model,
        }

        revised_sections = {s.section: s.content for s in snippet.sections}

        return {
            "revised_snippets": [snippet],
            "pipeline_metrics": [metrics],
            "deliberation_log": [
                {
                    "agent": self.agent_name,
                    "round": 1,
                    "issue": qa_issue,
                    "original_sections": original_sections,
                    "revised_sections": revised_sections,
                }
            ],
        }

    def _self_critique(
        self,
        snippet: AgentSnippet,
    ) -> tuple[AgentSnippet, dict[str, Any], list[dict[str, Any]]]:
        """Lightweight self-critique of the agent's output (single pass).

        Does NOT re-send source data or the full agent system prompt.
        The agent already processed source data in step 1. Self-critique
        catches reasoning errors only — numeric fidelity is QA's job.
        Total prompt: ~1-1.5k tokens.

        Returns:
            Tuple of (possibly revised snippet, usage metrics dict, trace events list).
        """
        sections_text = "\n".join(
            f"[{s.section}]: {s.content}" for s in snippet.sections
        )
        sections_list = ', '.join(f'"{s}"' for s in self.target_sections)

        critique_message = (
            f"You just produced this clinical summary as the {self.agent_name} agent:\n\n"
            f"{sections_text}\n\n"
            f"Check it for:\n"
            f"1. Hallucinated values — doses, labs, device settings, or clinical "
            f"events you are not confident were in the source data\n"
            f"2. Temporal misattribution — findings described as active/current "
            f"that may actually be historical ('history of', 'resolved', 'previously'). "
            f"Historical items should NOT appear in active sections (S, P, U)\n"
            f"3. Missing required sections: {sections_list}\n\n"
            f"If you find issues, produce a REVISED output and list each "
            f"correction in warnings. If NO issues are found, reproduce your "
            f"output unchanged with an EMPTY warnings array. Do NOT add "
            f"'no issues found' messages — only list actual corrections.\n\n"
            f"Respond with JSON:\n"
            f'{{\n'
            f'  "agent_name": "{self.agent_name}",\n'
            f'  "sections": [\n'
            f'    {{\n'
            f'      "section": "<section_key>",\n'
            f'      "content": "<clinical text>",\n'
            f'      "confidence": 0.0-1.0,\n'
            f'      "data_sources_used": ["<key1>", "<key2>"]\n'
            f'    }}\n'
            f'  ],\n'
            f'  "warnings": []\n'
            f'}}'
        )

        _CRITIQUE_SYSTEM = (
            f"You are a clinical QA reviewer checking the {self.agent_name} "
            f"agent's output for errors. Return valid JSON only."
        )

        trace_events = []
        try:
            revised_llm = self.llm.invoke(
                system=_CRITIQUE_SYSTEM,
                user=critique_message,
                response_format=AgentSnippetLLM,
            )
            # Drop "no issues found" / no-op confirmations — keep only actual
            # corrections that the agent narrated.
            _NO_ISSUE_PATTERNS = ["no ", "not found", "not detected", "all required", "none"]
            revised_llm.warnings = [
                w for w in revised_llm.warnings
                if not any(p in w.lower() for p in _NO_ISSUE_PATTERNS)
            ]
            # Self-critique narrations are wording revisions, not new patient
            # safety signals — route to audit-only.
            revised = wrap_llm_snippet(
                revised_llm,
                category=WarningCategory.EDITORIAL_REVISION,
                severity=WarningSeverity.INFO,
                source_agent=self.agent_name,
            )
            logger.info(
                f"{self.agent_name} self-critique: "
                f"{len(revised.sections)} sections, {len(revised.warnings)} warnings"
            )
            trace_events.append({
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "type": "self_critique",
                "node": self.agent_name,
                "level": "info",
                "message": f"Self-critique complete: {len(revised.warnings)} corrections noted",
                "data": {
                    "corrections": [w.message for w in revised.warnings],
                },
            })
            result_snippet = revised
        except Exception as e:
            logger.warning(f"{self.agent_name} self-critique failed: {e}, keeping original")
            trace_events.append({
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "type": "self_critique",
                "node": self.agent_name,
                "level": "warning",
                "message": f"Self-critique failed: {e}",
                "data": {},
            })
            result_snippet = snippet

        usage = self.llm.last_usage
        metrics = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
        }
        return result_snippet, metrics, trace_events

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent within the LangGraph workflow.

        Args:
            state: The current GraphState dict.

        Returns:
            Dict with 'agent_snippets' key containing a list with one AgentSnippet.
        """
        # Compression sub-study: the agent message is composed from two
        # independent axes — structured_axis ∈ {s0,s1,s2}, notes_axis ∈ {n0,n1,n2}
        # — plus the deterministic CRITICAL FLAGS prefix (injected in ALL arms,
        # byte-identical, because it reads the shared patient_context_text).
        structured_axis, notes_axis = self._resolve_axes(state)
        trace_events = []

        agent_contexts = state.get("agent_context_text", {})
        context = agent_contexts.get(
            self.agent_name, state.get("patient_context_text", {})
        )

        # Prepend cross-domain critical flags (deterministic, ~150 tokens).
        from icu_pause.data.context import build_critical_flags
        patient_ctx = state.get("patient_context_text", {})
        critical_flags = build_critical_flags(patient_ctx)

        if structured_axis == "s0" and notes_axis == "n0":
            # Byte-identical to the historical early_fusion / hybrid_v1 agent
            # message (notes kept in their original position within
            # required_context_keys — do NOT reconstruct from split blocks).
            relevant_data = {
                k: context.get(k, None) for k in self.required_context_keys
            }
            body = self._build_user_message(relevant_data)
            nonempty_keys = [k for k, v in relevant_data.items() if v]
        else:
            structured_block = self._structured_block(structured_axis, context, state)
            notes_block = self._notes_block(notes_axis, context, state)
            body = self._build_user_message_composed(structured_block, notes_block)
            nonempty_keys = [
                lbl for lbl, blk in (
                    (f"structured:{structured_axis}", structured_block),
                    (f"notes:{notes_axis}", notes_block),
                ) if blk
            ]

        user_message = critical_flags + "\n\n" + body
        scribe_pins = self._format_scribe_pins(state)
        if scribe_pins:
            user_message = scribe_pins + "\n\n" + user_message

        # Trace: log the resolved cell + what this agent received.
        trace_events.append({
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "type": "agent_input",
            "node": self.agent_name,
            "level": "info",
            "message": (
                f"cell ({structured_axis},{notes_axis}): "
                f"received {len(nonempty_keys)} blocks: "
                f"{', '.join(nonempty_keys) if nonempty_keys else 'none'}"
            ),
            "data": {
                "fusion_mode": state.get("fusion_mode", "early_fusion"),
                "structured_axis": structured_axis,
                "notes_axis": notes_axis,
                "input_keys": nonempty_keys,
            },
        })

        try:
            llm_snippet = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=AgentSnippetLLM,
            )
            # First-pass agent warnings are genuine safety signals (the prompts
            # steer agents to flag risks, not narrate edits) — route to the
            # clinician panel by default.
            snippet = wrap_llm_snippet(
                llm_snippet,
                category=WarningCategory.SAFETY_FLAG,
                severity=WarningSeverity.CLINICAL,
                source_agent=self.agent_name,
            )
            logger.info(f"{self.agent_name} agent produced {len(snippet.sections)} sections")
        except Exception as e:
            # Retry once with explicit JSON instruction on parse failure
            logger.warning(f"{self.agent_name} agent first attempt failed: {e}. Retrying...")
            try:
                retry_message = (
                    user_message + "\n\n"
                    "IMPORTANT: Your previous response could not be parsed as valid JSON. "
                    "Respond ONLY with a valid JSON object matching the AgentSnippet schema. "
                    "Do not include any text outside the JSON object."
                )
                llm_snippet = self.llm.invoke(
                    system=self.system_prompt,
                    user=retry_message,
                    response_format=AgentSnippetLLM,
                )
                snippet = wrap_llm_snippet(
                    llm_snippet,
                    category=WarningCategory.SAFETY_FLAG,
                    severity=WarningSeverity.CLINICAL,
                    source_agent=self.agent_name,
                )
                logger.info(
                    f"{self.agent_name} agent retry succeeded: "
                    f"{len(snippet.sections)} sections"
                )
            except Exception as e2:
                logger.error(f"{self.agent_name} agent retry also failed: {e2}")
                snippet = AgentSnippet(
                    agent_name=self.agent_name,
                    sections=[],
                    warnings=[
                        Warning(
                            category=WarningCategory.QA_PROCESS,
                            severity=WarningSeverity.INFO,
                            message=f"Agent execution failed: {str(e)}",
                            source_agent=self.agent_name,
                        )
                    ],
                )
                trace_events.append({
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "type": "agent_error",
                    "node": self.agent_name,
                    "level": "error",
                    "message": f"Agent failed after retries: {e2}",
                    "data": {
                        "first_error": str(e),
                        "final_error": str(e2),
                        "error_type": type(e2).__name__,
                    },
                })

        usage = self.llm.last_usage
        agent_failed = len(snippet.sections) == 0 and any(
            "failed" in w.message.lower() for w in snippet.warnings
        )
        metrics = {
            "agent": self.agent_name,
            "prompt_version": self.prompt_version,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "model": usage.model,
        }
        if agent_failed:
            metrics["agent_failed"] = True
            metrics["error"] = snippet.warnings[0].message if snippet.warnings else "unknown"

        # --- Truncation detection ---
        if usage.output_tokens >= self._agent_max_tokens - 10:
            metrics["truncated"] = True
            logger.warning(
                "%s: output likely truncated (%d/%d tokens)",
                self.agent_name, usage.output_tokens, self._agent_max_tokens,
            )
            trace_events.append({
                "timestamp": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "type": "truncation_warning",
                "node": self.agent_name,
                "level": "warning",
                "message": (
                    f"Output may be truncated: {usage.output_tokens}/{self._agent_max_tokens} tokens"
                ),
                "data": {
                    "output_tokens": usage.output_tokens,
                    "max_tokens": self._agent_max_tokens,
                },
            })
        else:
            metrics["truncated"] = False

        # --- Optional self-critique: one pass of self-review against source data ---
        if self.settings.agent_self_critique and snippet.sections:
            snippet, critique_metrics, critique_trace = self._self_critique(
                snippet,
            )
            # Accumulate tokens/latency into the agent's metrics
            metrics["input_tokens"] += critique_metrics["input_tokens"]
            metrics["output_tokens"] += critique_metrics["output_tokens"]
            metrics["latency_ms"] = round(
                metrics["latency_ms"] + critique_metrics["latency_ms"], 1
            )
            metrics["self_critique"] = True
            trace_events.extend(critique_trace)

        # Trace: log output
        trace_events.append({
            "timestamp": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat(),
            "type": "agent_output",
            "node": self.agent_name,
            "level": "info",
            "message": f"Produced {len(snippet.sections)} sections: {', '.join(s.section for s in snippet.sections)}",
            "data": {
                "prompt_version": self.prompt_version,
                "sections": {
                    s.section: {
                        "content": s.content,
                        "confidence": s.confidence,
                        "data_sources_used": s.data_sources_used,
                    }
                    for s in snippet.sections
                },
                "warnings": [w.model_dump() for w in snippet.warnings],
                "metrics": metrics,
            },
        })

        return {
            "agent_snippets": [snippet],
            "pipeline_metrics": [metrics],
            "trace_events": trace_events,
        }

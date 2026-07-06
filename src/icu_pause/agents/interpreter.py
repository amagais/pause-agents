"""Interpreter agents for Cross-Referenced Dual-Stream Fusion (CR-DSF).

In CR-DSF mode, two interpreter agents pre-process raw data into clinical
summaries before domain agents see them:

  Stream 1: Structured Data → StructuredInterpreterAgent → per-domain summaries
  Stream 2: Clinical Notes  → NoteInterpreterAgent       → per-domain summaries
                                      ↓
                Both Summaries → Domain Agent → Output

This mirrors cross-attention from multimodal ML: two modality-specific
processing streams produce intermediate representations, which domain
agents then cross-reference for agreement, disagreement, and gap-filling.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from icu_pause.agents import note_chunking
from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm
from icu_pause.schemas.icu_pause import InterpreterOutput, SalienceOutput

logger = logging.getLogger(__name__)

# Domain agent descriptions for interpreter prompts
DOMAIN_AGENT_SPECS = {
    "nurse": {
        "label": "Nursing",
        "focus": "Vital signs, nursing assessments, patient demographics, ADT movements, SOFA scores",
        "sections": "I (ICU course), U_uncertainty (diagnostic uncertainty), S (summary/to-dos)",
    },
    "respiratory": {
        "label": "Respiratory Therapy",
        "focus": "Respiratory support devices/settings, ventilator parameters, ABGs, oxygenation, positioning",
        "sections": "I (ICU course), P (pending tests), A (active consults), E (exam at transfer)",
    },
    "pharmacy": {
        "label": "Pharmacy",
        "focus": "Medications (continuous infusions, intermittent meds), lab values (drug levels, renal/hepatic function), diagnoses",
        "sections": "U_unprescribing (high-risk meds), S (summary), E (exam/data review)",
    },
    "dietitian": {
        "label": "Nutrition/Dietetics",
        "focus": "Lab values (albumin, prealbumin, electrolytes), medications (TPN, insulin), vital signs, nursing assessments",
        "sections": "P (pending tests), A (active consults), S (summary)",
    },
    "case_manager": {
        "label": "Case Management",
        "focus": "Diagnoses, code status, goals of care, ADT movements, patient demographics, discharge planning",
        "sections": "C (code status/GOC), A (active consults), S (summary)",
    },
    "therapist": {
        "label": "Physical/Occupational Therapy",
        "focus": "Patient assessments (mobility, functional status), procedures, positioning history",
        "sections": "P (pending), A (active consults), U_uncertainty (diagnostic uncertainty), E (exam)",
    },
}


# Deterministic sentinel emitted WITHOUT an LLM call when a domain agent has no
# structured data routed to it. Handing the model an empty prompt is a prime
# confabulation trigger and a wasted call; the downstream domain agent reads
# this string in place of a summary/view. Worded window-neutrally on purpose —
# structured data is not gated on the 48h notes lookback.
NO_STRUCTURED_DATA = "No structured data available for this domain in the retrieval window."


def _structured_fields(agent_ctx: dict[str, Any]) -> dict[str, Any]:
    """The non-notes, non-null structured slice routed to one domain agent."""
    return {k: v for k, v in agent_ctx.items() if k != "notes" and v is not None}


def _extract_single_summary(domain_summaries: dict[str, str], agent_name: str) -> str:
    """Pull this agent's summary from a (now single-key) domain_summaries dict,
    tolerating a model that keyed it differently but returned exactly one entry."""
    ds = domain_summaries or {}
    if agent_name in ds:
        return ds[agent_name]
    if len(ds) == 1:
        return next(iter(ds.values()))
    return ""


class StructuredInterpreterAgent:
    """Interprets raw structured CLIF data into per-domain clinical summaries.

    Produces a dict mapping each domain agent name to a natural-language
    clinical summary of the structured data relevant to that agent's scope.
    """

    agent_name = "structured_interpreter"

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.settings = settings
        self._load_prompt(settings)

    def _load_prompt(self, settings: Settings):
        from pathlib import Path
        import yaml

        path = Path(settings.prompts_dir) / "structured_interpreter.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get("system_prompt", self._default_prompt())
        else:
            self.system_prompt = self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "You are a clinical data analyst specializing in ICU structured data interpretation. "
            "Given raw structured ICU data (vitals, labs, medications, respiratory support, etc.), "
            "produce a focused clinical summary for each downstream domain specialist. "
            "Each summary should highlight the clinically relevant findings for that specialist's scope. "
            "Use only the data provided. Do not infer diagnoses not supported by the data. "
            "Output valid JSON matching the InterpreterOutput schema."
        )

    def _build_user_message(
        self, agent_name: str, spec: dict[str, str], structured: dict[str, Any]
    ) -> str:
        # ONE domain per call. S1 reads only this agent's tiered slice
        # (agent_context_text[role]) — the per-agent routing/column-selection
        # already bounds the size, so each call stays well under the context
        # window. The previous all-six-in-one-call batching peaked at ~125k
        # input, overflowed ~1/3 of the cohort, and confounded S0-vs-S1 with a
        # single cross-domain call (pre-reg §F #15). Notes are the other stream.
        data_json = json.dumps(structured, indent=2, default=str)
        return (
            f"Produce a concise clinical summary (2-5 sentences) of the structured "
            f"data for the {spec['label']} specialist, using ONLY the data below. "
            f"Highlight abnormals, trends, and clinically actionable findings within "
            f"their scope — do not infer diagnoses the data does not support.\n\n"
            f"Focus: {spec['focus']}\n\n"
            f"## STRUCTURED DATA\n```json\n{data_json}\n```\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "agent_name": "structured_interpreter",\n'
            f'  "domain_summaries": {{ "{agent_name}": "<summary>" }},\n'
            f'  "warnings": ["<any data quality issues>"]\n'
            f'}}'
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        agent_contexts = state.get("agent_context_text", {})

        summaries: dict[str, str] = {}
        metrics_list: list[dict[str, Any]] = []
        n_summarized = n_empty = n_failed = 0

        for agent_name, spec in DOMAIN_AGENT_SPECS.items():
            structured = _structured_fields(agent_contexts.get(agent_name, {}))
            if not structured:
                # No data routed → deterministic sentinel, NO LLM call.
                summaries[agent_name] = NO_STRUCTURED_DATA
                n_empty += 1
                continue

            user_message = self._build_user_message(agent_name, spec, structured)
            try:
                output = self.llm.invoke(
                    system=self.system_prompt,
                    user=user_message,
                    response_format=InterpreterOutput,
                )
                summaries[agent_name] = _extract_single_summary(
                    output.domain_summaries, agent_name
                )
                usage = self.llm.last_usage
                metrics_list.append({
                    "agent": self.agent_name,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_ms": round(usage.latency_ms, 1),
                    "model": usage.model,
                })
                n_summarized += 1
            except Exception as e:
                # Fault-isolated: one domain's failure no longer zeroes the rest.
                logger.error(f"Structured interpreter failed: [{agent_name}] {e}")
                summaries[agent_name] = ""
                n_failed += 1

        logger.info(
            f"Structured interpreter: {n_summarized} summarized, {n_empty} no-data, "
            f"{n_failed} failed across {len(DOMAIN_AGENT_SPECS)} domains"
        )

        return {
            "structured_summaries": summaries,
            "pipeline_metrics": metrics_list,
        }


class StructuredSalienceSelectorAgent:
    """S2 — LLM salience selection over the per-agent tiered structured tables.

    Substitutive structural mirror of N2 (pre-reg compression sub-study, decision
    2026-06-04). Reads the SAME per-agent tiered tables S0/S1 read
    (agent_context_text[role]) and emits a per-agent *selected view* — the model
    LOCATES and SELECTS the salient rows/columns/time-windows (it does NOT
    paraphrase the values). The agent then reasons over this view IN PLACE OF the
    S0 tables (base.py s2 arm). Absent-but-requested fields render as the explicit
    "Not documented" null sentinel so a gap is visible, never hallucinated. The
    view contains only selected values — no raw-table JSON dump (substitutive;
    no-raw-table-leakage is asserted by tests/test_s2_substitutive.py, §F #23).
    """

    agent_name = "structured_salience"

    # Canonical fields the selector must surface per domain when present, else
    # render as "Not documented". Keeps the selection schema-bounded + auditable.
    CANONICAL_FIELDS = (
        "current vent settings (mode, FiO2, PEEP, set RR, set TV)",
        "active vasopressors/inotropes (drug + dose + trend)",
        "active sedation/analgesia infusions (drug + dose)",
        "most-recent + trend for critical labs (K, Na, Cr, Hgb, Plt, INR, lactate, glucose)",
        "active antibiotics (drug + day-of-therapy)",
        "CRRT/ECMO/HD status",
        "latest GCS/RASS, SOFA",
    )

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.settings = settings
        self._load_prompt(settings)

    def _load_prompt(self, settings: Settings):
        from pathlib import Path
        import yaml

        path = Path(settings.prompts_dir) / "structured_salience.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get("system_prompt", self._default_prompt())
        else:
            self.system_prompt = self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "You are a clinical data triage analyst for an ICU handoff. Given a "
            "domain specialist's routed structured data, SELECT the values that are "
            "salient to that specialist's scope and emit them verbatim in a compact, "
            "readable view. You LOCATE and SELECT values — you do NOT paraphrase, "
            "round, summarize, or invent them. Copy numbers and units exactly as they "
            'appear. For any canonical field that is absent, write "Not documented" — '
            "never omit it silently and never guess. Output valid JSON matching the "
            "InterpreterOutput schema (domain_summaries = the selected view per domain)."
        )

    def _build_user_message(
        self, agent_name: str, spec: dict[str, str], structured: dict[str, Any]
    ) -> str:
        # ONE domain per call (mirrors S1 / N2). The per-agent routed slice
        # bounds the size; the prior all-six-in-one-call batching overflowed
        # ~1/3 of the cohort and broke per-agent isolation (pre-reg §F #15/#23).
        data_json = json.dumps(structured, indent=2, default=str)
        canonical = "\n".join(f"  - {f}" for f in self.CANONICAL_FIELDS)
        return (
            f"Produce a SELECTED VIEW of the {spec['label']} specialist's routed "
            f"structured data below. Select the rows, columns, and time-windows most "
            f"salient to their scope and copy the values EXACTLY (no paraphrasing, no "
            f"rounding). Where present, surface these canonical fields; if a field is "
            f"absent, write \"Not documented\":\n"
            f"{canonical}\n\n"
            f"Focus: {spec['focus']}\n\n"
            f"## STRUCTURED DATA\n```json\n{data_json}\n```\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "agent_name": "structured_salience",\n'
            f'  "domain_summaries": {{ "{agent_name}": "<selected view>" }},\n'
            f'  "warnings": ["<any data quality issues>"]\n'
            f'}}'
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        agent_contexts = state.get("agent_context_text", {})

        views: dict[str, str] = {}
        metrics_list: list[dict[str, Any]] = []
        n_selected = n_empty = n_failed = 0

        for agent_name, spec in DOMAIN_AGENT_SPECS.items():
            structured = _structured_fields(agent_contexts.get(agent_name, {}))
            if not structured:
                # No data routed → deterministic sentinel, NO LLM call.
                views[agent_name] = NO_STRUCTURED_DATA
                n_empty += 1
                continue

            user_message = self._build_user_message(agent_name, spec, structured)
            try:
                output = self.llm.invoke(
                    system=self.system_prompt,
                    user=user_message,
                    response_format=SalienceOutput,
                )
                views[agent_name] = _extract_single_summary(
                    output.domain_summaries, agent_name
                )
                usage = self.llm.last_usage
                metrics_list.append({
                    "agent": self.agent_name,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "latency_ms": round(usage.latency_ms, 1),
                    "model": usage.model,
                })
                n_selected += 1
            except Exception as e:
                # Fault-isolated: one domain's failure no longer zeroes the rest.
                logger.error(f"Structured salience selector failed: [{agent_name}] {e}")
                views[agent_name] = ""
                n_failed += 1

        logger.info(
            f"Structured salience selector: {n_selected} selected, {n_empty} no-data, "
            f"{n_failed} failed across {len(DOMAIN_AGENT_SPECS)} domains"
        )

        return {
            "structured_views": views,
            "pipeline_metrics": metrics_list,
        }


class NoteInterpreterAgent:
    """Interprets clinical notes into per-domain clinical summaries.

    Uses the per-agent note routing to produce domain-specific summaries
    from each agent's assigned note types.
    """

    agent_name = "note_interpreter"

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.settings = settings
        self._load_prompt(settings)

    def _load_prompt(self, settings: Settings):
        from pathlib import Path
        import yaml

        path = Path(settings.prompts_dir) / "note_interpreter.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get("system_prompt", self._default_prompt())
        else:
            self.system_prompt = self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "You are a clinical documentation analyst specializing in ICU clinical notes. "
            "Given clinical notes routed to specific domain specialists, produce a focused "
            "clinical summary for each specialist highlighting the key findings from their "
            "assigned notes. Extract clinically relevant information — do not copy notes verbatim. "
            "If no notes are available for a specialist, state that explicitly. "
            "Output valid JSON matching the InterpreterOutput schema."
        )

    def _build_user_message(self, agent_contexts: dict[str, dict[str, Any]]) -> str:
        notes_by_domain = []
        for agent_name, spec in DOMAIN_AGENT_SPECS.items():
            agent_ctx = agent_contexts.get(agent_name, {})
            notes_data = agent_ctx.get("notes")
            if notes_data:
                notes_json = json.dumps(notes_data, indent=2, default=str)
                notes_by_domain.append(
                    f"### {agent_name} ({spec['label']})\n"
                    f"Focus: {spec['focus']}\n"
                    f"Sections: {spec['sections']}\n"
                    f"```json\n{notes_json}\n```"
                )
            else:
                notes_by_domain.append(
                    f"### {agent_name} ({spec['label']})\nNo notes routed to this agent."
                )

        return (
            f"Produce a clinical summary of the notes for each domain specialist.\n\n"
            f"## CLINICAL NOTES BY DOMAIN\n\n"
            f"{''.join(notes_by_domain)}\n\n"
            f"For each domain specialist, write a concise clinical summary (2-5 sentences) "
            f"of the relevant notes. Focus on active problems, recent changes, and "
            f"clinical decision-making documented in the notes.\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "agent_name": "note_interpreter",\n'
            f'  "domain_summaries": {{\n'
            f'    "nurse": "<summary from nursing notes>",\n'
            f'    "respiratory": "<summary from respiratory-relevant notes>",\n'
            f'    "pharmacy": "<summary from pharmacy-relevant notes>",\n'
            f'    "dietitian": "<summary from nutrition-relevant notes>",\n'
            f'    "case_manager": "<summary from case management notes>",\n'
            f'    "therapist": "<summary from therapy notes>"\n'
            f'  }},\n'
            f'  "warnings": ["<any data quality issues>"]\n'
            f'}}'
        )

    def _render_domain_block(
        self,
        agent_name: str,
        spec: dict[str, Any],
        agent_contexts: dict[str, dict[str, Any]],
        max_chars: int | None = None,
    ) -> str:
        """Render one domain's routed-notes block.

        Truncates the notes JSON to ``max_chars`` (with an explicit marker and a
        WARNING) only when a single domain's notes alone exceed the per-call
        budget -- an explicit, logged degradation, never a silent drop.
        """
        agent_ctx = agent_contexts.get(agent_name, {})
        notes_data = agent_ctx.get("notes")
        if not notes_data:
            return f"### {agent_name} ({spec['label']})\nNo notes routed to this agent."
        notes_json = json.dumps(notes_data, indent=2, default=str)
        if max_chars is not None and len(notes_json) > max_chars:
            notes_json = (
                notes_json[:max_chars]
                + "\n... [TRUNCATED: this domain's notes exceeded the per-call "
                "context budget; summary may be incomplete] ..."
            )
            logger.warning(
                f"Note interpreter truncated {agent_name} notes to fit the "
                f"per-call context budget ({max_chars} chars)"
            )
        return (
            f"### {agent_name} ({spec['label']})\n"
            f"Focus: {spec['focus']}\n"
            f"Sections: {spec['sections']}\n"
            f"```json\n{notes_json}\n```"
        )

    def _build_user_message_for_domains(
        self,
        agent_contexts: dict[str, dict[str, Any]],
        domains: list[str],
        truncate_chars: dict[str, int] | None = None,
    ) -> str:
        """Build a note-summary prompt for a SUBSET of domains (chunked path).

        Used only when the full set of routed notes would overflow the context
        window. The full-set path keeps using ``_build_user_message`` unchanged,
        so non-overflowing cases stay byte-identical to pre-change behavior.
        """
        truncate_chars = truncate_chars or {}
        notes_by_domain = [
            self._render_domain_block(
                name, DOMAIN_AGENT_SPECS[name], agent_contexts,
                max_chars=truncate_chars.get(name),
            )
            for name in domains
        ]
        example_lines = ",\n".join(
            f'    "{name}": "<summary from {DOMAIN_AGENT_SPECS[name]["label"]} notes>"'
            for name in domains
        )
        return (
            f"Produce a clinical summary of the notes for each domain specialist.\n\n"
            f"## CLINICAL NOTES BY DOMAIN\n\n"
            f"{''.join(notes_by_domain)}\n\n"
            f"For each domain specialist, write a concise clinical summary (2-5 sentences) "
            f"of the relevant notes. Focus on active problems, recent changes, and "
            f"clinical decision-making documented in the notes.\n\n"
            f"Respond with a JSON object:\n"
            f'{{\n'
            f'  "agent_name": "note_interpreter",\n'
            f'  "domain_summaries": {{\n'
            f'{example_lines}\n'
            f'  }},\n'
            f'  "warnings": ["<any data quality issues>"]\n'
            f'}}'
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        agent_contexts = state.get("agent_context_text", {})

        # --- Context-window guard (fusion-mode only) -------------------------
        # N1's single all-domains call overflows the served window on long-LOS
        # cases, which previously triggered a silent all-domain notes drop.
        # Plan domain-level chunks so each call stays within budget. When
        # everything fits one chunk (the common case) we take the ORIGINAL
        # single-call path below -- byte-identical to pre-change behavior.
        budget = int(getattr(self.settings, "fusion_note_input_budget", 115000))
        token_counts = {
            name: note_chunking.estimate_tokens(
                self._render_domain_block(name, spec, agent_contexts)
            )
            for name, spec in DOMAIN_AGENT_SPECS.items()
        }
        chunks = note_chunking.plan_domain_chunks(token_counts, budget)
        oversized = any(t > budget for t in token_counts.values())

        in_tok = out_tok = 0
        lat = 0.0
        model = None
        try:
            if len(chunks) <= 1 and not oversized:
                # Common case: everything fits -> original single-call path.
                user_message = self._build_user_message(agent_contexts)
                output = self.llm.invoke(
                    system=self.system_prompt,
                    user=user_message,
                    response_format=InterpreterOutput,
                )
                summaries = output.domain_summaries
                u = self.llm.last_usage
                in_tok, out_tok, lat, model = (
                    u.input_tokens, u.output_tokens, u.latency_ms, u.model,
                )
                logger.info(
                    f"Note interpreter produced summaries for "
                    f"{len(summaries)} domains"
                )
            else:
                # Overflow path: summarize domain-chunks separately, union the
                # per-domain results (each domain lives in exactly one chunk, so
                # no cross-chunk merge call is needed).
                logger.warning(
                    f"Note interpreter: routed notes exceed the per-call budget "
                    f"({budget} tokens); splitting into {len(chunks)} chunk(s) "
                    f"to avoid context-window overflow"
                )
                summaries = {}
                for chunk_domains in chunks:
                    truncate_chars = {}
                    if (
                        len(chunk_domains) == 1
                        and token_counts[chunk_domains[0]] > budget
                    ):
                        truncate_chars[chunk_domains[0]] = note_chunking.char_budget(budget)
                    user_message = self._build_user_message_for_domains(
                        agent_contexts, chunk_domains, truncate_chars
                    )
                    output = self.llm.invoke(
                        system=self.system_prompt,
                        user=user_message,
                        response_format=InterpreterOutput,
                    )
                    for d in chunk_domains:
                        summaries[d] = output.domain_summaries.get(d, "")
                    u = self.llm.last_usage
                    in_tok += u.input_tokens
                    out_tok += u.output_tokens
                    lat += u.latency_ms
                    model = u.model
                for name in DOMAIN_AGENT_SPECS:
                    summaries.setdefault(name, "")
                logger.info(
                    f"Note interpreter produced summaries for {len(summaries)} "
                    f"domains across {len(chunks)} chunk(s)"
                )
        except Exception as e:
            logger.error(f"Note interpreter failed: {e}")
            summaries = {name: "" for name in DOMAIN_AGENT_SPECS}

        metrics = {
            "agent": self.agent_name,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": round(lat, 1),
            "model": model,
        }

        return {
            "note_summaries": summaries,
            "pipeline_metrics": [metrics],
        }


# ---------------------------------------------------------------------------
# Predefined extraction schemas for CR-DSF+ mode
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMAS: dict[str, list[str]] = {
    "pharmacy": [
        "active_medications (name, dose, route, frequency, temporal_status: ACTIVE/STOPPED/TO_RESTART)",
        "anticoagulation_details (drug, dose, indication)",
        "antibiotic_details (drug, indication, start_date, planned_duration)",
    ],
    "therapist": [
        "slp_evaluation_status (completed/pending, date, result)",
        "diet_clearance (cleared_for: regular/thin_liquids/NPO, date)",
        "pt_ot_functional_assessment (evaluation_status, recommendations)",
        "mobility_level (independent/standby_assist/mod_assist/max_assist/bedbound)",
        "activity_restrictions (current restrictions, NOT superseded ones)",
    ],
    "respiratory": [
        "tracheostomy_details (type, size, cuff_status)",
        "current_vent_mode_settings (mode, FiO2, PEEP, TV, PS)",
        "weaning_plan (current_plan, next_steps)",
    ],
    "case_manager": [
        "code_status (full_code/DNR/DNI/DNR_DNI/comfort_care/not_documented)",
        "dpoa_contact (name, relationship, phone_if_available)",
        "disposition_plan (home/SNF/rehab/LTAC, details)",
    ],
    "dietitian": [
        "feeding_route_and_formula (route: TPN/enteral/PO, formula_name, rate)",
        "slp_swallow_result (cleared_for, date_of_evaluation)",
        "nutrition_labs (albumin, prealbumin, phosphorus — most recent values)",
    ],
    "nurse": [
        "active_lines_drains_wounds (type, site, condition for each)",
        "pain_assessment (score, location, current_management)",
    ],
}


class StructuredExtractorAgent:
    """Extracts predefined discrete fields from clinical notes (CR-DSF+ mode).

    Unlike the NoteInterpreterAgent which produces narrative summaries, this
    agent extracts specific factual fields per domain. These serve as anchors
    that prevent information loss during narrative summarization.
    """

    agent_name = "structured_extractor"

    def __init__(self, settings: Settings):
        self.llm: BaseLLM = create_llm(settings)
        self.settings = settings
        self._load_prompt(settings)

    def _load_prompt(self, settings: Settings):
        from pathlib import Path
        import yaml

        path = Path(settings.prompts_dir) / "structured_extractor.yaml"
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            self.system_prompt = data.get("system_prompt", self._default_prompt())
        else:
            self.system_prompt = self._default_prompt()

    def _default_prompt(self) -> str:
        return (
            "You are a clinical data extractor. Given clinical notes, extract "
            "specific predefined fields for each domain specialist. Output "
            "exact values — not narrative summaries. If a field is not found "
            "in the notes, write 'Not documented'. Output valid JSON."
        )

    def _build_user_message(self, agent_contexts: dict[str, dict[str, Any]]) -> str:
        parts = []
        parts.append("Extract the following SPECIFIC fields from the clinical notes.\n")
        parts.append("For each field, output the EXACT value found in the notes.")
        parts.append("If a field is not documented, write 'Not documented'.\n")

        for agent_name, fields in EXTRACTION_SCHEMAS.items():
            agent_ctx = agent_contexts.get(agent_name, {})
            notes_data = agent_ctx.get("notes")

            parts.append(f"### {agent_name}")
            parts.append("Fields to extract:")
            for field in fields:
                parts.append(f"  - {field}")

            if notes_data:
                notes_json = json.dumps(notes_data, indent=2, default=str)
                parts.append(f"Notes:\n```json\n{notes_json}\n```")
            else:
                parts.append("Notes: None available")
            parts.append("")

        parts.append("Respond with a JSON object:")
        parts.append('{')
        parts.append('  "agent_name": "structured_extractor",')
        parts.append('  "extraction_fields": {')
        for i, (agent_name, fields) in enumerate(EXTRACTION_SCHEMAS.items()):
            field_names = [f.split(" (")[0] for f in fields]
            field_json = ", ".join(f'"{fn}": "<value>"' for fn in field_names)
            comma = "," if i < len(EXTRACTION_SCHEMAS) - 1 else ""
            parts.append(f'    "{agent_name}": {{{field_json}}}{comma}')
        parts.append('  },')
        parts.append('  "warnings": ["<any extraction issues>"]')
        parts.append('}')

        return "\n".join(parts)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        from icu_pause.schemas.icu_pause import ExtractorOutput

        agent_contexts = state.get("agent_context_text", {})
        user_message = self._build_user_message(agent_contexts)

        try:
            output = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=ExtractorOutput,
            )
            logger.info(
                f"Structured extractor produced fields for "
                f"{len(output.extraction_fields)} domains"
            )
            fields = output.extraction_fields
        except Exception as e:
            logger.error(f"Structured extractor failed: {e}")
            fields = {name: {} for name in EXTRACTION_SCHEMAS}

        usage = self.llm.last_usage
        metrics = {
            "agent": self.agent_name,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "latency_ms": round(usage.latency_ms, 1),
            "model": usage.model,
        }

        return {
            "extraction_fields": fields,
            "pipeline_metrics": [metrics],
        }

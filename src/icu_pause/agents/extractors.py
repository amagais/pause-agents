"""Per-domain extractors — hybrid_v1 Option B core.

Pre-registered in PRE_REGISTRATION_compression_redesign.md §1.3 step B: each
domain has its own LLM extractor that reads its pre-routed raw notes (+ the
structured anchors visible in patient_context_text) and emits a flat dict
of extracted anchor fields. The fan-out is N parallel calls (one per
domain), not the single multi-domain call used in cr_dsf_plus.

This module is **net-new**. ``StructuredExtractorAgent`` in
``interpreter.py`` is intentionally left untouched so the ``cr_dsf_plus``
mode remains byte-identical for reproducibility (expert lock-pass §"Two
scope concerns" #1, 2026-05-31).

Field definitions reuse the v1.0 ``EXTRACTION_SCHEMAS`` from
``interpreter.py`` until pre-reg §1.5 schema expansion lands. Per-domain
prompts use the same hardened JSON-output discipline as the rest of the
compression layer (direct-JSON instruction, temperature 0, ``response_format``
on the local provider).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from icu_pause.agents.interpreter import EXTRACTION_SCHEMAS
from icu_pause.config import Settings
from icu_pause.llm.provider import BaseLLM, create_llm

logger = logging.getLogger(__name__)


# Domain agents that get a per-domain extractor in hybrid_v1.
DOMAIN_NAMES: tuple[str, ...] = (
    "nurse",
    "respiratory",
    "pharmacy",
    "dietitian",
    "case_manager",
    "therapist",
)


# Hardened system prompt mirroring the compression-stage spec locked in
# pre-reg §1.4 (vLLM xgrammar + direct-JSON + few-shot + temp 0):
# - "Respond with ONLY the JSON object."
# - "No preamble. First character must be `{`."
# - "If a field is not found, write 'Not documented'."
_DEFAULT_SYSTEM_PROMPT = """\
You are a clinical data extractor for the {domain} role on an ICU handoff team.

Read the clinical notes and extract the SPECIFIC predefined fields for the {domain} \
specialist. Output exact values pulled verbatim from the notes where possible — not \
narrative summaries. If a field is not documented in the notes, write "Not documented".

CRITICAL: Respond with ONLY the JSON object matching the schema below. No preamble. \
No phrases like "Here is" or "Sure". No markdown code fences. The first character of \
your response must be `{{`.

Example shape (illustrative; field names vary by domain):
{{"agent_name": "{domain}_extractor", "extraction_fields": {{"field_a": "value", "field_b": "Not documented"}}, "warnings": []}}

Fields to extract for {domain}:
{fields_list}
"""


class PerDomainExtractor:
    """One extractor per domain. Reads raw notes routed to this domain + the
    shared structured patient context, emits a flat ``{field: value}`` dict.

    Output is merged into ``state["per_domain_extractions"]`` keyed by
    ``domain_name`` so downstream Stage E can resolve anchors per-agent.
    """

    def __init__(
        self,
        settings: Settings,
        domain_name: str,
        fields: list[str],
    ) -> None:
        self.settings = settings
        self.domain_name = domain_name
        self.fields = fields
        # Each per-domain extractor gets its own LLM with temperature 0.0
        # (locked compression-stage setting per pre-reg §1.4) and structured
        # output via response_format. Domain agents keep the production
        # temperature unchanged — only the extractors pass the override.
        agent_max = settings.agent_max_tokens.get(
            self.agent_name, settings.llm_max_tokens
        )
        self.llm: BaseLLM = create_llm(
            settings,
            max_tokens_override=agent_max,
            temperature_override=0.0,
        )
        self.system_prompt = _DEFAULT_SYSTEM_PROMPT.format(
            domain=domain_name,
            fields_list="\n".join(f"  - {f}" for f in fields),
        )

    @property
    def agent_name(self) -> str:
        return f"{self.domain_name}_extractor"

    def _build_user_message(self, agent_contexts: dict[str, dict[str, Any]]) -> str:
        agent_ctx = agent_contexts.get(self.domain_name, {})
        notes_data = agent_ctx.get("notes")
        parts: list[str] = []
        parts.append(f"## DOMAIN\n{self.domain_name}")
        parts.append("")
        parts.append("## FIELDS TO EXTRACT")
        for field in self.fields:
            parts.append(f"- {field}")
        parts.append("")
        parts.append("## ROUTED CLINICAL NOTES")
        if notes_data:
            parts.append("```json")
            parts.append(json.dumps(notes_data, indent=2, default=str))
            parts.append("```")
        else:
            parts.append("No notes routed to this domain.")
        parts.append("")
        parts.append(
            "Return JSON: "
            '{"agent_name": "%s", "extraction_fields": {...}, "warnings": [...]}'
            % self.agent_name
        )
        return "\n".join(parts)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node: extract this domain's anchors from routed notes."""
        from icu_pause.schemas.icu_pause import ExtractorOutput

        agent_contexts = state.get("agent_context_text", {})
        user_message = self._build_user_message(agent_contexts)
        fields_out: dict[str, str] = {}

        try:
            output = self.llm.invoke(
                system=self.system_prompt,
                user=user_message,
                response_format=ExtractorOutput,
            )
            # ExtractorOutput.extraction_fields is dict[domain, dict[field, value]].
            # Per-domain extractors typically return their own domain's slice;
            # tolerate both shapes (nested under domain_name OR already flat).
            raw = getattr(output, "extraction_fields", {}) or {}
            if isinstance(raw, dict) and self.domain_name in raw:
                fields_out = dict(raw[self.domain_name])
            elif isinstance(raw, dict):
                # Flat dict — treat as this domain's fields directly.
                fields_out = {
                    k: v for k, v in raw.items()
                    if isinstance(v, str)
                }
            logger.info(
                f"Per-domain extractor {self.domain_name}: extracted "
                f"{len(fields_out)} fields"
            )
        except Exception as e:
            logger.error(f"Per-domain extractor {self.domain_name} failed: {e}")
            fields_out = {}

        usage = getattr(self.llm, "last_usage", None)
        metrics_payload: list[dict[str, Any]] = []
        if usage is not None:
            metrics_payload = [{
                "agent": self.agent_name,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "latency_ms": round(usage.latency_ms, 1),
                "model": usage.model,
            }]

        return {
            # Single-domain slice merged into the per_domain_extractions
            # dict by the state reducer.
            "per_domain_extractions": {self.domain_name: fields_out},
            "pipeline_metrics": metrics_payload,
        }


def build_per_domain_extractors(settings: Settings) -> dict[str, PerDomainExtractor]:
    """Construct one extractor per domain for hybrid_v1.

    Uses the v1.0 ``EXTRACTION_SCHEMAS`` from ``interpreter.py``. When pre-reg
    §1.5 schema expansion lands, add the new fields to ``EXTRACTION_SCHEMAS``
    in interpreter.py — they will flow through automatically.
    """
    return {
        domain: PerDomainExtractor(
            settings=settings,
            domain_name=domain,
            fields=EXTRACTION_SCHEMAS[domain],
        )
        for domain in DOMAIN_NAMES
        if domain in EXTRACTION_SCHEMAS
    }

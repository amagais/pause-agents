"""Dietitian Agent: nutrition, labs (albumin), weight → A, S sections."""

from icu_pause.agents.base import BaseDomainAgent


class DietitianAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "dietitian"

    @property
    def required_context_keys(self) -> list[str]:
        return [
            "labs",
            "vitals",
            "assessments",
            "notes",
            # meds removed: Northwestern's CLIF medication_admin tables do not
            # represent nutrition administration at any layer (med_category,
            # med_route_name, or med_name). CLIF v2.1 also does not define a
            # medication_orders table, so order-level nutrition data is not
            # available either. See docs/clif_data_gaps_investigation.md.
            # Re-add if CLIF v2.x adds a dedicated nutrition domain or a
            # medication_orders table that captures TPN/EN orders.
        ]

    @property
    def target_sections(self) -> list[str]:
        return ["S"]

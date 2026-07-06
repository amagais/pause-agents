"""Respiratory Agent: vent settings, weaning trials → U_uncertainty, S, E sections."""

from icu_pause.agents.base import BaseDomainAgent


class RespiratoryAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "respiratory"

    @property
    def required_context_keys(self) -> list[str]:
        return ["respiratory", "vitals", "labs", "position", "microbiology", "notes"]

    @property
    def target_sections(self) -> list[str]:
        return ["S", "E"]

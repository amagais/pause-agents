"""Therapist Agent: mobility scores, functional assessments → P, A, U_uncertainty, S, E sections."""

from icu_pause.agents.base import BaseDomainAgent


class TherapistAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "therapist"

    @property
    def required_context_keys(self) -> list[str]:
        return ["assessments", "position", "notes"]

    @property
    def target_sections(self) -> list[str]:
        return ["A", "S", "E"]

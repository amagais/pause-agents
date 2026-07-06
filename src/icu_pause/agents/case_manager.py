"""Case Manager Agent: diagnoses, code status, discharge planning → C, A, S sections."""

from icu_pause.agents.base import BaseDomainAgent


class CaseManagerAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "case_manager"

    @property
    def required_context_keys(self) -> list[str]:
        return ["code_status", "adt", "demographics", "notes"]

    @property
    def target_sections(self) -> list[str]:
        return ["C", "A", "S"]

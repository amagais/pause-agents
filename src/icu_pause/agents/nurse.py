"""Nurse Agent: vitals, assessments → U_uncertainty, S, E sections."""

from icu_pause.agents.base import BaseDomainAgent


class NurseAgent(BaseDomainAgent):
    @property
    def agent_name(self) -> str:
        return "nurse"

    @property
    def required_context_keys(self) -> list[str]:
        # ``meds`` here delivers ONLY the classified infusion-state view
        # (``meds.states.records`` from med_state classifier). Raw
        # medication_admin_continuous rows are stripped for nurse in
        # workflow.py — nurse must not infer infusion activity from
        # individual admin events. See nurse.yaml STATE OF INFUSIONS block.
        return ["vitals", "assessments", "demographics", "adt", "sofa", "notes", "meds"]

    @property
    def target_sections(self) -> list[str]:
        return ["S", "E"]

"""Deliberation Node: orchestrates agent discussion when QA finds contradictions."""

from __future__ import annotations

import logging
from typing import Any

from icu_pause.agents.base import BaseDomainAgent
from icu_pause.config import Settings
from icu_pause.schemas.icu_pause import AgentSnippet

logger = logging.getLogger(__name__)


class DeliberationNode:
    """Orchestrates targeted agent revision when QA flags contradictions.

    When QA identifies conflicting claims between agents, this node:
    1. Parses QA issue strings to identify which agents are in conflict
    2. Invokes each conflicting agent's revise() with the other's output
    3. Collects revised snippets and logs all changes for transparency
    """

    def __init__(self, settings: Settings, agent_instances: dict[str, BaseDomainAgent]):
        self.settings = settings
        self.agents = agent_instances
        self.max_rounds = settings.max_deliberation_rounds

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Parse QA issues, identify conflicting agents, invoke revisions."""
        qa_issues: list[str] = state.get("qa_issues", [])
        snippets: list[AgentSnippet] = state.get("agent_snippets", [])

        all_revised: list[AgentSnippet] = []
        all_log: list[dict] = []
        all_metrics: list[dict] = []
        revised_agent_names: set[str] = set()

        for issue in qa_issues:
            involved_agents = self._extract_agents(issue, snippets)
            if len(involved_agents) < 2:
                # Need at least two agents for a meaningful conflict
                logger.debug(f"Skipping issue (< 2 agents identified): {issue[:80]}...")
                continue

            for agent_name in involved_agents:
                if agent_name not in self.agents:
                    continue
                # Only revise each agent once even if mentioned in multiple issues
                if agent_name in revised_agent_names:
                    logger.debug(f"Agent {agent_name} already revised, skipping")
                    continue

                agent = self.agents[agent_name]
                other_outputs = self._get_conflicting_outputs(
                    agent_name, involved_agents, snippets
                )

                logger.info(
                    f"Deliberation: asking {agent_name} to revise "
                    f"(conflict with {[a for a in involved_agents if a != agent_name]})"
                )

                result = agent.revise(state, qa_issue=issue, conflicting_output=other_outputs)
                all_revised.extend(result.get("revised_snippets", []))
                all_log.extend(result.get("deliberation_log", []))
                all_metrics.extend(result.get("pipeline_metrics", []))
                revised_agent_names.add(agent_name)

        logger.info(
            f"Deliberation complete: {len(revised_agent_names)} agents revised, "
            f"{len(all_log)} log entries"
        )

        return {
            "revised_snippets": all_revised,
            "deliberation_log": all_log,
            "pipeline_metrics": all_metrics,
        }

    @staticmethod
    def _extract_agents(issue: str, snippets: list[AgentSnippet]) -> list[str]:
        """Parse a QA issue string to find which agents are mentioned.

        Matches patterns like "Nurse agent", "pharmacy agent", "[nurse]",
        or bare agent names in the issue text.
        """
        found = []
        agent_names = {s.agent_name for s in snippets}
        issue_lower = issue.lower()
        for name in agent_names:
            # Match "nurse agent", "[nurse]", or standalone "nurse"
            # Use word-boundary-like matching to avoid false positives
            name_lower = name.lower()
            if (
                f"{name_lower} agent" in issue_lower
                or f"[{name_lower}]" in issue_lower
                or f"{name_lower}:" in issue_lower
            ):
                found.append(name)
        return found

    @staticmethod
    def _get_conflicting_outputs(
        agent_name: str, involved_agents: list[str], snippets: list[AgentSnippet]
    ) -> str:
        """Get output text from other agents involved in the conflict."""
        parts = []
        for snippet in snippets:
            if snippet.agent_name in involved_agents and snippet.agent_name != agent_name:
                for sec in snippet.sections:
                    if sec.content and sec.content != "Not enough information from structured data.":
                        parts.append(f"[{snippet.agent_name} → {sec.section}]: {sec.content}")
        return "\n".join(parts)

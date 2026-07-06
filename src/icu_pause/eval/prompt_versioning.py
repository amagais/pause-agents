"""Prompt versioning and snapshot utilities.

Provides stable identifiers for prompt versions (SHA-1 hash) and snapshot
utilities for capturing pipeline state at key checkpoints:
- Pre-QA: domain agent outputs
- Post-QA: revised outputs after deliberation
- Post-Intensivist: final harmonized output

This enables:
1. Reproducibility: re-run against the exact prompt that produced a result
2. Deliberation delta: measure what fraction of cases QA actually changed
3. Regression testing: prove that changing one agent's prompt doesn't break others
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt hashing
# ---------------------------------------------------------------------------


def get_prompt_hash(agent_name: str, prompt: str) -> str:
    """Generate a stable short identifier for a specific prompt version.

    Args:
        agent_name: Agent name (e.g., "nurse", "pharmacy").
        prompt: The full system prompt text.

    Returns:
        String like "nurse-a3f8c1b2" (agent name + 8-char SHA-1 prefix).
    """
    sha = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{agent_name}-{sha}"


def hash_all_prompts(prompts_dir: str) -> dict[str, str]:
    """Hash all prompt YAML files in the prompts directory.

    Args:
        prompts_dir: Path to the directory containing prompt YAML files.

    Returns:
        Dict mapping agent name to prompt hash (e.g., {"nurse": "nurse-a3f8c1b2"}).
    """
    prompts_path = Path(prompts_dir)
    hashes: dict[str, str] = {}

    if not prompts_path.exists():
        return hashes

    for yaml_file in sorted(prompts_path.glob("*.yaml")):
        agent_name = yaml_file.stem  # e.g., "nurse" from "nurse.yaml"
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            prompt_text = data.get("system_prompt", "")
            hashes[agent_name] = get_prompt_hash(agent_name, prompt_text)
        except Exception as e:
            logger.warning(f"Could not hash prompt for {agent_name}: {e}")

    return hashes


def get_all_prompt_versions(prompts_dir: str) -> dict[str, str]:
    """Read the human-readable version field from all prompt YAML files.

    Args:
        prompts_dir: Path to the directory containing prompt YAML files.

    Returns:
        Dict mapping agent name to version string (e.g., {"nurse": "1.0"}).
    """
    prompts_path = Path(prompts_dir)
    versions: dict[str, str] = {}

    if not prompts_path.exists():
        return versions

    for yaml_file in sorted(prompts_path.glob("*.yaml")):
        agent_name = yaml_file.stem
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            versions[agent_name] = str(data.get("version", "unknown"))
        except Exception as e:
            logger.warning(f"Could not read version for {agent_name}: {e}")

    return versions


# ---------------------------------------------------------------------------
# Pipeline snapshot
# ---------------------------------------------------------------------------


class AgentSnapshot(BaseModel):
    """Snapshot of a single agent's output at a checkpoint."""

    agent_name: str
    prompt_hash: str
    sections: dict[str, str]  # section_key -> content
    warnings: list[str] = Field(default_factory=list)


class PipelineSnapshot(BaseModel):
    """Snapshot of the full pipeline state at a checkpoint."""

    checkpoint: str  # "pre_qa", "post_qa", "post_intensivist"
    timestamp: str
    prompt_hashes: dict[str, str]  # agent_name -> prompt_hash
    prompt_versions: dict[str, str] = Field(default_factory=dict)  # agent_name -> human-readable version
    agent_snapshots: list[AgentSnapshot]
    qa_issues: list[str] = Field(default_factory=list)
    qa_passed: bool | None = None


class DeliberationDelta(BaseModel):
    """Measures what changed between pre-QA and post-QA snapshots."""

    agents_revised: list[str]
    sections_changed: int
    sections_unchanged: int
    change_rate: float  # sections_changed / total
    per_agent_changes: dict[str, dict[str, str]]  # agent -> {section -> "added"|"modified"|"unchanged"}


def capture_snapshot(
    checkpoint: str,
    agent_snippets: list[Any],
    prompt_hashes: dict[str, str],
    qa_issues: list[str] | None = None,
    qa_passed: bool | None = None,
    prompt_versions: dict[str, str] | None = None,
) -> PipelineSnapshot:
    """Capture the pipeline state at a checkpoint.

    Args:
        checkpoint: Name of the checkpoint ("pre_qa", "post_qa", "post_intensivist").
        agent_snippets: List of AgentSnippet objects.
        prompt_hashes: Dict of agent_name -> prompt_hash.
        qa_issues: QA issues found (for post_qa checkpoint).
        qa_passed: Whether QA passed (for post_qa checkpoint).
        prompt_versions: Dict of agent_name -> human-readable version string.

    Returns:
        PipelineSnapshot with all agent outputs frozen.
    """
    snapshots = []
    for snippet in agent_snippets:
        sections = {
            sec.section: sec.content
            for sec in snippet.sections
        }
        snapshots.append(AgentSnapshot(
            agent_name=snippet.agent_name,
            prompt_hash=prompt_hashes.get(snippet.agent_name, "unknown"),
            sections=sections,
            warnings=[w.message for w in snippet.warnings],
        ))

    return PipelineSnapshot(
        checkpoint=checkpoint,
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_hashes=prompt_hashes,
        prompt_versions=prompt_versions or {},
        agent_snapshots=snapshots,
        qa_issues=qa_issues or [],
        qa_passed=qa_passed,
    )


def compute_deliberation_delta(
    pre_qa: PipelineSnapshot,
    post_qa: PipelineSnapshot,
) -> DeliberationDelta:
    """Compare pre-QA and post-QA snapshots to measure deliberation impact.

    Args:
        pre_qa: Snapshot taken before QA/deliberation.
        post_qa: Snapshot taken after QA/deliberation.

    Returns:
        DeliberationDelta with detailed change analysis.
    """
    # Build lookup: agent_name -> {section -> content}
    pre_agents: dict[str, dict[str, str]] = {}
    for snap in pre_qa.agent_snapshots:
        pre_agents[snap.agent_name] = snap.sections

    post_agents: dict[str, dict[str, str]] = {}
    for snap in post_qa.agent_snapshots:
        post_agents[snap.agent_name] = snap.sections

    agents_revised: list[str] = []
    sections_changed = 0
    sections_unchanged = 0
    per_agent_changes: dict[str, dict[str, str]] = {}

    all_agents = set(pre_agents.keys()) | set(post_agents.keys())
    for agent_name in sorted(all_agents):
        pre_sections = pre_agents.get(agent_name, {})
        post_sections = post_agents.get(agent_name, {})
        all_section_keys = set(pre_sections.keys()) | set(post_sections.keys())

        agent_changes: dict[str, str] = {}
        agent_had_changes = False

        for section_key in sorted(all_section_keys):
            pre_content = pre_sections.get(section_key, "")
            post_content = post_sections.get(section_key, "")

            if pre_content == post_content:
                agent_changes[section_key] = "unchanged"
                sections_unchanged += 1
            elif not pre_content and post_content:
                agent_changes[section_key] = "added"
                sections_changed += 1
                agent_had_changes = True
            else:
                agent_changes[section_key] = "modified"
                sections_changed += 1
                agent_had_changes = True

        per_agent_changes[agent_name] = agent_changes
        if agent_had_changes:
            agents_revised.append(agent_name)

    total = sections_changed + sections_unchanged
    change_rate = sections_changed / total if total > 0 else 0.0

    return DeliberationDelta(
        agents_revised=agents_revised,
        sections_changed=sections_changed,
        sections_unchanged=sections_unchanged,
        change_rate=round(change_rate, 4),
        per_agent_changes=per_agent_changes,
    )


def save_snapshot(snapshot: PipelineSnapshot, output_dir: str, hosp_id: str) -> Path:
    """Save a pipeline snapshot to disk as JSON.

    Args:
        snapshot: The snapshot to save.
        output_dir: Directory to write to.
        hosp_id: Hospitalization ID for the filename.

    Returns:
        Path to the saved file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = f"{hosp_id}_{snapshot.checkpoint}.json"
    file_path = out_path / filename

    with open(file_path, "w") as f:
        json.dump(snapshot.model_dump(), f, indent=2, default=str)

    logger.info(f"Snapshot saved: {file_path}")
    return file_path

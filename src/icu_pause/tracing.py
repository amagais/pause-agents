"""Run tracing infrastructure for ICU-PAUSE pipeline.

Captures per-run debug information — data loaded, agent inputs/outputs,
metrics, timing — and saves as JSON for reproducibility and review.

Each run produces a trace file at:
  output/runs/{hospitalization_id}_{timestamp}.trace.json
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RunTrace:
    """Collects trace events for a single pipeline run."""

    def __init__(self, hospitalization_id: str, run_dir: str | None = None):
        self.hospitalization_id = hospitalization_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.events: list[dict[str, Any]] = []
        self._run_dir = run_dir or os.environ.get(
            "ICUPAUSE_RUN_DIR", "output/runs"
        )

    def log(
        self,
        event_type: str,
        node: str,
        *,
        message: str = "",
        data: Any = None,
        level: str = "info",
    ) -> dict[str, Any]:
        """Add a trace event and return it (for SSE streaming)."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "node": node,
            "level": level,
            "message": message,
        }
        if data is not None:
            event["data"] = data
        self.events.append(event)

        # Also log to Python logger
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[trace:{node}] {message}")

        return event

    def log_data_loaded(
        self,
        node: str,
        table_name: str,
        row_count: int,
        columns: list[str] | None = None,
        source_path: str | None = None,
    ) -> dict[str, Any]:
        """Log a data table/note being loaded."""
        data = {
            "table": table_name,
            "rows": row_count,
        }
        if columns:
            data["columns"] = columns
        if source_path:
            data["source"] = source_path
        return self.log(
            "data_loaded",
            node,
            message=f"Loaded {table_name}: {row_count} rows",
            data=data,
        )

    def log_agent_input(
        self,
        agent_name: str,
        input_keys: list[str],
        input_preview: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log what data an agent received."""
        data: dict[str, Any] = {"input_keys": input_keys}
        if input_preview:
            data["preview"] = input_preview
        return self.log(
            "agent_input",
            agent_name,
            message=f"Agent received {len(input_keys)} data keys: {', '.join(input_keys)}",
            data=data,
        )

    def log_agent_output(
        self,
        agent_name: str,
        sections: list[str],
        warnings: list[str] | None = None,
        confidence: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log what an agent produced."""
        data: dict[str, Any] = {"sections": sections}
        if warnings:
            data["warnings"] = warnings
        if confidence:
            data["confidence"] = confidence
        return self.log(
            "agent_output",
            agent_name,
            message=f"Produced {len(sections)} sections: {', '.join(sections)}",
            data=data,
        )

    def log_note_routing(
        self,
        agent_name: str,
        note_types: dict[str, int],
    ) -> dict[str, Any]:
        """Log which notes were routed to an agent and how many rows each."""
        total = sum(note_types.values())
        detail = ", ".join(f"{k}({v})" for k, v in note_types.items())
        return self.log(
            "note_routing",
            agent_name,
            message=f"Routed {total} note rows: {detail}" if total > 0 else "No notes routed",
            data={"note_types": note_types, "total_rows": total},
        )

    def log_metrics(
        self,
        agent_name: str,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Log agent metrics (tokens, latency, model)."""
        return self.log(
            "metrics",
            agent_name,
            message=(
                f"model={metrics.get('model', '?')} "
                f"tokens={metrics.get('input_tokens', 0)}in/{metrics.get('output_tokens', 0)}out "
                f"latency={metrics.get('latency_ms', 0):.0f}ms"
            ),
            data=metrics,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full trace."""
        return {
            "hospitalization_id": self.hospitalization_id,
            "started_at": self.started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "event_count": len(self.events),
            "events": self.events,
        }

    def save(self) -> str | None:
        """Save trace to a JSON file. Returns the file path."""
        try:
            run_dir = Path(self._run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{self.hospitalization_id}.trace.json"
            path = run_dir / filename

            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

            logger.info(f"Trace saved to {path}")
            return str(path)
        except Exception as e:
            logger.warning(f"Failed to save trace: {e}")
            return None

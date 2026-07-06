"""LangGraph Studio entry point.

Exposes a compiled graph instance that LangGraph Studio can discover
via langgraph.json. This module builds the graph with default settings
so Studio can visualize the topology and run interactive traces.

Usage:
    langgraph dev   # auto-discovers via langgraph.json → localhost:8123
"""

from __future__ import annotations

import os

from icu_pause.config import Settings
from icu_pause.graph.workflow import build_graph

# Build graph with settings from environment / .env file.
# LangGraph Studio imports this module and looks for a ``graph`` variable.
_clif_dir = os.environ.get("ICUPAUSE_CLIF_DATA_DIR", "")
if not _clif_dir:
    # Provide a sensible fallback so Studio can at least show the topology
    # even without data configured. The graph will error at runtime if you
    # try to invoke without valid data, but the visual still works.
    _clif_dir = "/tmp/clif_data"

_settings = Settings(clif_data_dir=_clif_dir)
graph = build_graph(_settings)

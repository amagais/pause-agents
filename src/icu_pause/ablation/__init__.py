"""Decomposition-ablation arms (additive; production pipeline untouched).

This package implements the arms for the "does section-owned decomposition beat
a single generalist LLM?" ablation. Nothing here is imported by the production
pipeline — it composes the existing pipeline pieces (DataRetriever, agents,
build_graph) into ablated configurations and adds two monolith baselines.

Arms:
    full                 -- build_graph(settings), early_fusion (production path)
    monolith_best_effort -- one expert prompt, free-form best all-sections brief
    monolith_templated   -- one call given the 8-section schema (isolates structure)
    extract_only         -- Scribe extraction -> mechanical synthesis (no domain agents)
    no_intensivist       -- domain agents + mechanical merge (PRIMARY metric only)
    no_qa                -- domain agents -> intensivist directly (skip QA node)

All graph-based arms run early_fusion (the compression/fusion sub-study is out of
scope for this paper). Identical data bundle, model, and temperature=0 across arms.
"""

from icu_pause.ablation.arms import ARM_KEYS, build_arm  # noqa: F401

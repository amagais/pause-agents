# ICU-PAUSE Agents

**A multi-agent LLM system for auto-generating structured ICU-to-ward transfer handoff briefs, with a full evaluation harness.**

ICU-PAUSE reads structured ICU data in the [CLIF](https://clif-consortium.github.io/website/) (Common Longitudinal ICU Format) schema plus clinical notes, and produces a structured transfer summary organized around the **ICU-PAUSE mnemonic**. A panel of specialized domain agents runs in parallel, a QA agent cross-validates for internal consistency, an intensivist agent synthesizes the clinical narrative, and a deterministic merger compiles the final brief. The repository also ships the evaluation code used to assess brief quality (PDSQI-9, grounding/hallucination, and an LLM-as-judge calibration harness) and a human-review web app used to collect clinician ratings.

> **Status:** research code accompanying a preprint. This is a cleaned public release of the pipeline and evaluation code. It contains **no patient data and no protected health information (PHI)** — see [Data availability](#data-availability).

---

## System overview

```
CLIF parquet + clinical notes
          │
          ▼
   Data retrieval  ──►  scribe / structured extraction
          │
          ▼
   ┌───────────────────────────────────────────────┐
   │  Domain agents (parallel)                       │
   │  nurse · respiratory · pharmacy · dietitian ·   │
   │  case manager · therapist                       │
   └───────────────────────────────────────────────┘
          │
          ▼
   Resident (cross-domain analysis)
          │
          ▼
   QA agent (deterministic safety tools + LLM consistency)
          │
          ▼
   Intensivist (clinical synthesis & harmonization)
          │
          ▼
   Orchestrator / section merger  ──►  structured brief
```

- **Domain agents** each own a clinical lane and cite the source rows they draw from.
- **Deterministic safety tools** (drug interactions, device dwell-time, lab reference ranges) run before the LLM QA pass, so high-severity checks never depend on model behavior.
- **Intensivist** rewrites and harmonizes agent output into a coherent narrative.
- **Orchestrator** applies deterministic to-do and section-ownership rules and emits the final brief.

---

## Repository layout

```
src/icu_pause/
  agents/          # Domain agents (nurse, respiratory, pharmacy, dietitian,
                   #   case_manager, therapist), plus intensivist, resident,
                   #   QA, deliberation, scribe, extractors, orchestrator
  data/            # CLIF retrieval, note routing, context assembly
  graph/           # LangGraph pipeline definition
  llm/             # LLM provider abstraction (local / OpenAI / Anthropic / Azure)
  eval/            # Evaluation: PDSQI-9, grounding, HQI, judge calibration,
                   #   concordance, numeric fidelity, batch runner
  ablation/        # Ablation arms + monolith baselines + scoring
  tools/           # Deterministic safety tools (drug interactions, device
                   #   dwell, lab ranges, med state, canonicalization)
  safety/          # Clinical-context reframing + guards
  rendering/       # Brief formatter, citation rendering, doc export
  schemas/         # Pydantic models + graph state
  config.py        # Settings (environment-driven)
  main.py          # CLI entry point

config/prompts/    # YAML prompt template per agent
config/settings.yaml

review_app/        # Streamlit human-review app used to collect clinician
                   #   ratings (PDSQI-9, hallucination checks, omissions).
                   #   Reads generated briefs; writes ratings to blob storage.
```

The React web UI, internal design docs, experiment outputs, and all patient-derived
artifacts are intentionally **not** included in this release.

---

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/amagais/pause-agents.git
cd pause-agents

# core pipeline
pip install -e .

# with the human-review app and evaluation extras
pip install -e ".[review,anthropic]"
```

(If you use [uv](https://docs.astral.sh/uv/): `uv sync --extra review`.)

---

## Configuration

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Key settings:

| Variable | Purpose |
|----------|---------|
| `ICUPAUSE_CLIF_DATA_DIR` | Directory of CLIF-formatted parquet tables |
| `ICUPAUSE_NOTES_DATA_DIR` | (optional) separate directory for clinical-note CSVs |
| `ICUPAUSE_LLM_PROVIDER` | `local` (Ollama/vLLM, default) · `openai` · `anthropic` · `azure` |
| `ICUPAUSE_LLM_MODEL` | Model name for the selected provider |

### LLM providers

| Provider | `ICUPAUSE_LLM_PROVIDER` | Example model | Key |
|----------|--------------------------|---------------|-----|
| Local (Ollama / vLLM, OpenAI-compatible) | `local` | any served model | none |
| OpenAI | `openai` | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-sonnet-4-...` | `ANTHROPIC_API_KEY` |
| Azure OpenAI | `azure` | your deployment | `AZURE_OPENAI_API_KEY` + endpoint |

The system defaults to a local model so it can run fully open-source with no API keys.

### Expected data

The pipeline reads CLIF parquet tables (`patient`, `hospitalization`, `adt`, `vitals`,
`labs`, `medication_admin_continuous`/`_intermittent`, `respiratory_support`,
`patient_assessments`, `code_status`, `microbiology_culture`, `crrt_therapy`,
`ecmo_mcs`, `position`) and, optionally, clinical-note CSVs matched by fuzzy filename.
**No data files are shipped with this repository** — you supply your own CLIF dataset.

---

## Usage

### Generate a brief (CLI)

```bash
icu-pause --hospitalization-id <hospitalization_id>
icu-pause --hospitalization-id <hospitalization_id> --lookback-hours 24 --output-format text
icu-pause --hospitalization-id <hospitalization_id> --lookback-hours 0   # entire stay
```

The `--reference-dttm` flag anchors *when* the brief is considered "written" for
retrospective evaluation: all structured data and notes at or after that timestamp are
excluded, so the system sees exactly what the physician saw when they wrote the
real transfer note.

```bash
icu-pause --hospitalization-id <hospitalization_id> --reference-dttm "2024-07-15T14:00:00"
```

### Batch evaluation

```bash
python -m icu_pause.eval.batch --ids cases.txt --output-dir results/
```

---

## Evaluation harness

The `eval/` package contains the metrics used to assess brief quality:

- **PDSQI-9** (`eval/pdsqi9.py`) — the validated Provider Documentation Summarization
  Quality Instrument (cited, accurate, thorough, useful, organized, comprehensible,
  succinct, synthesized, non-stigmatizing) applied as an LLM-as-judge.
- **Grounding / hallucination** (`eval/grounding.py`) — flags claims not supported by
  the source data.
- **HQI** (`eval/hqi.py`) — ICU-PAUSE handoff quality index.
- **Judge calibration & concordance** (`eval/judge_calibration.py`,
  `eval/concordance.py`, `eval/judge_leaderboard.py`) — compares LLM judges against
  human ratings.
- **Ablation** (`ablation/`) — decomposition arms plus monolith baselines and scoring.

---

## Human-review app

`review_app/` is a Streamlit application used to collect clinician ratings on generated
briefs. Reviewers see a rendered brief alongside its source notes and complete a
structured form: **PDSQI-9** scores, **hallucination** claim-level checks, and
**omissions**. Responses are written to Azure Blob Storage.

```bash
cd review_app
pip install -r requirements.txt
cp .env.example .env      # configure blob storage + auth
streamlit run app.py
```

The app expects generated briefs and source bundles to be provided via its storage
backend; it ships with **no cases and no patient data**.

---

## Deterministic safety tools

Three no-LLM checks run before the LLM QA pass:

| Tool | File | Checks |
|------|------|--------|
| Drug-interaction checker | `tools/drug_interactions.py` | ICU-critical interaction table (+ optional openFDA fallback) |
| Device dwell-time | `tools/device_dwell.py` | Lines/catheters exceeding clinical thresholds |
| Lab reference ranges | `tools/lab_ranges.py` | Critical values and agent mischaracterizations |

---

## Data availability

This repository contains **only source code**. It does **not** contain any patient
data, protected health information (PHI), model outputs, reviewer responses, or
identifiers of any kind. The clinical dataset used in the associated study cannot be
shared publicly because it contains PHI governed by institutional data-use agreements
and IRB oversight. To run the pipeline you must supply your own CLIF-formatted dataset.

---

## Citation

If you use this code, please cite the accompanying preprint (details forthcoming).

```bibtex
@misc{icupause_agents,
  title  = {PAUSE-Agents: A Clinician-in-the-Loop Multi-Agent AI Pipeline for ICU-to-Ward Handoff Briefs},
  author = {Amagai, Saki and colleagues},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Released under the [MIT License](LICENSE).

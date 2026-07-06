# PHI redaction (philter-ucsf)

Redacts names and phone numbers from `source_bundle.json` and `output.json`
before they're uploaded to Azure for the reviewer app. **Dates are preserved**
(timestamps and date strings inside notes stay readable).

## Why a separate Python 3.11 venv

philter-ucsf v1.0.3 doesn't run on Python 3.12:

- Its CLI imports `distutils`, which was removed from the stdlib in 3.12.
- The bundled regex pattern files use mid-pattern global flags like `(?i)`,
  which 3.11+ `re.compile` rejects (must appear at the start of the regex).

The icu_pause_agents project venv is 3.12. So redaction lives in a tiny side
venv (3.11) called via subprocess.

## One-time setup

```bash
# 1. Install Python 3.11 (e.g. `brew install python@3.11` or use system python3.11)

# 2. Create the side-venv
python3.11 -m venv .venv-redact

# 3. Install philter-ucsf and its (undeclared) transitive deps
.venv-redact/bin/pip install philter-ucsf nltk chardet xmltodict

# 4. Download the NLTK data philter needs
.venv-redact/bin/python -m nltk.downloader punkt averaged_perceptron_tagger

# 5. Point the main venv's prepare_cases at this python
echo "ICUPAUSE_REDACT_PYTHON=$(pwd)/.venv-redact/bin/python" >> review_app/.env
```

Verify the env var is set in whichever shell you run `prepare_cases.py` from:

```bash
echo $ICUPAUSE_REDACT_PYTHON
```

## How it's wired

```
prepare_cases.py                       # main 3.12 venv
  └── redaction.redact_case_payload()  # walks payload, batches strings
        └── redact_strings()
              └── subprocess: $ICUPAUSE_REDACT_PYTHON run_philter.py  # 3.11 venv
                    └── philter_ucsf.Philter().transform()
```

`run_philter.py` loads philter's default config (`philter_delta.json`),
drops every entry whose `phi_type` is `DATE`, and runs philter against the
input directory. Output is asterisk-format (length-preserved), so citation
byte-offsets remain valid.

## Disabling redaction (dev/debug)

```bash
python review_app/scripts/prepare_cases.py --outputs-dir … --hosp-ids … --no-redact
```

When `--no-redact` is passed, no sub-venv is required.

## What gets redacted

- `source_bundle.json`: every `clinical_notes[<type>][*].note_text`.
- `output.json`:
  - `sections.<id>` (agent-written narrative).
  - `todo_checklist[*]` strings (or string fields of dict items).
  - `warnings[*]`, `qa_issues[*]` strings.
  - `metadata.source_data.clinical_notes[<type>][*].note_text`.
  - `metadata.agent_source_data.<role>.clinical_notes[<type>][*].note_text`.

What is **not** redacted: structured fields (`note_id`, `creation_dttm`,
`revision_dttm`, ICD codes, lab values, vitals, demographics dict, citation
indices). Anything that is not a free-text English string stays as-is.

## Local `output.json` is the unredacted judge source

`prepare_cases.py` redacts the COPY uploaded to Azure. The local pipeline
output at `<outputs-dir>/{hosp_id}.json` is **not** modified. Run LLM-as-judge
against the local copy.

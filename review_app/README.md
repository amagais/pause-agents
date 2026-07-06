# ICU PAUSE Review App

Streamlit app for clinicians to review and score AI-generated ICU-PAUSE handoff notes.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example env file:

```bash
cp .env.example .env
```

Open `.env` and fill in the values:

```
AZURE_BLOB_CONNECTION_STRING=...   # Get this from the Azure Storage Account → Access keys
BLOB_CONTAINER_NAME=icupause-review
REVIEW_APP_PASSWORD=...            # Shared password you give to all reviewers
ADMIN_PASSWORD=...                 # Separate password for the Admin panel (study coordinator only)
REVIEW_APP_AUTH_MODE=password_only
```

> **REVIEW_APP_AUTH_MODE**
> - `password_only` — reviewers just enter the study password (simplest, use for pilots)
> - `entra_plus_password` — requires NU NetID login first, then the study password (for full deployment)

### 3. Run the app

```bash
cd /path/to/icupause-review
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Logging in

1. Open the app in your browser
2. Enter the study password (set in `REVIEW_APP_PASSWORD`)
3. Select your name from the dropdown
4. Start reviewing

## Admin panel

Click **Admin** in the sidebar and enter the admin password (set in `ADMIN_PASSWORD`).

From here you can:
- Bootstrap and manage the pilot manifest (batch workflow)
- Add batches as the pipeline advances (Batch 1..5)
- Open exactly one batch at a time to reviewers (gate the dashboard + review page)
- Track completion progress (per-batch summary + per-case detail)
- Export review data as CSV (now includes `batch` and `pipeline_version` columns)

## Pilot batch workflow

The pilot study runs as 5 sequential batches of 5 notes each (25 unique patients
total). Each batch uses a different pipeline version (`v1`..`v5`); after
annotation, the pipeline is updated and the next batch is generated. While a
batch is open, all 6 clinicians annotate every note in it. When the next batch
is opened, the previous one closes — its cases disappear from the reviewer
dashboard and the review page refuses to render them.

Run order in the Admin panel → **Assignment Setup**:

1. **Tab 1a — Bootstrap & Roster.** Create an empty manifest with the reviewer
   roster. Re-run to update roster/seed without touching batches. A guarded
   "Reset & bootstrap" button wipes batches/assignments if you need to start
   fresh.
2. **Tab 1b — Add batch.** When Batch N's notes are ready in blob storage,
   submit `batch_number=N`, `pipeline_version="vN"`, optional date window, and
   the 5 hosp_ids. Adding a batch does NOT open it.
3. **Tab 1c — Active batch.** Advance the active batch when reviewers should
   start on Batch N. Confirmation-gated. `0` closes all batches.

`Tab 1d — Final-phase (post-pilot)` retains the legacy all-at-once
pilot/IRR/round-robin generator for use after the pilot is complete.

### Reviewer experience

- Dashboard shows only the active batch's cases, with a header reading
  `Batch N of 5 — pipeline vN — <date window>`.
- Review-page header shows a `Batch N · vN` chip on every case.
- Once a batch closes, reviewers can no longer access its cases — even via
  stale browser state. Saved drafts/submissions remain on blob for analysis.

### Backward compatibility

Existing manifests/responses without batch fields load with `batch=0` and
`pipeline_version=""`. A pre-pilot manifest will render the empty-state on the
reviewer dashboard until you bootstrap and open a batch. Legacy responses
appear in CSVs under `batch=0`.

## Changing passwords

Open `.env` in a text editor and change `REVIEW_APP_PASSWORD` or `ADMIN_PASSWORD`, then restart the app.

From terminal:
```bash
nano .env
```
Ctrl+O to save, Ctrl+X to exit. Then restart Streamlit.

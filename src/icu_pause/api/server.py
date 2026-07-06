"""FastAPI backend for the ICU-PAUSE Transfer Note Generator."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from icu_pause.config import Settings
from icu_pause.graph.workflow import build_graph
from icu_pause.rendering.doc_export import export_docx, export_pdf, render_emr_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
_dotenv = dotenv_values(_ENV_FILE) if _ENV_FILE.exists() else {}


def _get_settings() -> Settings:
    """Build Settings from .env file."""
    clif_dir = os.environ.get(
        "ICUPAUSE_CLIF_DATA_DIR", _dotenv.get("ICUPAUSE_CLIF_DATA_DIR", "")
    )
    if not clif_dir:
        raise HTTPException(status_code=400, detail="ICUPAUSE_CLIF_DATA_DIR is not configured")
    os.environ["ICUPAUSE_CLIF_DATA_DIR"] = clif_dir
    return Settings(
        _env_file=_ENV_FILE if _ENV_FILE.exists() else None,
        clif_data_dir=clif_dir,
    )


# ---------------------------------------------------------------------------
# Clinician-facing stage labels
# ---------------------------------------------------------------------------

# Maps internal LangGraph node names to (clinician_label, stage_number)
# Stage numbers group parallel nodes so the UI shows coarse progress.
STAGE_LABELS: dict[str, tuple[str, int]] = {
    "data_retrieval": ("Retrieving patient data", 1),
    # Interpreter agents (CR-DSF mode — stage 2)
    "structured_interpreter": ("Interpreting structured data", 2),
    "note_interpreter": ("Interpreting clinical notes", 2),
    "structured_extractor": ("Extracting discrete clinical facts", 2),
    # Domain agents (stage 2 in early fusion, stage 2 in CR-DSF after interpreters)
    "nurse": ("Reviewing nursing assessments", 2),
    "respiratory": ("Reviewing respiratory status", 2),
    "pharmacy": ("Reviewing medications and lab values", 2),
    "dietitian": ("Reviewing nutrition status", 2),
    "case_manager": ("Reviewing discharge planning", 2),
    "therapist": ("Reviewing mobility and therapy", 2),
    "risk_predictor": ("Running risk prediction model", 2),
    "qa_check": ("Checking for inconsistencies", 3),
    "deliberation": ("Resolving discrepancies", 3),
    "resident": ("Resident pre-synthesis review", 4),
    "intensivist": ("Synthesizing clinical plan", 4),
    "merge_and_render": ("Compiling handoff brief", 5),
}

# Labels for stage groups (shown as primary status)
STAGE_GROUP_LABELS: dict[int, str] = {
    1: "Retrieving patient data",
    2: "Generating specialist summaries",
    3: "Checking for inconsistencies",
    4: "Synthesizing clinical plan",
    5: "Compiling handoff brief",
}

TOTAL_STAGES = 5


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="ICU-PAUSE API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AuthRequest(BaseModel):
    password: str


class GenerateRequest(BaseModel):
    hospitalization_id: str
    lookback_hours: int | None = 48  # None = entire stay
    reference_dttm: str | None = None  # ISO-8601; None = prospective / auto-detect
    notes_lookback_hours: int | None = None  # Override; falls back to settings default
    # LLM selection
    llm_provider: str | None = None  # "local" | "openai" | "anthropic"
    llm_model: str | None = None  # e.g. "qwen2.5:32b", "gpt-4o", etc.
    # Fusion mode
    fusion_mode: str | None = None  # "early_fusion" | "cr_dsf"
    # Data modality toggles
    structured_data_enabled: bool | None = None  # None = use settings default
    notes_enabled: bool | None = None
    # Pipeline feature toggles
    risk_predictor_enabled: bool | None = None
    deliberation_enabled: bool | None = None
    resident_enabled: bool | None = None
    qa_ensemble_passes: int | None = None
    agent_self_critique: bool | None = None


class ExportRequest(BaseModel):
    output: dict[str, Any]


class EvalRequest(BaseModel):
    output: dict[str, Any]
    # Per-evaluator model overrides (optional)
    pdsqi9_llm_provider: str | None = None
    pdsqi9_llm_model: str | None = None
    hqi_llm_provider: str | None = None
    hqi_llm_model: str | None = None
    grounding_llm_provider: str | None = None
    grounding_llm_model: str | None = None


class StandaloneEvalRequest(BaseModel):
    note_text: str
    hospitalization_id: str | None = None  # Optional: load source data for grounding
    note_type: str = "ai_generated"  # "ai_generated" or "human_written"


def _build_initial_state(req: GenerateRequest) -> dict[str, Any]:
    """Build the initial LangGraph state from a generate request."""
    return {
        "hospitalization_id": req.hospitalization_id,
        "lookback_hours": 48,  # Fixed at 48h — context window constraint
        "reference_dttm": req.reference_dttm,
        "notes_lookback_hours": req.notes_lookback_hours,
        "patient_context_text": {},
        "agent_context_text": {},
        "cite_registry": {},
        "agent_snippets": [],
        "pipeline_metrics": [],
        "risk_score": None,
        "qa_issues": [],
        "qa_scope_issues": [],
        "qa_passed": False,
        "revised_snippets": [],
        "deliberation_log": [],
        "fusion_mode": "early_fusion",
        "structured_summaries": {},
        "note_summaries": {},
        "extraction_fields": {},
        "resident_pre_brief": None,
        "scribe_extraction": None,
        "intensivist_output": None,
        "icu_pause_output": {},
        "trace_events": [],
    }


def _apply_toggles(settings: Settings, req: GenerateRequest) -> None:
    """Apply per-request toggle overrides to settings."""
    if req.llm_provider is not None:
        settings.llm_provider = req.llm_provider
    if req.llm_model is not None:
        settings.llm_model = req.llm_model
    if req.fusion_mode is not None:
        settings.fusion_mode = req.fusion_mode
    if req.structured_data_enabled is not None:
        settings.structured_data_enabled = req.structured_data_enabled
    if req.notes_enabled is not None:
        settings.notes_enabled = req.notes_enabled
    if req.risk_predictor_enabled is not None:
        settings.risk_predictor_enabled = req.risk_predictor_enabled
    if req.deliberation_enabled is not None:
        settings.deliberation_enabled = req.deliberation_enabled
    if req.resident_enabled is not None:
        settings.resident_enabled = req.resident_enabled
    if req.qa_ensemble_passes is not None:
        settings.qa_ensemble_passes = req.qa_ensemble_passes
    if req.agent_self_critique is not None:
        settings.agent_self_critique = req.agent_self_critique


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth")
def authenticate(req: AuthRequest):
    expected = os.environ.get("ICUPAUSE_UI_PASSWORD", _dotenv.get("ICUPAUSE_UI_PASSWORD", ""))
    if not expected:
        return {"authenticated": True}
    if req.password == expected:
        return {"authenticated": True}
    raise HTTPException(status_code=401, detail="Incorrect password")


@app.post("/api/generate")
def generate(req: GenerateRequest):
    """Non-streaming generate endpoint (kept for backward compatibility)."""
    settings = _get_settings()
    _apply_toggles(settings, req)

    window_desc = f"last {req.lookback_hours}h" if req.lookback_hours is not None else "entire stay"
    logger.info(
        f"Generating brief for {req.hospitalization_id} ({window_desc}) "
        f"via {settings.llm_provider}/{settings.llm_model} "
        f"[structured={settings.structured_data_enabled}, notes={settings.notes_enabled}, "
        f"risk={settings.risk_predictor_enabled}, deliberation={settings.deliberation_enabled}, "
        f"resident={settings.resident_enabled}]"
    )

    graph = build_graph(settings)
    result = graph.invoke(_build_initial_state(req))
    return result.get("icu_pause_output", {})


@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Streaming generate endpoint — sends SSE stage updates to the frontend."""
    settings = _get_settings()
    _apply_toggles(settings, req)

    window_desc = f"last {req.lookback_hours}h" if req.lookback_hours is not None else "entire stay"
    logger.info(
        f"Streaming brief for {req.hospitalization_id} ({window_desc}) "
        f"via {settings.llm_provider}/{settings.llm_model}"
    )

    graph = build_graph(settings)
    initial_state = _build_initial_state(req)

    async def event_generator():
        import queue
        import threading

        emitted_stages: set[int] = set()
        q: queue.Queue = queue.Queue()

        # Collect intermediate states for debug persistence
        collected_snippets: list = []
        collected_intensivist = None
        collected_resident = None
        collected_qa_issues: list[str] = []
        collected_qa_scope_issues: list[str] = []
        collected_agent_contexts: dict = {}
        collected_patient_context: dict = {}

        def _stream_worker():
            try:
                for chunk in graph.stream(initial_state, stream_mode="updates"):
                    q.put(("chunk", chunk))
                q.put(("done", None))
            except Exception as e:
                q.put(("error", str(e)))

        # Emit stage 1 immediately so the UI shows activity right away
        yield f"data: {json.dumps({'type': 'stage_start', 'stage': 1, 'total_stages': TOTAL_STAGES, 'label': STAGE_GROUP_LABELS[1]})}\n\n"
        emitted_stages.add(1)

        thread = threading.Thread(target=_stream_worker, daemon=True)
        thread.start()

        while True:
            try:
                msg_type, payload = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: q.get(timeout=900)  # 15 min — pipeline with self-critique can take 7-10 min
                )
            except Exception:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Pipeline timed out'})}\n\n"
                break

            if msg_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'message': payload})}\n\n"
                break

            if msg_type == "done":
                yield f"data: {json.dumps({'type': 'complete', 'label': 'Brief ready'})}\n\n"
                break

            if msg_type == "chunk":
                for node_name, node_output in payload.items():
                    if node_name not in STAGE_LABELS:
                        continue

                    label, stage_num = STAGE_LABELS[node_name]

                    # Emit stage group start BEFORE the first node_complete in that group
                    if stage_num not in emitted_stages:
                        emitted_stages.add(stage_num)
                        group_label = STAGE_GROUP_LABELS.get(stage_num, label)
                        yield f"data: {json.dumps({'type': 'stage_start', 'stage': stage_num, 'total_stages': TOTAL_STAGES, 'label': group_label})}\n\n"

                    # Emit node completion (with metrics for dev mode)
                    node_event: dict[str, Any] = {
                        "type": "node_complete",
                        "node": node_name,
                        "label": label,
                        "stage": stage_num,
                    }
                    # Include pipeline_metrics if present (for dev mode)
                    metrics = node_output.get("pipeline_metrics", [])
                    if metrics:
                        node_event["metrics"] = metrics

                    yield f"data: {json.dumps(node_event, default=str)}\n\n"

                    # Collect data contexts (what each agent was sent)
                    if "agent_context_text" in node_output:
                        collected_agent_contexts = node_output["agent_context_text"]
                    if "patient_context_text" in node_output:
                        collected_patient_context = node_output["patient_context_text"]

                    # Collect intermediate states for debug JSON
                    for snippet in node_output.get("agent_snippets", []):
                        collected_snippets.append(
                            snippet.model_dump() if hasattr(snippet, "model_dump") else snippet
                        )
                    if "intensivist_output" in node_output and node_output["intensivist_output"]:
                        io = node_output["intensivist_output"]
                        collected_intensivist = io.model_dump() if hasattr(io, "model_dump") else io
                    if "resident_pre_brief" in node_output:
                        collected_resident = node_output["resident_pre_brief"]
                    collected_qa_issues.extend(node_output.get("qa_issues", []))
                    collected_qa_scope_issues.extend(node_output.get("qa_scope_issues", []))

                    # Stream trace events (for dev mode trace viewer)
                    trace_events = node_output.get("trace_events", [])
                    for tevt in trace_events:
                        yield f"data: {json.dumps({'type': 'trace', 'event': tevt}, default=str)}\n\n"

                    # Surface QA issues
                    if node_name == "qa_check":
                        qa_passed = node_output.get("qa_passed", True)
                        qa_issues = node_output.get("qa_issues", [])
                        if not qa_passed and qa_issues:
                            yield f"data: {json.dumps({'type': 'info', 'label': 'Inconsistency detected — resolving automatically', 'details': qa_issues})}\n\n"

                    # Send final output when merge completes
                    if node_name == "merge_and_render":
                        output = node_output.get("icu_pause_output", {})
                        # Save trace file — include both trace events and pipeline metrics
                        all_trace = []
                        all_metrics = []
                        for _n, _o in payload.items():
                            all_trace.extend(_o.get("trace_events", []))
                            for m in _o.get("pipeline_metrics", []):
                                all_metrics.append(m)
                        trace_path = None
                        if all_trace or all_metrics:
                            from icu_pause.tracing import RunTrace
                            rt = RunTrace(req.hospitalization_id)
                            rt.events = all_trace
                            # Add metrics as a trace event so they're in the export
                            if all_metrics:
                                import datetime as _dt
                                rt.events.append({
                                    "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                                    "type": "pipeline_metrics",
                                    "node": "summary",
                                    "level": "info",
                                    "message": f"{len(all_metrics)} agents completed",
                                    "data": {"metrics": all_metrics},
                                })
                            trace_path = rt.save()

                        # Save full debug JSON with intermediate states
                        run_json_path = None
                        try:
                            import datetime as _dt2
                            run_dir = Path(os.environ.get("ICUPAUSE_RUN_DIR", "output/runs"))
                            run_dir.mkdir(parents=True, exist_ok=True)
                            ts2 = _dt2.datetime.now().strftime("%Y%m%d_%H%M%S")
                            run_json_path = str(run_dir / f"{ts2}_{req.hospitalization_id}.json")
                            # Build agent_inputs summary (what each agent saw)
                            agent_inputs = {}
                            for _agent_name, _agent_ctx in collected_agent_contexts.items():
                                agent_inputs[_agent_name] = {
                                    "data_keys": list(_agent_ctx.keys()),
                                    "notes": {
                                        nt: len(nl) if isinstance(nl, list) else 0
                                        for nt, nl in (_agent_ctx.get("notes") or {}).items()
                                    },
                                }

                            run_data = {
                                "hospitalization_id": req.hospitalization_id,
                                "pipeline_output": output,
                                "agent_snippets": collected_snippets,
                                "intensivist_output": collected_intensivist,
                                "resident_pre_brief": collected_resident,
                                "qa_issues": collected_qa_issues,
                                "qa_scope_issues": collected_qa_scope_issues,
                                "pipeline_metrics": all_metrics,
                                "agent_inputs": agent_inputs,
                                "trace_path": trace_path,
                            }
                            with open(run_json_path, "w") as _rf:
                                json.dump(run_data, _rf, indent=2, default=str)
                            logger.info(f"Run debug JSON saved to {run_json_path}")
                        except Exception as _e:
                            logger.warning(f"Failed to save run debug JSON: {_e}")

                        yield f"data: {json.dumps({'type': 'result', 'output': output, 'trace_path': trace_path, 'run_json_path': run_json_path}, default=str)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/export/pdf")
def export_pdf_endpoint(req: ExportRequest):
    buf = export_pdf(req.output)
    hosp_id = req.output.get("hospitalization_id", "note")
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="ICU_PAUSE_{hosp_id}.pdf"'},
    )


@app.post("/api/export/docx")
def export_docx_endpoint(req: ExportRequest):
    buf = export_docx(req.output)
    hosp_id = req.output.get("hospitalization_id", "note")
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="ICU_PAUSE_{hosp_id}.docx"'},
    )


@app.post("/api/export/emr")
def export_emr_endpoint(req: ExportRequest):
    """Return plain text optimized for EMR copy-paste."""
    from fastapi.responses import PlainTextResponse
    text = render_emr_text(req.output)
    return PlainTextResponse(text)


def _apply_eval_overrides(settings: Settings, req) -> None:
    """Apply per-evaluator model overrides from a request."""
    for prefix in ("pdsqi9", "hqi", "grounding"):
        for suffix in ("llm_provider", "llm_model"):
            key = f"{prefix}_{suffix}"
            val = getattr(req, key, None)
            if val:
                setattr(settings, key, val)


@app.post("/api/evaluate/pdsqi9")
def evaluate_pdsqi9(req: EvalRequest):
    """Run PDSQI-9 LLM-as-judge evaluation on a generated ICU-PAUSE note."""
    from icu_pause.eval.pdsqi9 import PDSQI9Evaluator
    from icu_pause.rendering.formatter import render_icu_pause_text

    settings = _get_settings()
    _apply_eval_overrides(settings, req)
    evaluator = PDSQI9Evaluator(settings)

    summary_text = render_icu_pause_text(req.output)

    # Load source data directly from CLIF files instead of round-tripping
    # through the frontend (which can lose the large metadata.source_data field)
    hosp_id = req.output.get("hospitalization_id")
    source_text = ""
    if hosp_id:
        loaded = _load_source_data_for_eval(hosp_id, settings)
        if loaded:
            source_text = loaded

    if not source_text:
        # Fallback: try metadata, then sections
        metadata = req.output.get("metadata", {})
        source_data = metadata.get("source_data", {})
        if source_data:
            source_text = json.dumps(source_data, indent=2, default=str)
        else:
            logger.warning("No source_data available — PDSQI-9 accuracy check will be limited")
            source_text = json.dumps(req.output.get("sections", {}), indent=2, default=str)

    sections = dict(req.output.get("sections", {}))

    # Enrich S section with rendered todo_checklist so PDSQI-9 scores the
    # complete S section (narrative + to-dos) as defined by ICU-PAUSE.
    todo_checklist = req.output.get("todo_checklist", [])
    if todo_checklist and "S" in sections:
        from icu_pause.rendering.formatter import render_todo_checklist
        todo_lines = render_todo_checklist(todo_checklist)
        if todo_lines:
            sections["S"] = sections["S"].rstrip() + "\n\nTo-do list:\n" + "\n".join(todo_lines)

    try:
        result = evaluator.evaluate_full(source_text, summary_text, sections)
        data = result.model_dump()
        # Sanitize: ensure all string values are clean for JSON transport
        sanitized = json.dumps(data, ensure_ascii=True, default=str)
        logger.info(f"PDSQI-9 result: total={data.get('overall', {}).get('total_score', 'N/A')}, "
                     f"response_bytes={len(sanitized)}")
        from starlette.responses import Response
        return Response(content=sanitized, media_type="application/json")
    except Exception as e:
        logger.error(f"PDSQI-9 evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/api/evaluate/hqi")
def evaluate_hqi(req: EvalRequest):
    """Run ICU-PAUSE-HQI evaluation on a generated transition brief."""
    from icu_pause.eval.hqi import HQIEvaluator
    from icu_pause.rendering.formatter import render_icu_pause_text

    settings = _get_settings()
    _apply_eval_overrides(settings, req)
    evaluator = HQIEvaluator(settings)

    summary_text = render_icu_pause_text(req.output)

    hosp_id = req.output.get("hospitalization_id")
    source_text = ""
    if hosp_id:
        loaded = _load_source_data_for_eval(hosp_id, settings)
        if loaded:
            source_text = loaded

    if not source_text:
        metadata = req.output.get("metadata", {})
        source_data = metadata.get("source_data", {})
        if source_data:
            source_text = json.dumps(source_data, indent=2, default=str)
        else:
            logger.warning("No source_data available — HQI accuracy check will be limited")
            source_text = json.dumps(req.output.get("sections", {}), indent=2, default=str)

    sections = req.output.get("sections", {})
    todo_checklist = req.output.get("todo_checklist", [])

    try:
        result = evaluator.evaluate(
            source_text, summary_text,
            sections=sections, todo_checklist=todo_checklist,
        )
        sanitized = json.dumps(result.model_dump(), ensure_ascii=True, default=str)
        from starlette.responses import Response
        return Response(content=sanitized, media_type="application/json")
    except Exception as e:
        logger.error(f"ICU-PAUSE-HQI evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


def _load_source_data_for_eval(hospitalization_id: str, settings) -> str | None:
    """Load patient source data for standalone evaluation grounding."""
    try:
        from icu_pause.data.context import PatientContext, serialize_to_json
        from icu_pause.data.retriever import DataRetriever

        retriever = DataRetriever(settings)
        ctx = retriever.retrieve(hospitalization_id, lookback_hours=48)
        context_text = serialize_to_json(ctx, lookback_hours=48)
        return json.dumps(context_text, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Could not load source data for {hospitalization_id}: {e}")
        return None


@app.post("/api/evaluate/standalone/pdsqi9")
def evaluate_standalone_pdsqi9(req: StandaloneEvalRequest):
    """Run PDSQI-9 evaluation on a standalone note (pasted text)."""
    from icu_pause.eval.pdsqi9 import PDSQI9Evaluator

    settings = _get_settings()
    evaluator = PDSQI9Evaluator(settings)

    # Load source data if hospitalization ID provided
    source_text = ""
    if req.hospitalization_id:
        source_text = _load_source_data_for_eval(req.hospitalization_id, settings) or ""

    if not source_text:
        source_text = "(No source data available — evaluate based on internal consistency only)"

    try:
        result = evaluator.evaluate(source_text, req.note_text)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Standalone PDSQI-9 evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/api/evaluate/standalone/hqi")
def evaluate_standalone_hqi(req: StandaloneEvalRequest):
    """Run ICU-PAUSE-HQI evaluation on a standalone note (pasted text)."""
    from icu_pause.eval.hqi import HQIEvaluator

    settings = _get_settings()
    evaluator = HQIEvaluator(settings)

    # Load source data if hospitalization ID provided
    source_text = ""
    if req.hospitalization_id:
        source_text = _load_source_data_for_eval(req.hospitalization_id, settings) or ""

    if not source_text:
        source_text = "(No source data available — evaluate based on internal consistency only)"

    try:
        result = evaluator.evaluate(source_text, req.note_text)
        return result.model_dump()
    except Exception as e:
        logger.error(f"Standalone HQI evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Debug / Run history endpoints
# ---------------------------------------------------------------------------

_RUN_DIR = Path(os.environ.get("ICUPAUSE_RUN_DIR", "output/runs"))


@app.get("/api/runs/{hosp_id}")
def get_run(hosp_id: str):
    """Return the most recent saved run JSON for a hospitalization ID."""
    if not _RUN_DIR.exists():
        raise HTTPException(status_code=404, detail="No runs found")

    # Filenames: {timestamp}_{hosp_id}.json (new) or {hosp_id}_{timestamp}.json (legacy)
    matches = sorted(
        [p for p in _RUN_DIR.glob("*.json") if hosp_id in p.stem],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise HTTPException(status_code=404, detail=f"No run found for {hosp_id}")

    with open(matches[0]) as f:
        return json.load(f)


@app.get("/api/runs/{hosp_id}/compare")
def get_run_comparison(hosp_id: str):
    """Return a structured agent-vs-intensivist comparison for debugging."""
    run_data = get_run(hosp_id)

    snippets = run_data.get("agent_snippets", [])
    int_out = run_data.get("intensivist_output")
    final_sections = run_data.get("pipeline_output", {}).get("sections", {})
    rewrite_rate = run_data.get("pipeline_output", {}).get("metadata", {}).get(
        "intensivist_rewrite_rate", {}
    )

    sections: dict[str, Any] = {}
    for sec_key in ["I", "C", "U_unprescribing", "P", "A", "U_uncertainty", "S", "E"]:
        agents = []
        for snippet in snippets:
            for sec in snippet.get("sections", []):
                if (sec.get("section") == sec_key
                        and sec.get("content")
                        and sec["content"] != "Not enough information from structured data."):
                    agents.append({
                        "agent": snippet.get("agent_name", ""),
                        "confidence": sec.get("confidence", 0),
                        "content": sec["content"],
                        "data_sources": sec.get("data_sources_used", []),
                    })

        intensivist_text = ""
        if int_out:
            for sec in int_out.get("sections", []):
                if sec.get("section") == sec_key and sec.get("content"):
                    intensivist_text = sec["content"]
                    break

        sections[sec_key] = {
            "rewrite_status": rewrite_rate.get(sec_key, "unknown"),
            "agents": agents,
            "intensivist": intensivist_text,
            "final": final_sections.get(sec_key, ""),
        }

    return {
        "hospitalization_id": hosp_id,
        "sections": sections,
        "qa_issues": run_data.get("qa_issues", []),
        "resident_pre_brief": run_data.get("resident_pre_brief"),
    }


@app.get("/api/runs")
def list_runs(limit: int = 20):
    """List recent runs with basic metadata."""
    if not _RUN_DIR.exists():
        return {"runs": []}

    files = sorted(
        _RUN_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    runs = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            meta = data.get("pipeline_output", {}).get("metadata", {})
            runs.append({
                "file": f.name,
                "hospitalization_id": data.get("hospitalization_id", ""),
                "timestamp": f.stem.split("_")[0] if "_" in f.stem else "",
                "word_count": meta.get("total_word_count"),
                "qa_issue_count": len(data.get("qa_issues", [])),
                "agent_count": len(data.get("agent_snippets", [])),
            })
        except Exception:
            continue

    return {"runs": runs}


# ---------------------------------------------------------------------------
# Autoresearch endpoints
# ---------------------------------------------------------------------------

_AUTORESEARCH_DIR = Path(__file__).resolve().parents[3] / "autoresearch"


class AutoresearchCaseConfig(BaseModel):
    hospitalization_id: str
    reference_dttm: str | None = None  # Per-case data cutoff time


class AutoresearchRunRequest(BaseModel):
    tag: str
    cases: list[AutoresearchCaseConfig] | None = None  # Per-case configs from frontend
    lookback_hours: int = 48           # Data window (hours)
    llm_provider: str | None = None    # "local" | "azure"
    llm_model: str | None = None       # Model ID
    fusion_mode: str | None = None     # "early_fusion" | "cr_dsf" | "cr_dsf_plus"


@app.get("/api/autoresearch/results")
def get_autoresearch_results():
    """Read autoresearch results.tsv and return as JSON."""
    import csv as csv_mod

    tsv_path = _AUTORESEARCH_DIR / "results.tsv"
    if not tsv_path.exists():
        return {"experiments": []}

    experiments = []
    with open(tsv_path) as f:
        reader = csv_mod.DictReader(f, delimiter="\t")
        for row in reader:
            # Convert numeric fields
            for key in ("composite", "pdsqi9", "hqi", "hallucination_rate", "data_utilization"):
                if key in row and row[key]:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = 0.0
            experiments.append(row)

    return {"experiments": experiments}


@app.post("/api/autoresearch/run")
def run_autoresearch_experiment(req: AutoresearchRunRequest):
    """Run a single autoresearch experiment."""
    import sys as _sys

    # Add project root to path for autoresearch imports
    project_root = str(Path(__file__).resolve().parents[3])
    if project_root not in _sys.path:
        _sys.path.insert(0, project_root)

    from autoresearch.tune import run_experiment

    # Build per-case configs from frontend, or fall back to dev_cases.txt
    case_configs = None
    cases_file = None
    if req.cases:
        case_configs = [
            {"hospitalization_id": c.hospitalization_id, "reference_dttm": c.reference_dttm}
            for c in req.cases
        ]
    else:
        cases_file = str(_AUTORESEARCH_DIR / "dev_cases.txt")
        if not Path(cases_file).exists():
            raise HTTPException(
                status_code=400,
                detail="No cases provided and autoresearch/dev_cases.txt not found.",
            )

    try:
        result = run_experiment(
            tag=req.tag,
            cases_file=cases_file,
            autoresearch_dir=_AUTORESEARCH_DIR,
            case_configs=case_configs,
            lookback_hours=req.lookback_hours,
            llm_provider=req.llm_provider,
            llm_model=req.llm_model,
            fusion_mode=req.fusion_mode,
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Autoresearch experiment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Experiment failed: {str(e)}")

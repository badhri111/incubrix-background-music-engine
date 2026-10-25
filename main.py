"""
IncuBrix AI Background Music Engine – FastAPI application entry point.

Endpoints
---------
GET  /health
GET  /ready
POST /api/v1/music/jobs               – submit a generation request (202)
GET  /api/v1/jobs/{job_id}            – poll job status / result
GET  /api/v1/jobs/{job_id}/manifest   – download JSON manifest
GET  /api/v1/jobs/{job_id}/downloads/{path:path}  – download any output file
POST /api/v1/jobs/{job_id}/stems/{stem_name}/regenerate – re-synthesise one stem
"""

import asyncio
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from logger import logger, correlation_id_ctx, job_id_ctx
from models import (
    ErrorDetails,
    ErrorResponse,
    JobResponse,
    JobStatus,
    MusicJobRequest,
    RegenerateStemRequest,
)
from music_engine import MusicEngine

# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis / DB for production)
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}
_idempotency: Dict[str, str] = {}  # idempotency_key → job_id
_lock = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_job(job_id: str, request: MusicJobRequest) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": JobStatus.QUEUED,
        "progress": 0,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "request": request,
        "downloads": None,
        "manifest": None,
        "error": None,
    }


def _update(job: Dict[str, Any], **kwargs) -> None:
    job.update(kwargs)
    job["updated_at"] = _now_iso()


def _job_response(job: Dict[str, Any]) -> JobResponse:
    return JobResponse(**{k: v for k, v in job.items()})


def _build_download_links(job_id: str, job_dir: Path) -> Dict[str, str]:
    base = f"/api/v1/jobs/{job_id}/downloads"
    links: Dict[str, str] = {}
    if (job_dir / "mix.wav").exists():
        links["mix_wav"] = f"{base}/mix.wav"
    if (job_dir / "mix.mp3").exists():
        links["mix_mp3"] = f"{base}/mix.mp3"
    if (job_dir / "mix_ducked.wav").exists():
        links["mix_ducked_wav"] = f"{base}/mix_ducked.wav"
    stems_dir = job_dir / "stems"
    if stems_dir.exists():
        for f in sorted(stems_dir.glob("*.wav")):
            key = f"stem_{f.stem}"
            links[key] = f"{base}/stems/{f.name}"
    links["manifest"] = f"/api/v1/jobs/{job_id}/manifest"
    return links


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, request: MusicJobRequest) -> None:
    """Execute the music generation in a background thread."""
    job_id_ctx.set(job_id)
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        _update(job, status=JobStatus.RUNNING, progress=5)

    try:
        logger.info(f"Starting job {job_id}: '{request.prompt[:60]}'")
        with _lock:
            _update(_jobs[job_id], progress=10)

        engine = MusicEngine()

        with _lock:
            _update(_jobs[job_id], progress=25)

        job_dir, manifest = engine.generate_music_job(request, job_id)

        with _lock:
            _update(_jobs[job_id], progress=90)

        links = _build_download_links(job_id, job_dir)

        with _lock:
            _update(
                _jobs[job_id],
                status=JobStatus.SUCCEEDED,
                progress=100,
                downloads=links,
                manifest=manifest.model_dump(),
            )
        logger.info(f"Job {job_id} succeeded.")

    except Exception as exc:
        logger.exception(f"Job {job_id} failed: {exc}")
        with _lock:
            _update(
                _jobs[job_id],
                status=JobStatus.FAILED,
                progress=0,
                error=ErrorDetails(
                    code="GENERATION_FAILED",
                    message=str(exc),
                    retryable=True,
                    details={"exception_type": type(exc).__name__},
                ).model_dump(),
            )


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(
        f"IncuBrix Music Engine starting. Provider={settings.MUSIC_PROVIDER} "
        f"OutputDir={settings.OUTPUT_DIR}"
    )
    yield
    logger.info("IncuBrix Music Engine shutting down.")


app = FastAPI(
    title="IncuBrix AI Background Music Engine",
    description=(
        "Generates instrumental music stems and mixes with a local, rule-based "
        "procedural DSP synthesizer. This is not a trained AI music model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_error_response(request: Request, exc: HTTPException) -> JSONResponse:
    """Keep application errors in the documented, consistent JSON shape."""
    if isinstance(exc.detail, dict) and {"code", "message", "retryable"} <= exc.detail.keys():
        error = exc.detail
    else:
        error = {
            "code": "NOT_FOUND" if exc.status_code == 404 else "REQUEST_FAILED",
            "message": str(exc.detail),
            "retryable": exc.status_code >= 500,
            "details": None,
        }
    return JSONResponse(status_code=exc.status_code, content=error, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_response(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = [
        {"location": list(error["loc"]), "message": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "code": "INVALID_REQUEST",
            "message": "The request body is invalid.",
            "retryable": False,
            "details": {"errors": errors},
        },
    )


# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
    correlation_id_ctx.set(cid)
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({ms} ms)")
    response.headers["X-Correlation-ID"] = cid
    return response


# ---------------------------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health():
    """Basic liveness check."""
    return {"status": "ok", "service": "incubrix-music-engine", "version": "1.0.0"}


@app.get("/ready", tags=["System"])
def ready():
    """Readiness check – verifies output directory is writable."""
    out = Path(settings.OUTPUT_DIR)
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".probe"
        probe.write_text("ok")
        probe.unlink()
        return {"status": "ready", "output_dir": str(out)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Not ready: {exc}")


# ---------------------------------------------------------------------------
# Music job endpoints
# ---------------------------------------------------------------------------

def _validate_request(req: MusicJobRequest) -> Optional[ErrorDetails]:
    if not req.prompt.strip():
        return ErrorDetails(
            code="INVALID_PROMPT",
            message="prompt must contain a non-whitespace character.",
            retryable=False,
        )
    if not math.isfinite(req.duration_s):
        return ErrorDetails(
            code="INVALID_DURATION",
            message="duration_s must be a finite number.",
            retryable=False,
        )
    if req.duration_s < settings.MIN_DURATION_S:
        return ErrorDetails(
            code="INVALID_DURATION",
            message=f"duration_s must be ≥ {settings.MIN_DURATION_S}s.",
            retryable=False,
            details={"min": settings.MIN_DURATION_S, "received": req.duration_s},
        )
    if req.duration_s > settings.MAX_DURATION_S:
        return ErrorDetails(
            code="INVALID_DURATION",
            message=f"duration_s must be ≤ {settings.MAX_DURATION_S}s.",
            retryable=False,
            details={"max": settings.MAX_DURATION_S, "received": req.duration_s},
        )
    if req.tempo_bpm is not None:
        if req.tempo_bpm < settings.MIN_TEMPO_BPM or req.tempo_bpm > settings.MAX_TEMPO_BPM:
            return ErrorDetails(
                code="INVALID_TEMPO",
                message=f"tempo_bpm must be between {settings.MIN_TEMPO_BPM} and {settings.MAX_TEMPO_BPM}.",
                retryable=False,
                details={"min": settings.MIN_TEMPO_BPM, "max": settings.MAX_TEMPO_BPM, "received": req.tempo_bpm},
            )
    if req.sections:
        for sec in req.sections:
            if not sec.mood.strip():
                return ErrorDetails(
                    code="INVALID_SECTION",
                    message="Each section mood must contain a non-whitespace character.",
                    retryable=False,
                )
            if sec.start_s < 0:
                return ErrorDetails(
                    code="INVALID_SECTION",
                    message=f"Section start_s must be ≥ 0 (got {sec.start_s}).",
                    retryable=False,
                )
            if sec.start_s >= req.duration_s:
                return ErrorDetails(
                    code="INVALID_SECTION",
                    message=f"Section start_s ({sec.start_s}s) must be < duration_s ({req.duration_s}s).",
                    retryable=False,
                )
    out = req.output
    if out:
        if not math.isfinite(out.target_lufs) or not -36.0 <= out.target_lufs <= 0.0:
            return ErrorDetails(
                code="INVALID_LOUDNESS",
                message="target_lufs must be a finite value between -36 and 0.",
                retryable=False,
                details={
                    "min": -36.0,
                    "max": 0.0,
                    "received": out.target_lufs if math.isfinite(out.target_lufs) else str(out.target_lufs),
                },
            )
        if out.sample_rate != 48000:
            return ErrorDetails(
                code="INVALID_SAMPLE_RATE",
                message="sample_rate must be 48000 Hz.",
                retryable=False,
            )
        invalid_formats = sorted(set(out.formats) - {"wav", "mp3"})
        if invalid_formats or "wav" not in out.formats:
            return ErrorDetails(
                code="INVALID_FORMATS",
                message="formats must include wav and may also include mp3.",
                retryable=False,
                details={"allowed": ["wav", "mp3"], "received": out.formats},
            )
    return None


@app.post(
    "/api/v1/music/jobs",
    status_code=202,
    response_model=JobResponse,
    tags=["Music"],
    summary="Submit a music-generation job",
)
def create_job(
    request: MusicJobRequest,
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
) -> JobResponse:
    # Idempotency – return existing job for duplicate key
    if idempotency_key:
        with _lock:
            existing_id = _idempotency.get(idempotency_key)
            if existing_id and existing_id in _jobs:
                return _job_response(_jobs[existing_id])

    err = _validate_request(request)
    if err:
        raise HTTPException(status_code=422, detail=err.model_dump())

    job_id = str(uuid.uuid4())
    job = _new_job(job_id, request)

    with _lock:
        if idempotency_key:
            existing_id = _idempotency.get(idempotency_key)
            if existing_id and existing_id in _jobs:
                return _job_response(_jobs[existing_id])
        _jobs[job_id] = job
        if idempotency_key:
            _idempotency[idempotency_key] = job_id

    response = _job_response(job)
    # Fire and forget in a thread
    threading.Thread(target=_run_job, args=(job_id, request), daemon=True).start()

    return response


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobResponse,
    tags=["Music"],
    summary="Poll job status",
)
def get_job(job_id: str) -> JobResponse:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return _job_response(job)


@app.get(
    "/api/v1/jobs/{job_id}/manifest",
    tags=["Music"],
    summary="Download the JSON manifest for a completed job",
)
def get_manifest(job_id: str):
    path = Path(settings.OUTPUT_DIR) / job_id / "music_manifest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Manifest not found.")
    return FileResponse(str(path), media_type="application/json", filename="music_manifest.json")


@app.get(
    "/api/v1/jobs/{job_id}/downloads/{file_path:path}",
    tags=["Music"],
    summary="Download an output file (mix or stem)",
)
def download_file(job_id: str, file_path: str):
    # Security: reject path traversal
    base = Path(settings.OUTPUT_DIR) / job_id
    target = (base / file_path).resolve()
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File '{file_path}' not found.")
    base_resolved = base.resolve()
    if not target.is_relative_to(base_resolved):
        raise HTTPException(status_code=400, detail="Invalid path.")
    suffix = target.suffix.lower()
    media_map = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".json": "application/json"}
    return FileResponse(str(target), media_type=media_map.get(suffix, "application/octet-stream"), filename=target.name)


@app.post(
    "/api/v1/jobs/{job_id}/stems/{stem_name}/regenerate",
    response_model=JobResponse,
    tags=["Music"],
    summary="Re-synthesise a single stem and re-mix",
)
def regenerate_stem(job_id: str, stem_name: str, body: RegenerateStemRequest):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != JobStatus.SUCCEEDED:
        raise HTTPException(status_code=409, detail="Job must have status 'succeeded' to regenerate a stem.")
    valid_stems = {"drums", "bass", "harmony", "melody"}
    if stem_name not in valid_stems:
        raise HTTPException(status_code=422, detail=f"stem_name must be one of {sorted(valid_stems)}.")

    try:
        engine = MusicEngine()
        engine.regenerate_stem(job_id, stem_name, new_seed=body.seed)
        links = _build_download_links(job_id, Path(settings.OUTPUT_DIR) / job_id)
        with _lock:
            _update(_jobs[job_id], downloads=links)
        return _job_response(_jobs[job_id])
    except Exception as exc:
        logger.exception(f"Stem regeneration failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Static frontend – served last so API routes take priority
# ---------------------------------------------------------------------------

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

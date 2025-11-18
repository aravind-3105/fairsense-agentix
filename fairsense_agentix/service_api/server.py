"""FastAPI surface for FairSense-AgentiX."""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any, cast

import anyio
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from fairsense_agentix import FairSense
from fairsense_agentix.service_api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchItem,
    BatchRequest,
    BatchStatus,
)
from fairsense_agentix.service_api.utils import (
    InputType,
    detect_input_type,
    normalize_input_hint,
)
from fairsense_agentix.services import telemetry
from fairsense_agentix.services.event_bus import AgentEventBus


app = FastAPI(title="FairSense AgentiX API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = FairSense()
event_bus = AgentEventBus(telemetry)
batch_jobs: dict[str, BatchStatus] = {}
batch_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup() -> None:
    loop = asyncio.get_running_loop()
    event_bus.attach_loop(loop)


def get_engine() -> FairSense:
    """Dependency injector for shared FairSense instance."""
    return engine


@app.get("/v1/health")
async def health() -> dict[str, str]:
    """Return readiness probe details."""
    return {"status": "ok"}


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    fs: Annotated[FairSense, Depends(get_engine)],
) -> AnalyzeResponse:
    """Run analysis for JSON payloads."""
    hinted = normalize_input_hint(request.input_type)
    input_type = hinted or detect_input_type(request.content)
    return await _run_analysis(fs, input_type, request.content, request.options)


@app.post("/v1/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    fs: Annotated[FairSense, Depends(get_engine)],
    file: Annotated[UploadFile, File(...)],
    input_type: InputType | None = None,
) -> AnalyzeResponse:
    """Run analysis for uploaded binary/text files."""
    data = await file.read()
    detected = input_type or detect_input_type(file.filename or data)
    payload: str | bytes = data if detected == "image" else data.decode("utf-8")
    return await _run_analysis(fs, detected, payload, {})


@app.post("/v1/batch", response_model=BatchStatus, status_code=202)
async def start_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(get_engine)],
) -> BatchStatus:
    """Create a new batch job and return initial metadata."""
    job_id = str(uuid.uuid4())
    status = BatchStatus(
        job_id=job_id,
        status="pending",
        total=len(request.items),
        completed=0,
        errors=[],
        results=[],
    )
    async with batch_lock:
        batch_jobs[job_id] = status
    background_tasks.add_task(_process_batch, job_id, request.items, fs)
    return status


@app.get("/v1/batch/{job_id}", response_model=BatchStatus)
async def get_batch(job_id: str) -> BatchStatus:
    """Retrieve current status (or final result) for batch job."""
    async with batch_lock:
        job = batch_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch job not found")
        return job


@app.websocket("/v1/stream/{run_id}")
async def stream_events(ws: WebSocket, run_id: str) -> None:
    """Push telemetry events to UI clients for visualization."""
    await ws.accept()
    try:
        async for payload in event_bus.stream(run_id):
            await ws.send_json(payload)
    except WebSocketDisconnect:
        return


async def _process_batch(
    job_id: str,
    items: list[BatchItem],
    fs: FairSense,
) -> None:
    """Background task that runs each batch entry sequentially."""
    async with batch_lock:
        status = batch_jobs[job_id]
    status.status = "running"
    for item in items:
        try:
            hinted = normalize_input_hint(item.input_type)
            input_type = hinted or detect_input_type(item.content)
            result = await _run_analysis(fs, input_type, item.content, item.options)
            status.results.append(result)
        except Exception as exc:  # pragma: no cover - best effort logging
            status.errors.append(str(exc))
        finally:
            status.completed += 1
    status.status = "failed" if status.errors else "completed"


async def _run_analysis(
    fs: FairSense,
    input_type: InputType,
    content: str | bytes,
    options: dict[str, Any],
) -> AnalyzeResponse:
    """Dispatch to the correct FairSense API based on input type."""
    if input_type == "image":
        image_bytes = (
            content
            if isinstance(content, (bytes, bytearray))
            else str(content).encode("utf-8")
        )
        bias_result = await anyio.to_thread.run_sync(
            lambda: fs.analyze_image(cast(bytes, image_bytes), **options)
        )
        return AnalyzeResponse.from_bias_result(bias_result)

    text_content = (
        content
        if isinstance(content, str)
        else content.decode("utf-8", errors="ignore")
    )
    if input_type == "csv":
        risk_result = await anyio.to_thread.run_sync(
            lambda: fs.assess_risk(text_content, **options)
        )
        return AnalyzeResponse.from_risk_result(risk_result)

    bias_result = await anyio.to_thread.run_sync(
        lambda: fs.analyze_text(text_content, **options)
    )
    return AnalyzeResponse.from_bias_result(bias_result)

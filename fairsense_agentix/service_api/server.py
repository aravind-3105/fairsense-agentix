"""FastAPI surface for FairSense-AgentiX."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator

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

# Import logging config for side effects (suppress verbose HTTP logs)
from fairsense_agentix import (
    FairSense,
    __version__,
    logging_config,  # noqa: F401 (imported for side effects)
)
from fairsense_agentix.service_api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeStartResponse,
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


logger = logging.getLogger(__name__)


# Module-level state (initialized in lifespan)
engine: FairSense
event_bus: AgentEventBus
batch_jobs: dict[str, BatchStatus] = {}
batch_lock = asyncio.Lock()
# Store analysis results for async retrieval
analysis_results: dict[str, AnalyzeResponse | None] = {}
analysis_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup warmup and shutdown cleanup.

    Startup Phase:
    - Preloads all models and tools (LLM, OCR, embeddings, FAISS indexes)
    - This takes 30-45s but makes subsequent requests instant
    - Attaches event loop for WebSocket streaming

    Shutdown Phase:
    - Cleanup resources if needed
    """
    global engine, event_bus  # noqa: PLW0603

    # STARTUP: Preload models and tools
    logger.info("🔥 FairSense AgentiX starting up...")
    logger.info("⏳ Preloading models and tools (this takes 30-45s)...")

    # Initialize FairSense (triggers eager tool loading via api.py __init__)
    engine = FairSense()
    logger.info("✅ FairSense engine initialized with all models loaded")

    # Initialize event bus for WebSocket streaming
    event_bus = AgentEventBus(telemetry)
    loop = asyncio.get_running_loop()
    event_bus.attach_loop(loop)
    logger.info("✅ Event bus attached to event loop")

    logger.info("🚀 Server ready! All requests will be fast now.")

    yield  # Server runs here

    # SHUTDOWN: Cleanup
    logger.info("🛑 Shutting down FairSense AgentiX...")


app = FastAPI(
    title="FairSense AgentiX API",
    version=__version__,
    lifespan=lifespan,  # Use modern lifespan pattern
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> FairSense:
    """Dependency injector for shared FairSense instance."""
    return engine


@app.get("/v1/health")
async def health() -> dict[str, str]:
    """Return readiness probe details."""
    return {"status": "ok"}


@app.post("/v1/shutdown")
async def shutdown(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Gracefully shutdown the server.

    This endpoint triggers a clean shutdown of the backend server.
    The launcher process will detect the shutdown and clean up ports.

    Returns immediately with a success message, then shuts down after 1 second.
    """

    def _shutdown() -> None:
        """Background task to exit cleanly after response is sent."""
        time.sleep(1)  # Give time for response to be sent
        logger.info("🛑 Shutdown requested via API - exiting gracefully...")
        os._exit(0)  # Clean exit - launcher will detect and cleanup ports

    background_tasks.add_task(_shutdown)
    return {"status": "shutting_down", "message": "Server will shutdown in 1 second"}


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
    # Convert filename string to Path for proper extension detection
    detected = input_type or detect_input_type(
        Path(file.filename) if file.filename else data,
    )

    # Handle image vs text with proper error handling
    if detected == "image":
        payload: str | bytes = data
    else:
        # Attempt UTF-8 decode for text/csv files
        try:
            payload = data.decode("utf-8")
        except UnicodeDecodeError as e:
            # File appears to be binary but was detected as text
            # This can happen with images that have wrong file extensions
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File '{file.filename}' could not be decoded as text. "
                    "If this is an image, please rename it with a proper extension "
                    "(.png, .jpg, .jpeg, .gif, .bmp, .webp) or explicitly set "
                    "input_type='image' in your request."
                ),
            ) from e

    return await _run_analysis(fs, detected, payload, {})


@app.post("/v1/analyze/start", response_model=AnalyzeStartResponse)
async def analyze_start(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(get_engine)],
) -> AnalyzeStartResponse:
    """Start analysis and return run_id immediately for WebSocket connection.

    This endpoint solves the WebSocket race condition by:
    1. Generating run_id before analysis starts
    2. Returning run_id immediately for WebSocket connection
    3. Running analysis in background task that publishes events
    """
    run_id = str(uuid.uuid4())

    # Mark analysis as pending
    async with analysis_lock:
        analysis_results[run_id] = None

    # Detect input type
    hinted = normalize_input_hint(request.input_type)
    input_type = hinted or detect_input_type(request.content)

    # Start analysis in background
    background_tasks.add_task(
        _run_analysis_background,
        run_id,
        fs,
        input_type,
        request.content,
        request.options,
    )

    return AnalyzeStartResponse(run_id=run_id)


@app.post("/v1/analyze/upload/start", response_model=AnalyzeStartResponse)
async def analyze_upload_start(
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(get_engine)],
    file: Annotated[UploadFile, File(...)],
    input_type: InputType | None = None,
) -> AnalyzeStartResponse:
    """Start file upload analysis and return run_id immediately.

    Returns run_id for WebSocket connection.
    """
    run_id = str(uuid.uuid4())

    # Mark analysis as pending
    async with analysis_lock:
        analysis_results[run_id] = None

    # Read and process file
    data = await file.read()
    detected = input_type or detect_input_type(
        Path(file.filename) if file.filename else data,
    )

    # Handle image vs text with proper error handling
    if detected == "image":
        payload: str | bytes = data
    else:
        try:
            payload = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"File '{file.filename}' could not be decoded as text. "
                    "If this is an image, please rename it with a proper extension "
                    "(.png, .jpg, .jpeg, .gif, .bmp, .webp) or explicitly set "
                    "input_type='image' in your request."
                ),
            ) from e

    # Start analysis in background
    background_tasks.add_task(
        _run_analysis_background,
        run_id,
        fs,
        detected,
        payload,
        {},
    )

    return AnalyzeStartResponse(run_id=run_id)


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
    run_id: str | None = None,
) -> AnalyzeResponse:
    """Dispatch to the correct FairSense API based on input type.

    Parameters
    ----------
    fs : FairSense
        FairSense instance
    input_type : InputType
        Type of input (text, image, csv)
    content : str | bytes
        Input content
    options : dict[str, Any]
        Analysis options
    run_id : str | None, optional
        Pre-generated run ID for tracing (fixes WebSocket event streaming)
    """
    if input_type == "image":
        image_bytes = (
            content
            if isinstance(content, (bytes, bytearray))
            else str(content).encode("utf-8")
        )
        bias_result = await anyio.to_thread.run_sync(
            lambda: fs.analyze_image(image_bytes, run_id=run_id, **options),
        )
        return AnalyzeResponse.from_bias_result(bias_result)

    text_content = (
        content
        if isinstance(content, str)
        else content.decode("utf-8", errors="ignore")
    )
    if input_type == "csv":
        risk_result = await anyio.to_thread.run_sync(
            lambda: fs.assess_risk(text_content, run_id=run_id, **options),
        )
        return AnalyzeResponse.from_risk_result(risk_result)

    bias_result = await anyio.to_thread.run_sync(
        lambda: fs.analyze_text(text_content, run_id=run_id, **options),
    )
    return AnalyzeResponse.from_bias_result(bias_result)


async def _run_analysis_background(
    run_id: str,
    fs: FairSense,
    input_type: InputType,
    content: str | bytes,
    options: dict[str, Any],
) -> None:
    """Background task that runs analysis and publishes result through WebSocket.

    This function:
    1. Runs the analysis with our pre-generated run_id
       (fixes WebSocket event streaming!)
    2. Stores the result in analysis_results dict
    3. Publishes a final 'complete' event through WebSocket with the full result
    """
    try:
        logger.debug("Background task started for run_id=%s", run_id)

        # Run the analysis WITH our run_id so all telemetry uses it
        result = await _run_analysis(fs, input_type, content, options, run_id=run_id)

        logger.debug(
            "Analysis completed, result run_id=%s, our run_id=%s",
            result.metadata["run_id"],
            run_id,
        )

        # Store result for potential polling
        async with analysis_lock:
            analysis_results[run_id] = result

        # Publish completion event through WebSocket
        completion_payload = {
            "run_id": run_id,
            "timestamp": time.time(),  # Use Unix timestamp, not event loop time
            "event": "analysis_complete",
            "level": "info",
            "context": {
                "message": "Analysis completed successfully",
                "result": result.model_dump(),
            },
        }

        # Manually inject completion event into event_bus queue
        if event_bus._loop:
            queue = event_bus._queues.get(run_id)
            if queue:
                event_bus._loop.call_soon_threadsafe(
                    event_bus._enqueue,
                    queue,
                    completion_payload,
                )

    except Exception as e:  # pragma: no cover - best effort error handling
        # Publish error event through WebSocket
        error_payload = {
            "run_id": run_id,
            "timestamp": time.time(),  # Use Unix timestamp, not event loop time
            "event": "analysis_error",
            "level": "error",
            "context": {
                "message": f"Analysis failed: {str(e)}",
                "error_type": type(e).__name__,
            },
        }

        if event_bus._loop:
            queue = event_bus._queues.get(run_id)
            if queue:
                event_bus._loop.call_soon_threadsafe(
                    event_bus._enqueue,
                    queue,
                    error_payload,
                )

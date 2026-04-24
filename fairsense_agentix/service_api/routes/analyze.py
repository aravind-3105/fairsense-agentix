"""Analysis routes: JSON and file upload, sync and async variants."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from fairsense_agentix import FairSense
from fairsense_agentix.service_api import app_state
from fairsense_agentix.service_api.helpers import run_analysis, run_analysis_background
from fairsense_agentix.service_api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnalyzeStartResponse,
)
from fairsense_agentix.service_api.utils import (
    InputType,
    detect_input_type,
    normalize_input_hint,
)


router = APIRouter()


@router.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    fs: Annotated[FairSense, Depends(app_state.get_engine)],
) -> AnalyzeResponse:
    """Run analysis for JSON payloads."""
    hinted = normalize_input_hint(request.input_type)
    input_type = hinted or detect_input_type(request.content)
    return await run_analysis(fs, input_type, request.content, request.options)


@router.post("/v1/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(
    fs: Annotated[FairSense, Depends(app_state.get_engine)],
    file: Annotated[UploadFile, File(...)],
    input_type: InputType | None = None,
) -> AnalyzeResponse:
    """Run analysis for uploaded binary/text files."""
    data = await file.read()
    detected = input_type or detect_input_type(
        Path(file.filename) if file.filename else data,
    )

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

    return await run_analysis(fs, detected, payload, {})


@router.post("/v1/analyze/start", response_model=AnalyzeStartResponse)
async def analyze_start(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(app_state.get_engine)],
) -> AnalyzeStartResponse:
    """Start analysis and return run_id immediately for WebSocket connection.

    Solves the WebSocket race condition by:
    1. Generating run_id before analysis starts
    2. Returning run_id immediately for WebSocket connection
    3. Running analysis in background task that publishes events
    """
    run_id = str(uuid.uuid4())

    async with app_state.analysis_lock:
        app_state.analysis_results[run_id] = None

    hinted = normalize_input_hint(request.input_type)
    input_type = hinted or detect_input_type(request.content)

    background_tasks.add_task(
        run_analysis_background,
        run_id,
        fs,
        input_type,
        request.content,
        request.options,
    )

    return AnalyzeStartResponse(run_id=run_id)


@router.post("/v1/analyze/upload/start", response_model=AnalyzeStartResponse)
async def analyze_upload_start(
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(app_state.get_engine)],
    file: Annotated[UploadFile, File(...)],
    input_type: InputType | None = None,
) -> AnalyzeStartResponse:
    """Start file upload analysis and return run_id immediately."""
    run_id = str(uuid.uuid4())

    async with app_state.analysis_lock:
        app_state.analysis_results[run_id] = None

    data = await file.read()
    detected = input_type or detect_input_type(
        Path(file.filename) if file.filename else data,
    )

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

    background_tasks.add_task(
        run_analysis_background,
        run_id,
        fs,
        detected,
        payload,
        {},
    )

    return AnalyzeStartResponse(run_id=run_id)

"""Batch processing routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from fairsense_agentix import FairSense
from fairsense_agentix.service_api import app_state
from fairsense_agentix.service_api.helpers import run_analysis
from fairsense_agentix.service_api.schemas import BatchItem, BatchRequest, BatchStatus
from fairsense_agentix.service_api.utils import detect_input_type, normalize_input_hint


router = APIRouter()


@router.post("/v1/batch", response_model=BatchStatus, status_code=202)
async def start_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    fs: Annotated[FairSense, Depends(app_state.get_engine)],
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
    async with app_state.batch_lock:
        app_state.batch_jobs[job_id] = status
    background_tasks.add_task(_process_batch, job_id, request.items, fs)
    return status


@router.get("/v1/batch/{job_id}", response_model=BatchStatus)
async def get_batch(job_id: str) -> BatchStatus:
    """Retrieve current status (or final result) for batch job."""
    async with app_state.batch_lock:
        job = app_state.batch_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Batch job not found")
        return job


async def _process_batch(
    job_id: str,
    items: list[BatchItem],
    fs: FairSense,
) -> None:
    """Background task that runs each batch entry sequentially."""
    async with app_state.batch_lock:
        status = app_state.batch_jobs[job_id]
    status.status = "running"
    for item in items:
        try:
            hinted = normalize_input_hint(item.input_type)
            input_type = hinted or detect_input_type(item.content)
            result = await run_analysis(fs, input_type, item.content, item.options)
            status.results.append(result)
        except Exception as exc:  # pragma: no cover - best effort logging
            status.errors.append(str(exc))
        finally:
            status.completed += 1
    status.status = "failed" if status.errors else "completed"

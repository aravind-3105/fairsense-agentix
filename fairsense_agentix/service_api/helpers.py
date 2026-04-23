"""Shared analysis helpers used by multiple route modules."""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio

from fairsense_agentix import FairSense
from fairsense_agentix.service_api import app_state
from fairsense_agentix.service_api.schemas import AnalyzeResponse
from fairsense_agentix.service_api.utils import InputType


logger = logging.getLogger(__name__)


async def run_analysis(
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


async def run_analysis_background(
    run_id: str,
    fs: FairSense,
    input_type: InputType,
    content: str | bytes,
    options: dict[str, Any],
) -> None:
    """Background task that runs analysis and publishes result through WebSocket.

    1. Runs the analysis with the pre-generated run_id (fixes WebSocket streaming)
    2. Stores the result in analysis_results dict
    3. Publishes a final 'complete' event through the WebSocket
    """
    try:
        logger.debug("Background task started for run_id=%s", run_id)

        result = await run_analysis(fs, input_type, content, options, run_id=run_id)

        logger.debug(
            "Analysis completed, result run_id=%s, our run_id=%s",
            result.metadata["run_id"],
            run_id,
        )

        async with app_state.analysis_lock:
            app_state.analysis_results[run_id] = result

        completion_payload = {
            "run_id": run_id,
            "timestamp": time.time(),
            "event": "analysis_complete",
            "level": "info",
            "context": {
                "message": "Analysis completed successfully",
                "result": result.model_dump(),
            },
        }

        if app_state.event_bus._loop:
            queue = app_state.event_bus._queues.get(run_id)
            if queue:
                app_state.event_bus._loop.call_soon_threadsafe(
                    app_state.event_bus._enqueue,
                    queue,
                    completion_payload,
                )

    except Exception as e:  # pragma: no cover - best effort error handling
        error_payload = {
            "run_id": run_id,
            "timestamp": time.time(),
            "event": "analysis_error",
            "level": "error",
            "context": {
                "message": f"Analysis failed: {str(e)}",
                "error_type": type(e).__name__,
            },
        }

        if app_state.event_bus._loop:
            queue = app_state.event_bus._queues.get(run_id)
            if queue:
                app_state.event_bus._loop.call_soon_threadsafe(
                    app_state.event_bus._enqueue,
                    queue,
                    error_payload,
                )

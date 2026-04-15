"""Health and lifecycle routes."""

from __future__ import annotations

import logging
import os
import time

from fastapi import APIRouter, BackgroundTasks


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/v1/health")
async def health() -> dict[str, str]:
    """Return readiness probe details."""
    return {"status": "ok"}


@router.post("/v1/shutdown")
async def shutdown(background_tasks: BackgroundTasks) -> dict[str, str]:
    """Gracefully shutdown the server.

    Returns immediately with a success message, then shuts down after 1 second.
    """

    def _shutdown() -> None:
        """Background task to exit cleanly after response is sent."""
        time.sleep(1)  # Give time for response to be sent
        logger.info("🛑 Shutdown requested via API - exiting gracefully...")
        os._exit(0)  # Clean exit - launcher will detect and cleanup ports

    background_tasks.add_task(_shutdown)
    return {"status": "shutting_down", "message": "Server will shutdown in 1 second"}

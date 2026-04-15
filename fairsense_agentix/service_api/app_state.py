"""Shared module-level state for the FastAPI application.

Holds the singleton FairSense engine, event bus, and in-memory stores for
batch jobs and async analysis results. Initialised during lifespan startup.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from fairsense_agentix.service_api.schemas import AnalyzeResponse, BatchStatus


if TYPE_CHECKING:
    from fairsense_agentix import FairSense
    from fairsense_agentix.services.event_bus import AgentEventBus


# Initialised in lifespan; declared here so all route modules share one reference.
# Type-annotated only — assigned at runtime, so ruff F821 is suppressed.
engine: "FairSense"  # noqa: F821
event_bus: "AgentEventBus"  # noqa: F821

batch_jobs: dict[str, BatchStatus] = {}
batch_lock = asyncio.Lock()

analysis_results: dict[str, AnalyzeResponse | None] = {}
analysis_lock = asyncio.Lock()


def get_engine() -> "FairSense":
    """FastAPI dependency that returns the shared FairSense engine."""
    return engine  # noqa: F821

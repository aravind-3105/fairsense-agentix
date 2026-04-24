"""WebSocket-friendly event bus for streaming telemetry events."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, AsyncIterator

from fairsense_agentix.services.telemetry import TelemetryService


class AgentEventBus:
    """Publishes telemetry events to per-run queues for streaming to clients."""

    def __init__(self, telemetry: TelemetryService) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(
            asyncio.Queue,
        )
        telemetry.register_observer(self._handle_event)

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the running event loop (FastAPI startup)."""
        self._loop = loop

    def _handle_event(self, payload: dict[str, Any]) -> None:
        """Receive telemetry payloads and push onto per-run queues."""
        run_id = payload.get("run_id")
        if not run_id or run_id == "unknown" or self._loop is None:
            return

        queue = self._queues.setdefault(run_id, asyncio.Queue(maxsize=200))
        self._loop.call_soon_threadsafe(self._enqueue, queue, payload)

    @staticmethod
    def _enqueue(queue: asyncio.Queue[dict[str, Any]], payload: dict[str, Any]) -> None:
        """Push payload into queue, dropping oldest if full."""
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(payload)

    async def stream(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield telemetry events for a run as they arrive."""
        queue = self._queues.setdefault(run_id, asyncio.Queue())
        try:
            while True:
                payload = await queue.get()
                yield payload
        finally:
            if queue.qsize() == 0:
                self._queues.pop(run_id, None)

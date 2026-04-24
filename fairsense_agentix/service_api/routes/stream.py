"""WebSocket streaming route."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from fairsense_agentix.service_api import app_state


router = APIRouter()


@router.websocket("/v1/stream/{run_id}")
async def stream_events(ws: WebSocket, run_id: str) -> None:
    """Push telemetry events to UI clients for visualization."""
    await ws.accept()
    try:
        async for payload in app_state.event_bus.stream(run_id):
            await ws.send_json(payload)
    except WebSocketDisconnect:
        return

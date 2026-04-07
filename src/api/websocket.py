"""WebSocket streaming endpoints for trajectory decoding."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.decoder import decode
from src.api.schemas import SpikeBuffer

router = APIRouter()


@router.websocket("/ws/decode")
async def websocket_decode(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            buffer = SpikeBuffer.model_validate(payload)
            model = websocket.app.state.model
            pca = websocket.app.state.pca
            points = decode(buffer, model, pca)
            for point in points:
                await websocket.send_json(point.model_dump())
    except WebSocketDisconnect:
        return

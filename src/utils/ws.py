from fastapi import WebSocket

from src.schemas.ws_messages import (
    ServerMessage,
    ServerMessageType,
)


def _server_msg(msg_type: ServerMessageType, **kwargs: object) -> str:
    return ServerMessage(type=msg_type, payload=dict(kwargs)).to_json()


async def _send_text(
    ws: WebSocket, msg_type: ServerMessageType, **kwargs: object
) -> None:
    await ws.send_text(_server_msg(msg_type, **kwargs))

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.services.events import sse_message

router = APIRouter()


@router.get("/events")
async def events(request: Request, client: str = "panel"):
    state = request.app.state
    sub = state.bus.subscribe(client if client in ("team", "history") else "panel")

    async def stream():
        try:
            yield "retry: 2000\n\n"
            # Full state on every (re)connection, so a phone that lost Wi-Fi catches up.
            yield sse_message(
                "hello",
                {
                    "incidents": [i.public() for i in state.incidents.open_incidents()],
                    "large_object": state.incidents.large_object,
                    "clients": state.bus.counts(),
                },
            )
            while True:
                if await request.is_disconnected():
                    break
                try:
                    yield await asyncio.wait_for(sub.queue.get(), timeout=10)
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            state.bus.unsubscribe(sub)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

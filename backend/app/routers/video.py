import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from app.vision.annotate import encode_jpeg

router = APIRouter()

BOUNDARY = "frame"


@router.get("/video")
async def video(request: Request):
    vision = request.app.state.vision

    async def stream():
        last = -1
        while True:
            if await request.is_disconnected():
                break
            seq, jpeg = vision.latest_jpeg()
            if seq != last and jpeg:
                last = seq
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
            await asyncio.sleep(0.03)

    return StreamingResponse(
        stream(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-cache, no-store", "Pragma": "no-cache"},
    )


@router.get("/snapshot")
def snapshot(request: Request, raw: bool = False):
    """Annotated frame by default; `?raw=1` returns the clean frame (used by calibration)."""
    vision = request.app.state.vision
    if raw:
        frame = vision.raw_frame()
        jpeg = encode_jpeg(frame, 90) if frame is not None else vision.latest_jpeg()[1]
    else:
        jpeg = vision.latest_jpeg()[1]
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

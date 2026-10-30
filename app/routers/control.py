"""Status, health, 'Pista e curată acum', tuning panel and the phone QR code."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.services.team_alerts import qr_svg, team_url

router = APIRouter()


@router.get("/health")
def health(request: Request):
    vision = request.app.state.vision
    return {"ok": True, "video_connected": vision.capture.connected, "fps": round(vision.fps, 1)}


@router.get("/api/status")
def status(request: Request):
    st = request.app.state
    s = st.settings
    return {
        **st.vision.status(),
        "data": st.store.status(),
        "clients": st.bus.counts(),
        "claude": "pornit" if s.enable_claude and s.anthropic_api_key else "oprit",
        "runway": s.runway_name,
        "team_url": team_url(s.port),
        "demo_mode": s.demo_mode,
        "fake_detections": s.demo_fake_detections,
        "large_object": st.incidents.large_object,
    }


@router.post("/reference")
def reset_reference(request: Request):
    request.app.state.vision.reset_reference()
    return {"ok": True, "message": "Referința se reface: pista e considerată curată"}


class Tuning(BaseModel):
    diff_threshold: int | None = Field(None, ge=1, le=255)
    min_area_px: int | None = Field(None, ge=1, le=100_000)
    persist_frames: int | None = Field(None, ge=1, le=300)
    yolo_conf: float | None = Field(None, gt=0, lt=1)


@router.post("/tuning")
def tune(request: Request, values: Tuning):
    if not request.app.state.settings.demo_mode:
        raise HTTPException(403, "Reglajul e disponibil doar în DEMO_MODE")
    clean = {k: v for k, v in values.model_dump().items() if v is not None}
    return request.app.state.vision.tune(clean)


@router.get("/qr.svg")
def team_qr(request: Request):
    return Response(qr_svg(team_url(request.app.state.settings.port)), media_type="image/svg+xml")

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.templating import templates
from app.vision.calibration import CORNER_NAMES, Calibration

router = APIRouter()


@router.get("/calibrate", response_class=HTMLResponse)
def calibrate_page(request: Request):
    vision = request.app.state.vision
    cal = vision.calibration
    size = vision.frame_size()
    return templates.TemplateResponse(
        request,
        "calibrate.html",
        {
            "corners": cal.corners if cal else [],
            "corner_names": CORNER_NAMES,
            "frame_size": size or (vision.s.frame_width, vision.s.frame_width * 9 // 16),
            "geometry": vision.geometry,
        },
    )


class CalibrationIn(BaseModel):
    corners: list[tuple[float, float]] = Field(min_length=4, max_length=4)


@router.post("/calibrate")
def save_calibration(request: Request, body: CalibrationIn):
    st = request.app.state
    vision = st.vision
    size = vision.frame_size()
    if size is None:
        raise HTTPException(409, "Nu există încă niciun cadru video")
    try:
        cal = Calibration(body.corners, size, vision.geometry)
    except Exception:
        raise HTTPException(422, "Colțurile nu formează un patrulater valid")
    cal.save(st.settings.calibration_file)
    vision.set_calibration(cal)
    vision.clear_ignored()
    st.store.update_camera(
        {"id": st.settings.camera_id, "name": st.settings.camera_name, "source": st.settings.video_source,
         "calibration": cal.to_dict()}
    )
    # Show where the corners land on the runway, as a sanity check.
    center = cal.to_pixel(vision.geometry.length_m / 2, vision.geometry.width_m / 2)
    return {"ok": True, "message": "Calibrare salvată", "center_px": center}


@router.post("/calibrate/reset")
def reset_calibration(request: Request):
    st = request.app.state
    f = st.settings.calibration_file
    if f.exists():
        f.unlink()
    st.vision.set_calibration(None)
    st.vision.clear_ignored()
    return {"ok": True, "message": "Calibrarea a fost ștearsă: toată imaginea e considerată pistă"}

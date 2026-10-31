"""Test upload: run detection on a photo (YOLO) or a clip (both layers), without creating incidents."""

import base64
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse

from app.services.risk import assess_risk
from app.templating import templates
from app.vision.annotate import annotate, encode_jpeg
from app.vision.capture import resize_to_width
from app.vision.engine import DetectionEngine, EngineParams, NewIncident
from app.vision.fusion import Track

router = APIRouter()

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_CLIP_SECONDS = 120
MAX_UPLOAD_MB = 500


def _data_uri(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", {})


def _analyse_image(request: Request, data: bytes) -> dict:
    vision = request.app.state.vision
    s = request.app.state.settings
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Imaginea nu a putut fi citită"}
    if frame.shape[1] > s.frame_width:  # a small photo is not blown up: YOLO sees it as it was taken
        frame = resize_to_width(frame, s.frame_width)
    if vision.yolo is None:
        return {
            "error": "Pentru o poză e nevoie de modelul YOLO (ENABLE_YOLO=true). "
            "Detecția schimbărilor are nevoie de un clip, ca să învețe pista curată."
        }
    cands = vision.yolo.detect(frame)
    tracks, risks, rows = [], {}, []
    for i, c in enumerate(cands):
        t = Track(id=i + 1, cand=c, anchor=c.center, anchor_area=c.area, confirmed=True, best_yolo=c)
        risk = assess_risk(c.class_key, None, s.unknown_critical_cm)
        tracks.append(t)
        risks[t.id] = risk
        rows.append({"object_class": c.label, "confidence": c.confidence, "risk": risk, "source": "yolo",
                     "sector": None, "time_s": None})
    image = annotate(frame, tracks, risks, None, None, "Upload")
    return {"kind": "poză", "image": _data_uri(encode_jpeg(image, 85)), "rows": rows, "frames": 1}


def _analyse_clip(request: Request, path: Path, use_calibration: bool) -> dict:
    vision = request.app.state.vision
    s = request.app.state.settings
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Clipul nu a putut fi deschis"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cal = vision.calibration if use_calibration else None
    engine = DetectionEngine(params=EngineParams.from_settings(s), yolo=vision.yolo, calibration=cal)
    rows, best = [], None
    n = 0
    while n < fps * MAX_CLIP_SECONDS:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        frame = resize_to_width(frame, s.frame_width)
        result = engine.process(frame)
        for ev in result.events:
            if isinstance(ev, NewIncident):
                d = ev.detection
                rows.append({"object_class": d.object_class, "confidence": d.confidence, "risk": d.risk,
                             "source": d.source, "sector": d.sector, "time_s": round(n / fps, 1),
                             "crop": f"data:image/jpeg;base64,{d.crop_jpeg_b64}" if d.crop_jpeg_b64 else None})
                best = annotate(frame, result.tracks, result.risks, cal, None, f"Upload · {n / fps:.1f} s")
        last = (frame, result)
    cap.release()
    if n == 0:
        return {"error": "Clipul nu conține cadre"}
    if best is None:
        frame, result = last
        best = annotate(frame, result.tracks, result.risks, cal, None, "Upload · ultimul cadru")
    return {"kind": "clip", "image": _data_uri(encode_jpeg(best, 85)), "rows": rows, "frames": n,
            "seconds": round(n / fps, 1), "truncated": n >= fps * MAX_CLIP_SECONDS}


@router.post("/upload", response_class=HTMLResponse)
def upload(request: Request, file: UploadFile = File(...), use_calibration: bool = Form(False)):
    ext = Path(file.filename or "").suffix.lower()
    if file.size is not None and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        result = {"error": f"Fișierul e prea mare (maximum {MAX_UPLOAD_MB} MB). Tăiați clipul la câteva zeci de secunde."}
    elif ext in IMAGE_EXT:
        result = _analyse_image(request, file.file.read())
    elif ext in VIDEO_EXT:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            result = _analyse_clip(request, tmp_path, use_calibration)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        result = {"error": "Format nesuportat. Încarcă o poză (JPG, PNG, WEBP) sau un clip (MP4, AVI, MOV, MKV)."}
    result["filename"] = file.filename
    return templates.TemplateResponse(request, "partials/upload_result.html", {"r": result})

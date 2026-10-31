"""Video source page: a clip, a webcam or a surveillance (IP) camera, picked from the browser."""

import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.config import ROOT_DIR
from app.templating import templates
from app.vision.capture import parse_source, probe_cameras, test_source

router = APIRouter()

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_MB = 500


def _videos_dir(request: Request) -> Path:
    return request.app.state.settings.data_path / "videos"


def _source_of(path: Path) -> str:
    """How a clip is written as a source: relative to the project when it is inside it."""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path.resolve())


def _clips(request: Request) -> list[dict]:
    folder = _videos_dir(request)
    clips = sorted(p for p in folder.glob("*") if p.suffix.lower() in VIDEO_EXT and ".part" not in p.name) \
        if folder.exists() else []
    current = request.app.state.vision.capture.primary
    playing = Path(str(current.target)).resolve() if current.kind == "file" else None
    return [{"name": p.name, "source": _source_of(p), "mb": round(p.stat().st_size / 1e6, 1),
             "in_use": p.resolve() == playing} for p in clips]


@router.get("/source", response_class=HTMLResponse)
def source_page(request: Request):
    return templates.TemplateResponse(request, "source.html", {})


@router.get("/api/sources")
def sources(request: Request):
    return {"current": request.app.state.vision.capture.status(), "clips": _clips(request)}


@router.get("/api/sources/cameras")
def cameras(request: Request):
    """Which webcams answer (takes a few seconds: each index is opened once)."""
    cap = request.app.state.vision.capture
    in_use = cap.primary.target if cap.primary.kind == "camera" else None
    return {"cameras": probe_cameras(skip=in_use)}


class SourceIn(BaseModel):
    source: str


@router.post("/api/source/test")
def check(request: Request, body: SourceIn):
    s = request.app.state.settings
    return test_source(body.source.strip(), s.camera_width, s.camera_height)


@router.post("/api/source")
def use(request: Request, body: SourceIn):
    raw = body.source.strip()
    spec = parse_source(raw)
    if spec.kind == "none":
        raise HTTPException(422, "Alegeți un clip, o cameră sau scrieți adresa camerei")
    if spec.kind == "file" and not Path(str(spec.target)).exists():
        raise HTTPException(404, "Clipul nu există")
    request.app.state.vision.set_source(raw)
    return {"ok": True, "message": f"Sursa video: {spec.label}. Pista curată se învață din nou."}


@router.post("/api/source/upload")
def upload_clip(request: Request, file: UploadFile = File(...)):
    """A clip from this computer becomes the live source (saved in data/videos/)."""
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(422, "Format nesuportat: MP4, AVI, MOV, MKV sau WEBM")
    if file.size is not None and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Clipul e prea mare (maximum {MAX_UPLOAD_MB} MB)")
    safe = re.sub(r"[^\w.-]+", "_", Path(name).stem).strip("._") or "clip"
    folder = _videos_dir(request)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{safe}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    raw = _source_of(dest)
    result = test_source(raw)
    if not result["ok"]:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, "Clipul nu poate fi citit: " + result["message"])
    request.app.state.vision.set_source(raw)
    return {"ok": True, "message": f"Rulează clipul {dest.name}. Pista curată se învață din nou.", "source": raw}

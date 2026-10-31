from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.vision.calibration import Calibration
from app.vision.fusion import Track
from app.vision.types import BBox

# BGR colours by risk.
RISK_COLORS = {
    "critic": (68, 68, 239),
    "mediu": (11, 158, 245),
    "de verificat": (224, 199, 62),
}
TENTATIVE_COLOR = (150, 150, 150)
LARGE_COLOR = (11, 158, 245)
MASK_COLOR = (224, 199, 62)
GRID_COLOR = (120, 110, 60)

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


@lru_cache(maxsize=512)
def _label_bitmap(text: str, bg: tuple[int, int, int], fg: tuple[int, int, int], size: int) -> np.ndarray:
    """Render a label (with Romanian diacritics, which cv2.putText can't draw) to a BGR patch."""
    font = _font(size)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 10, bottom - top + 8
    img = Image.new("RGB", (w, h), bg[::-1])
    ImageDraw.Draw(img).text((5 - left, 4 - top), text, font=font, fill=fg[::-1])
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def put_label(frame: np.ndarray, text: str, x: int, y: int, bg, fg=(255, 255, 255), size: int = 15) -> None:
    patch = _label_bitmap(text, tuple(bg), tuple(fg), size)
    ph, pw = patch.shape[:2]
    H, W = frame.shape[:2]
    x = int(min(max(0, x), max(0, W - pw)))
    y = int(min(max(0, y), max(0, H - ph)))
    frame[y : y + ph, x : x + pw] = patch[: H - y, : W - x]


def _box(frame, b: BBox, color, thickness: int) -> None:
    cv2.rectangle(frame, (b.x, b.y), (b.x2, b.y2), color, thickness, cv2.LINE_AA)


def draw_runway(frame: np.ndarray, calibration: Calibration | None) -> None:
    if calibration is None:
        return
    poly = np.round(np.array(calibration.corners)).astype(np.int32)
    for line in calibration.grid_lines():
        cv2.polylines(frame, [line], False, GRID_COLOR, 1, cv2.LINE_AA)
    cv2.polylines(frame, [poly], True, MASK_COLOR, 1, cv2.LINE_AA)


def annotate(
    frame: np.ndarray,
    tracks: list[Track],
    risks: dict[int, str],
    calibration: Calibration | None,
    status_text: str | None = None,
    camera_label: str = "",
) -> np.ndarray:
    out = frame.copy()
    draw_runway(out, calibration)

    for t in tracks:
        b = t.cand.bbox
        if t.large:
            color = LARGE_COLOR
            _box(out, b, color, 2 if t.confirmed else 1)
            if t.confirmed:
                put_label(out, "obiect mare pe pistă", b.x, b.y - 24, color, (0, 0, 0))
            continue
        if not t.confirmed:
            # Seen but not persistent yet: thin grey box, no alarm.
            _box(out, b, TENTATIVE_COLOR, 1)
            continue
        risk = risks.get(t.id, "de verificat")
        color = RISK_COLORS.get(risk, RISK_COLORS["de verificat"])
        pad = 4
        _box(out, BBox(x=b.x - pad, y=b.y - pad, w=b.w + 2 * pad, h=b.h + 2 * pad), color, 2)
        if t.best_yolo is not None:
            text = f"{t.best_yolo.label} {round(t.best_yolo.confidence * 100)}%"
        else:
            text = "necunoscut"
        fg = (255, 255, 255) if risk == "critic" else (0, 0, 0)
        put_label(out, text, b.x - pad, b.y - pad - 24, color, fg)

    hud = f"{camera_label}  {datetime.now().strftime('%H:%M:%S')}".strip()
    put_label(out, hud, 10, 10, (19, 14, 10), (230, 220, 210), 14)
    if status_text:
        put_label(out, status_text, 10, 38, (0, 120, 200), (0, 0, 0), 14)
    return out


def crop_jpeg(frame: np.ndarray, bbox: BBox, min_side: int = 96, quality: int = 88) -> bytes:
    """Crop around an object with generous padding, upscaled so small objects are visible."""
    H, W = frame.shape[:2]
    side = max(min_side, int(max(bbox.w, bbox.h) * 2.2))
    cx, cy = bbox.center
    x1 = max(0, min(W - side, cx - side // 2))
    y1 = max(0, min(H - side, cy - side // 2))
    crop = frame[y1 : y1 + side, x1 : x1 + side]
    if crop.size == 0:
        crop = frame
    target = 320
    if crop.shape[1] < target:
        scale = target / crop.shape[1]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def encode_jpeg(frame: np.ndarray, quality: int = 78) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def placeholder_frame(width: int, height: int, title: str, subtitle: str = "") -> np.ndarray:
    frame = np.full((height, width, 3), (19, 14, 10), dtype=np.uint8)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (30, 24, 18), 1)
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (30, 24, 18), 1)
    put_label(frame, title, width // 2 - 150, height // 2 - 30, (19, 14, 10), (0, 169, 242), 26)
    if subtitle:
        put_label(frame, subtitle, width // 2 - 150, height // 2 + 12, (19, 14, 10), (200, 190, 180), 16)
    return frame

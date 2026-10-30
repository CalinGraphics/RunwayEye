"""Data contract between the vision side and the dashboard side.

`Detection` is produced by the vision pipeline (or by the fake generator) and
consumed by the incident service, the Command Center and the team page.
"""

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Source = Literal["yolo", "change", "both"]
Risk = Literal["critic", "mediu", "de verificat"]

UNKNOWN_KEY = "unknown"
UNKNOWN_LABEL = "obiect necunoscut"

# Class keys of our trained model -> label shown in the UI (Romanian, with diacritics).
CLASS_LABELS: dict[str, str] = {
    "surub": "șurub",
    "piulita": "piuliță",
    "cheie": "cheie",
    "metal": "piesă metalică",
    "plastic": "plastic",
    UNKNOWN_KEY: UNKNOWN_LABEL,
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


class BBox(BaseModel):
    """Bounding box in pixels of the processed frame (FRAME_WIDTH wide)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return self.w * self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x2 and self.y <= py <= self.y2

    def iou(self, other: "BBox") -> float:
        ix = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        iy = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


class Detection(BaseModel):
    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utcnow)
    source: Source
    object_class: str  # Romanian label, e.g. "șurub" or "obiect necunoscut"
    class_key: str  # machine key, e.g. "surub" or "unknown"
    confidence: float
    bbox: BBox
    center_px: tuple[int, int]
    runway_distance_m: float | None = None  # from threshold 09; None until calibrated
    centerline_offset_m: float | None = None  # negative = left of centerline
    sector: str | None = None  # e.g. "C2"
    size_estimate_cm: float | None = None
    risk: Risk
    crop_jpeg_b64: str | None = None

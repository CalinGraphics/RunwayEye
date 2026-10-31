"""Data contract between the vision side and the dashboard side.

`Detection` is produced by the vision pipeline (or by the fake generator) and
consumed by the incident service, the Command Center and the team page.
"""

import re
import unicodedata
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

Source = Literal["yolo", "change", "both"]
Risk = Literal["critic", "mediu", "de verificat"]

UNKNOWN_KEY = "unknown"
UNKNOWN_LABEL = "obiect necunoscut"

# The 31 classes of the FOD-A dataset (Munyer et al., 2021), which models/best.pt is trained on
# (training/prepare_fod_a.py writes them in this order).
# key -> (label shown in the UI, coarse group used by the evaluation annotations).
FOD_CLASSES: dict[str, tuple[str, str]] = {
    "adjustable_clamp": ("clemă reglabilă", "metal"),
    "adjustable_wrench": ("cheie reglabilă", "cheie"),
    "battery": ("baterie", "metal"),
    "bolt": ("bolț", "surub"),
    "bolt_nut_set": ("șurub cu piuliță", "surub"),
    "bolt_washer": ("șurub cu șaibă", "surub"),
    "clamp_part": ("bucată de clemă", "metal"),
    "cutter": ("cutter", "metal"),
    "fuel_cap": ("bușon rezervor", "metal"),
    "hammer": ("ciocan", "metal"),
    "hose": ("furtun", "plastic"),
    "label": ("etichetă", "plastic"),
    "luggage_part": ("bucată de bagaj", "plastic"),
    "luggage_tag": ("etichetă de bagaj", "plastic"),
    "metal_part": ("piesă metalică", "metal"),
    "metal_sheet": ("tablă", "metal"),
    "nail": ("cui", "metal"),
    "nut": ("piuliță", "piulita"),
    "paint_chip": ("fulg de vopsea", "plastic"),
    "pen": ("pix", "plastic"),
    "plastic_part": ("piesă de plastic", "plastic"),
    "pliers": ("clește", "cheie"),
    "rock": ("piatră", "piatra"),
    "screw": ("șurub", "surub"),
    "screwdriver": ("șurubelniță", "cheie"),
    "soda_can": ("doză", "metal"),
    "tape": ("bandă adezivă", "plastic"),
    "washer": ("șaibă", "piulita"),
    "wire": ("sârmă", "metal"),
    "wood": ("bucată de lemn", "plastic"),
    "wrench": ("cheie fixă", "cheie"),
}

# What FOD-A has no pictures of (wildlife above all), for the open-vocabulary model models/general.pt
# (YOLOE, built by scripts/build_general_model.py). key -> (label, group); group "animal" is critical
# (bird strikes, animals on the runway).
GENERAL_CLASSES: dict[str, tuple[str, str]] = {
    "pasare": ("pasăre", "animal"),
    "animal": ("animal", "animal"),
    "anvelopa": ("bucată de anvelopă", "anvelopa"),
    "bagaj": ("bagaj", "bagaj"),
    "punga": ("pungă de plastic", "plastic"),
    "sticla": ("sticlă de plastic", "plastic"),
    "carton": ("carton / hârtie", "plastic"),
    "manusa": ("mănușă", "plastic"),
}
# Text prompt (what the model is asked to find, in English) -> key above.
GENERAL_PROMPTS: dict[str, str] = {
    "bird": "pasare",
    "dead bird": "pasare",
    "pigeon": "pasare",
    "seagull": "pasare",
    "dog": "animal",
    "cat": "animal",
    "fox": "animal",
    "rabbit": "animal",
    "tire fragment": "anvelopa",
    "suitcase": "bagaj",
    "backpack": "bagaj",
    "plastic bag": "punga",
    "plastic bottle": "sticla",
    "cardboard box": "carton",
    "glove": "manusa",
}

# Coarse groups: evaluation annotations, fake detections and models trained on our 5 classes.
COARSE_CLASSES = ("surub", "piulita", "cheie", "metal", "plastic", "piatra")

# Class key -> label shown in the UI (Romanian, with diacritics).
CLASS_LABELS: dict[str, str] = {
    "surub": "șurub",
    "piulita": "piuliță",
    "cheie": "cheie",
    "metal": "piesă metalică",
    "plastic": "plastic",
    "piatra": "piatră",
    **{key: label for key, (label, _) in FOD_CLASSES.items()},
    **{key: label for key, (label, _) in GENERAL_CLASSES.items()},
    UNKNOWN_KEY: UNKNOWN_LABEL,
}


def class_key(name: str) -> str:
    """Model or dataset class name -> key: 'BoltNutSet' -> 'bolt_nut_set', 'Șurub ' -> 'surub'."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    norm = unicodedata.normalize("NFKD", name.lower())
    return re.sub(r"[\s\-]+", "_", "".join(ch for ch in norm if not unicodedata.combining(ch)))


def class_group(class_key: str) -> str:
    """'bolt' -> 'surub', 'pasare' -> 'animal'; coarse keys and 'unknown' map to themselves."""
    for table in (FOD_CLASSES, GENERAL_CLASSES):
        if class_key in table:
            return table[class_key][1]
    return class_key


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

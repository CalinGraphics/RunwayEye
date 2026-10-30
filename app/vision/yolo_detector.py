import logging
import shutil
import threading
import unicodedata
from pathlib import Path

import numpy as np

from app.vision.fusion import Candidate
from app.vision.types import CLASS_LABELS, UNKNOWN_KEY, BBox

log = logging.getLogger(__name__)

# Provisional pretrained (COCO) model: only classes that can plausibly be FOD.
COCO_FOD_CLASSES: dict[str, tuple[str, str]] = {
    "scissors": ("metal", "foarfecă"),
    "knife": ("metal", "cuțit"),
    "fork": ("metal", "furculiță"),
    "spoon": ("metal", "lingură"),
    "bottle": ("plastic", "sticlă"),
    "cup": ("plastic", "pahar"),
    "cell phone": (UNKNOWN_KEY, "telefon"),
    "remote": (UNKNOWN_KEY, "telecomandă"),
}


def _key(name: str) -> str:
    """'Șurub' / 'surub ' -> 'surub'."""
    norm = unicodedata.normalize("NFKD", name.strip().lower())
    return "".join(ch for ch in norm if not unicodedata.combining(ch)).replace(" ", "_")


class YoloDetector:
    def __init__(self, model_path: Path, fallback_path: Path | None, conf: float = 0.35, imgsz: int = 640):
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.model_name = ""
        self.provisional = False
        self.error: str | None = None
        self._class_map: dict[int, tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._load(model_path, fallback_path)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _load(self, model_path: Path, fallback_path: Path | None) -> None:
        if model_path.exists():
            path, self.provisional = model_path, False
        elif fallback_path is not None:
            # Missing file name like "yolo11n.pt" is downloaded by Ultralytics on first use.
            path, self.provisional = fallback_path, True
        else:
            self.error = f"Modelul {model_path} lipsește"
            return
        try:
            from ultralytics import YOLO

            if self.provisional and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                model = YOLO(path.name)
                # Ultralytics downloads into the current folder; keep the copy in models/.
                src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else None
                if src and src.exists() and src.resolve() != path.resolve():
                    shutil.move(str(src), str(path))
            else:
                model = YOLO(str(path))
            self.model = model
            self.model_name = path.name
            self._class_map = self._build_class_map(model.names)
            log.info("YOLO loaded: %s (provisional=%s)", path, self.provisional)
        except Exception as exc:  # model problems must never stop the app
            self.error = f"Modelul YOLO nu s-a putut încărca: {exc}"
            log.exception("YOLO load failed")

    def _build_class_map(self, names: dict[int, str]) -> dict[int, tuple[str, str]]:
        mapping: dict[int, tuple[str, str]] = {}
        for idx, name in names.items():
            if self.provisional:
                if name in COCO_FOD_CLASSES:
                    mapping[idx] = COCO_FOD_CLASSES[name]
            else:
                key = _key(name)
                mapping[idx] = (key, CLASS_LABELS.get(key, name))
        return mapping

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        if self.model is None:
            return []
        # The live pipeline and /upload share one model; Ultralytics predict isn't thread-safe.
        with self._lock:
            results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        out: list[Candidate] = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes):
                if cls not in self._class_map:
                    continue
                key, label = self._class_map[cls]
                b = BBox(x=int(x1), y=int(y1), w=max(1, int(x2 - x1)), h=max(1, int(y2 - y1)))
                out.append(
                    Candidate(b, b.center, b.area, "yolo", key, label, round(float(conf), 3))
                )
        return out

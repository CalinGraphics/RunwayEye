import logging
import shutil
import threading
from pathlib import Path

import numpy as np

from app.vision.fusion import Candidate
from app.vision.types import CLASS_LABELS, GENERAL_PROMPTS, UNKNOWN_KEY, BBox, class_group, class_key

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
                # Open-vocabulary models (models/general.pt) name their classes by the text prompt.
                key = GENERAL_PROMPTS.get(name) or class_key(name)
                mapping[idx] = (key, CLASS_LABELS.get(key, name))
        return mapping

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        if self.model is None:
            return []
        # Never upscale: a 300 px photo is run at 320, like the FOD-A images the model learned on.
        imgsz = min(self.imgsz, -(-max(frame.shape[:2]) // 32) * 32)
        # The live pipeline and /upload share one model; Ultralytics predict isn't thread-safe.
        with self._lock:
            results = self.model.predict(frame, imgsz=imgsz, conf=self.conf, verbose=False)
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


# Groups of the general model that FOD-A has no examples of (see CombinedDetector).
OUTSIDE_FOD_A = {"animal", "anvelopa"}
RUNNER_UP_CONF = 0.5


class CombinedDetector:
    """The FOD model (models/best.pt, 31 FOD-A classes) plus the open-vocabulary model
    (models/general.pt: birds, animals, tyre pieces, bags...). Same interface as YoloDetector.
    Where both see the same object, the FOD model's finer class wins, except for what FOD-A has no
    pictures of at all (birds, animals, tyre pieces): there the FOD model can only guess the nearest
    hardware class (a dead pigeon came out as "clamp part"), while the general model sometimes takes a
    bolt with a washer for a bird. The more confident one wins; a confident runner-up stays in the
    label ("bucată de clemă / posibil pasăre"). Both are critical, so the alert goes out either way."""

    def __init__(self, primary: YoloDetector, general: YoloDetector | None, general_every: int = 1):
        self.primary = primary
        self.general = general if general is not None and general.loaded else None
        # The general model is ~4x slower: run it every Nth call and reuse its boxes in between.
        self.general_every = max(1, general_every)
        self._calls = 0
        self._general_cache: list[Candidate] = []
        self.provisional = primary.provisional
        self.error = primary.error

    @property
    def loaded(self) -> bool:
        return self.primary.loaded or self.general is not None

    @property
    def model_name(self) -> str:
        names = [d.model_name for d in (self.primary, self.general) if d is not None and d.loaded]
        return " + ".join(names)

    @property
    def conf(self) -> float:
        return self.primary.conf

    @conf.setter
    def conf(self, value: float) -> None:  # live tuning panel: only the FOD model's threshold
        self.primary.conf = value

    def detect(self, frame: np.ndarray) -> list[Candidate]:
        out = self.primary.detect(frame)
        if self.general is not None:
            if self._calls % self.general_every == 0:
                self._general_cache = self.general.detect(frame)
            self._calls += 1
            for c in self._general_cache:
                overlapping = [p for p in out if c.bbox.iou(p.bbox) >= 0.5]
                if not overlapping:
                    out.append(c)
                elif class_group(c.class_key) in OUTSIDE_FOD_A:
                    best = max(overlapping, key=lambda p: p.confidence)
                    if c.confidence >= best.confidence:
                        out = [p for p in out if p not in overlapping] + [c]
                    elif c.confidence >= RUNNER_UP_CONF:
                        best.label = f"{best.label} / posibil {c.label}"
        return out


def build_detector(s, conf: float | None = None) -> CombinedDetector:
    """Both models as configured in .env (YOLO_*, GENERAL_*)."""
    primary = YoloDetector(
        s.path(s.yolo_model_path),
        s.path(s.yolo_fallback_model) if s.yolo_fallback_model else None,
        conf=s.yolo_conf if conf is None else conf,
        imgsz=s.yolo_imgsz,
    )
    general = None
    if s.enable_general_model and s.path(s.general_model_path).exists():
        general = YoloDetector(s.path(s.general_model_path), None, conf=s.general_conf, imgsz=s.yolo_imgsz)
    return CombinedDetector(primary, general, s.general_every_n)

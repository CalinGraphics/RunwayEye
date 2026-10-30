"""Fake detections (DEMO_FAKE_DETECTIONS=true), so the dashboard can be built and shown without vision."""

import base64
import logging
import random
import threading

import cv2
import numpy as np

from app.services.risk import assess_risk
from app.vision.annotate import placeholder_frame
from app.vision.calibration import RunwayGeometry
from app.vision.types import CLASS_LABELS, BBox, Detection

log = logging.getLogger(__name__)

FAKE_CLASSES = ["surub", "piulita", "cheie", "metal", "plastic", "unknown"]


def fake_crop(class_key: str, rng: random.Random) -> bytes:
    img = np.random.default_rng(rng.randint(0, 10_000)).normal(105, 9, (240, 240)).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    color = (40, 160, 230) if class_key == "plastic" else (200, 205, 210)
    if class_key in ("surub", "metal"):
        cv2.rectangle(img, (85, 112), (160, 128), color, -1)
        cv2.rectangle(img, (75, 104), (90, 136), color, -1)
    elif class_key == "piulita":
        pts = np.array([[120 + 26 * np.cos(a), 120 + 26 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]], np.int32)
        cv2.fillPoly(img, [pts], color)
        cv2.circle(img, (120, 120), 10, (90, 90, 90), -1)
    elif class_key == "cheie":
        cv2.line(img, (70, 160), (160, 85), color, 12)
        cv2.circle(img, (165, 80), 18, color, -1)
        cv2.circle(img, (172, 72), 9, (100, 100, 100), -1)
    else:
        cv2.ellipse(img, (120, 120), (40, 22), 30, 0, 360, color, -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else b""


def fake_snapshot(det: Detection, crop_jpeg: bytes, width: int = 960, height: int = 540) -> bytes:
    frame = placeholder_frame(width, height, "Detecție de test", "DEMO_FAKE_DETECTIONS=true")
    crop = cv2.imdecode(np.frombuffer(crop_jpeg, np.uint8), cv2.IMREAD_COLOR)
    if crop is not None:
        b = det.bbox
        side = max(b.w, b.h) * 3
        x, y = max(0, min(width - side, b.center[0] - side // 2)), max(0, min(height - side, b.center[1] - side // 2))
        frame[y : y + side, x : x + side] = cv2.resize(crop, (side, side))
        cv2.rectangle(frame, (x, y), (x + side, y + side), (0, 0, 230), 2)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def make_fake_detection(geometry: RunwayGeometry, unknown_critical_cm: float, rng: random.Random | None = None) -> Detection:
    rng = rng or random.Random()
    key = rng.choice(FAKE_CLASSES)
    u = rng.uniform(50, geometry.length_m - 50)
    v = rng.uniform(3, geometry.width_m - 3)
    size = round(rng.uniform(3, 25), 1)
    x, y = rng.randint(100, 800), rng.randint(100, 420)
    bbox = BBox(x=x, y=y, w=rng.randint(12, 40), h=rng.randint(8, 30))
    source = "change" if key == "unknown" else rng.choice(["yolo", "both"])
    return Detection(
        source=source,
        object_class=CLASS_LABELS[key],
        class_key=key,
        confidence=round(rng.uniform(0.55, 0.97), 2),
        bbox=bbox,
        center_px=bbox.center,
        runway_distance_m=round(u, 1),
        centerline_offset_m=round(v - geometry.width_m / 2, 1),
        sector=geometry.sector(u, v),
        size_estimate_cm=size,
        risk=assess_risk(key, size, unknown_critical_cm),
        crop_jpeg_b64=base64.b64encode(fake_crop(key, rng)).decode("ascii"),
    )


class FakeDetectionGenerator:
    def __init__(self, incidents, geometry: RunwayGeometry, unknown_critical_cm: float, interval_s=(12, 22), max_open=3):
        self.incidents = incidents
        self.geometry = geometry
        self.unknown_critical_cm = unknown_critical_cm
        self.interval_s = interval_s
        self.max_open = max_open
        self._rng = random.Random()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fake-detections", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        self._stop.wait(4)
        while not self._stop.is_set():
            if len(self.incidents.open_incidents()) < self.max_open:
                det = make_fake_detection(self.geometry, self.unknown_critical_cm, self._rng)
                snapshot = fake_snapshot(det, base64.b64decode(det.crop_jpeg_b64))
                self.incidents.create_from_detection(det, snapshot, fake=True)
            self._stop.wait(self._rng.uniform(*self.interval_s))

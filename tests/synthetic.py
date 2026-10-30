"""Synthetic frames for tests: textured grey 'asphalt' plus simple objects."""

import numpy as np

from app.vision.engine import DetectionEngine, EngineParams, NewIncident
from app.vision.fusion import Candidate
from app.vision.types import BBox

W, H = 320, 240


def background(seed: int = 1, level: int = 110) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tex = rng.normal(level, 6, size=(H, W)).clip(0, 255)
    return np.repeat(tex[:, :, None], 3, axis=2).astype(np.float32)


class Scene:
    def __init__(self, seed: int = 1):
        self.base = background(seed)
        self.rng = np.random.default_rng(seed + 100)
        self.brightness = 0.0

    def frame(self, objects: list[tuple[int, int, int, int, int]] = ()) -> np.ndarray:
        """objects: (x, y, w, h, intensity) rectangles."""
        img = self.base + self.brightness + self.rng.normal(0, 2, size=self.base.shape)
        for x, y, w, h, v in objects:
            img[y : y + h, x : x + w] = v
        return img.clip(0, 255).astype(np.uint8)


def make_engine(**overrides) -> DetectionEngine:
    params = EngineParams(
        reference_frames=10,
        min_area_px=40,
        max_area_px=3000,
        persist_frames=6,
        large_persist_frames=12,
        gone_frames=10,
    )
    for k, v in overrides.pop("params", {}).items():
        setattr(params, k, v)
    return DetectionEngine(params=params, **overrides)


def learn(engine: DetectionEngine, scene: Scene) -> None:
    for _ in range(engine.params.reference_frames):
        engine.process(scene.frame())


def run(engine: DetectionEngine, frames) -> list:
    events = []
    for f in frames:
        events.extend(engine.process(f).events)
    return events


def incidents(events) -> list[NewIncident]:
    return [e for e in events if isinstance(e, NewIncident)]


class FakeYolo:
    """Returns fixed boxes, like a YOLO model that always sees the same object."""

    def __init__(self, boxes: list[tuple[BBox, str, str, float]]):
        self.boxes = boxes

    def detect(self, frame) -> list[Candidate]:
        return [Candidate(b, b.center, b.area, "yolo", key, label, conf) for b, key, label, conf in self.boxes]

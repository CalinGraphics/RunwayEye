"""Runway calibration: 4 clicked corners -> homography, runway mask and sectors.

Runway coordinates: u = metres from threshold 09 along the runway (0..L),
v = metres across from the left edge seen from threshold 09 (0..W).
The centerline is at v = W/2; offset < 0 means left of the centerline.
"""

import json
import math
import string
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.vision.types import BBox

CORNER_NAMES = ["prag 09 stânga", "prag 09 dreapta", "prag 27 dreapta", "prag 27 stânga"]


@dataclass
class RunwayGeometry:
    length_m: float
    width_m: float
    sectors_along: int = 6
    sectors_across: int = 3
    # Physical size of the painted mock-up runway (cm); 0 = a real runway.
    mockup_length_cm: float = 0.0
    mockup_width_cm: float = 0.0

    def sector(self, u: float, v: float) -> str:
        # Multiply before dividing (plus a tiny epsilon) so exact boundaries land in the next sector.
        i = math.floor(u * self.sectors_along / self.length_m + 1e-9)
        j = math.floor(v * self.sectors_across / self.width_m + 1e-9)
        i = min(max(i, 0), self.sectors_along - 1)
        j = min(max(j, 0), self.sectors_across - 1)
        return f"{string.ascii_uppercase[i]}{j + 1}"

    def sector_names(self) -> list[str]:
        return [
            f"{string.ascii_uppercase[i]}{j + 1}"
            for i in range(self.sectors_along)
            for j in range(self.sectors_across)
        ]

    def physical_cm(self, du_m: float, dv_m: float) -> tuple[float, float]:
        """Convert a runway-scale extent to the real object size in cm."""
        if self.mockup_length_cm > 0 and self.mockup_width_cm > 0:
            return (
                du_m / self.length_m * self.mockup_length_cm,
                dv_m / self.width_m * self.mockup_width_cm,
            )
        return du_m * 100.0, dv_m * 100.0


@dataclass
class RunwayPosition:
    distance_m: float
    offset_m: float
    sector: str
    inside: bool


class Calibration:
    def __init__(
        self,
        corners: list[tuple[float, float]],
        frame_size: tuple[int, int],
        geometry: RunwayGeometry,
    ):
        if len(corners) != 4:
            raise ValueError("Calibration needs exactly 4 corners")
        self.corners = [(float(x), float(y)) for x, y in corners]
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.geometry = geometry
        L, W = geometry.length_m, geometry.width_m
        src = np.array(self.corners, dtype=np.float32)
        dst = np.array([(0, 0), (0, W), (L, W), (L, 0)], dtype=np.float32)
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.H_inv = cv2.getPerspectiveTransform(dst, src)
        self._polygon = np.array(self.corners, dtype=np.float32)

    def to_runway(self, x: float, y: float) -> tuple[float, float]:
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self.H)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def to_pixel(self, u: float, v: float) -> tuple[float, float]:
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), self.H_inv)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def pixel_to_runway(self, x: float, y: float) -> RunwayPosition:
        g = self.geometry
        u, v = self.to_runway(x, y)
        inside = 0 <= u <= g.length_m and 0 <= v <= g.width_m
        uc = min(max(u, 0.0), g.length_m)
        vc = min(max(v, 0.0), g.width_m)
        return RunwayPosition(
            distance_m=uc,
            offset_m=vc - g.width_m / 2,
            sector=g.sector(uc, vc),
            inside=inside,
        )

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self._polygon, (float(x), float(y)), False) >= 0

    def size_cm(self, bbox: BBox) -> float:
        pts = [(bbox.x, bbox.y), (bbox.x2, bbox.y), (bbox.x2, bbox.y2), (bbox.x, bbox.y2)]
        uv = [self.to_runway(px, py) for px, py in pts]
        us = [p[0] for p in uv]
        vs = [p[1] for p in uv]
        du_cm, dv_cm = self.geometry.physical_cm(max(us) - min(us), max(vs) - min(vs))
        return max(du_cm, dv_cm)

    def mask(self, width: int, height: int) -> np.ndarray:
        m = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(self._polygon).astype(np.int32)], 255)
        return m

    def grid_lines(self, steps: int = 12) -> list[np.ndarray]:
        """Sector boundaries projected into the image, as polylines (int32)."""
        g = self.geometry
        lines = []
        for i in range(1, g.sectors_along):
            u = g.length_m * i / g.sectors_along
            pts = [self.to_pixel(u, g.width_m * k / steps) for k in range(steps + 1)]
            lines.append(np.round(np.array(pts)).astype(np.int32))
        for j in range(1, g.sectors_across):
            v = g.width_m * j / g.sectors_across
            pts = [self.to_pixel(g.length_m * k / steps, v) for k in range(steps + 1)]
            lines.append(np.round(np.array(pts)).astype(np.int32))
        return lines

    def to_dict(self) -> dict:
        return {
            "corners": [list(c) for c in self.corners],
            "corner_names": CORNER_NAMES,
            "frame_size": list(self.frame_size),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(
        cls, path: Path, geometry: RunwayGeometry, frame_size: tuple[int, int] | None = None
    ) -> "Calibration | None":
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        corners = [tuple(c) for c in data["corners"]]
        saved_size = tuple(data.get("frame_size") or frame_size or (0, 0))
        if frame_size and saved_size[0] and tuple(saved_size) != tuple(frame_size):
            sx = frame_size[0] / saved_size[0]
            sy = frame_size[1] / saved_size[1]
            corners = [(x * sx, y * sy) for x, y in corners]
            saved_size = frame_size
        return cls(corners, saved_size, geometry)

"""A clip can carry its own runway: `<clip>.scene.json` next to `<clip>.mp4`.

For real footage. Such a clip shows only part of a runway, so its corners are not the real
thresholds, and it is filmed at real scale, unlike the mock-up that .env describes. While the
clip plays, its scene replaces the geometry from .env and data/calibration.json, and its optional
"tuning" the detection sensitivity: a real camera sees objects a few pixels across, where the
mock-up's thresholds would ignore them."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.vision.calibration import CORNER_NAMES, Calibration, RunwayGeometry


@dataclass
class Scene:
    path: Path
    geometry: RunwayGeometry
    data: dict

    @classmethod
    def for_source(cls, kind: str, target) -> "Scene | None":
        if kind != "file" or target is None:
            return None
        path = Path(str(target)).with_suffix(".scene.json")
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        r = data["runway"]
        geometry = RunwayGeometry(
            length_m=float(r["length_m"]),
            width_m=float(r["width_m"]),
            sectors_along=int(r.get("sectors_along", 4)),
            sectors_across=int(r.get("sectors_across", 3)),
        )
        return cls(path, geometry, data)

    @property
    def tuning(self) -> dict:
        return dict(self.data.get("tuning") or {})

    def calibration(self, frame_size: tuple[int, int]) -> Calibration | None:
        if len(self.data.get("corners") or []) != 4:
            return None
        return Calibration.from_data(self.data, self.geometry, frame_size)

    @property
    def has_original(self) -> bool:
        return bool(self.data.get("original"))

    def restore_original(self) -> bool:
        """Back to the corners the clip came with, after a manual calibration went wrong."""
        original = self.data.pop("original", None)
        if not original:
            return False
        self.data.update(original)
        self._write()
        return True

    def clear_corners(self) -> None:
        for key in ("corners", "outline", "corner_names"):
            self.data.pop(key, None)
        self._write()

    def save_corners(self, cal: Calibration) -> None:
        """The operator recalibrated while the clip played: the corners belong to the clip.
        The ones it came with are kept the first time, so a wrong calibration can be undone."""
        if not self.has_original and self.data.get("corners"):
            self.data["original"] = {k: self.data[k] for k in ("corners", "outline", "frame_size") if k in self.data}
        self.data.update(corners=[list(c) for c in cal.corners], outline=[list(c) for c in cal.outline],
                         corner_names=CORNER_NAMES, frame_size=list(cal.frame_size))
        self._write()

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

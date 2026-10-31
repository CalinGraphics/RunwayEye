import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import ROOT_DIR

log = logging.getLogger(__name__)


@dataclass
class SourceSpec:
    raw: str
    kind: str  # "camera" | "file" | "stream" | "none"
    target: int | str | None
    label: str


def parse_source(raw: str) -> SourceSpec:
    value = (raw or "").strip()
    if not value or value.lower() == "none":
        return SourceSpec(value, "none", None, "fără sursă")
    if value.isdigit():
        return SourceSpec(value, "camera", int(value), f"cameră {value}")
    if "://" in value:
        scheme = value.split("://", 1)[0].upper()
        return SourceSpec(value, "stream", value, f"flux {scheme}")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT_DIR / path
    return SourceSpec(value, "file", str(path), f"fișier {path.name}")


def _open(spec: SourceSpec, width: int, height: int) -> cv2.VideoCapture | None:
    if spec.kind == "none":
        return None
    if spec.kind == "file" and not Path(str(spec.target)).exists():
        return None
    if spec.kind == "camera":
        cap = cv2.VideoCapture(spec.target, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(spec.target)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(spec.target)
    if not cap.isOpened():
        cap.release()
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    if spec.kind == "file":
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap


def test_source(raw: str, width: int = 1280, height: int = 720) -> dict:
    """Open a source once and read a frame: lets the operator check a camera address before using it."""
    spec = parse_source(raw)
    if spec.kind == "none":
        return {"ok": False, "message": "Adresa e goală"}
    t0 = time.monotonic()
    cap = _open(spec, width, height)
    if cap is None:
        hint = " (fișierul nu există)" if spec.kind == "file" else ""
        return {"ok": False, "message": f"Nu se deschide: {spec.label}{hint}"}
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return {"ok": False, "message": f"{spec.label} s-a deschis, dar nu trimite imagine"}
    return {"ok": True, "message": f"{spec.label} · {frame.shape[1]}×{frame.shape[0]} · răspuns în "
                                   f"{time.monotonic() - t0:.1f} s"}


def probe_cameras(skip: int | None = None, max_index: int = 4) -> list[int]:
    """Indexes of the cameras attached to this computer (the one in use can't be opened twice)."""
    found = []
    for i in range(max_index):
        if i == skip:
            found.append(i)
            continue
        cap = _open(SourceSpec(str(i), "camera", i, f"cameră {i}"), 640, 480)
        if cap is not None:
            found.append(i)
            cap.release()
    return found


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, round(h * scale)), interpolation=interp)


class FrameSource:
    def __init__(self, source: str, fallback: str, width: int, height: int, frame_width: int):
        self.primary = parse_source(source)
        self.fallback = parse_source(fallback) if fallback else None
        self.width, self.height, self.frame_width = width, height, frame_width
        self.active = self.primary
        self.using_fallback = False
        self.connected = False
        self.fps = 0.0
        self.message = ""
        self._frame: np.ndarray | None = None
        self._pending: SourceSpec | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def switch(self, source: str) -> None:
        """Use another source from now on (the capture thread picks it up within a frame)."""
        spec = parse_source(source)
        with self._lock:
            self._pending = spec
            self.primary = spec  # status() shows the chosen source right away

    def rewind(self) -> None:
        """Play a clip from its start again (a live camera has nothing to rewind)."""
        if self.primary.kind == "file" and not self.using_fallback:
            self.switch(self.primary.raw)

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq, self._frame

    def status(self) -> dict:
        return {
            "source": self.primary.raw,
            "label": self.active.label,
            "kind": self.active.kind,
            "connected": self.connected,
            "using_fallback": self.using_fallback,
            "fps": round(self.fps, 1),
            "message": self.message,
        }

    def _switch_to_fallback(self) -> bool:
        if self.using_fallback or self.fallback is None or self.fallback.kind == "none":
            return False
        if self.fallback.kind == "file" and not Path(str(self.fallback.target)).exists():
            return False
        log.warning("Video source %s unavailable, switching to fallback %s", self.primary.raw, self.fallback.raw)
        self.active = self.fallback
        self.using_fallback = True
        self.message = f"Sursa {self.primary.label} nu răspunde: rulează clipul de rezervă"
        return True

    def _run(self) -> None:
        cap: cv2.VideoCapture | None = None
        failures = 0
        last_ts = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None:
                if cap is not None:
                    cap.release()
                    cap = None
                self.primary = self.active = pending
                self.using_fallback, failures, self.fps, self.message = False, 0, 0.0, ""
            if cap is None:
                self.connected = False
                if self.active.kind == "none":
                    self.message = "Nicio sursă video setată"
                    if not self._switch_to_fallback():
                        self._stop.wait(1.0)
                    continue
                cap = _open(self.active, self.width, self.height)
                if cap is None:
                    failures += 1
                    self.message = f"Nu pot deschide {self.active.label}, reîncerc…"
                    if failures >= 2 and self._switch_to_fallback():
                        failures = 0
                        continue
                    self._stop.wait(min(5.0, 0.5 * failures))
                    continue
                failures = 0
                self.connected = True
                if not self.using_fallback:
                    self.message = ""
                file_fps = cap.get(cv2.CAP_PROP_FPS) if self.active.kind == "file" else 0
                frame_interval = 1.0 / file_fps if 1 <= file_fps <= 120 else (1 / 25 if self.active.kind == "file" else 0)

            t0 = time.monotonic()
            ok, frame = cap.read()
            if not ok and self.active.kind == "file":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                failures += 1
                self.message = f"Semnal pierdut de la {self.active.label}, reconectare…"
                if failures >= 3:
                    self._switch_to_fallback()
                continue

            frame = resize_to_width(frame, self.frame_width)
            with self._lock:
                self._frame = frame
                self._seq += 1
            now = time.monotonic()
            dt = now - last_ts
            last_ts = now
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            if frame_interval:
                # Play files at their own speed, like a live camera.
                remaining = frame_interval - (time.monotonic() - t0)
                if remaining > 0:
                    self._stop.wait(remaining)
        if cap is not None:
            cap.release()

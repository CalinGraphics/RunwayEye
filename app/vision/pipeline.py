"""Vision worker thread: capture -> engine -> annotated JPEG + incident events.

FastAPI never waits for detection; it only reads the latest annotated JPEG.
"""

import json
import logging
import threading
import time
from typing import Callable

import numpy as np

from app.config import Settings
from app.vision.annotate import annotate, encode_jpeg, placeholder_frame
from app.vision.calibration import Calibration, RunwayGeometry
from app.vision.capture import FrameSource
from app.vision.engine import DetectionEngine, EngineParams

log = logging.getLogger(__name__)

TUNABLE = {"diff_threshold", "min_area_px", "persist_frames", "yolo_conf"}


def runway_geometry(s: Settings) -> RunwayGeometry:
    return RunwayGeometry(
        length_m=s.runway_length_m,
        width_m=s.runway_width_m,
        sectors_along=s.sectors_along,
        sectors_across=s.sectors_across,
        mockup_length_cm=s.mockup_length_cm,
        mockup_width_cm=s.mockup_width_cm,
    )


class VisionService:
    def __init__(self, settings: Settings, on_event: Callable):
        self.s = settings
        self.on_event = on_event
        self.geometry = runway_geometry(settings)
        self.capture = FrameSource(
            settings.video_source,
            settings.fallback_video,
            settings.camera_width,
            settings.camera_height,
            settings.frame_width,
        )
        params = EngineParams.from_settings(settings)
        self._load_tuning(params)
        self.engine = DetectionEngine(params=params)
        self.calibration: Calibration | None = None
        self.yolo = None
        self.yolo_state = "oprit" if not settings.enable_yolo else "se încarcă"
        self.yolo_error: str | None = None
        self.fps = 0.0
        self.last_result = None
        self._jpeg: bytes = encode_jpeg(placeholder_frame(settings.frame_width, settings.frame_width * 9 // 16, "Pornire…"))
        self._raw: np.ndarray | None = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vision", daemon=True)

    def start(self) -> None:
        self.capture.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.capture.stop()
        self._thread.join(timeout=3)

    def latest_jpeg(self) -> tuple[int, bytes]:
        with self._lock:
            return self._seq, self._jpeg

    def raw_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def frame_size(self) -> tuple[int, int] | None:
        raw = self._raw
        return None if raw is None else (raw.shape[1], raw.shape[0])

    def status(self) -> dict:
        r = self.last_result
        return {
            "video": self.capture.status(),
            "fps": round(self.fps, 1),
            "yolo": {
                "state": self.yolo_state,
                "model": getattr(self.yolo, "model_name", "") if self.yolo else "",
                "provisional": bool(getattr(self.yolo, "provisional", False)),
                "error": self.yolo_error,
            },
            "learning": bool(r.learning) if r else True,
            "learned_frames": r.learned_frames if r else 0,
            "reference_frames": self.engine.params.reference_frames,
            "calibrated": self.calibration is not None,
            "tuning": {k: getattr(self.engine.params, k) for k in sorted(TUNABLE)},
        }

    def reset_reference(self) -> None:
        self.engine.reset_reference()

    def close_incident(self, incident_id: str, absorb: bool) -> None:
        self.engine.close_incident(incident_id, absorb)

    def set_calibration(self, cal: Calibration | None) -> None:
        self.calibration = cal
        self.engine.set_calibration(cal)

    def tune(self, values: dict) -> dict:
        clean = {k: v for k, v in values.items() if k in TUNABLE}
        self.engine.update_params(**clean)
        current = {k: getattr(self.engine.params, k) for k in TUNABLE}
        current.update(clean)
        try:
            self.s.tuning_file.parent.mkdir(parents=True, exist_ok=True)
            self.s.tuning_file.write_text(json.dumps(current, indent=2), encoding="utf-8")
        except OSError:
            log.warning("Could not save tuning values")
        return current

    def _load_tuning(self, params: EngineParams) -> None:
        try:
            data = json.loads(self.s.tuning_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for k, v in data.items():
            if k in TUNABLE:
                setattr(params, k, type(getattr(params, k))(v))

    def _load_yolo(self) -> None:
        if not self.s.enable_yolo:
            return
        try:
            from app.vision.yolo_detector import YoloDetector

            det = YoloDetector(
                self.s.path(self.s.yolo_model_path),
                self.s.path(self.s.yolo_fallback_model) if self.s.yolo_fallback_model else None,
                conf=self.engine.params.yolo_conf,
                imgsz=self.s.yolo_imgsz,
            )
        except Exception as exc:
            det = None
            self.yolo_error = str(exc)
            log.exception("YOLO import failed")
        if det is not None and det.loaded:
            self.yolo = det
            self.yolo_state = "provizoriu" if det.provisional else "încărcat"
            self.engine.set_yolo(det)
        else:
            self.yolo_state = "eroare"
            self.yolo_error = self.yolo_error or (det.error if det else "necunoscută")

    def _load_calibration(self, size: tuple[int, int]) -> None:
        try:
            cal = Calibration.load(self.s.calibration_file, self.geometry, size)
        except Exception:
            log.exception("Invalid calibration file, ignoring it")
            cal = None
        if cal is not None:
            self.set_calibration(cal)

    def _run(self) -> None:
        # YOLO loads in this thread (it takes seconds), so the server starts instantly.
        self._load_yolo()
        last_seq = 0
        last_frame_at = time.monotonic()
        calibration_loaded = False
        while not self._stop.is_set():
            seq, frame = self.capture.latest()
            if frame is None or seq == last_seq:
                if time.monotonic() - last_frame_at > 1.5:
                    self._show_placeholder()
                    last_frame_at = time.monotonic()
                self._stop.wait(0.005)
                continue
            last_seq = seq
            if not calibration_loaded:
                self._load_calibration((frame.shape[1], frame.shape[0]))
                calibration_loaded = True
            t0 = time.monotonic()
            try:
                self._process(frame)
            except Exception:
                log.exception("Frame processing failed")
            dt = time.monotonic() - last_frame_at
            last_frame_at = time.monotonic()
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
            log.debug("frame %.1f ms", (time.monotonic() - t0) * 1000)

    def _process(self, frame: np.ndarray) -> None:
        result = self.engine.process(frame)
        self.last_result = result
        status = None
        if result.learning:
            status = f"Învăț pista curată… {result.learned_frames}/{self.engine.params.reference_frames}"
        elif result.global_change:
            status = "Schimbare de lumină: aștept stabilizarea"
        out = annotate(frame, result.tracks, result.risks, self.calibration, status, self.s.camera_name)
        jpeg = encode_jpeg(out)
        with self._lock:
            self._jpeg = jpeg
            self._raw = frame
            self._seq += 1
        for ev in result.events:
            try:
                self.on_event(ev, jpeg)
            except Exception:
                log.exception("Event handler failed")

    def _show_placeholder(self) -> None:
        cs = self.capture.status()
        w = self.s.frame_width
        img = placeholder_frame(w, w * 9 // 16, "Fără semnal video", cs.get("message") or cs["label"])
        with self._lock:
            self._jpeg = encode_jpeg(img)
            self._seq += 1

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
from app.vision.engine import DetectionEngine, EngineParams, IgnoredZones
from app.vision.scene import Scene
from app.vision.types import BBox

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
            self._saved_source() or settings.video_source,
            settings.fallback_video,
            settings.camera_width,
            settings.camera_height,
            settings.frame_width,
        )
        params = EngineParams.from_settings(settings)
        self._load_tuning(params)
        # Sensitivity outside any clip scene, to come back to when a scene's own tuning ends.
        self._base_tuning = {k: getattr(params, k) for k in TUNABLE}
        self.engine = DetectionEngine(params=params)
        self.calibration: Calibration | None = None
        self.scene: Scene | None = None
        self._calibrated_for = None  # the source the calibration was loaded for
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
        """'Pista e curată acum'. A clip starts over too, as at startup: its reference is learned
        from its clean first seconds, and what it shows after that is detected again."""
        self.capture.rewind()
        self.engine.reset_reference(cleared=True)

    def restore_incidents(self, incidents) -> None:
        for inc in incidents:
            if not inc.fake:
                self.engine.restore_incident(inc.id, tuple(inc.center_px), inc.bbox, inc.risk, inc.object_present)

    def close_incident(self, incident_id: str, absorb: bool) -> None:
        self.engine.close_incident(incident_id, absorb)

    def _saved_source(self) -> str | None:
        """Source picked on the "Sursă video" page; it wins over VIDEO_SOURCE from .env."""
        try:
            return json.loads(self.s.video_source_file.read_text(encoding="utf-8"))["source"] or None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def set_source(self, source: str) -> None:
        """Another camera or clip: a new scene, so the clean-runway reference is learned again."""
        self.s.video_source_file.parent.mkdir(parents=True, exist_ok=True)
        self.s.video_source_file.write_text(json.dumps({"source": source}), encoding="utf-8")
        self.capture.switch(source)
        self.engine.reset_reference()
        self.engine.forget_closed()

    def _load_ignored(self) -> None:
        try:
            zones = json.loads(self.s.ignored_zones_file.read_text(encoding="utf-8"))
            self.engine.set_ignored([BBox(**z) for z in zones])
        except (OSError, ValueError, TypeError):
            pass

    def _save_ignored(self, zones: list[BBox]) -> None:
        try:
            self.s.ignored_zones_file.parent.mkdir(parents=True, exist_ok=True)
            self.s.ignored_zones_file.write_text(json.dumps([z.model_dump() for z in zones]), encoding="utf-8")
        except OSError:
            log.warning("Could not save the ignored zones")

    def clear_ignored(self) -> None:
        """The camera view changed (new calibration): the dismissed spots no longer match."""
        self.engine.set_ignored([])
        self.s.ignored_zones_file.unlink(missing_ok=True)

    def set_calibration(self, cal: Calibration | None) -> None:
        self.calibration = cal
        self.engine.set_calibration(cal)

    def tune(self, values: dict) -> dict:
        clean = {k: v for k, v in values.items() if k in TUNABLE}
        self.engine.update_params(**clean)
        if self.scene is None:
            self._base_tuning.update(clean)
        current = {k: getattr(self.engine.params, k) for k in TUNABLE}
        current.update(clean)
        try:
            # A clip's scene tuning is the clip's; the file keeps the camera's own values.
            saved = current if self.scene is None else self._base_tuning
            self.s.tuning_file.parent.mkdir(parents=True, exist_ok=True)
            self.s.tuning_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
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
            from app.vision.yolo_detector import build_detector

            det = build_detector(self.s, conf=self.engine.params.yolo_conf)
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

    def reset_calibration(self) -> None:
        """Forget the corners of the source now playing: the clip's own, or the camera's."""
        if self.scene is not None:
            self.scene.clear_corners()
        else:
            self.s.calibration_file.unlink(missing_ok=True)
            self.clear_ignored()
        self.set_calibration(None)

    def save_calibration(self, cal: Calibration) -> None:
        if self.scene is not None:
            self.scene.save_corners(cal)
        else:
            cal.save(self.s.calibration_file)
        self.set_calibration(cal)

    def _load_calibration(self, size: tuple[int, int]) -> None:
        """Runway and corners for the source now playing: its own scene if it is a real clip
        that has one, otherwise .env and data/calibration.json."""
        active = self.capture.active
        try:
            self.scene = Scene.for_source(active.kind, active.target)
        except Exception:
            log.exception("Invalid scene file next to %s, ignoring it", active.target)
            self.scene = None
        self.geometry = self.scene.geometry if self.scene else runway_geometry(self.s)
        scene_tuning = self.scene.tuning if self.scene else {}
        self.engine.update_params(**{**self._base_tuning, **{k: v for k, v in scene_tuning.items() if k in TUNABLE}})
        try:
            cal = (self.scene.calibration(size) if self.scene
                   else Calibration.load(self.s.calibration_file, self.geometry, size))
        except Exception:
            log.exception("Invalid calibration, ignoring it")
            cal = None
        self.set_calibration(cal)
        # The camera's dismissed spots don't belong on a clip filmed elsewhere.
        if self.scene is not None:
            self.engine.set_ignored([])
        else:
            self._load_ignored()

    def _run(self) -> None:
        # YOLO loads in this thread (it takes seconds), so the server starts instantly.
        self._load_ignored()
        self._load_yolo()
        # A clip kept playing while the models loaded: start it over, so the clean runway is
        # learned from its first seconds and not halfway through a hand placing an object.
        self.capture.rewind()
        last_seq = 0
        last_frame_at = time.monotonic()
        while not self._stop.is_set():
            seq, frame = self.capture.latest()
            if frame is None or seq == last_seq:
                if time.monotonic() - last_frame_at > 1.5:
                    self._show_placeholder()
                    last_frame_at = time.monotonic()
                self._stop.wait(0.005)
                continue
            last_seq = seq
            if self._calibrated_for != self.capture.active:
                self._load_calibration((frame.shape[1], frame.shape[0]))
                self._calibrated_for = self.capture.active
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
        # Same thread as process(), so a plain attribute is enough.
        self.engine.replay = self.capture.active.kind == "file"
        result = self.engine.process(frame)
        self.last_result = result
        status = None
        if result.learning and result.settling:
            status = f"Mișcare pe pistă: aștept să fie liberă… {result.learned_frames}/{self.engine.params.reference_frames}"
        elif result.learning:
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
            # False alarms dismissed on a clip's scene stay with that clip, not the camera.
            if isinstance(ev, IgnoredZones) and self.scene is None:
                self._save_ignored(ev.zones)
                continue
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

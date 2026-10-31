"""Pure frame-in / events-out logic, without threads or HTTP, so it can be tested
on synthetic frames and reused for uploaded clips and `evaluate.py`.
Calls from other threads (reset, calibration, closing incidents) are queued and
applied at the start of the next `process()`."""

import base64
import math
import queue
from dataclasses import dataclass, field, fields
from typing import Protocol

import numpy as np

from app.services.risk import assess_risk
from app.vision.annotate import crop_jpeg
from app.vision.calibration import Calibration
from app.vision.change_detector import ChangeDetector, ChangeParams
from app.vision.fusion import Candidate, Track, Tracker, TrackerParams, fuse
from app.vision.types import UNKNOWN_KEY, UNKNOWN_LABEL, BBox, Detection, Risk, Source


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Candidate]: ...


@dataclass
class EngineParams:
    reference_frames: int = 15
    diff_threshold: int = 30
    blur_ksize: int = 3
    min_area_px: int = 60
    max_area_px: int = 6000
    learning_rate: float = 0.02
    global_change_frac: float = 0.45
    global_reset_frames: int = 15
    persist_frames: int = 8
    large_persist_frames: int = 30
    gone_frames: int = 20
    stationary_px: int = 14
    match_px: int = 40
    dedup_px: int = 40
    yolo_every_n: int = 2
    yolo_conf: float = 0.35
    # A YOLO box with no change against the clean runway is usually the runway itself misread
    # (threshold bars taken for a label, lights): alone, it must be this sure to raise an alert.
    yolo_only_conf: float = 0.7
    unknown_critical_cm: float = 15.0

    @classmethod
    def from_settings(cls, s) -> "EngineParams":
        return cls(
            reference_frames=s.reference_frames,
            diff_threshold=s.diff_threshold,
            blur_ksize=s.blur_ksize,
            min_area_px=s.min_area_px,
            max_area_px=s.max_area_px,
            learning_rate=s.bg_learning_rate,
            global_change_frac=s.global_change_frac,
            persist_frames=s.persist_frames,
            large_persist_frames=s.large_persist_frames,
            gone_frames=s.gone_frames,
            stationary_px=s.stationary_px,
            match_px=s.match_px,
            dedup_px=s.dedup_px,
            yolo_every_n=s.yolo_every_n,
            yolo_conf=s.yolo_conf,
            yolo_only_conf=s.yolo_only_conf,
            unknown_critical_cm=s.unknown_critical_cm,
        )

    def change_params(self) -> ChangeParams:
        return ChangeParams(
            reference_frames=self.reference_frames,
            diff_threshold=self.diff_threshold,
            blur_ksize=self.blur_ksize,
            min_area_px=self.min_area_px,
            max_area_px=self.max_area_px,
            learning_rate=self.learning_rate,
            global_change_frac=self.global_change_frac,
        )

    def tracker_params(self) -> TrackerParams:
        return TrackerParams(
            persist_frames=self.persist_frames,
            large_persist_frames=self.large_persist_frames,
            gone_frames=self.gone_frames,
            stationary_px=self.stationary_px,
            match_px=self.match_px,
        )


@dataclass
class NewIncident:
    detection: Detection


@dataclass
class ObjectGone:
    incident_id: str


@dataclass
class ObjectReturned:
    incident_id: str


@dataclass
class ClassUpdated:
    incident_id: str
    source: Source
    class_key: str
    object_class: str
    confidence: float
    risk: Risk


@dataclass
class LargeObject:
    active: bool
    bbox: BBox
    sector: str | None
    label: str | None = None  # "persoană pe pistă" when the general model saw who it is


@dataclass
class ReferenceReset:
    reason: str


@dataclass
class IgnoredZones:
    """The operator marked a YOLO-only detection as a false alarm: zones to save and keep ignoring."""

    zones: list[BBox]


EngineEvent = NewIncident | ObjectGone | ObjectReturned | ClassUpdated | LargeObject | ReferenceReset | IgnoredZones


@dataclass
class EngineResult:
    frame_index: int
    tracks: list[Track]
    risks: dict[int, str]
    events: list[EngineEvent]
    learning: bool
    learned_frames: int
    global_change: bool


@dataclass
class _Site:
    """An open incident location, used for deduplication."""

    center: tuple[int, int]
    present: bool = True
    bbox: BBox | None = None
    risk: str = "de verificat"


@dataclass
class DetectionEngine:
    params: EngineParams = field(default_factory=EngineParams)
    yolo: Detector | None = None
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        self.change = ChangeDetector(self.params.change_params())
        self.tracker = Tracker(self.params.tracker_params())
        self.frame_index = 0
        self._sites: dict[str, _Site] = {}
        self._yolo_cache: list[Candidate] = []
        self._global_frames = 0
        self._intrusion_frames = 0
        self._mask_shape: tuple[int, int] | None = None
        self._commands: queue.Queue = queue.Queue()
        # Spots where YOLO alone keeps seeing something that isn't FOD (a marking, a light, a crack)
        # and the operator said "false alarm". The change detector learns such a spot as background;
        # these do the same for YOLO, and survive restarts (data/ignored_zones.json).
        self.ignored: list[BBox] = []

    def reset_reference(self) -> None:
        self._commands.put(("reset", None))

    def set_calibration(self, calibration: Calibration | None) -> None:
        self._commands.put(("calibration", calibration))

    def update_params(self, **values) -> None:
        self._commands.put(("params", values))

    def close_incident(self, incident_id: str, absorb: bool = False) -> None:
        """Operator closed the incident. `absorb`: learn that spot as background (false alarm)."""
        self._commands.put(("close", (incident_id, absorb)))

    def restore_incident(self, incident_id: str, center: tuple[int, int], bbox: BBox, risk: str, present: bool) -> None:
        """An incident still open from before a restart, so the same object doesn't raise a second one."""
        self._commands.put(("restore", (incident_id, _Site(center=center, present=present, bbox=bbox, risk=risk))))

    def set_yolo(self, yolo: Detector | None) -> None:
        self._commands.put(("yolo", yolo))

    def set_ignored(self, zones: list[BBox]) -> None:
        self._commands.put(("ignored", list(zones)))

    def _apply_commands(self) -> list[EngineEvent]:
        events: list[EngineEvent] = []
        while True:
            try:
                cmd, arg = self._commands.get_nowait()
            except queue.Empty:
                return events
            if cmd == "reset":
                self.change.reset_reference()
                self.tracker.clear()
                self._sites = {k: s for k, s in self._sites.items() if s.present}
                events.append(ReferenceReset("operator"))
            elif cmd == "calibration":
                self.calibration = arg
                self._mask_shape = None
            elif cmd == "params":
                names = {f.name for f in fields(EngineParams)}
                for k, v in arg.items():
                    if k in names:
                        setattr(self.params, k, type(getattr(self.params, k))(v))
                self.change.params = self.params.change_params()
                self.tracker.params = self.params.tracker_params()
                if self.yolo is not None and hasattr(self.yolo, "conf"):
                    self.yolo.conf = self.params.yolo_conf
            elif cmd == "close":
                incident_id, absorb = arg
                site = self._sites.pop(incident_id, None)
                for t in list(self.tracker.tracks):
                    if t.incident_id == incident_id:
                        if absorb:
                            self.change.absorb(t.cand.bbox)
                        self.tracker.drop(t)
                if absorb and site is not None and site.bbox is not None:
                    self.change.absorb(site.bbox)
                    self.ignored.append(site.bbox)
                    events.append(IgnoredZones(list(self.ignored)))
            elif cmd == "restore":
                incident_id, site = arg
                self._sites.setdefault(incident_id, site)
            elif cmd == "yolo":
                self.yolo = arg
            elif cmd == "ignored":
                self.ignored = arg

    def process(self, frame: np.ndarray) -> EngineResult:
        events = self._apply_commands()
        self.frame_index += 1
        h, w = frame.shape[:2]

        if self._mask_shape != (h, w):
            self.change.set_mask(self.calibration.mask(w, h) if self.calibration else None)
            self._mask_shape = (h, w)

        protect = [t.cand.bbox for t in self.tracker.tracks if t.confirmed and not t.large]
        change = self.change.detect(frame, protect=protect)

        if change.global_change:
            self._global_frames += 1
            if self._global_frames >= self.params.global_reset_frames:
                self._global_frames = 0
                self.change.reset_reference()
                self.tracker.tracks = [t for t in self.tracker.tracks if t.confirmed and not t.large]
                events.append(ReferenceReset("lumina"))
        else:
            self._global_frames = 0

        yolo_boxes: list[Candidate] = []
        if self.yolo is not None and not change.global_change:
            if self.frame_index % max(1, self.params.yolo_every_n) == 1 or self.params.yolo_every_n <= 1:
                # A detector that knows about streams reuses its slow model's boxes between frames.
                stream = getattr(self.yolo, "detect_stream", None)
                boxes = stream(frame, id(self)) if stream else self.yolo.detect(frame)
                self._yolo_cache = [c for c in boxes if self._in_runway(c.center)]
            yolo_boxes = self._yolo_cache

        if change.learning or change.global_change:
            # Without a reference, only YOLO can see objects.
            cands = list(yolo_boxes)
        else:
            cands = fuse(change.regions, yolo_boxes)
        cands = [c for c in cands if c.source != "yolo" or
                 (c.confidence >= self.params.yolo_only_conf and not self._is_ignored(c.bbox))]

        # An arm or a vehicle coming in over the edge: wait until it is gone, so the hand that
        # places an object is never reported as the object. Limited, so something left at
        # the edge for good can't block the alerts.
        if change.intrusion or any(r.edge for r in change.regions):
            self._intrusion_frames += 1
        else:
            self._intrusion_frames = 0
        hold = 0 < self._intrusion_frames <= self.params.large_persist_frames * 5
        for ev in self.tracker.update(cands, hold=hold):
            events.extend(self._handle(ev.kind, ev.track, frame))

        risks = {}
        for t in self.tracker.tracks:
            if t.confirmed and t.incident_id in self._sites:
                risks[t.id] = self._sites[t.incident_id].risk

        return EngineResult(
            frame_index=self.frame_index,
            tracks=list(self.tracker.tracks),
            risks=risks,
            events=events,
            learning=change.learning,
            learned_frames=self.change.learned_frames,
            global_change=change.global_change,
        )

    def _is_ignored(self, b: BBox) -> bool:
        return any(b.iou(z) >= 0.3 or z.contains(*b.center) for z in self.ignored)

    def _in_runway(self, center: tuple[int, int]) -> bool:
        return self.calibration is None or self.calibration.contains(*center)

    def _sector(self, center: tuple[int, int]) -> str | None:
        if self.calibration is None:
            return None
        return self.calibration.pixel_to_runway(*center).sector

    def _handle(self, kind: str, t: Track, frame: np.ndarray) -> list[EngineEvent]:
        if kind == "large_on":
            label = t.cand.label if t.cand.class_key != UNKNOWN_KEY else None
            return [LargeObject(True, t.cand.bbox, self._sector(t.cand.center), label)]
        if kind == "large_off":
            return [LargeObject(False, t.cand.bbox, self._sector(t.cand.center))]
        if kind == "gone":
            site = self._sites.get(t.incident_id or "")
            others = any(o is not t and o.incident_id == t.incident_id for o in self.tracker.tracks)
            if site is not None and site.present and not others:
                site.present = False
                return [ObjectGone(t.incident_id)]
            return []
        if kind == "class_upgrade":
            site = self._sites.get(t.incident_id or "")
            if site is None or t.best_yolo is None:
                return []
            det = self._build_detection(t, frame)
            site.risk = det.risk
            return [ClassUpdated(t.incident_id, det.source, det.class_key, det.object_class, det.confidence, det.risk)]
        if kind == "confirmed":
            # Deduplication: an object where an incident is already open joins that incident.
            for incident_id, site in self._sites.items():
                if math.dist(site.center, t.cand.center) <= self.params.dedup_px:
                    t.incident_id = incident_id
                    if not site.present:
                        site.present = True
                        return [ObjectReturned(incident_id)]
                    return []
            det = self._build_detection(t, frame)
            t.incident_id = det.id
            self._sites[det.id] = _Site(center=t.cand.center, bbox=t.cand.bbox, risk=det.risk)
            return [NewIncident(det)]
        return []

    def _build_detection(self, t: Track, frame: np.ndarray) -> Detection:
        c = t.cand
        y = t.best_yolo
        if y is not None:
            source: Source = "both" if t.saw_change else "yolo"
            class_key, label, conf = y.class_key, y.label, y.confidence
        else:
            source, class_key, label, conf = "change", UNKNOWN_KEY, UNKNOWN_LABEL, c.confidence
        distance = offset = size = None
        sector = None
        if self.calibration is not None:
            pos = self.calibration.pixel_to_runway(*c.center)
            distance, offset, sector = round(pos.distance_m, 1), round(pos.offset_m, 1), pos.sector
            size = round(self.calibration.size_cm(c.bbox), 1)
        risk = assess_risk(class_key, size, self.params.unknown_critical_cm)
        crop = crop_jpeg(frame, c.bbox)
        return Detection(
            source=source,
            object_class=label,
            class_key=class_key,
            confidence=round(float(conf), 3),
            bbox=c.bbox,
            center_px=c.center,
            runway_distance_m=distance,
            centerline_offset_m=offset,
            sector=sector,
            size_estimate_cm=size,
            risk=risk,
            crop_jpeg_b64=base64.b64encode(crop).decode("ascii") if crop else None,
        )

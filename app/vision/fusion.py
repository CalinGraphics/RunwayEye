"""Fusion of YOLO boxes with change regions, persistence and tracking.

A candidate becomes an alert only after it stays in (almost) the same place for
`persist_frames` frames, so a hand placing an object, a bird or a passing
vehicle never raises an alarm.
"""

import math
from dataclasses import dataclass, field
from typing import Literal

from app.vision.change_detector import Region
from app.vision.types import UNKNOWN_KEY, UNKNOWN_LABEL, BBox, Source


@dataclass
class Candidate:
    """One object seen in one frame (before persistence)."""

    bbox: BBox
    center: tuple[int, int]
    area: int
    source: Source
    class_key: str = UNKNOWN_KEY
    label: str = UNKNOWN_LABEL
    confidence: float = 0.0
    large: bool = False


def fuse(regions: list[Region], yolo_boxes: list[Candidate], iou_min: float = 0.1) -> list[Candidate]:
    """Merge change regions with YOLO boxes.

    - region overlapping a YOLO box -> YOLO class, source "both"
    - region without a YOLO box -> "obiect necunoscut", source "change"
    - YOLO box without a region -> keeps source "yolo"
    """
    used: set[int] = set()
    out: list[Candidate] = []
    for r in regions:
        if r.large:
            # A vehicle / person / hand: never an FOD class, even if it holds one.
            for i, y in enumerate(yolo_boxes):
                if r.bbox.contains(*y.center):
                    used.add(i)
            out.append(Candidate(r.bbox, r.center, r.area, "change", confidence=r.score, large=True))
            continue
        best_i, best_score = -1, 0.0
        for i, y in enumerate(yolo_boxes):
            if i in used:
                continue
            iou = r.bbox.iou(y.bbox)
            inside = y.bbox.contains(*r.center) or r.bbox.contains(*y.center)
            score = max(iou, 0.5 if inside else 0.0)
            if (iou >= iou_min or inside) and score > best_score:
                best_i, best_score = i, score
        if best_i >= 0:
            used.add(best_i)
            y = yolo_boxes[best_i]
            out.append(Candidate(r.bbox, r.center, r.area, "both", y.class_key, y.label, y.confidence))
        else:
            out.append(Candidate(r.bbox, r.center, r.area, "change", confidence=r.score))
    for i, y in enumerate(yolo_boxes):
        if i not in used:
            out.append(y)
    return out


@dataclass
class TrackerParams:
    persist_frames: int = 8
    large_persist_frames: int = 30
    gone_frames: int = 20
    stationary_px: int = 14
    match_px: int = 40
    tentative_max_misses: int = 2


@dataclass
class Track:
    id: int
    cand: Candidate
    anchor: tuple[int, int]
    anchor_area: int
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    incident_id: str | None = None
    best_yolo: Candidate | None = None
    saw_change: bool = False
    upgraded: bool = False

    @property
    def large(self) -> bool:
        return self.cand.large


TrackEventKind = Literal["confirmed", "gone", "large_on", "large_off", "class_upgrade"]


@dataclass
class TrackEvent:
    kind: TrackEventKind
    track: Track


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Tracker:
    params: TrackerParams = field(default_factory=TrackerParams)
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1
    _hold: bool = False

    def clear(self) -> None:
        self.tracks = []

    def drop(self, track: Track) -> None:
        self.tracks = [t for t in self.tracks if t is not track]

    def _radius(self, t: Track) -> float:
        b = t.cand.bbox
        return max(self.params.match_px, 0.5 * max(b.w, b.h))

    def update(self, cands: list[Candidate], hold: bool = False) -> list[TrackEvent]:
        """`hold`: someone reaches into the picture, so nothing new is confirmed until they leave."""
        p = self.params
        self._hold = hold
        events: list[TrackEvent] = []

        # Greedy nearest-neighbour matching within the same kind (small / large).
        pairs = []
        for ti, t in enumerate(self.tracks):
            for ci, c in enumerate(cands):
                if c.large != t.large:
                    continue
                ref = t.anchor if t.confirmed else t.cand.center
                d = _dist(c.center, ref)
                if d <= self._radius(t):
                    pairs.append((d, ti, ci))
        pairs.sort()
        matched_t: set[int] = set()
        matched_c: set[int] = set()
        for _, ti, ci in pairs:
            if ti in matched_t or ci in matched_c:
                continue
            matched_t.add(ti)
            matched_c.add(ci)
            self._update_track(self.tracks[ti], cands[ci], events)

        survivors: list[Track] = []
        for ti, t in enumerate(self.tracks):
            if ti in matched_t:
                survivors.append(t)
                continue
            t.misses += 1
            if t.confirmed:
                if t.misses >= p.gone_frames:
                    events.append(TrackEvent("large_off" if t.large else "gone", t))
                    continue
            elif t.misses > p.tentative_max_misses:
                continue
            survivors.append(t)

        for ci, c in enumerate(cands):
            if ci in matched_c:
                continue
            t = Track(id=self._next_id, cand=c, anchor=c.center, anchor_area=c.area)
            self._next_id += 1
            self._note_sources(t, c)
            survivors.append(t)
            self._maybe_confirm(t, events)
        self.tracks = survivors
        return events

    def _note_sources(self, t: Track, c: Candidate) -> None:
        if c.source in ("change", "both"):
            t.saw_change = True
        if c.source in ("yolo", "both") and (t.best_yolo is None or c.confidence > t.best_yolo.confidence):
            t.best_yolo = c

    def _update_track(self, t: Track, c: Candidate, events: list[TrackEvent]) -> None:
        p = self.params
        had_class = t.best_yolo is not None
        t.cand = c
        t.misses = 0
        self._note_sources(t, c)
        if t.confirmed:
            if not had_class and t.best_yolo is not None and not t.upgraded:
                t.upgraded = True
                events.append(TrackEvent("class_upgrade", t))
            return
        moved = _dist(c.center, t.anchor) > p.stationary_px
        ratio = c.area / max(1, t.anchor_area)
        if moved or not (0.5 <= ratio <= 2.0) or (self._hold and not t.large):
            # Moving or changing shape (e.g. a hand): restart the persistence count.
            t.anchor, t.anchor_area, t.hits = c.center, c.area, 1
        else:
            t.hits += 1
        self._maybe_confirm(t, events)

    def _maybe_confirm(self, t: Track, events: list[TrackEvent]) -> None:
        needed = self.params.large_persist_frames if t.large else self.params.persist_frames
        if not t.confirmed and t.hits >= needed:
            t.confirmed = True
            events.append(TrackEvent("large_on" if t.large else "confirmed", t))

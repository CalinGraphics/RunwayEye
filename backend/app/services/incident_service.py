"""Incident life cycle: nou -> confirmat / echipă trimisă -> rezolvat, or alarmă falsă."""

import base64
import logging
import threading
from datetime import datetime
from typing import Callable

from pydantic import BaseModel

from app.services.events import EventBus
from app.config import get_settings
from app.services.risk import RISK_ORDER, severity
from app.services.supabase_service import DataStore
from app.vision.engine import (
    ClassUpdated,
    LargeObject,
    NewIncident,
    ObjectGone,
    ObjectReturned,
    ReferenceReset,
)
from app.vision.types import BBox, Detection, new_id, utcnow

log = logging.getLogger(__name__)

STATUS_LABELS = {
    "new": "Nou",
    "confirmed": "Confirmat",
    "dispatched": "Echipă trimisă",
    "resolved": "Rezolvat",
    "false_alarm": "Alarmă falsă",
}
OPEN_STATUSES = {"new", "confirmed", "dispatched"}

ACTIONS = {
    # action: (allowed from, new status, timestamp field)
    "confirm": ({"new"}, "confirmed", "confirmed_at"),
    "dispatch": ({"new", "confirmed"}, "dispatched", "dispatched_at"),
    "ack": ({"new", "confirmed", "dispatched"}, "dispatched", "team_ack_at"),
    "resolve": ({"new", "confirmed", "dispatched"}, "resolved", "resolved_at"),
    "false_alarm": ({"new", "confirmed", "dispatched"}, "false_alarm", "false_alarm_at"),
}
ACTION_LABELS = {
    "detected": "Detectat",
    "confirm": "Confirmat de operator",
    "dispatch": "Echipă trimisă",
    "ack": "Echipa a preluat alerta",
    "resolve": "Obiect eliminat",
    "false_alarm": "Marcat ca alarmă falsă",
    "object_gone": "Obiect dispărut din imagine",
    "object_returned": "Obiectul a reapărut",
    "class_updated": "Clasă actualizată de YOLO",
    "ai_description": "Descriere AI",
}


class InvalidAction(Exception):
    pass


class Incident(BaseModel):
    id: str
    camera_id: str
    detected_at: datetime
    status: str = "new"
    source: str
    object_class: str
    class_key: str
    confidence: float
    risk: str
    bbox: BBox
    center_px: tuple[int, int]
    sector: str | None = None
    runway_distance_m: float | None = None
    centerline_offset_m: float | None = None
    size_estimate_cm: float | None = None
    object_present: bool = True
    confirmed_at: datetime | None = None
    dispatched_at: datetime | None = None
    team_ack_at: datetime | None = None
    resolved_at: datetime | None = None
    false_alarm_at: datetime | None = None
    updated_at: datetime | None = None
    ai_description: str | None = None
    fake: bool = False

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def closed_at(self) -> datetime | None:
        return self.resolved_at or self.false_alarm_at

    def display_status(self) -> str:
        if self.status == "dispatched" and self.team_ack_at:
            return "Echipă în drum"
        return STATUS_LABELS.get(self.status, self.status)

    def severity_info(self) -> dict | None:
        """How serious the object is for an aircraft (backend/app/services/severity.json)."""
        return severity(self.class_key, self.size_estimate_cm, get_settings().unknown_critical_cm)

    def public(self) -> dict:
        d = self.model_dump(mode="json")
        d["severity"] = self.severity_info()
        d["status_label"] = self.display_status()
        d["is_open"] = self.is_open
        d["crop_url"] = f"/incidents/{self.id}/crop.jpg"
        d["snapshot_url"] = f"/incidents/{self.id}/snapshot.jpg"
        return d


class IncidentService:
    def __init__(self, store: DataStore, bus: EventBus, camera_id: str, describer=None):
        self.store = store
        self.describer = describer
        self.bus = bus
        self.camera_id = camera_id
        self.on_close: Callable[[str, bool], None] | None = None
        self.large_object: dict | None = None
        self._lock = threading.RLock()
        self._open: dict[str, Incident] = {}
        # Incidents of other cameras (pulled from Supabase) are in the history, not on this Command Center.
        for data in store.local.load_incidents(
            "WHERE status IN ('new','confirmed','dispatched') AND json_extract(data, '$.camera_id')=?", (camera_id,)
        ):
            inc = Incident.model_validate(data)
            self._open[inc.id] = inc

    def close_fake(self) -> None:
        """Test detections left open from a run with DEMO_FAKE_DETECTIONS=true."""
        with self._lock:
            fake = [inc for inc in self._open.values() if inc.fake]
            for inc in fake:
                inc.status = "false_alarm"
                inc.false_alarm_at = utcnow()
                self._open.pop(inc.id)
                self._save(inc, "false_alarm", note="detecție de test")

    def open_incidents(self) -> list[Incident]:
        with self._lock:
            newest = sorted(self._open.values(), key=lambda i: i.detected_at, reverse=True)
        return sorted(newest, key=lambda i: RISK_ORDER.get(i.risk, len(RISK_ORDER)))

    def get(self, incident_id: str) -> Incident | None:
        with self._lock:
            if incident_id in self._open:
                return self._open[incident_id]
        rows = self.store.local.load_incidents("WHERE id=?", (incident_id,), limit=1)
        return Incident.model_validate(rows[0]) if rows else None

    def history(self, status: str = "", risk: str = "", sector: str = "", since: str = "", limit: int = 300) -> list[Incident]:
        clauses, params = [], []
        if status == "open":
            clauses.append("status IN ('new','confirmed','dispatched')")
        elif status:
            clauses.append("status=?")
            params.append(status)
        if risk:
            clauses.append("risk=?")
            params.append(risk)
        if sector:
            clauses.append("sector LIKE ?")
            params.append(f"{sector.upper()}%")
        if since:
            clauses.append("detected_at>=?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return [Incident.model_validate(d) for d in self.store.local.load_incidents(where, tuple(params), limit)]

    def events(self, incident_id: str) -> list[dict]:
        evs = self.store.local.events_for(incident_id)
        for e in evs:
            e["label"] = ACTION_LABELS.get(e["action"], e["action"])
        return evs

    def image(self, incident_id: str, kind: str) -> bytes | None:
        return self.store.local.read_image(incident_id, kind)

    def remote_image_url(self, incident_id: str, kind: str) -> str | None:
        """Public Supabase Storage URL of an image taken by another camera."""
        rows = self.store.local.load_incidents("WHERE id=?", (incident_id,), limit=1)
        return (rows[0].get("remote_images") or {}).get(kind) if rows else None

    def _save(self, inc: Incident, action: str, note: str | None = None, publish: str = "incident.updated") -> None:
        inc.updated_at = utcnow()
        self.store.local.upsert_incident(inc.model_dump(mode="json"))
        self.store.local.add_event(
            {"id": new_id(), "incident_id": inc.id, "at": inc.updated_at.isoformat(), "action": action, "note": note}
        )
        self.store.changed()
        self.bus.publish(publish, inc.public())

    def create_from_detection(self, det: Detection, snapshot_jpeg: bytes | None = None, fake: bool = False) -> Incident:
        inc = Incident(
            id=det.id,
            camera_id=self.camera_id,
            detected_at=det.timestamp,
            source=det.source,
            object_class=det.object_class,
            class_key=det.class_key,
            confidence=det.confidence,
            risk=det.risk,
            bbox=det.bbox,
            center_px=det.center_px,
            sector=det.sector,
            runway_distance_m=det.runway_distance_m,
            centerline_offset_m=det.centerline_offset_m,
            size_estimate_cm=det.size_estimate_cm,
            fake=fake,
        )
        crop = base64.b64decode(det.crop_jpeg_b64) if det.crop_jpeg_b64 else None
        if crop:
            self.store.local.save_image(inc.id, "crop", crop)
        self.store.local.save_image(inc.id, "snapshot", snapshot_jpeg)
        with self._lock:
            self._open[inc.id] = inc
            self._save(inc, "detected", note=f"{det.object_class} · {det.source}", publish="incident.new")
        log.info("New incident %s: %s sector=%s risk=%s", inc.id, inc.object_class, inc.sector, inc.risk)
        if self.describer is not None and crop and not fake:
            self.describer.describe_later(crop, inc.object_class, lambda text: self.set_description(inc.id, text))
        return inc

    def set_description(self, incident_id: str, text: str) -> None:
        with self._lock:
            inc = self._open.get(incident_id) or self.get(incident_id)
            if inc is None:
                return
            inc.ai_description = text
            self._save(inc, "ai_description")

    def act(self, incident_id: str, action: str) -> Incident:
        if action not in ACTIONS:
            raise InvalidAction("Acțiune necunoscută")
        allowed, new_status, ts_field = ACTIONS[action]
        with self._lock:
            inc = self._open.get(incident_id)
            if inc is None:
                existing = self.get(incident_id)
                if existing is None:
                    raise KeyError(incident_id)
                raise InvalidAction(f"Incidentul e deja închis ({existing.display_status().lower()})")
            if action == "ack" and inc.team_ack_at is not None:
                return inc  # double tap on the phone
            if inc.status not in allowed:
                raise InvalidAction(f"Nu se poate din starea „{inc.display_status()}”")
            now = utcnow()
            inc.status = new_status
            setattr(inc, ts_field, now)
            if action == "ack" and inc.dispatched_at is None:
                inc.dispatched_at = now
            if not inc.is_open:
                self._open.pop(inc.id, None)
            self._save(inc, action)
        if not inc.is_open and self.on_close:
            self.on_close(inc.id, action == "false_alarm")
        return inc

    def handle_engine_event(self, ev, snapshot_jpeg: bytes | None = None) -> None:
        if isinstance(ev, NewIncident):
            self.create_from_detection(ev.detection, snapshot_jpeg)
        elif isinstance(ev, (ObjectGone, ObjectReturned)):
            with self._lock:
                inc = self._open.get(ev.incident_id)
                if inc is None:
                    return
                inc.object_present = isinstance(ev, ObjectReturned)
                self._save(inc, "object_returned" if inc.object_present else "object_gone")
        elif isinstance(ev, ClassUpdated):
            with self._lock:
                inc = self._open.get(ev.incident_id)
                if inc is None:
                    return
                inc.source, inc.class_key, inc.object_class = ev.source, ev.class_key, ev.object_class
                inc.confidence, inc.risk = ev.confidence, ev.risk
                self._save(inc, "class_updated", note=ev.object_class)
        elif isinstance(ev, LargeObject):
            self.large_object = (
                {"active": True, "sector": ev.sector, "label": ev.label, "bbox": ev.bbox.model_dump(),
                 "at": utcnow().isoformat()}
                if ev.active
                else None
            )
            self.bus.publish("large_object", {"active": ev.active, "sector": ev.sector, "label": ev.label})
        elif isinstance(ev, ReferenceReset):
            self.bus.publish("reference", {"reason": ev.reason})

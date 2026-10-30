"""Incident storage: always local first (SQLite + JPEGs on disk), Supabase synced in the background.

The local copy is what the app reads, so detection, the dashboard and the team
page keep working without internet. When Supabase is enabled, a background
thread pushes every unsynced row and image; failures stay queued and are retried.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

INCIDENT_COLUMNS = [
    "id", "camera_id", "detected_at", "status", "source", "object_class", "class_key",
    "confidence", "risk", "bbox", "center_px", "sector", "runway_distance_m",
    "centerline_offset_m", "size_estimate_cm", "snapshot_url", "crop_url", "object_present",
    "confirmed_at", "dispatched_at", "team_ack_at", "resolved_at", "false_alarm_at", "updated_at",
]


class LocalStore:
    def __init__(self, data_dir: Path):
        self.images_dir = data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(data_dir / "runwayeye.db", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY,
                    detected_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    risk TEXT,
                    sector TEXT,
                    class_key TEXT,
                    data TEXT NOT NULL,
                    synced INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS incidents_detected_at ON incidents(detected_at);
                CREATE TABLE IF NOT EXISTS incident_events (
                    id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    note TEXT,
                    synced INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS events_incident ON incident_events(incident_id);
                CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, synced INTEGER DEFAULT 0);
                """
            )
            self._db.commit()

    def image_path(self, incident_id: str, kind: str) -> Path:
        return self.images_dir / f"{incident_id}_{kind}.jpg"

    def save_image(self, incident_id: str, kind: str, data: bytes | None) -> None:
        if data:
            self.image_path(incident_id, kind).write_bytes(data)

    def read_image(self, incident_id: str, kind: str) -> bytes | None:
        p = self.image_path(incident_id, kind)
        return p.read_bytes() if p.exists() else None

    def upsert_incident(self, inc: dict) -> None:
        with self._lock:
            self._db.execute(
                """INSERT INTO incidents (id, detected_at, status, risk, sector, class_key, data, synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                   ON CONFLICT(id) DO UPDATE SET status=excluded.status, risk=excluded.risk,
                     sector=excluded.sector, class_key=excluded.class_key, data=excluded.data, synced=0""",
                (inc["id"], inc["detected_at"], inc["status"], inc["risk"], inc.get("sector"),
                 inc.get("class_key"), json.dumps(inc, ensure_ascii=False)),
            )
            self._db.commit()

    def add_event(self, event: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO incident_events (id, incident_id, at, action, note, synced) VALUES (?,?,?,?,?,0)",
                (event["id"], event["incident_id"], event["at"], event["action"], event.get("note")),
            )
            self._db.commit()

    def load_incidents(self, where: str = "", params: tuple = (), limit: int = 500) -> list[dict]:
        sql = f"SELECT data FROM incidents {where} ORDER BY detected_at DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, (*params, limit)).fetchall()
        return [json.loads(r["data"]) for r in rows]

    def events_for(self, incident_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, incident_id, at, action, note FROM incident_events WHERE incident_id=? ORDER BY at",
                (incident_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_kv(self, key: str, value: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO kv (key, value, synced) VALUES (?, ?, 0) ON CONFLICT(key) DO UPDATE SET value=excluded.value, synced=0",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._db.commit()

    def unsynced(self, limit: int = 50) -> tuple[list[dict], list[dict], list[tuple[str, dict]]]:
        with self._lock:
            incs = [json.loads(r["data"]) for r in self._db.execute(
                "SELECT data FROM incidents WHERE synced=0 ORDER BY detected_at LIMIT ?", (limit,))]
            # Only events whose incident is already in Supabase (foreign key).
            evs = [dict(r) for r in self._db.execute(
                """SELECT id, incident_id, at, action, note FROM incident_events
                   WHERE synced=0 AND incident_id IN (SELECT id FROM incidents WHERE synced=1)
                   ORDER BY at LIMIT ?""",
                (limit,))]
            kvs = [(r["key"], json.loads(r["value"])) for r in self._db.execute("SELECT key, value FROM kv WHERE synced=0")]
        return incs, evs, kvs

    def mark_kv_synced(self, key: str) -> None:
        with self._lock:
            self._db.execute("UPDATE kv SET synced=1 WHERE key=?", (key,))
            self._db.commit()

    def pending_count(self) -> int:
        with self._lock:
            a = self._db.execute("SELECT COUNT(*) FROM incidents WHERE synced=0").fetchone()[0]
            b = self._db.execute("SELECT COUNT(*) FROM incident_events WHERE synced=0").fetchone()[0]
        return a + b

    def mark_synced(self, table: str, ids: list[str], key_col: str = "id", versions: dict[str, str] | None = None) -> None:
        if not ids:
            return
        with self._lock:
            for i in ids:
                if versions is not None:
                    # Only mark if the row was not changed again while uploading.
                    self._db.execute(f"UPDATE {table} SET synced=1 WHERE {key_col}=? AND data=?", (i, versions[i]))
                else:
                    self._db.execute(f"UPDATE {table} SET synced=1 WHERE {key_col}=?", (i,))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _iso(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


class SupabaseSync:
    """Background uploader. Never raises into the app; status is shown in the status bar."""

    def __init__(self, store: LocalStore, url: str, key: str, bucket: str, camera: dict):
        self.store = store
        self.url, self.key, self.bucket = url, key, bucket
        self.camera = camera
        self.state = "pornire"
        self.error: str | None = None
        self._client = None
        self._uploaded_images: set[str] = set()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = threading.Thread(target=self._run, name="supabase-sync", daemon=True)

    def start(self) -> None:
        self.store.set_kv(f"camera:{self.camera['id']}", self.camera)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=3)

    def poke(self) -> None:
        self._wake.set()

    def _connect(self):
        if self._client is None:
            from supabase import ClientOptions, create_client

            self._client = create_client(
                self.url, self.key, options=ClientOptions(postgrest_client_timeout=10, storage_client_timeout=15)
            )
        return self._client

    def _run(self) -> None:
        delay = 2.0
        while not self._stop.is_set():
            try:
                self._sync_once()
                self.state, self.error, delay = "conectat", None, 3.0
            except Exception as exc:  # network down, wrong key, missing table...
                self.state, self.error = "eroare", str(exc)[:200]
                log.warning("Supabase sync failed: %s", exc)
                delay = min(60.0, delay * 2)
            self._wake.wait(delay)
            self._wake.clear()

    def _upload_image(self, client, incident_id: str, kind: str) -> str | None:
        data = self.store.read_image(incident_id, kind)
        if not data:
            return None
        path = f"{incident_id}/{kind}.jpg"
        if path not in self._uploaded_images:
            client.storage.from_(self.bucket).upload(path, data, {"content-type": "image/jpeg", "upsert": "true"})
            self._uploaded_images.add(path)
        return client.storage.from_(self.bucket).get_public_url(path)

    def _sync_once(self) -> None:
        incs, evs, kvs = self.store.unsynced()
        if not (incs or evs or kvs) and self._client is not None:
            return
        client = self._connect()
        for key, value in kvs:
            if key.startswith("camera:"):
                client.table("cameras").upsert(
                    {"id": value["id"], "name": value.get("name"), "source": value.get("source"),
                     "calibration": value.get("calibration")}
                ).execute()
                self.store.mark_kv_synced(key)
        if incs:
            rows, versions = [], {}
            for inc in incs:
                versions[inc["id"]] = json.dumps(inc, ensure_ascii=False)
                row = {k: inc.get(k) for k in INCIDENT_COLUMNS}
                row["crop_url"] = self._upload_image(client, inc["id"], "crop")
                row["snapshot_url"] = self._upload_image(client, inc["id"], "snapshot")
                for k in ("detected_at", "confirmed_at", "dispatched_at", "team_ack_at", "resolved_at",
                          "false_alarm_at", "updated_at"):
                    row[k] = _iso(row[k])
                rows.append(row)
            client.table("incidents").upsert(rows).execute()
            self.store.mark_synced("incidents", list(versions), versions=versions)
        if evs:
            client.table("incident_events").upsert(evs).execute()
            self.store.mark_synced("incident_events", [e["id"] for e in evs])


class DataStore:
    """Facade used by the incident service."""

    def __init__(self, settings, camera: dict):
        self.settings = settings
        self.local = LocalStore(settings.data_path)
        self.sync: SupabaseSync | None = None
        if settings.supabase_active:
            self.sync = SupabaseSync(
                self.local, settings.supabase_url, settings.supabase_key, settings.supabase_bucket, camera
            )

    def start(self) -> None:
        if self.sync:
            self.sync.start()

    def stop(self) -> None:
        if self.sync:
            self.sync.stop()
        self.local.close()

    def changed(self) -> None:
        if self.sync:
            self.sync.poke()

    def update_camera(self, camera: dict) -> None:
        self.local.set_kv(f"camera:{camera['id']}", camera)
        self.changed()

    def status(self) -> dict:
        s = self.settings
        if not s.enable_supabase:
            return {"state": "oprit", "label": "Supabase oprit", "pending": 0}
        if s.use_local_data:
            return {"state": "local", "label": "date locale", "pending": 0}
        if self.sync is None:
            return {"state": "eroare", "label": "Supabase: lipsesc URL / cheia", "pending": 0}
        pending = self.local.pending_count()
        label = {"conectat": "Supabase conectat", "eroare": "Supabase indisponibil", "pornire": "Supabase…"}.get(
            self.sync.state, self.sync.state
        )
        if pending:
            label += f" · {pending} în coadă"
        return {"state": self.sync.state, "label": label, "pending": pending, "error": self.sync.error}

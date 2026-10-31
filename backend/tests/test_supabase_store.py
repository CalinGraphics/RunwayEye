import base64
import json

from app.services.supabase_service import LocalStore, key_role


def _jwt(role: str) -> str:
    body = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
    return f"eyJhbGciOiJIUzI1NiJ9.{body}.sig"


def _row(**kw) -> dict:
    row = {"id": "b9c6c7a2-1111-4a5b-9c2d-000000000001", "camera_id": "cam-02", "detected_at": "2026-10-31T10:00:00+00:00",
           "status": "new", "source": "yolo", "object_class": "bolț", "class_key": "bolt", "confidence": 0.9,
           "risk": "critic", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}, "center_px": [2, 4],
           "updated_at": "2026-10-31T10:00:01+00:00", "crop_url": "https://x/crop.jpg", "snapshot_url": None}
    row.update(kw)
    return row


def test_key_role():
    assert key_role("sb_secret_abc") == "secret"
    assert key_role("sb_publishable_abc") == "public"
    assert key_role(_jwt("service_role")) == "secret"
    assert key_role(_jwt("anon")) == "public"
    assert key_role("nonsense") == "?"


def test_remote_incident_lands_in_history_without_being_pushed_back(tmp_path):
    store = LocalStore(tmp_path)
    assert store.upsert_remote(_row())
    [inc] = store.load_incidents()
    assert inc["camera_id"] == "cam-02"
    assert inc["remote_images"]["crop"] == "https://x/crop.jpg"
    assert store.pending_count() == 0
    store.close()


def test_local_unsynced_change_wins_over_pull(tmp_path):
    store = LocalStore(tmp_path)
    store.upsert_remote(_row())
    # The operator here resolves it; before the push, an older copy arrives from Supabase.
    store.upsert_incident({**store.load_incidents()[0], "status": "resolved"})
    assert not store.upsert_remote(_row(status="confirmed"))
    assert store.load_incidents()[0]["status"] == "resolved"
    assert store.pending_count() == 1
    store.close()


def test_meta_roundtrip(tmp_path):
    store = LocalStore(tmp_path)
    assert store.get_meta("pulled_until") is None
    store.set_meta("pulled_until", "2026-10-31T10:00:01+00:00")
    assert store.get_meta("pulled_until") == "2026-10-31T10:00:01+00:00"
    store.close()

"""API smoke tests: pages render, incident life cycle (operator + team), errors in Romanian."""

import os
import random

import pytest

os.environ.update(
    {"VIDEO_SOURCE": "none", "FALLBACK_VIDEO": "", "ENABLE_YOLO": "false", "ENABLE_SUPABASE": "false",
     "DEMO_FAKE_DETECTIONS": "false"}
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
    get_settings.cache_clear()


def _new_incident(client):
    from app.services.fake_detections import make_fake_detection
    from app.vision.pipeline import runway_geometry

    s = client.app.state.settings
    det = make_fake_detection(runway_geometry(s), s.unknown_critical_cm, random.Random(3))
    return client.app.state.incidents.create_from_detection(det, b"\xff\xd8fake")


@pytest.mark.parametrize("path", ["/", "/team", "/incidents", "/calibrate", "/upload", "/health", "/api/status", "/qr.svg"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200, r.text


def test_incident_lifecycle_team_ack_and_resolve(client):
    inc = _new_incident(client)
    open_ = client.get("/api/incidents").json()
    assert [i["id"] for i in open_] == [inc.id]
    assert open_[0]["status_label"] == "Nou"

    r = client.post(f"/incidents/{inc.id}/ack")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "dispatched" and body["team_ack_at"]
    assert body["status_label"] == "Echipă în drum"

    # A second tap on the phone is harmless.
    assert client.post(f"/incidents/{inc.id}/ack").status_code == 200

    r = client.post(f"/incidents/{inc.id}/resolve")
    assert r.json()["status"] == "resolved"
    assert client.get("/api/incidents").json() == []

    r = client.post(f"/incidents/{inc.id}/confirm")
    assert r.status_code == 409
    assert "închis" in r.json()["detail"]

    assert client.get(f"/incidents/{inc.id}/crop.jpg").status_code == 200
    detail = client.get(f"/incidents/{inc.id}")
    assert "Echipa a preluat alerta" in detail.text and "Obiect eliminat" in detail.text


def test_operator_flow_and_false_alarm(client):
    inc = _new_incident(client)
    assert client.post(f"/incidents/{inc.id}/confirm").json()["status"] == "confirmed"
    assert client.post(f"/incidents/{inc.id}/dispatch").json()["status_label"] == "Echipă trimisă"
    assert client.post(f"/incidents/{inc.id}/false-alarm").json()["status"] == "false_alarm"
    hist = client.get("/incidents?period=all&status=false_alarm", headers={"HX-Request": "true"})
    assert "Alarmă falsă" in hist.text


def test_unknown_incident_and_action(client):
    assert client.post("/incidents/nope/resolve").status_code == 404
    inc = _new_incident(client)
    assert client.post(f"/incidents/{inc.id}/explode").status_code == 404


def test_calibration_roundtrip(client):
    # No frame yet without a video source.
    r = client.post("/calibrate", json={"corners": [[0, 0], [0, 100], [600, 100], [600, 0]]})
    assert r.status_code == 409
    assert client.post("/calibrate/reset").json()["ok"]


def test_tuning_in_demo_mode(client):
    r = client.post("/tuning", json={"diff_threshold": 42})
    assert r.status_code == 200
    assert r.json()["diff_threshold"] == 42


def test_tuning_rejects_out_of_range(client):
    assert client.post("/tuning", json={"diff_threshold": -1}).status_code == 422
    assert client.post("/tuning", json={"persist_frames": 0}).status_code == 422
    assert client.post("/tuning", json={"yolo_conf": 1.5}).status_code == 422


def test_open_incidents_most_dangerous_first(client):
    from app.services.fake_detections import make_fake_detection
    from app.vision.pipeline import runway_geometry

    s = client.app.state.settings
    rng = random.Random(1)
    svc = client.app.state.incidents
    for _ in range(12):
        svc.create_from_detection(make_fake_detection(runway_geometry(s), s.unknown_critical_cm, rng))
    order = {"critic": 0, "mediu": 1, "de verificat": 2}
    risks = [i["risk"] for i in client.get("/api/incidents").json()]
    assert risks == sorted(risks, key=order.get)


def test_history_partial_and_back_button(client):
    _new_incident(client)
    partial = client.get("/incidents?period=all", headers={"HX-Request": "true"})
    assert "<html" not in partial.text and partial.headers["vary"] == "HX-Request"
    restore = client.get("/incidents?period=all", headers={"HX-Request": "true", "HX-History-Restore-Request": "true"})
    assert "<html" in restore.text


def test_missing_page_is_html_for_browsers(client):
    page = client.get("/nu-exista", headers={"Accept": "text/html"})
    assert page.status_code == 404 and "Pagina nu există" in page.text
    assert client.get("/nu-exista").headers["content-type"].startswith("application/json")


def test_history_csv_export_and_detail_timings(client):
    inc = _new_incident(client)
    client.post(f"/incidents/{inc.id}/ack")
    client.post(f"/incidents/{inc.id}/resolve")
    r = client.get("/incidents.csv?period=all")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.content.decode("utf-8-sig").splitlines()
    assert lines[0].startswith("id;detectat;obiect")
    assert lines[1].startswith(inc.id) and ";Rezolvat;" in lines[1]
    assert client.get("/incidents.csv?period=all&status=false_alarm").content.decode("utf-8-sig").count("\n") == 1

    detail = client.get(f"/incidents/{inc.id}").text
    assert "Până la preluarea echipei" in detail and "Până la eliminare" in detail
    page = client.get("/incidents?period=all").text
    assert 'href="/incidents.csv?period=all"' in page

import csv
import io
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from app.services.incident_service import STATUS_LABELS
from app.templating import LOCAL_TZ, _local, fmt_seconds, templates


def fmt_datetime_full(dt) -> str:
    d = _local(dt)
    return d.strftime("%Y-%m-%d %H:%M:%S") if d else ""

router = APIRouter()

ACTION_SLUGS = {
    "confirm": "confirm",
    "false-alarm": "false_alarm",
    "dispatch": "dispatch",
    "ack": "ack",
    "resolve": "resolve",
}


@router.get("/api/incidents")
def list_incidents(request: Request, open: bool = True):
    svc = request.app.state.incidents
    items = svc.open_incidents() if open else svc.history(limit=100)
    return [i.public() for i in items]


@router.post("/incidents/{incident_id}/{action}")
def incident_action(request: Request, incident_id: str, action: str):
    if action not in ACTION_SLUGS:
        raise HTTPException(404, "Acțiune necunoscută")
    try:
        inc = request.app.state.incidents.act(incident_id, ACTION_SLUGS[action])
    except KeyError:
        raise HTTPException(404, "Incidentul nu există")
    return inc.public()


@router.get("/incidents/{incident_id}/{kind}.jpg")
def incident_image(request: Request, incident_id: str, kind: str):
    if kind not in ("crop", "snapshot"):
        raise HTTPException(404)
    data = request.app.state.incidents.image(incident_id, kind)
    if data is None:
        url = request.app.state.incidents.remote_image_url(incident_id, kind)
        if url:
            return RedirectResponse(url)
        raise HTTPException(404, "Imaginea nu există")
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


def _since(period: str) -> str:
    now = datetime.now(LOCAL_TZ)
    if period == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "hour":
        start = now - timedelta(hours=1)
    elif period == "week":
        start = now - timedelta(days=7)
    else:
        return ""
    # Stored timestamps are UTC ISO strings ("...Z"), so compare in the same format.
    return start.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _summary(items: list) -> dict:
    resolved = [(i.resolved_at - i.detected_at).total_seconds() for i in items if i.resolved_at]
    return {
        "total": len(items),
        "open": sum(1 for i in items if i.is_open),
        "critical": sum(1 for i in items if i.risk == "critic"),
        "false_alarms": sum(1 for i in items if i.status == "false_alarm"),
        "avg_resolve": fmt_seconds(sum(resolved) / len(resolved)) if resolved else None,
    }


def _filtered(request: Request, status: str, risk: str, sector: str, period: str) -> list:
    return request.app.state.incidents.history(status=status, risk=risk, sector=sector.strip(), since=_since(period))


@router.get("/incidents", response_class=HTMLResponse)
def history(request: Request, status: str = "", risk: str = "", sector: str = "", period: str = "today"):
    items = _filtered(request, status, risk, sector, period)
    filters = {"status": status, "risk": risk, "sector": sector, "period": period}
    ctx = {
        "items": items,
        "summary": _summary(items),
        "filters": filters,
        "export_query": urlencode({k: v for k, v in filters.items() if v}),
        "status_labels": STATUS_LABELS,
        "geometry": request.app.state.vision.geometry,
    }
    # htmx asks for the full page when restoring history (Back button), so only plain filter requests get the table.
    partial = request.headers.get("HX-Request") and not request.headers.get("HX-History-Restore-Request")
    name = "partials/history_table.html" if partial else "history.html"
    response = templates.TemplateResponse(request, name, ctx)
    response.headers["Vary"] = "HX-Request"
    return response


CSV_COLUMNS = [
    "id", "detectat", "obiect", "sursa", "incredere_pct", "risc", "sector", "distanta_prag_09_m",
    "abatere_ax_m", "marime_cm", "stare", "confirmat", "echipa_trimisa", "echipa_a_preluat",
    "rezolvat", "alarma_falsa", "secunde_pana_la_inchidere",
]


@router.get("/incidents.csv")
def history_csv(request: Request, status: str = "", risk: str = "", sector: str = "", period: str = "today"):
    """The filtered history as CSV for reports. `;` and a BOM so Excel (Romanian locale) opens it correctly."""
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(CSV_COLUMNS)
    for i in _filtered(request, status, risk, sector, period):
        closed = i.closed_at
        w.writerow([
            i.id, fmt_datetime_full(i.detected_at), i.object_class, i.source, round(i.confidence * 100),
            i.risk, i.sector or "", _num(i.runway_distance_m), _num(i.centerline_offset_m), _num(i.size_estimate_cm),
            i.display_status(), fmt_datetime_full(i.confirmed_at), fmt_datetime_full(i.dispatched_at),
            fmt_datetime_full(i.team_ack_at), fmt_datetime_full(i.resolved_at), fmt_datetime_full(i.false_alarm_at),
            int((closed - i.detected_at).total_seconds()) if closed else "",
        ])
    stamp = datetime.now(LOCAL_TZ).strftime("%Y%m%d-%H%M")
    return Response(
        "﻿" + buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="runwayeye-incidente-{stamp}.csv"'},
    )


def _num(value: float | None) -> str:
    # Decimal comma, as Excel expects with a Romanian locale.
    return "" if value is None else f"{value:.1f}".replace(".", ",")


def _step(start, end) -> str | None:
    return fmt_seconds((end - start).total_seconds()) if start and end else None


@router.get("/incidents/{incident_id}", response_class=HTMLResponse)
def incident_detail(request: Request, incident_id: str):
    svc = request.app.state.incidents
    inc = svc.get(incident_id)
    if inc is None:
        raise HTTPException(404, "Incidentul nu există")
    events = svc.events(incident_id)
    for e in events:
        e["offset"] = _step(inc.detected_at, datetime.fromisoformat(e["at"]))
    timings = [
        ("Până la preluarea echipei", _step(inc.detected_at, inc.team_ack_at)),
        ("Până la eliminare", _step(inc.detected_at, inc.resolved_at)),
        ("Până la alarmă falsă", _step(inc.detected_at, inc.false_alarm_at)),
    ]
    return templates.TemplateResponse(
        request,
        "partials/incident_detail.html",
        {"inc": inc, "events": events, "timings": [(label, v) for label, v in timings if v]},
    )

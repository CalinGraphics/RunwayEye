"""Pre-demo checklist: video source, model, calibration, Supabase, phone access on the local network.

Run before every presentation:   python backend/scripts/demo_check.py
For every problem it says which .env setting works around it.
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import ROOT_DIR as ROOT  # noqa: E402  (project root: data/, samples/, .env)

from app.config import get_settings  # noqa: E402
from app.services.team_alerts import lan_ips  # noqa: E402
from app.vision.capture import _open, parse_source  # noqa: E402

OK, WARN, FAIL = "  OK ", "  !! ", " NU  "
results: list[tuple[str, str, str, str]] = []


def report(status: str, name: str, detail: str, fix: str = "") -> None:
    results.append((status, name, detail, fix))
    print(f"[{status}] {name}: {detail}")
    if fix:
        print(f"         → {fix}")


def check_files(s) -> None:
    if not (ROOT / ".env").exists():
        report(WARN, ".env", "lipsește: se folosesc valorile implicite", "python backend/scripts/setup.py --skip-install")
    css = ROOT / "frontend" / "static" / "css" / "app.css"
    vendor = [ROOT / "frontend" / "static" / "vendor" / f for f in ("htmx.min.js", "alpine.min.js")]
    if not css.exists() or not all(v.exists() for v in vendor):
        report(FAIL, "Pagini", "CSS-ul sau HTMX/Alpine lipsesc", "git checkout -- frontend/static")
    else:
        report(OK, "Pagini", "CSS, HTMX și Alpine sunt locale (merg fără internet)")
    if not (ROOT / "frontend" / "static" / "sounds" / "alarm.wav").exists():
        report(FAIL, "Sunet", "alarm.wav lipsește", "git checkout -- frontend/static/sounds")


def check_video(s) -> None:
    for label, raw in (("Sursa video", s.video_source), ("Clip de rezervă", s.fallback_video)):
        spec = parse_source(raw)
        if spec.kind == "none":
            report(WARN, label, "nesetată")
            continue
        t0 = time.monotonic()
        cap = _open(spec, s.camera_width, s.camera_height)
        if cap is None:
            if label == "Sursa video":
                fb = parse_source(s.fallback_video)
                has_fb = fb.kind == "file" and Path(str(fb.target)).exists()
                fix = ("pornește automat clipul de rezervă" if has_fb else
                       "setează VIDEO_SOURCE=<cale către un clip cu macheta> sau alt index de cameră (1, 2…)")
                report(FAIL, label, f"{spec.label} nu răspunde", fix)
            else:
                report(WARN, label, f"{spec.label} lipsește",
                       "python backend/scripts/make_demo_clip.py, sau un clip filmat cu macheta în FALLBACK_VIDEO")
            continue
        ok, frame = cap.read()
        fps = cap.get(5) or 0
        cap.release()
        if ok:
            rate = f" · {fps:.0f} cadre/s" if fps > 0 else ""
            report(OK, label, f"{spec.label} · {frame.shape[1]}×{frame.shape[0]}{rate} · "
                              f"deschis în {time.monotonic() - t0:.1f} s")
        else:
            report(FAIL, label, f"{spec.label} s-a deschis, dar nu dă cadre", "verificați cablul / aplicația camerei")


def check_model(s) -> None:
    if not s.enable_yolo:
        report(WARN, "Model YOLO", "oprit (ENABLE_YOLO=false): merge doar detecția schimbărilor")
        return
    from app.vision.yolo_detector import build_detector

    t0 = time.monotonic()
    det = build_detector(s)
    if not det.loaded:
        report(FAIL, "Model YOLO", det.error or "nu se încarcă", "ENABLE_YOLO=false (detecția schimbărilor merge oricum)")
        return
    import numpy as np

    frame = np.zeros((s.frame_width * 9 // 16, s.frame_width, 3), np.uint8)
    det.detect(frame)
    t1 = time.monotonic()
    for _ in range(5):
        det.detect(frame)
    ms = (time.monotonic() - t1) / 5 * 1000
    note = f"{det.model_name} · {ms:.0f} ms/cadru pe acest laptop · încărcat în {t1 - t0:.1f} s"
    if det.provisional:
        report(WARN, "Model YOLO", f"PROVIZORIU {note}: models/best.pt lipsește",
               "models/best.pt vine cu repo-ul (git pull); altfel training/README.md")
    else:
        report(OK, "Model YOLO", note)
    if det.general is None:
        report(WARN, "Model general", "models/general.pt lipsește: păsările și animalele intră doar ca «obiect necunoscut»",
               "python backend/scripts/build_general_model.py")
    if ms > 150:
        report(WARN, "Viteză YOLO", "laptopul e lent pentru YOLO pe fiecare cadru", "YOLO_EVERY_N=3 (sau 4)")


def check_calibration(s) -> None:
    f = s.calibration_file
    if not f.exists():
        report(WARN, "Calibrare", "lipsește: alertele nu au sector și metri",
               "deschideți /calibrate și dați clic pe cele 4 colțuri ale pistei")
        return
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data["corners"]) == 4
        report(OK, "Calibrare", f"salvată {data.get('saved_at', '')[:16].replace('T', ' ')} · "
                                f"pista {s.runway_name}, {s.runway_length_m:.0f} × {s.runway_width_m:.0f} m")
    except Exception:
        report(FAIL, "Calibrare", "fișier invalid", "refaceți calibrarea în /calibrate")


def check_supabase(s) -> None:
    from app.services.supabase_service import PUBLIC_KEY_HELP, key_role

    if not s.enable_supabase or s.use_local_data:
        report(WARN, "Supabase", "oprit în .env: datele rămân doar pe acest laptop",
               "ENABLE_SUPABASE=true și USE_LOCAL_DATA=false")
        return
    if not (s.supabase_url and s.supabase_key):
        report(WARN, "Supabase", "lipsesc SUPABASE_URL sau SUPABASE_KEY: datele rămân doar pe acest laptop",
               "completați .env (supabase/schema.sql creează tabelele într-un proiect nou)")
        return
    if key_role(s.supabase_key) == "public":
        report(FAIL, "Supabase", "cheia din .env e cea publică", PUBLIC_KEY_HELP)
        return
    try:
        from supabase import ClientOptions, create_client

        client = create_client(s.supabase_url, s.supabase_key, options=ClientOptions(postgrest_client_timeout=8))
        n = client.table("incidents").select("id", count="exact").limit(1).execute().count
        # The app writes its camera row at start anyway: a real write test.
        client.table("cameras").upsert({"id": s.camera_id, "name": s.camera_name}).execute()
        report(OK, "Supabase", f"conectat, citire și scriere · {n} incidente în tabel")
    except Exception as exc:
        report(FAIL, "Supabase", f"nu răspunde: {str(exc)[:120]}",
               "ENABLE_SUPABASE=false sau USE_LOCAL_DATA=true (aplicația merge local; sincronizează când revine)")


def firewall_allows(port: int) -> bool | None:
    try:
        out = subprocess.run(["netsh", "advfirewall", "firewall", "show", "rule", "name=all", "dir=in"],
                             capture_output=True, text=True, timeout=15, encoding="oem", errors="ignore").stdout
    except Exception:
        return None
    blocks = out.split("\n\n")
    for b in blocks:
        if f"{port}" in b and ("Allow" in b or "Permite" in b) and ("Yes" in b or "Da" in b):
            return True
    return False


def check_network(s) -> None:
    ip = lan_ips()[0]
    if ip.startswith("127."):
        report(FAIL, "Rețea", "laptopul nu e conectat la nicio rețea",
               "porniți Hotspot mobil pe laptop (Setări → Rețea și internet) și conectați telefonul la el")
        return
    url = f"http://{ip}:{s.port}"
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            data = json.load(r)
        report(OK, "Server", f"pornit · telefonul deschide {url}/team (video conectat: {data.get('video_connected')})")
    except Exception:
        try:
            with socket.create_connection(("127.0.0.1", s.port), timeout=2):
                pass
            report(FAIL, "Server", f"pornit doar local, nu pe {ip}", "HOST=0.0.0.0 în .env")
        except OSError:
            report(WARN, "Server", f"nu rulează acum (portul {s.port})", "porniți start_demo.bat și rulați din nou")
    if s.host not in ("0.0.0.0", ip):
        report(FAIL, "Server", f"HOST={s.host}: telefonul nu se poate conecta", "HOST=0.0.0.0")
    fw = firewall_allows(s.port)
    if fw is False:
        report(WARN, "Firewall", f"nu am găsit o regulă care permite portul {s.port}",
               f'ca administrator: netsh advfirewall firewall add rule name="RunwayEye {s.port}" '
               f"dir=in action=allow protocol=TCP localport={s.port}")
    elif fw:
        report(OK, "Firewall", f"portul {s.port} e permis")


def main() -> None:
    s = get_settings()
    print("RunwayEye · verificare înainte de demo\n")
    check_files(s)
    check_video(s)
    check_model(s)
    check_calibration(s)
    check_supabase(s)
    check_network(s)
    fails = [r for r in results if r[0] == FAIL]
    warns = [r for r in results if r[0] == WARN]
    print()
    if fails:
        print(f"{len(fails)} probleme de rezolvat, {len(warns)} avertismente.")
        sys.exit(1)
    print(f"Totul e pregătit pentru demo. {len(warns)} avertismente.")


if __name__ == "__main__":
    main()

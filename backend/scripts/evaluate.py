"""Measure detection on annotated clips: detection rate, false alarms per hour, time to alert.

Annotations: one JSON per clip in data/eval/ (same name as the clip, .json):

    {
      "clip": "macheta_01.mp4",
      "frame_width": 960,            # width the zones were drawn at (the app's FRAME_WIDTH)
      "objects": [
        {"label": "surub", "appear_s": 4.0, "disappear_s": 20.0,
         "zone": {"x": 360, "y": 220, "w": 60, "h": 50}}
      ]
    }

`appear_s` = the moment the object is released on the runway; `disappear_s` = picked up
(optional). Optional "corners": the 4 runway corners (calibration order) at frame_width;
without them the saved calibration (data/calibration.json) is used, as in the live app. The clip is processed frame by frame (not real time), with the same engine and
settings as the live app (.env + data/tuning.json).

    python backend/scripts/evaluate.py                # all clips in data/eval/
    python backend/scripts/evaluate.py --no-yolo      # change detection only
    python backend/scripts/evaluate.py --no-calibration
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import ROOT_DIR as ROOT  # noqa: E402  (project root: data/, samples/, .env)

import cv2  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.vision.calibration import Calibration  # noqa: E402
from app.vision.capture import resize_to_width  # noqa: E402
from app.vision.engine import DetectionEngine, EngineParams, NewIncident  # noqa: E402
from app.vision.pipeline import TUNABLE, runway_geometry  # noqa: E402
from app.vision.types import COARSE_CLASSES, FOD_CLASSES, class_group  # noqa: E402

ZONE_MARGIN_PX = 15


def load_params(s) -> EngineParams:
    params = EngineParams.from_settings(s)
    try:
        for k, v in json.loads(s.tuning_file.read_text(encoding="utf-8")).items():
            if k in TUNABLE:
                setattr(params, k, type(getattr(params, k))(v))
    except (OSError, ValueError):
        pass
    return params


def clip_calibration(ann: dict, s, size: tuple[int, int], use_saved: bool) -> Calibration | None:
    geometry = runway_geometry(s)
    if ann.get("corners"):
        k = s.frame_width / ann.get("frame_width", s.frame_width)
        return Calibration([(x * k, y * k) for x, y in ann["corners"]], size, geometry)
    if use_saved:
        return Calibration.load(s.calibration_file, geometry, size)
    return None


def run_clip(path: Path, ann: dict, s, yolo, use_calibration: bool = True) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Nu pot deschide {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    engine = DetectionEngine(params=load_params(s), yolo=yolo)
    scale = s.frame_width / ann.get("frame_width", s.frame_width)
    alerts = []
    n = 0
    busy = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        frame = resize_to_width(frame, s.frame_width)
        if n == 1 and use_calibration:
            engine.set_calibration(clip_calibration(ann, s, (frame.shape[1], frame.shape[0]), use_calibration))
        t0 = time.perf_counter()
        result = engine.process(frame)
        busy += time.perf_counter() - t0
        for ev in result.events:
            if isinstance(ev, NewIncident):
                d = ev.detection
                alerts.append({"t": n / fps, "center": d.center_px, "class_key": d.class_key, "source": d.source})
    cap.release()
    duration = n / fps

    objects = []
    used = set()
    for obj in ann["objects"]:
        z = obj["zone"]
        x1, y1 = z["x"] * scale - ZONE_MARGIN_PX, z["y"] * scale - ZONE_MARGIN_PX
        x2, y2 = (z["x"] + z["w"]) * scale + ZONE_MARGIN_PX, (z["y"] + z["h"]) * scale + ZONE_MARGIN_PX
        end = obj.get("disappear_s") or duration
        match = None
        for i, a in enumerate(alerts):
            cx, cy = a["center"]
            if i not in used and obj["appear_s"] <= a["t"] <= end and x1 <= cx <= x2 and y1 <= cy <= y2:
                match = i
                break
        if match is not None:
            used.add(match)
        a = alerts[match] if match is not None else None
        objects.append({
            "label": obj["label"],
            "detected": a is not None,
            "latency_s": round(a["t"] - obj["appear_s"], 2) if a else None,
            "class_key": a["class_key"] if a else None,
            "source": a["source"] if a else None,
        })
    false_alarms = [a for i, a in enumerate(alerts) if i not in used]
    return {"clip": path.name, "about": ann.get("about", ""), "synthetic": bool(ann.get("synthetic")),
            "duration_s": round(duration, 1), "frames": n, "ms_per_frame": round(busy / max(n, 1) * 1000, 1), "objects": objects, "false_alarms": len(false_alarms),
            "false_alarm_times_s": [round(a["t"], 1) for a in false_alarms]}


def summarize(results: list[dict], yolo_name: str) -> dict:
    objs = [o for r in results for o in r["objects"]]
    detected = [o for o in objs if o["detected"]]
    lat = [o["latency_s"] for o in detected]
    hours = sum(r["duration_s"] for r in results) / 3600
    fa = sum(r["false_alarms"] for r in results)
    known = [o for o in detected if o["label"] in COARSE_CLASSES or o["label"] in FOD_CLASSES]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": yolo_name,
        "clips": len(results),
        "contains_synthetic_clips": any(r["synthetic"] for r in results),
        "objects": len(objs),
        "detected": len(detected),
        "detection_rate": round(len(detected) / len(objs), 3) if objs else None,
        "false_alarms": fa,
        "total_hours": round(hours, 3),
        "false_alarms_per_hour": round(fa / hours, 1) if hours else None,
        "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
        "latency_median_s": round(statistics.median(lat), 2) if lat else None,
        "latency_max_s": round(max(lat), 2) if lat else None,
        # Annotations use a coarse group ("surub") or an exact FOD-A class ("bolt"); a model
        # predicting "bolt" is right for both.
        "class_correct": sum(1 for o in known if o["class_key"] and o["label"] in (o["class_key"], class_group(o["class_key"]))),
        "ms_per_frame": round(statistics.mean(r["ms_per_frame"] for r in results), 1) if results else None,
        "class_total": len(known),
        "per_clip": results,
    }


def markdown(s: dict) -> str:
    def fmt(v, suffix=""):
        return "—" if v is None else f"{v}{suffix}"

    lines = [
        "# RunwayEye · rezultatele testului de detecție",
        "",
        f"Generat: {s['generated_at']} · model: {s['model']} · {s['clips']} clipuri · {s['total_hours'] * 60:.1f} min",
        "",
    ]
    if s["contains_synthetic_clips"]:
        lines += ["> **Atenție:** setul conține clipuri SINTETICE. Aceste cifre nu sunt pentru slide.", ""]
    rate = None if s["detection_rate"] is None else round(s["detection_rate"] * 100, 1)
    lines += [
        "| Indicator | Valoare |",
        "|---|---|",
        f"| Rata de detecție | {fmt(rate, ' %')} ({s['detected']}/{s['objects']}) |",
        f"| Alarme false | {s['false_alarms']} ({fmt(s['false_alarms_per_hour'], ' / oră')}) |",
        f"| Timp până la alertă (medie / mediană / maxim) | {fmt(s['latency_mean_s'], ' s')} / {fmt(s['latency_median_s'], ' s')} / {fmt(s['latency_max_s'], ' s')} |",
        f"| Clasă corectă (obiecte cunoscute) | {s['class_correct']}/{s['class_total']} |",
        f"| Timp de procesare pe cadru (pe acest laptop) | {fmt(s['ms_per_frame'], ' ms')} |",
        "",
        "Timpul până la alertă se măsoară de la momentul în care obiectul e lăsat pe pistă (inclusiv retragerea mâinii).",
        "",
        "## Pe clipuri",
        "",
        "| Clip | Ce testează | Detectate | Alarme false | Timp până la alertă |",
        "|---|---|---|---|---|",
    ]
    for r in s["per_clip"]:
        found = [o for o in r["objects"] if o["detected"]]
        lat = ", ".join(f"{o['latency_s']} s" for o in found) or "—"
        missed = [o["label"] for o in r["objects"] if not o["detected"]]
        note = f" (ratat: {', '.join(missed)})" if missed else ""
        lines.append(f"| {r['clip']} | {r['about'] or '—'} | {len(found)}/{len(r['objects'])}{note} | {r['false_alarms']} | {lat} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/eval")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--no-calibration", action="store_true", help="toată imaginea e considerată pistă")
    args = ap.parse_args()
    s = get_settings()
    folder = ROOT / args.dir
    anns = sorted(folder.glob("*.json"))
    anns = [a for a in anns if a.name != "summary.json"]
    if not anns:
        raise SystemExit(f"Nicio adnotare în {folder}. Vezi formatul în docstring-ul acestui script.")

    yolo, yolo_name = None, "fără YOLO (doar detecția schimbărilor)"
    if s.enable_yolo and not args.no_yolo:
        from app.vision.yolo_detector import build_detector

        det = build_detector(s)
        if det.loaded:
            yolo = det
            yolo_name = det.model_name + (" (provizoriu)" if det.provisional else "")

    results = []
    for a in anns:
        ann = json.loads(a.read_text(encoding="utf-8"))
        clip = folder / ann.get("clip", a.with_suffix(".mp4").name)
        print(f"· {clip.name} …", flush=True)
        results.append(run_clip(clip, ann, s, yolo, use_calibration=not args.no_calibration))

    summary = summarize(results, yolo_name)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(summary)
    (folder / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Salvat: {folder / 'summary.md'} și summary.json")


if __name__ == "__main__":
    main()

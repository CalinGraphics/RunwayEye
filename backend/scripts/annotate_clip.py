"""Annotate an evaluation clip for evaluate.py (writes data/eval/<clip>.json).

    python backend/scripts/annotate_clip.py data/eval/macheta_01.mp4

Keys in the video window:
    SPACE  pause / play          ← →  step one frame (when paused)       [ ]  jump 1 s back / forward
    a      at the moment an object is released: draw its zone, then type its label in the console
    d      at the moment the last object is picked up: mark its disappear time
    u      undo the last object       s  save and quit       q  quit without saving
"""

import json
import sys
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import ROOT_DIR as ROOT  # noqa: E402  (project root: data/, samples/, .env)

from app.config import get_settings  # noqa: E402
from app.vision.capture import resize_to_width  # noqa: E402

LABELS = ["surub", "piulita", "cheie", "metal", "plastic", "necunoscut"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    clip = Path(sys.argv[1])
    width = get_settings().frame_width
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = clip.with_suffix(".json")
    objects = json.loads(out.read_text(encoding="utf-8"))["objects"] if out.exists() else []
    paused, idx = False, 0
    frame = None
    win = "RunwayEye · adnotare"
    while True:
        if not paused or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, raw = cap.read()
            if not ok:
                paused, idx = True, max(0, total - 1)
                continue
            frame = resize_to_width(raw, width)
        view = frame.copy()
        t = idx / fps
        for o in objects:
            z = o["zone"]
            active = o["appear_s"] <= t <= (o.get("disappear_s") or 1e9)
            color = (0, 200, 255) if active else (90, 90, 90)
            cv2.rectangle(view, (z["x"], z["y"]), (z["x"] + z["w"], z["y"] + z["h"]), color, 2)
            cv2.putText(view, o["label"], (z["x"], z["y"] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(view, f"{t:6.2f} s  {'PAUZA' if paused else ''}  obiecte: {len(objects)}  (a/d/u/s/q)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKeyEx(int(1000 / fps) if not paused else 0)
        if not paused:
            idx = min(idx + 1, total - 1)
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (2555904, ord(".")):  # right arrow
            idx, paused = min(idx + 1, total - 1), True
        elif key in (2424832, ord(",")):  # left arrow
            idx, paused = max(idx - 1, 0), True
        elif key == ord("]"):
            idx = min(idx + int(fps), total - 1)
        elif key == ord("["):
            idx = max(idx - int(fps), 0)
        elif key == ord("a"):
            paused = True
            x, y, w, h = cv2.selectROI(win, frame, showCrosshair=True)
            if w and h:
                label = input(f"Eticheta ({', '.join(LABELS)}): ").strip() or "necunoscut"
                objects.append({"label": label, "appear_s": round(t, 2), "disappear_s": None,
                                "zone": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}})
        elif key == ord("d") and objects:
            objects[-1]["disappear_s"] = round(t, 2)
        elif key == ord("u") and objects:
            objects.pop()
        elif key == ord("s"):
            out.write_text(json.dumps({"clip": clip.name, "frame_width": width, "objects": objects},
                                      indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Salvat {out} ({len(objects)} obiecte)")
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

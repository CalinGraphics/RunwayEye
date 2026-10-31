import argparse
import re
import sys
import unicodedata
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.config import ROOT_DIR as ROOT  # noqa: E402  (project root: data/, samples/, .env)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.risk import assess_risk  # noqa: E402
from app.vision.types import CLASS_LABELS, class_group  # noqa: E402
from app.vision.yolo_detector import YoloDetector  # noqa: E402

TILE = 220
GREEN, ORANGE, RED = (80, 200, 80), (0, 165, 255), (60, 60, 230)


def truth_of(path: Path) -> str:
    """'bolt_nut_set_012345.jpg' -> 'bolt_nut_set'."""
    return re.sub(r"_[^_]*$", "", path.stem)


def draw(img: np.ndarray, cands, truth: str) -> tuple[np.ndarray, str]:
    out = img.copy()
    top = max(cands, key=lambda c: c.confidence) if cands else None
    verdict = "ratat" if top is None else "corect" if top.class_key == truth else \
        "grupă corectă" if class_group(top.class_key) == class_group(truth) else "greșit"
    for c in cands:
        color = GREEN if c.class_key == truth else ORANGE if class_group(c.class_key) == class_group(truth) else RED
        b = c.bbox
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 2)
    return out, verdict


def ascii_text(text: str) -> str:
    """OpenCV fonts have no diacritics: 'șurub' -> 'surub'."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def tile(img: np.ndarray, title: str, sub: str, color) -> np.ndarray:
    title, sub = ascii_text(title), ascii_text(sub)
    t = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_AREA)
    t = cv2.copyMakeBorder(t, 0, 38, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(t, title[:28], (4, TILE + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(t, sub[:28], (4, TILE + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "samples/fod-a-test")
    ap.add_argument("--out", type=Path, default=ROOT / "data/samples_result")
    ap.add_argument("--conf", type=float, help="pragul de încredere (implicit YOLO_CONF din .env)")
    args = ap.parse_args()

    s = get_settings()
    det = YoloDetector(s.path(s.yolo_model_path), None, args.conf or s.yolo_conf, s.yolo_imgsz)
    if not det.loaded:
        sys.exit(det.error or "Modelul nu s-a încărcat")
    images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        sys.exit(f"Nicio imagine în {args.dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    rows, tiles = [], []
    for p in images:
        img = cv2.imread(str(p))
        truth = truth_of(p)
        cands = det.detect(img)
        annotated, verdict = draw(img, cands, truth)
        cv2.imwrite(str(args.out / p.name), annotated)
        top = max(cands, key=lambda c: c.confidence) if cands else None
        rows.append((p.name, truth, top, verdict))
        color = {"corect": GREEN, "grupă corectă": ORANGE}.get(verdict, RED)
        sub = f"{top.label} {top.confidence:.0%}" if top else "nimic detectat"
        tiles.append(tile(annotated, CLASS_LABELS.get(truth, truth), sub, color))

    cols = 10
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(args.out / "sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    n = len(rows)
    count = {v: sum(1 for r in rows if r[3] == v) for v in ("corect", "grupă corectă", "greșit", "ratat")}
    lines = [
        "# RunwayEye · test pe imagini nevăzute la antrenare",
        "",
        f"Model: {det.model_name} · {n} imagini din `{(args.dir.relative_to(ROOT) if args.dir.is_relative_to(ROOT) else args.dir).as_posix()}`"
        f" · prag {det.conf}",
        "",
        "| Rezultat | Imagini |",
        "|---|---|",
        f"| Obiect găsit | {n - count['ratat']}/{n} ({(n - count['ratat']) / n:.0%}) |",
        f"| Clasa exactă | {count['corect']}/{n} ({count['corect'] / n:.0%}) |",
        f"| Clasa exactă sau aceeași grupă (ex. bolț / șurub) | {count['corect'] + count['grupă corectă']}/{n} |",
        f"| Risc corect (critic / mediu) | {sum(1 for _, t, top, _ in rows if top and assess_risk(top.class_key, None, 15) == assess_risk(t, None, 15))}/{n} |",
        "",
        "| Imagine | Adevărat | Detectat | Încredere | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name, truth, top, verdict in rows:
        lines.append(f"| {name} | {CLASS_LABELS.get(truth, truth)} | {top.label if top else '—'} | "
                     f"{f'{top.confidence:.0%}' if top else '—'} | {verdict} |")
    md = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Planșa cu toate imaginile: {args.out / 'sheet.jpg'}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()

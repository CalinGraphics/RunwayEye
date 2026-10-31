"""Write samples/fod-a-test/boxes.json: where the object is in each test image (for the demo clip).

The FOD-A labels are not kept with the repo, so the trained model marks the object once (it gets
all 93 images right, see test_samples.py); make_demo_clip.py cuts the objects out with these boxes.

    python backend/scripts/label_samples.py
"""

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402

from app.config import ROOT_DIR as ROOT  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.vision.yolo_detector import YoloDetector  # noqa: E402

SAMPLES = ROOT / "samples/fod-a-test"


def main() -> None:
    s = get_settings()
    det = YoloDetector(s.path(s.yolo_model_path), None, 0.25, s.yolo_imgsz)
    if not det.loaded:
        sys.exit(det.error or "Modelul nu s-a încărcat")
    boxes = {}
    for p in sorted(SAMPLES.glob("*.jpg")):
        cands = det.detect(cv2.imread(str(p)))
        if cands:
            best = max(cands, key=lambda c: c.confidence)
            boxes[p.name] = {"class_key": best.class_key, **best.bbox.model_dump()}
    (SAMPLES / "boxes.json").write_text(json.dumps(boxes, indent=1), encoding="utf-8")
    print(f"{len(boxes)} imagini cu obiect marcat -> {SAMPLES / 'boxes.json'}")


if __name__ == "__main__":
    main()

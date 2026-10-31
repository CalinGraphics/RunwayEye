"""Build models/general.pt: an open-vocabulary detector (YOLOE) for what FOD-A has no pictures of.

FOD-A teaches models/best.pt 31 kinds of hardware and baggage debris, but no birds, animals or tyre
pieces. YOLOE finds objects described in text. The prompts in app/vision/types.py (GENERAL_PROMPTS:
"bird", "dead bird", "dog", "tire fragment", "plastic bag"...) are turned into the model's classes
once, here; the saved model then runs offline like any YOLO model, on any computer.

    python scripts/build_general_model.py               # YOLOE-26s (default)
    python scripts/build_general_model.py --size n      # smaller and faster, less accurate

Needs internet once (YOLOE weights + the MobileCLIP text encoder, ~150 MB, into models/cache/).
The result is committed with the repo, so nobody else has to run this.
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision.types import CLASS_LABELS, GENERAL_PROMPTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", default="s", choices=list("nsmlx"))
    ap.add_argument("--out", type=Path, default=ROOT / "models/general.pt")
    args = ap.parse_args()

    cache = ROOT / "models/cache"  # downloads land in the current folder: keep them out of the repo root
    cache.mkdir(parents=True, exist_ok=True)
    os.chdir(cache)
    from ultralytics import YOLO, YOLOE

    base = f"yoloe-26{args.size}-seg.pt"
    model = YOLOE(base)
    names = list(GENERAL_PROMPTS)
    print(f"Clase din text ({len(names)}): {', '.join(names)}", flush=True)
    model.set_classes(names, model.get_text_pe(names))
    tmp = cache / "general.pt"
    model.save(str(tmp))
    shutil.move(str(tmp), args.out)

    # The saved model must work on its own (no text encoder): reload it and run once.
    import numpy as np

    check = YOLO(str(args.out))
    assert list(check.names.values()) == names, check.names
    frame = np.zeros((540, 960, 3), np.uint8)
    check.predict(frame, verbose=False)
    t0 = time.perf_counter()
    for _ in range(3):
        check.predict(frame, imgsz=640, verbose=False)
    ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"Gata: {args.out} ({args.out.stat().st_size / 1e6:.0f} MB, {ms:.0f} ms/cadru pe acest calculator)")
    for prompt, key in GENERAL_PROMPTS.items():
        print(f"  {prompt:16s} -> {CLASS_LABELS[key]}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    main()

"""Convert FOD-A (Pascal VOC, v2.1) to a YOLO dataset with a leak-free train/val/test split.

FOD-A: Munyer et al., "FOD-A: A Dataset for Foreign Object Debris in Airports", ICMLA 2021.
https://github.com/FOD-UNOmaha/FOD-data (Pascal VOC download, 412 MB, 300x300 images).

The images are frames of ~55 videos (one or two per class), so neighbouring frames are
near-duplicates. The official trainval/test lists are a random split per frame, which puts the
same frames in both sets and inflates mAP. Splitting whole videos is not possible either: most
classes have a single video, so they would be missing from test. Here every video is cut into
contiguous blocks of frames, the blocks are assigned to train / val / test, and GAP frames are
dropped next to every boundary between different splits, so no test frame has a near-copy in train.

    python training/prepare_fod_a.py datasets/fod-a/raw/FODPascalVOCFormat-V.2.1/VOC2007
    python training/prepare_fod_a.py <VOC2007> --out datasets/fod-a/yolo --val 0.1 --test 0.15

It also copies a few test images per class to samples/fod-a-test/ (committed with the repo),
for trying the model on /upload with pictures it never saw during training.
"""

import argparse
import json
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.vision.types import FOD_CLASSES, class_key  # noqa: E402

CLASSES = list(FOD_CLASSES)
INDEX = {key: i for i, key in enumerate(CLASSES)}
GAP = 15  # frames dropped on each side of a boundary between two splits (~0.5 s of video)


def read_voc(xml_path: Path) -> list[tuple[int, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes = []
    for obj in root.iter("object"):
        key = class_key(obj.findtext("name"))
        if key not in INDEX:
            raise SystemExit(f"{xml_path.name}: clasă necunoscută {obj.findtext('name')!r}")
        b = obj.find("bndbox")
        x1, y1 = max(0.0, float(b.findtext("xmin"))), max(0.0, float(b.findtext("ymin")))
        x2, y2 = min(w, float(b.findtext("xmax"))), min(h, float(b.findtext("ymax")))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((INDEX[key], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return boxes


def videos(frames: list[tuple[str, list]]) -> list[list[int]]:
    """Indices of consecutive frames of one video: a new video starts when no class carries over."""
    seqs: list[list[int]] = []
    prev: set[int] = set()
    for i, (_, boxes) in enumerate(frames):
        classes = {b[0] for b in boxes}
        if not seqs or not classes & prev:
            seqs.append([])
        seqs[-1].append(i)
        prev = classes or prev
    return seqs


def split_video(idx: list[int], val: float, test: float, rng: random.Random) -> dict[int, str]:
    """Frame index -> split for one video: contiguous blocks, GAP frames dropped at split changes."""
    n_blocks = max(8, min(20, len(idx) // 60))
    size = -(-len(idx) // n_blocks)
    blocks = [idx[i:i + size] for i in range(0, len(idx), size)]
    order = list(range(len(blocks)))
    rng.shuffle(order)
    n_test = max(1, round(test * len(blocks)))
    n_val = max(1, round(val * len(blocks)))
    parts = ["train"] * len(blocks)
    for b in order[:n_test]:
        parts[b] = "test"
    for b in order[n_test:n_test + n_val]:
        parts[b] = "val"

    gap = min(GAP, size // 3)  # very short videos (Tape, Screw) would otherwise lose all test frames
    out: dict[int, str] = {}
    for b, block in enumerate(blocks):
        keep = block
        # Train keeps its frames; val/test lose the ones touching a different split.
        if parts[b] != "train":
            if b > 0 and parts[b - 1] != parts[b]:
                keep = keep[gap:]
            if b + 1 < len(blocks) and parts[b + 1] != parts[b]:
                keep = keep[:-gap] if len(keep) > gap else []
        for i in keep:
            out[i] = parts[b]
    return out


def place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)  # same drive: no extra space, instant
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voc", type=Path, help="folderul VOC2007 (cu Annotations/ și JPEGImages/)")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/fod-a/yolo")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=3, help="imagini de test per clasă copiate în samples/")
    args = ap.parse_args()

    xmls = sorted((args.voc / "Annotations").glob("*.xml"))
    if not xmls:
        raise SystemExit(f"Niciun XML în {args.voc / 'Annotations'}")
    print(f"Citesc {len(xmls)} adnotări …", flush=True)
    frames = [(x.stem, read_voc(x)) for x in xmls]

    rng = random.Random(args.seed)
    seqs = videos(frames)
    split: dict[int, str] = {}
    for idx in seqs:
        split.update(split_video(idx, args.val, args.test, rng))

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)  # a previous run with another seed must not leave frames in two splits
    for part in ("train", "val", "test"):
        for sub in ("images", "labels"):
            (out / sub / part).mkdir(parents=True)
    stats = {p: {"images": 0, "boxes": Counter()} for p in ("train", "val", "test")}
    test_by_class: dict[str, list[str]] = {c: [] for c in CLASSES}
    for i, part in sorted(split.items()):
        stem, boxes = frames[i]
        place(args.voc / "JPEGImages" / f"{stem}.jpg", out / "images" / part / f"{stem}.jpg")
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in boxes]
        (out / "labels" / part / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[part]["images"] += 1
        stats[part]["boxes"].update(CLASSES[b[0]] for b in boxes)
        if part == "test" and len(boxes) == 1:
            test_by_class[CLASSES[boxes[0][0]]].append(stem)

    yaml_lines = [
        "# FOD-A v2.1 (Munyer et al., ICMLA 2021), converted by training/prepare_fod_a.py",
        "# No 'path:' on purpose: Ultralytics resolves the folders next to this file, on any machine.",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
        *[f"  {i}: {name}" for i, name in enumerate(CLASSES)],
    ]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    report = {
        "source": "FOD-A v2.1 Pascal VOC",
        "split": f"blocuri de cadre consecutive din fiecare video, {GAP} cadre eliminate la fiecare graniță",
        "seed": args.seed,
        "videos": len(seqs),
        "dropped_frames": len(frames) - len(split),
        **{p: {"images": v["images"], "boxes": dict(sorted(v["boxes"].items()))} for p, v in stats.items()},
    }
    (out / "split.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.samples:
        samples = ROOT / "samples/fod-a-test"
        if samples.exists():
            shutil.rmtree(samples)
        samples.mkdir(parents=True)
        for key, stems in test_by_class.items():
            # Spread over the test frames of the class, not neighbours.
            picks = stems[:: max(1, len(stems) // args.samples)][: args.samples]
            for stem in picks:
                shutil.copy2(args.voc / "JPEGImages" / f"{stem}.jpg", samples / f"{key}_{stem}.jpg")
        print(f"Imagini pentru demo: {samples}")

    print(f"{len(seqs)} videoclipuri, {report['dropped_frames']} cadre eliminate la granițe")
    for p in ("train", "val", "test"):
        print(f"  {p:5s} {stats[p]['images']:6d} imagini  {sum(stats[p]['boxes'].values()):6d} obiecte")
    missing = [c for c in CLASSES if not stats["test"]["boxes"][c] or not stats["val"]["boxes"][c]]
    if missing:
        print("ATENȚIE: clase fără exemple în val/test:", ", ".join(missing))
    print(f"Gata: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()

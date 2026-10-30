"""Builds the YOLO training set from FOD-A (Pascal VOC export) plus our own mock-up photos.

FOD-A: https://github.com/FOD-UNOmaha/FOD-data (MIT), "Pascal VOC format" download, ~400 MB.
Unzip it anywhere, then:

    python scripts/fod_to_yolo.py --voc D:\\FOD-A --list          # classes found and how many
    python scripts/fod_to_yolo.py --voc D:\\FOD-A --zip           # data/dataset_fod/ + dataset_fod.zip

Our own photos (exported from Roboflow as YOLOv8, same 5 class names) go in with
--extra <folder>; they matter most, because the model must see OUR mock-up and lighting.
The zip goes to the Colab notebook (training/train_colab.ipynb, variant B).
"""

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["surub", "piulita", "cheie", "metal", "plastic"]

# FOD-A category -> our class (compared lowercase, without spaces, "_" and "-").
# Anything not listed is left out of the boxes; check the names with --list.
MAPPING = {
    "bolt": "surub", "screw": "surub", "boltwasher": "surub", "boltnutset": "surub",
    "nut": "piulita",
    "wrench": "cheie", "adjustablewrench": "cheie",
    "metalpart": "metal", "metalsheet": "metal", "washer": "metal", "nail": "metal",
    "clamppart": "metal", "adjustableclamp": "metal", "sodacan": "metal",
    "plasticpart": "plastic", "fuelcap": "plastic",
}


def norm(name: str) -> str:
    return name.lower().replace(" ", "").replace("_", "").replace("-", "")


def find(voc: Path, name: str) -> Path:
    hits = [p for p in voc.rglob(name) if p.is_dir()]
    if not hits:
        sys.exit(f"Nu găsesc folderul {name} în {voc}")
    return hits[0]


def read_xml(path: Path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    boxes = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.findtext("name").strip(), x1, y1, x2, y2))
    return root.findtext("filename"), w, h, boxes


def split_lists(voc: Path) -> dict[str, str]:
    """FOD-A ships train/val lists in ImageSets/Main; we keep its split when present."""
    split = {}
    for name, target in (("train.txt", "train"), ("val.txt", "val")):
        for f in voc.rglob(name):
            for line in f.read_text(encoding="utf-8").split():
                split[line.strip()] = target
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", required=True, help="folderul dezarhivat FOD-A (format Pascal VOC)")
    ap.add_argument("--out", default="data/dataset_fod")
    ap.add_argument("--per-class", type=int, default=700, help="câte poze cel mult pentru fiecare clasă")
    ap.add_argument("--val", type=float, default=0.15, help="partea de validare, dacă FOD-A nu are liste")
    ap.add_argument("--extra", action="append", default=[], help="export Roboflow YOLOv8 cu pozele noastre")
    ap.add_argument("--list", action="store_true", help="doar afișează clasele găsite")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    voc = Path(args.voc)
    ann_dir, img_dir = find(voc, "Annotations"), find(voc, "JPEGImages")
    xmls = sorted(ann_dir.glob("*.xml"))
    print(f"{len(xmls)} adnotări în {ann_dir}")

    if args.list:
        counts = Counter(norm(n) for x in xmls for n, *_ in read_xml(x)[3])
        for name, n in counts.most_common():
            print(f"  {name:20s} {n:6d}  -> {MAPPING.get(name, '(ignorat)')}")
        return

    rng = random.Random(7)
    rng.shuffle(xmls)
    split = split_lists(voc)
    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    for part in ("train", "val"):
        (out / part / "images").mkdir(parents=True)
        (out / part / "labels").mkdir(parents=True)

    used = Counter()
    boxes_per_class = Counter()
    skipped = Counter()
    written = Counter()
    for x in xmls:
        filename, w, h, boxes = read_xml(x)
        lines, classes = [], set()
        for name, x1, y1, x2, y2 in boxes:
            cls = MAPPING.get(norm(name))
            if cls is None:
                skipped[norm(name)] += 1
                continue
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            classes.add(cls)
        # Keep the classes balanced: an image counts for its rarest class still under the cap.
        if not lines or all(used[c] >= args.per_class for c in classes):
            continue
        image = img_dir / (filename or x.with_suffix(".jpg").name)
        if not image.exists():
            image = img_dir / x.with_suffix(".jpg").name
            if not image.exists():
                continue
        for c in classes:
            used[c] += 1
        for line in lines:
            boxes_per_class[CLASSES[int(line.split()[0])]] += 1
        part = split.get(x.stem) or ("val" if rng.random() < args.val else "train")
        shutil.copy2(image, out / part / "images" / image.name)
        (out / part / "labels" / f"{image.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[part] += 1

    for extra in args.extra:
        src = Path(extra)
        n = 0
        for part_dir in ("train", "valid", "val", "test"):
            imgs = src / part_dir / "images"
            if not imgs.is_dir():
                continue
            part = "train" if part_dir in ("train", "test") else "val"
            for img in imgs.iterdir():
                label = src / part_dir / "labels" / f"{img.stem}.txt"
                if not label.exists():
                    continue
                shutil.copy2(img, out / part / "images" / f"own_{img.name}")
                shutil.copy2(label, out / part / "labels" / f"own_{img.stem}.txt")
                written[part] += 1
                n += 1
        print(f"Adăugat {n} poze proprii din {src}")

    (out / "data.yaml").write_text(
        "train: train/images\nval: val/images\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n", encoding="utf-8")

    print(f"\nScris în {out}: {written['train']} poze de antrenare, {written['val']} de validare")
    for c in CLASSES:
        print(f"  {c:8s} {used[c]:5d} poze · {boxes_per_class[c]:5d} cutii")
    if skipped:
        print("Ignorate (nu sunt în MAPPING):", ", ".join(f"{k} {v}" for k, v in skipped.most_common(8)))
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", out)
        print(f"Arhivă pentru Colab: {archive}")


if __name__ == "__main__":
    main()

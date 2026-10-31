"""Draws the demo runway clip used when there is no camera: a hand drops real objects on the runway.

The objects are photos from samples/fod-a-test/ (FOD-A test images the model never trained on), cut
out with samples/fod-a-test/boxes.json: a soda can, a cutter, a battery and pliers, named by the FOD
model, and a plastic part it does not recognise here, caught as an unknown object by the change
detector. Later a hand picks one up and another passes over the runway without leaving anything.
Without the samples it falls back to drawn shapes.

    python backend/scripts/make_demo_clip.py            # data/videos/synthetic_demo.mp4, 90 s
Only for showing the flow, not for the jury numbers.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # project root: the clip goes to data/videos/
SAMPLES = ROOT / "samples/fod-a-test"
W, H, FPS = 960, 540, 15

RUNWAY = np.array([[90, 150], [870, 150], [870, 400], [90, 400]], np.int32)

# (photo in samples/fod-a-test/, where it lands, when the hand starts bringing it, in seconds).
# Photos the detector names right on this drawn runway: on its darker asphalt some FOD-A classes
# (wrenches, bolts, rocks) get mixed up, which a model fine-tuned on the real camera would not do.
DROPS = [("soda_can_030258.jpg", (300, 225), 5.0), ("cutter_003109.jpg", (560, 320), 18.0),
         ("battery_000141.jpg", (720, 220), 31.0), ("pliers_007913.jpg", (410, 330), 44.0),
         ("plastic_part_026209.jpg", (650, 205), 57.0)]
PICK_UP = ("soda_can_030258.jpg", 70.0)  # the team removes the first object
EMPTY_PASS = (500, 78.0)  # a hand sweeps over the runway and leaves nothing
SCALE = 1.0  # FOD-A photo size: smaller and the FOD model mixes classes up, bigger looks like a vehicle


def background(rng: np.random.Generator) -> np.ndarray:
    table = rng.normal(150, 4, (H, W)).astype(np.float32)
    img = np.repeat(table[:, :, None], 3, axis=2) * np.array([0.78, 0.92, 1.0], np.float32)
    asphalt = rng.normal(72, 7, (H, W)).astype(np.float32)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [RUNWAY], 255)
    img[mask > 0] = np.repeat(asphalt[:, :, None], 3, axis=2)[mask > 0]
    img = cv2.GaussianBlur(img, (3, 3), 0).clip(0, 255).astype(np.uint8)

    white = (225, 225, 225)
    for x in range(150, 790, 60):
        cv2.rectangle(img, (x, 272), (x + 32, 278), white, -1)
    for x0 in (100, 830):
        for i in range(6):
            y = 170 + i * 37
            cv2.rectangle(img, (x0, y), (x0 + 30, y + 14), white, -1)
    cv2.putText(img, "09", (140, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, white, 3)
    cv2.putText(img, "27", (770, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.1, white, 3)
    cv2.line(img, (90, 152), (870, 152), white, 2)
    cv2.line(img, (90, 398), (870, 398), white, 2)
    return img.astype(np.float32)


def draw_bolt(img: np.ndarray, cx: int, cy: int) -> None:
    cv2.rectangle(img, (cx - 14, cy - 3), (cx + 14, cy + 3), (185, 190, 195), -1)
    cv2.rectangle(img, (cx - 18, cy - 7), (cx - 12, cy + 7), (170, 175, 180), -1)


def draw_clip(img: np.ndarray, cx: int, cy: int) -> None:
    cv2.ellipse(img, (cx, cy), (16, 6), 0, 0, 360, (60, 150, 230), 2)
    cv2.ellipse(img, (cx + 3, cy), (10, 3), 0, 0, 360, (60, 150, 230), 2)


def draw_hand(img: np.ndarray, x: float, y: float) -> None:
    skin = (120, 160, 215)
    cv2.rectangle(img, (int(x) - 22, -10), (int(x) + 22, int(y) - 30), (90, 70, 60), -1)
    cv2.ellipse(img, (int(x), int(y) - 15), (34, 42), 0, 0, 360, skin, -1)
    for dx in (-18, -6, 6, 18):
        cv2.ellipse(img, (int(x) + dx, int(y) + 22), (6, 16), 0, 0, 360, skin, -1)


def hand_path(t: float, start: float, target: tuple[int, int]) -> tuple[float, float] | None:
    """Hand comes down from the top edge, holds for a moment, then goes back up."""
    local = t - start
    if not 0 <= local <= 4.0:
        return None
    tx, ty = target
    if local < 1.5:
        k = local / 1.5
    elif local < 2.5:
        k = 1.0
    else:
        k = 1.0 - (local - 2.5) / 1.5
    return tx, -60 + (ty + 60) * k


class Cutout:
    """A real object cut out of a FOD-A photo, with soft edges, ready to paste on the runway."""

    def __init__(self, photo: Path, box: dict):
        img = cv2.imread(str(photo)).astype(np.float32)
        m = int(max(box["w"], box["h"]) * 0.18) + 4
        y1, y2 = max(0, box["y"] - m), min(img.shape[0], box["y"] + box["h"] + m)
        x1, x2 = max(0, box["x"] - m), min(img.shape[1], box["x"] + box["w"] + m)
        patch = cv2.resize(img[y1:y2, x1:x2], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
        h, w = patch.shape[:2]
        alpha = np.zeros((h, w), np.float32)
        cv2.ellipse(alpha, (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 1.0, -1)
        self.alpha = cv2.GaussianBlur(alpha, (0, 0), max(2.0, min(w, h) * 0.08))[:, :, None]
        # Only the object's difference from its own tarmac is carried over (the photo's asphalt is
        # lighter than the drawn runway): no visible patch, and the object keeps its contrast.
        k = max(3, (int(min(w, h) * 0.6) // 2) * 2 + 1)
        tarmac = cv2.medianBlur(patch.clip(0, 255).astype(np.uint8), min(k, 255)).astype(np.float32)
        self.delta = (patch - tarmac) * self.alpha
        self.patch = patch

    def paste(self, frame: np.ndarray, cx: int, cy: int) -> None:
        h, w = self.patch.shape[:2]
        x1, y1 = cx - w // 2, cy - h // 2
        if x1 < 0 or y1 < 0 or x1 + w > frame.shape[1] or y1 + h > frame.shape[0]:
            return
        region = frame[y1:y1 + h, x1:x1 + w].astype(np.float32)
        frame[y1:y1 + h, x1:x1 + w] = (region + self.delta).clip(0, 255).astype(np.uint8)


def load_cutouts() -> dict[str, Cutout]:
    """Photo name -> cutout, for the photos in DROPS that are present."""
    try:
        boxes = json.loads((SAMPLES / "boxes.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: Cutout(SAMPLES / name, boxes[name]) for name, _, _ in DROPS
            if name in boxes and (SAMPLES / name).exists()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/videos/synthetic_demo.mp4")
    ap.add_argument("--seconds", type=int, default=90)
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name and renamed at the end: an interrupted run never leaves a broken clip.
    part = out.with_name(out.stem + ".part.mp4")
    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        sys.exit(f"Nu pot scrie {out}")

    cutouts = load_cutouts()
    drawn = [draw_bolt, draw_clip]

    def draw_object(frame, i: int, key: str, cx: int, cy: int) -> None:
        if key in cutouts:
            cutouts[key].paste(frame, cx, cy)
        else:
            drawn[i % 2](frame, cx, cy)

    rng = np.random.default_rng(7)
    base = background(rng)
    pick_key, pick_start = PICK_UP
    for n in range(args.seconds * FPS):
        t = n / FPS
        frame = (base + rng.normal(0, 1.5, base.shape).astype(np.float32)).clip(0, 255).astype(np.uint8)
        for i, (key, (cx, cy), start) in enumerate(DROPS):
            picked = key == pick_key and t >= pick_start + 2.0
            if t >= start + 2.0 and not picked:
                draw_object(frame, i, key, cx, cy)
            pos = hand_path(t, start, (cx, cy))
            if pos is not None:
                if t < start + 2.0:
                    draw_object(frame, i, key, int(pos[0]), int(pos[1]) + 30)
                draw_hand(frame, *pos)
            if key == pick_key:
                pos = hand_path(t, pick_start, (cx, cy))
                if pos is not None:
                    if t >= pick_start + 2.0:
                        draw_object(frame, i, key, int(pos[0]), int(pos[1]) + 30)
                    draw_hand(frame, *pos)
        pos = hand_path(t, EMPTY_PASS[1], (EMPTY_PASS[0], 300))
        if pos is not None:
            draw_hand(frame, *pos)
        writer.write(frame)
    writer.release()
    part.replace(out)
    kind = "obiecte reale: " + ", ".join(k.rsplit("_", 1)[0] for k, _, _ in DROPS if k in cutouts) if cutouts \
        else "obiecte desenate"
    print(f"Am scris {out} ({args.seconds} s, {W}x{H}, {FPS} cadre/s, {kind})")


if __name__ == "__main__":
    main()

"""Draws a mock-up runway clip for testing without a camera: a hand drops a bolt, later a paper clip.

Run once:   python scripts/make_demo_clip.py
Then in .env:  VIDEO_SOURCE=data/videos/synthetic_demo.mp4
Only for testing the flow, not for the jury numbers.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
W, H, FPS = 960, 540, 15

RUNWAY = np.array([[90, 150], [870, 150], [870, 400], [90, 400]], np.int32)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/videos/synthetic_demo.mp4")
    ap.add_argument("--seconds", type=int, default=60)
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name and renamed at the end: an interrupted run never leaves a broken clip.
    part = out.with_name(out.stem + ".part.mp4")
    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        sys.exit(f"Nu pot scrie {out}")

    rng = np.random.default_rng(7)
    base = background(rng)
    drops = [(8.0, (330, 230), draw_bolt), (28.0, (640, 330), draw_clip)]

    for i in range(args.seconds * FPS):
        t = i / FPS
        img = base + rng.normal(0, 1.5, base.shape).astype(np.float32)
        frame = img.clip(0, 255).astype(np.uint8)
        for start, (cx, cy), draw in drops:
            if t >= start + 2.0:
                draw(frame, cx, cy)
            pos = hand_path(t, start, (cx, cy))
            if pos is not None:
                if t < start + 2.0:
                    draw(frame, int(pos[0]), int(pos[1]) + 30)
                draw_hand(frame, *pos)
        writer.write(frame)
    writer.release()
    part.replace(out)
    print(f"Am scris {out} ({args.seconds} s, {W}x{H}, {FPS} cadre/s)")


if __name__ == "__main__":
    main()
